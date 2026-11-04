"""SOAP and SOAP-H (SOAP + Hyperball).

SOAP (Vyas et al., 2025, "SOAP: Improving and stabilizing Shampoo using Adam",
arXiv:2409.11321) runs Adam in the slowly rotating eigenbasis of Shampoo's
left/right preconditioners (Gupta et al., 2018) and refreshes the eigenbasis
every ``precondition_frequency`` steps with one power iteration + QR. This is a
2D-only port of the official implementation (github.com/nikhilvyas/SOAP).

SOAP-H takes the same SOAP-preconditioned direction but applies a
norm-conserving Hyperball update instead of an Adam step (no weight decay, no
bias correction), following modded-nanogpt PR #302. It was trained with a
power-law LR schedule ``lr ~ (1 - t/T)^1.25`` and no warmup.

Preconditioner statistics (L, R, Q and the rotated second moment) are kept in
float32 regardless of the parameter dtype; bf16 accumulation degrades the
eigenbasis.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from ._adam import adamw_group_step
from .hyperball import hyperball_step_


def _eigvecs_desc(m: torch.Tensor) -> torch.Tensor:
    """Eigenvectors of a symmetric matrix, ordered by descending eigenvalue."""
    m = m.float()
    eye = torch.eye(m.shape[0], device=m.device, dtype=torch.float32)
    try:
        _, q = torch.linalg.eigh(m + 1e-30 * eye)
    except RuntimeError:
        _, q = torch.linalg.eigh(m.double() + 1e-30 * eye.double())
        q = q.float()
    return torch.flip(q, dims=[1]).contiguous()


# ----------------------------------------------------------------------------
# SOAP (official algorithm)
# ----------------------------------------------------------------------------
# state["GG"] = [L or None, R or None]; state["Q"] = [QL or None, QR or None].
# A side whose dimension exceeds max_precond_dim is left unpreconditioned.

def _project(x: torch.Tensor, Q: List[Optional[torch.Tensor]]) -> torch.Tensor:
    """Rotate into the eigenbasis: ``QL^T X QR``. A ``None`` side is the identity."""
    for mat in Q:
        if mat is not None:
            x = torch.tensordot(x, mat, dims=[[0], [0]])
        else:
            x = x.permute(list(range(1, x.ndim)) + [0])
    return x


def _project_back(x: torch.Tensor, Q: List[Optional[torch.Tensor]]) -> torch.Tensor:
    for mat in Q:
        if mat is not None:
            x = torch.tensordot(x, mat, dims=[[0], [1]])
        else:
            x = x.permute(list(range(1, x.ndim)) + [0])
    return x


def _refresh_basis_qr(state: dict) -> List[Optional[torch.Tensor]]:
    """One power iteration + QR per side. Also permutes ``exp_avg_sq`` to the
    new eigenvalue order, since it is a diagonal second moment in the rotated frame."""
    exp_avg_sq = state["exp_avg_sq"]
    out: List[Optional[torch.Tensor]] = []
    for dim, (m, o) in enumerate(zip(state["GG"], state["Q"])):
        if m is None:
            out.append(None)
            continue
        order = torch.argsort(torch.diag(o.T @ m @ o), descending=True)
        exp_avg_sq = exp_avg_sq.index_select(dim, order)
        q, _ = torch.linalg.qr(m @ o[:, order])
        out.append(q.contiguous())
    state["exp_avg_sq"] = exp_avg_sq
    return out


def _update_preconditioner(g: torch.Tensor, state: dict, frequency: int) -> None:
    """EMA the Kronecker factors with the new gradient; build or refresh Q and
    carry ``exp_avg`` across the basis change."""
    if state["Q"] is not None:
        state["exp_avg"] = _project_back(state["exp_avg"], state["Q"])
    beta = state["shampoo_beta"]
    for dim, gg in enumerate(state["GG"]):
        if gg is not None:
            outer = g @ g.T if dim == 0 else g.T @ g
            gg.lerp_(outer, 1.0 - beta)
    if state["Q"] is None:
        state["Q"] = [None if gg is None else _eigvecs_desc(gg) for gg in state["GG"]]
    if state["step"] > 0 and state["step"] % frequency == 0:
        state["Q"] = _refresh_basis_qr(state)
    if state["step"] > 0:
        state["exp_avg"] = _project(state["exp_avg"], state["Q"])


class SOAPAdamW(torch.optim.Optimizer):
    """SOAP on body matrices, AdamW on everything else.

    Args:
        soap_lr: learning rate of the SOAP group.
        adamw_lr: learning rate of the AdamW group (embeddings, norms, biases).
        betas: SOAP Adam betas (official default ``(0.95, 0.95)``).
        shampoo_beta: EMA coefficient of the Kronecker factors; ``-1`` uses ``betas[1]``.
        weight_decay: decoupled weight decay on the SOAP group.
        precondition_frequency: eigenbasis refresh period ``f``.
        max_precond_dim: sides larger than this are left unpreconditioned.
    """

    def __init__(
        self,
        body_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        *,
        soap_lr: float = 3e-3,
        adamw_lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        correct_bias: bool = True,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        param_groups = [
            dict(params=[p for p in body_params if p.requires_grad], lr=soap_lr,
                 kind="soap", betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                 weight_decay=weight_decay,
                 precondition_frequency=precondition_frequency,
                 max_precond_dim=max_precond_dim, correct_bias=correct_bias),
            dict(params=[p for p in adamw_params if p.requires_grad], lr=adamw_lr,
                 kind="adamw", betas=adamw_betas, eps=adamw_eps, weight_decay=0.0),
        ]
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "soap":
                self._step_soap(group)
            else:
                adamw_group_step(self, group)
        return loss

    def _step_soap(self, group: dict) -> None:
        b1, b2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        freq = group["precondition_frequency"]
        shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else b2
        for p in group["params"]:
            if p.grad is None:
                continue
            assert p.ndim == 2, "SOAP here is defined for 2D matrices only"
            g = p.grad.float()
            state = self.state[p]
            if "Q" not in state:
                # First step: build the basis from the first gradient, then skip.
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
                state["GG"] = [
                    None if d > group["max_precond_dim"]
                    else torch.zeros(d, d, device=g.device, dtype=torch.float32)
                    for d in g.shape
                ]
                state["Q"] = None
                state["shampoo_beta"] = shampoo_beta
                _update_preconditioner(g, state, freq)
                continue

            g_rot = _project(g, state["Q"])
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(b1).add_(g_rot, alpha=1.0 - b1)
            exp_avg_sq.mul_(b2).add_(g_rot.square(), alpha=1.0 - b2)
            step_size = lr
            if group["correct_bias"]:
                t = state["step"]
                step_size = lr * (1.0 - b2 ** t) ** 0.5 / (1.0 - b1 ** t)
            update = _project_back(exp_avg / exp_avg_sq.sqrt().add_(eps), state["Q"])
            p.add_(update.to(p.dtype), alpha=-step_size)
            if wd > 0.0:
                p.add_(p, alpha=-(lr * wd))
            _update_preconditioner(g, state, freq)


# ----------------------------------------------------------------------------
# SOAP-H
# ----------------------------------------------------------------------------

class SOAPHyperball(torch.optim.Optimizer):
    """SOAP-preconditioned direction + Hyperball update on body matrices,
    plain Adam on everything else. No weight decay and no bias correction.

    Differences from :class:`SOAPAdamW`: the eigenbasis is seeded from the
    first gradient and used immediately; the numerator is the momentum in the
    original frame re-projected each step; the basis is refreshed every
    ``q_freq`` steps without re-ordering the second moment.
    """

    def __init__(
        self,
        body_params: Iterable[nn.Parameter],
        adam_params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-2,
        adam_lr: float = 1e-3,
        momentum: float = 0.95,
        ema_beta: float = 0.9,
        eps: float = 1e-8,
        q_freq: int = 1,
        max_precond_dim: int = 10000,
        radius_mult: float = 1.0,
        hyperball_eps: float = 1e-10,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        param_groups = [
            dict(params=[p for p in body_params if p.requires_grad], lr=lr,
                 kind="soaph", momentum=momentum, ema_beta=ema_beta, eps=eps,
                 q_freq=q_freq, max_precond_dim=max_precond_dim,
                 radius_mult=radius_mult, hyperball_eps=hyperball_eps),
            dict(params=[p for p in adam_params if p.requires_grad], lr=adam_lr,
                 kind="adam", betas=adam_betas, eps=adam_eps, weight_decay=0.0),
        ]
        super().__init__(param_groups, defaults={})
        for p in self.param_groups[0]["params"]:
            self.state[p]["radius"] = p.detach().float().norm().clone()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "soaph":
                self._step_soaph(group)
            else:
                adamw_group_step(self, group)
        return loss

    @staticmethod
    def _init_state(g: torch.Tensor, state: dict, max_dim: int) -> None:
        m, n = g.shape
        gf = g.float()
        state["step"] = 0
        state["momentum_buffer"] = torch.zeros_like(g)
        state["square_buffer"] = torch.zeros(m, n, dtype=torch.float32, device=g.device)
        state["left"] = m <= max_dim
        state["right"] = n <= max_dim
        state["L"] = torch.zeros(m, m, dtype=torch.float32, device=g.device) if state["left"] else None
        state["R"] = torch.zeros(n, n, dtype=torch.float32, device=g.device) if state["right"] else None
        state["QL"] = _eigvecs_desc(gf @ gf.T) if state["left"] else None
        state["QR"] = _eigvecs_desc(gf.T @ gf) if state["right"] else None

    def _step_soaph(self, group: dict) -> None:
        mom, beta, eps = group["momentum"], group["ema_beta"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            assert p.ndim == 2, "SOAP-H here is defined for 2D matrices only"
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                self._init_state(g, state, group["max_precond_dim"])
            state["step"] += 1

            state["momentum_buffer"].lerp_(g, 1.0 - mom)
            gf = g.float()
            if state["left"]:
                state["L"].mul_(beta).addmm_(gf, gf.T, alpha=1.0 - beta)
            if state["right"]:
                state["R"].mul_(beta).addmm_(gf.T, gf, alpha=1.0 - beta)

            def rotate(x: torch.Tensor) -> torch.Tensor:
                x = x.float()
                if state["left"]:
                    x = state["QL"].T @ x
                if state["right"]:
                    x = x @ state["QR"]
                return x

            g_rot = rotate(g)
            state["square_buffer"].mul_(beta).addcmul_(g_rot, g_rot, value=1.0 - beta)
            u = rotate(state["momentum_buffer"]) / state["square_buffer"].sqrt().add(eps)
            if state["left"]:
                u = state["QL"] @ u
            if state["right"]:
                u = u @ state["QR"].T

            if state["step"] % group["q_freq"] == 0:
                if state["left"]:
                    state["QL"] = torch.linalg.qr(state["L"] @ state["QL"])[0].contiguous()
                if state["right"]:
                    state["QR"] = torch.linalg.qr(state["R"] @ state["QR"])[0].contiguous()

            hyperball_step_(p, u, state["radius"] * group["radius_mult"],
                            group["lr"], group["hyperball_eps"])
