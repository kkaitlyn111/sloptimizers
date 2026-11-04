"""MuonEq-R: Muon with row equilibration before orthogonalization.

Muon (Jordan et al., 2024; analysed in Chen, Li & Liu, 2025, "Muon optimizes
under spectral norm constraints") replaces the momentum matrix of a weight
matrix with its nearest semi-orthogonal matrix, computed with a few
Newton-Schulz iterations. MuonEq (Chang et al., 2026, "MuonEq: Balancing before
orthogonalization with lightweight equilibration") first rescales the rows
(and/or columns) of the momentum to unit norm, a cheap zeroth-order
whitening that improves finite-step Newton-Schulz. We use the row variant
("MuonEq-R"), following the reference implementation at
https://github.com/MaeChd/muon-eq.

Per step, for each body matrix::

    buf     <- mu * buf + g
    u       <- g + mu * buf                (Nesterov)
    u       <- RowNormalize(u)
    O       <- NewtonSchulz5(u)
    W       <- (1 - lr * wd) * W - lr * 0.2 * sqrt(max(m, n)) * O

All other parameters (embeddings, norms, biases) are updated with AdamW.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import nn

from ._adam import adamw_group_step


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor of ``G`` with quintic Newton-Schulz.

    Coefficients (3.4445, -4.7750, 2.0315) follow Keller Jordan's Muon. Runs in
    bfloat16 on the shorter side of the matrix.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def _scale(sq_sum: torch.Tensor, exponent: float, eps: float) -> torch.Tensor:
    return (sq_sum + eps).pow(exponent)


def equilibrate(g: torch.Tensor, mode: str = "row", exponent: float = -0.5,
                eps: float = 1e-8) -> torch.Tensor:
    """MuonEq equilibration of a 2D matrix.

    ``row``: divide each row by its L2 norm (with ``exponent=-0.5``).
    ``col``: same for columns. ``rowcol``: ``D_r^e G D_c^e`` (reference uses
    ``exponent=-0.25`` here). ``none``: identity, which recovers plain Muon.
    """
    if mode == "none":
        return g
    sq = g.square()
    if mode == "row":
        return g * _scale(sq.sum(dim=1), exponent, eps)[:, None]
    if mode == "col":
        return g * _scale(sq.sum(dim=0), exponent, eps)[None, :]
    if mode == "rowcol":
        return (g * _scale(sq.sum(dim=1), exponent, eps)[:, None]
                * _scale(sq.sum(dim=0), exponent, eps)[None, :])
    raise ValueError(f"unknown equilibrate mode {mode!r}")


@torch.no_grad()
def muoneq_direction(p: torch.Tensor, state: dict, momentum: float,
                     nesterov: bool, ns_steps: int, mode: str,
                     exponent: float, eps: float) -> torch.Tensor:
    """Momentum + equilibration + Newton-Schulz direction for ``p.grad``."""
    g = p.grad
    if g.ndim > 2:
        g = g.view(g.size(0), -1)
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(g)
    buf = state["momentum_buffer"]
    buf.mul_(momentum).add_(g)
    u = g.add(buf, alpha=momentum) if nesterov else buf
    u = equilibrate(u, mode, exponent, eps)
    return zeropower_via_newtonschulz5(u, steps=ns_steps).view_as(p)


class MuonEqAdamW(torch.optim.Optimizer):
    """MuonEq-R on body matrices, AdamW on everything else.

    Args:
        muon_params: 2D body matrices (see :func:`split_body_params`).
        adamw_params: embeddings, LM head, norm gains, biases.
        muon_lr: base learning rate of the Muon group. The applied step is
            scaled by ``0.2 * sqrt(max(m, n))`` per matrix, as in the reference.
        adamw_lr: learning rate of the AdamW group.
        weight_decay: decoupled weight decay on the Muon group.
        adamw_weight_decay: decoupled weight decay on the AdamW group
            (default 0: embeddings and norms are not decayed).
    """

    def __init__(
        self,
        muon_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        *,
        muon_lr: float = 1e-2,
        adamw_lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        equilibrate_mode: str = "row",
        equilibrate_exponent: float = -0.5,
        equilibrate_eps: float = 1e-8,
        weight_decay: float = 0.0,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ):
        param_groups = [
            dict(params=[p for p in muon_params if p.requires_grad], lr=muon_lr,
                 kind="muon", momentum=momentum, nesterov=nesterov,
                 ns_steps=ns_steps, mode=equilibrate_mode,
                 exponent=equilibrate_exponent, eq_eps=equilibrate_eps,
                 weight_decay=weight_decay),
            dict(params=[p for p in adamw_params if p.requires_grad], lr=adamw_lr,
                 kind="adamw", betas=adamw_betas, eps=adamw_eps,
                 weight_decay=adamw_weight_decay),
        ]
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "muon":
                self._step_muon(group)
            else:
                adamw_group_step(self, group)
        return loss

    def _step_muon(self, group: dict) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            u = muoneq_direction(p, self.state[p], group["momentum"],
                                 group["nesterov"], group["ns_steps"],
                                 group["mode"], group["exponent"],
                                 group["eq_eps"])
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            m, n = p.shape[0], p[0].numel()
            p.add_(u.to(p.dtype), alpha=-lr * 0.2 * math.sqrt(max(m, n)))
