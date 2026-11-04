"""Hyperball: replace weight decay with a fixed Frobenius-norm constraint.

Hyperball wraps a base update direction ``u_t`` and keeps every body matrix on
the sphere of its initial norm ``R = ||W_0||_F``::

    W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t)),   Normalize(X) = X / ||X||_F

The learning rate becomes a *relative* step size (the step is ``lr`` times the
weight norm regardless of ``||u_t||``), and ordinary weight decay is no longer
used on those matrices. Embeddings, norms and biases stay on plain Adam.

Instantiations in this package:
  * :class:`AdamHyperball`  ("AdamH"): Adam direction.
  * :class:`MuonEqHyperball` ("MuonH"): MuonEq-R direction.
  * :class:`sloptimizers.soap.SOAPHyperball` ("SOAP-H"): SOAP direction.

The SOAP-H recipe follows modded-nanogpt PR #302 (SOAP + Hyperball). The
Hyperball idea of decoupling a weight's scale from its directional learning
speed is related to analyses of weight decay as an effective-learning-rate
mechanism (Zhang et al., 2018, "Three mechanisms of weight decay
regularization").
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

from ._adam import adam_direction, adamw_group_step
from .muoneq import muoneq_direction


@torch.no_grad()
def hyperball_step_(p: torch.Tensor, update: torch.Tensor, radius: torch.Tensor,
                    lr: float, eps: float = 1e-12) -> None:
    """In-place Hyperball update of ``p`` toward ``-update`` on the radius-``R`` sphere."""
    u_norm = update.float().norm()
    if u_norm.item() != 0.0:
        p.add_(update.to(p.dtype), alpha=-float(lr * radius / (u_norm + eps)))
    p.mul_((radius / (p.float().norm() + eps)).to(p.dtype))


class _HyperballBase(torch.optim.Optimizer):
    """Two groups: ``kind="hyperball"`` body matrices and ``kind="adam"`` others."""

    def __init__(self, body_group: dict, adam_params: Iterable[nn.Parameter],
                 adam_lr: float, adam_betas: Tuple[float, float], adam_eps: float,
                 radius_mult: float, hyperball_eps: float):
        body_group.update(kind="hyperball", radius_mult=radius_mult,
                          hyperball_eps=hyperball_eps)
        body_group["params"] = [p for p in body_group["params"] if p.requires_grad]
        adam_group = dict(params=[p for p in adam_params if p.requires_grad],
                          lr=adam_lr, kind="adam", betas=adam_betas, eps=adam_eps,
                          weight_decay=0.0)
        super().__init__([body_group, adam_group], defaults={})
        # Record R = ||W_0||_F once, at construction.
        for p in self.param_groups[0]["params"]:
            self.state[p]["radius"] = p.detach().float().norm().clone()

    def _direction(self, p: torch.Tensor, group: dict) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "adam":
                adamw_group_step(self, group)
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                radius = state["radius"] * group["radius_mult"]
                hyperball_step_(p, self._direction(p, group), radius,
                                group["lr"], group["hyperball_eps"])
        return loss


class AdamHyperball(_HyperballBase):
    """AdamH: Adam direction + Hyperball on body matrices, plain Adam elsewhere."""

    def __init__(self, body_params: Iterable[nn.Parameter],
                 adam_params: Iterable[nn.Parameter], *, lr: float = 1e-2,
                 adam_lr: float | None = None,
                 betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 radius_mult: float = 1.0, hyperball_eps: float = 1e-12):
        super().__init__(dict(params=list(body_params), lr=lr, betas=betas, eps=eps),
                         adam_params, adam_lr if adam_lr is not None else lr,
                         betas, eps, radius_mult, hyperball_eps)

    def _direction(self, p, group):
        return adam_direction(p, self.state[p], group["betas"], group["eps"])


class MuonEqHyperball(_HyperballBase):
    """MuonH: MuonEq-R direction + Hyperball on body matrices, plain Adam elsewhere."""

    def __init__(self, body_params: Iterable[nn.Parameter],
                 adam_params: Iterable[nn.Parameter], *, muon_lr: float = 3e-2,
                 adam_lr: float = 1e-3, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 equilibrate_mode: str = "row",
                 equilibrate_exponent: float = -0.5,
                 adam_betas: Tuple[float, float] = (0.9, 0.95),
                 adam_eps: float = 1e-8, radius_mult: float = 1.0,
                 hyperball_eps: float = 1e-12):
        super().__init__(dict(params=list(body_params), lr=muon_lr,
                              momentum=momentum, nesterov=nesterov,
                              ns_steps=ns_steps, mode=equilibrate_mode,
                              exponent=equilibrate_exponent),
                         adam_params, adam_lr, adam_betas, adam_eps,
                         radius_mult, hyperball_eps)

    def _direction(self, p, group):
        return muoneq_direction(p, self.state[p], group["momentum"],
                                group["nesterov"], group["ns_steps"],
                                group["mode"], group["exponent"], 1e-8)
