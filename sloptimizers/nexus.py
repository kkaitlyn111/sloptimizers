"""Nexus (Chen et al., 2026, "Nexus: Same pretraining loss, better downstream
generalization via common minima"), Algorithm 3.

Within each outer step, an inner copy of the weights takes ``K`` normalized-SGD
steps, one per micro-batch (no inner momentum, per-tensor normalization). The
displacement of the inner copy is then fed to an outer optimizer (AdamW) as a
pseudo-gradient::

    snapshot <- w
    for k in 1..K:  w <- w - gamma * g_k / ||g_k||      (per tensor)
    grad     <- snapshot - w;  w <- snapshot
    outer_optimizer.step()

This maximizes gradient similarity across micro-batches, with an implicit
second-order regularization coefficient ``(K - 1) / (4K)``; ``K = 1`` reduces to
plain normalized SGD feeding AdamW. ``gamma`` (inner learning rate) is the only
Nexus-specific hyperparameter.

Usage, with ``K`` micro-batches per outer step::

    opt = Nexus(model.parameters(), torch.optim.AdamW(...), gamma=0.01)
    for x, y in micro_batches:          # K of them
        model(x, y).backward()
        opt.inner_step()                # NSGD step + zero_grad
    opt.step()                          # pseudo-gradient -> AdamW
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn


class Nexus:
    def __init__(self, params: Iterable[nn.Parameter], base_optimizer: torch.optim.Optimizer,
                 gamma: float = 0.01):
        self.params: List[nn.Parameter] = [p for p in params if p.requires_grad]
        self.base_optimizer = base_optimizer
        self.gamma = float(gamma)
        self._snapshot = [p.detach().clone() for p in self.params]

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def inner_step(self) -> None:
        """One normalized-SGD step on the inner weights, then clear the gradients."""
        for p in self.params:
            if p.grad is None:
                continue
            gnorm = p.grad.float().norm().item()
            if gnorm > 0.0 and gnorm == gnorm:
                p.add_(p.grad, alpha=-self.gamma / gnorm)
            p.grad = None

    @torch.no_grad()
    def step(self, max_grad_norm: Optional[float] = None) -> None:
        """Write ``snapshot - inner`` into ``.grad``, restore the snapshot,
        step the outer optimizer and start a new inner cycle. If
        ``max_grad_norm`` is set, the pseudo-gradient is clipped first."""
        for snap, p in zip(self._snapshot, self.params):
            p.grad = snap - p
            p.copy_(snap)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad(set_to_none=True)
        for snap, p in zip(self._snapshot, self.params):
            snap.copy_(p)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
