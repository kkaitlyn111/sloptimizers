"""AdamL1: Adam with a decoupled L1 (soft-threshold) penalty in place of L2.

After the Adam update, each parameter in a group with ``l1_lambda > 0`` is
passed through the proximal operator of the L1 norm::

    w <- sign(w) * max(|w| - lr * l1_lambda, 0)

The threshold scales with the scheduled learning rate (like AdamW's decoupled
decay), and weights with ``|w| <= lr * l1_lambda`` become exactly zero. There is
no L2 term. Set ``l1_lambda`` per group (body matrices > 0, embeddings / norms
/ biases = 0).
"""
from __future__ import annotations

from typing import Tuple

import torch


class AdamL1(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 l1_lambda: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, l1_lambda=l1_lambda))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (b1, b2), lr, eps = group["betas"], group["lr"], group["eps"]
            lam = group["l1_lambda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v.sqrt() / (1 - b2 ** t) ** 0.5).add_(eps)
                p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
                if lam != 0.0:
                    p.copy_(torch.sign(p) * (p.abs() - lr * lam).clamp_min(0.0))
        return loss
