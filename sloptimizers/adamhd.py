"""AdamHD: Adam with decoupled Huber decay (Guo & Fan, 2025, "AdamHD:
Decoupled Huber decay regularization for language model pre-training").

The Adam moments are exactly AdamW's. Only the decoupled regularization step
changes: instead of AdamW's shrink ``w <- w / (1 + tau)``-like step, AdamHD
applies the proximal operator of a Huber penalty with threshold ``delta``::

    y   = w - lr * m_hat / (sqrt(v_hat) + eps)        (plain Adam iterate)
    tau = lr * weight_decay
    w   = y / (1 + tau)                  if |y| <= (1 + tau) * delta    (l2-like)
    w   = y - tau * delta * sign(y)      otherwise                       (l1-like)

``delta = c * mean(|w|)`` (an EMA of it by default), so ``c`` interpolates
between l1-like behaviour (small ``c``) and l2-like behaviour (large ``c``).
As with AdamW, give embeddings / norms / biases their own group with
``weight_decay=0``.
"""
from __future__ import annotations

from typing import Tuple

import torch


class AdamHD(torch.optim.Optimizer):
    def __init__(self, params, *, lr: float = 3e-4,
                 betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1, huber_c: float = 8.0,
                 huber_beta0: float = 0.99, delta_mode: str = "ema"):
        if delta_mode not in ("ema", "mean"):
            raise ValueError(f"delta_mode must be 'ema' or 'mean', got {delta_mode!r}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        huber_c=huber_c, huber_beta0=huber_beta0, delta_mode=delta_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["mu_abs"] = torch.zeros((), device=p.device, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = v.sqrt().div_((1 - b2 ** t) ** 0.5).add_(eps)
                y = p.addcdiv(m, denom, value=-lr / (1 - b1 ** t))

                if wd == 0.0:
                    p.copy_(y)
                    continue

                mean_abs = p.abs().mean().float()
                if group["delta_mode"] == "ema":
                    state["mu_abs"].mul_(group["huber_beta0"]).add_(
                        mean_abs, alpha=1 - group["huber_beta0"])
                    delta = group["huber_c"] * state["mu_abs"]
                else:
                    delta = group["huber_c"] * mean_abs

                tau = lr * wd
                small = y.abs() <= (1.0 + tau) * delta
                p.copy_(torch.where(small, y / (1.0 + tau), y - tau * delta * y.sign()))
        return loss
