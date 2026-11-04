"""Shared Adam helpers used for the non-matrix parameter groups."""
from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def adam_direction(p: torch.Tensor, state: dict, betas: Tuple[float, float],
                   eps: float) -> torch.Tensor:
    """Bias-corrected Adam direction ``m_hat / (sqrt(v_hat) + eps)`` for ``p.grad``.

    Updates ``state`` in place (``step``, ``exp_avg``, ``exp_avg_sq``). Returns
    a new tensor; no learning rate or weight decay is applied.
    """
    b1, b2 = betas
    g = p.grad
    if "step" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
    state["step"] += 1
    t = state["step"]
    m, v = state["exp_avg"], state["exp_avg_sq"]
    m.lerp_(g, 1 - b1)
    v.lerp_(g.square(), 1 - b2)
    denom = v.sqrt().div_((1 - b2 ** t) ** 0.5).add_(eps)
    return m.div(1 - b1 ** t).div_(denom)


@torch.no_grad()
def adamw_group_step(opt: torch.optim.Optimizer, group: dict) -> None:
    """AdamW step (decoupled weight decay) over one param group of ``opt``."""
    lr = group["lr"]
    wd = group.get("weight_decay", 0.0)
    for p in group["params"]:
        if p.grad is None:
            continue
        update = adam_direction(p, opt.state[p], group["betas"], group["eps"])
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        p.add_(update, alpha=-lr)
