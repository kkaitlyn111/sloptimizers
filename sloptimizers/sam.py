"""SAM: Sharpness-Aware Minimization (Foret et al., 2021, arXiv:2010.01412).

SAM minimizes the worst-case loss in an L2 ball, ``min_w max_{||e|| <= rho} L(w + e)``,
biasing training toward flat minima. Each optimizer step takes two passes::

    1. g      = grad L(w)
    2. e      = rho * g / ||g||;  w <- w + e        (ascent)
    3. g_sam  = grad L(w + e)
    4. w <- w - e                                     (restore)
    5. base optimizer step with g_sam                 (we use AdamW)

:class:`SAM` wraps any base optimizer. Use :meth:`SAM.step` with a closure
that zeroes grads, runs forward + backward and returns the loss::

    opt = SAM(model.parameters(), torch.optim.AdamW(...), rho=0.1)
    def closure():
        opt.zero_grad()
        loss = model(x, y)
        loss.backward()
        return loss
    loss = opt.step(closure)

SAM is not combined with gradient accumulation here: both passes see the same
full batch.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch import nn


class SAM:
    def __init__(self, params: Iterable[nn.Parameter], base_optimizer: torch.optim.Optimizer,
                 rho: float = 0.05):
        self.params: List[nn.Parameter] = [p for p in params if p.requires_grad]
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self._eps: List[Optional[torch.Tensor]] = []

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def ascent_step(self) -> None:
        """Move to ``w + rho * g / ||g||`` using the current gradients."""
        assert not self._eps, "ascent_step called twice without restore"
        grads = [p.grad for p in self.params if p.grad is not None]
        gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2)))
        scale = self.rho / (gnorm + 1e-12)
        for p in self.params:
            if p.grad is None:
                self._eps.append(None)
                continue
            e = p.grad * scale
            p.add_(e)
            self._eps.append(e)

    @torch.no_grad()
    def restore(self) -> None:
        """Undo :meth:`ascent_step`, returning to ``w``."""
        for p, e in zip(self.params, self._eps):
            if e is not None:
                p.sub_(e)
        self._eps.clear()

    def step(self, closure: Callable[[], torch.Tensor],
             max_grad_norm: Optional[float] = None) -> torch.Tensor:
        """Full SAM step. ``closure`` must zero grads, run forward/backward and
        return the loss. Returns the loss at the unperturbed weights.

        If ``max_grad_norm`` is set, the second-pass gradient is clipped before
        the base step (the ascent itself always has length exactly ``rho``)."""
        loss = closure()
        self.ascent_step()
        closure()
        self.restore()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        self.base_optimizer.step()
        return loss

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
