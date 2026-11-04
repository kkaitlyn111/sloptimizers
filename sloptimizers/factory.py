"""One-call construction of every optimizer in the package for a given model.

``build_optimizer(name, model, **overrides)`` splits the model's parameters
with :func:`split_body_params` and returns a ready optimizer. The defaults are
the best hyperparameters we found for our smallest (40M) model on a 50M-token
data budget; they are a reasonable starting point, not universal values.

SAM and Nexus are returned as wrappers around AdamW and need their own loop
calls (see ``examples/train_tiny.py``); everything else is a standard
``torch.optim.Optimizer``. The schedulers in ``torch.optim.lr_scheduler``
(e.g. ``LambdaLR``) scale every param group's ``lr``, including the separate
body / non-body learning rates.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .adamhd import AdamHD
from .adaml1 import AdamL1
from .hyperball import AdamHyperball, MuonEqHyperball
from .muoneq import MuonEqAdamW
from .nexus import Nexus
from .param_groups import split_body_params
from .sam import SAM
from .soap import SOAPAdamW, SOAPHyperball

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "adamw":    dict(lr=3e-3, weight_decay=0.8),
    "muoneq-r": dict(muon_lr=1e-2, adamw_lr=1e-3, weight_decay=0.4),
    "muonh":    dict(muon_lr=3e-2, adam_lr=1e-3),
    "adamh":    dict(lr=1e-2),
    "soap":     dict(soap_lr=6e-3, adamw_lr=1e-3, weight_decay=0.4, precondition_frequency=4),
    "soap-h":   dict(lr=3e-2, adam_lr=1e-3),
    "sam":      dict(lr=3e-3, weight_decay=0.8, rho=0.1),
    "nexus":    dict(lr=3e-3, weight_decay=0.8, gamma=1.5e-3),
    "adamhd":   dict(lr=3e-3, weight_decay=0.8, huber_c=16.0),
    "adaml1":   dict(lr=3e-3, l1_lambda=0.05),
}
OPTIMIZERS = tuple(DEFAULTS)


def _adamw(body, other, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
    return torch.optim.AdamW(
        [dict(params=body, weight_decay=weight_decay), dict(params=other, weight_decay=0.0)],
        lr=lr, betas=betas, eps=eps)


def build_optimizer(name: str, model: nn.Module, **overrides: Any):
    """Build optimizer ``name`` (one of :data:`OPTIMIZERS`) for ``model``."""
    name = name.lower()
    if name not in DEFAULTS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMIZERS}")
    hp = {**DEFAULTS[name], **overrides}
    body, other = split_body_params(model)

    if name == "adamw":
        return _adamw(body, other, **hp)
    if name == "muoneq-r":
        return MuonEqAdamW(body, other, **hp)
    if name == "muonh":
        return MuonEqHyperball(body, other, **hp)
    if name == "adamh":
        return AdamHyperball(body, other, **hp)
    if name == "soap":
        return SOAPAdamW(body, other, **hp)
    if name == "soap-h":
        return SOAPHyperball(body, other, **hp)
    if name == "sam":
        rho = hp.pop("rho")
        return SAM(body + other, _adamw(body, other, **hp), rho=rho)
    if name == "nexus":
        gamma = hp.pop("gamma")
        return Nexus(body + other, _adamw(body, other, **hp), gamma=gamma)
    if name == "adamhd":
        wd = hp.pop("weight_decay")
        return AdamHD([dict(params=body, weight_decay=wd),
                       dict(params=other, weight_decay=0.0)], **hp)
    if name == "adaml1":
        lam = hp.pop("l1_lambda")
        return AdamL1([dict(params=body, l1_lambda=lam),
                       dict(params=other, l1_lambda=0.0)], **hp)
    raise AssertionError(name)
