"""Generalization measures for trained models.

Implements measures from Jiang et al., "Fantastic Generalization Measures and
Where to Find Them" (ICLR 2020, arXiv:1912.02178), adapted to language models
(the "error" is next-token cross-entropy):

  norm        products/sums of spectral and Frobenius norms, distance from init
  spectral    stable rank, effective rank (Roy & Vetterli, 2007), heavy-tailed
              power-law exponent alpha of each weight matrix's eigenvalue spectrum
              (Martin & Mahoney, 2021)
  fisher_rao  Fisher-Rao norm (Liang et al., 2019)
  sharpness   one-step m-sharpness (Foret et al., 2021) and its magnitude-aware
              (adaptive) variant (Kwon et al., 2021)
  pac_bayes   PAC-Bayes bounds with a Gaussian or magnitude-aware posterior

Norm and spectral measures use the 2D body matrices (see ``split_body_params``);
the others use all trainable parameters.

    from sloptimizers.measures import compute_measures
    m = compute_measures(model, batches, init_model=init, n_train=50_000_000)

``batches`` is a list of inputs for ``loss_fn(model, batch) -> scalar loss``
(default: ``model(*batch)``).
"""
from __future__ import annotations

import copy
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .param_groups import split_body_params

ALL_MEASURES = ("norm", "spectral", "fisher_rao", "sharpness", "pac_bayes")


def _default_loss(model, batch):
    return model(*batch)


def _body_matrices(model: nn.Module) -> Dict[str, torch.Tensor]:
    body = {id(p) for p in split_body_params(model)[0]}
    return {n: p.detach().float() for n, p in model.named_parameters() if id(p) in body}


def _params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def _mean_loss(model, batches, loss_fn) -> float:
    return float(np.mean([float(loss_fn(model, b)) for b in batches]))


# ----------------------------------------------------------------------------
# norm and spectral
# ----------------------------------------------------------------------------

def powerlaw_alpha(eigs: np.ndarray, n_xmin: int = 60) -> float:
    """MLE power-law exponent of the tail of ``eigs``, with x_min chosen by KS distance."""
    x = np.sort(eigs[eigs > 0])
    if x.size < 10:
        return float("nan")
    cands = np.unique(x[:-1])
    if cands.size > n_xmin:
        cands = cands[np.linspace(0, cands.size - 1, n_xmin).astype(int)]
    best_ks, best_alpha = np.inf, float("nan")
    for xmin in cands:
        tail = x[x >= xmin]
        slog = np.sum(np.log(tail / xmin))
        if tail.size < 5 or slog <= 0:
            continue
        alpha = 1.0 + tail.size / slog
        cdf_emp = np.arange(1, tail.size + 1) / tail.size
        ks = np.max(np.abs(cdf_emp - (1.0 - (tail / xmin) ** (1.0 - alpha))))
        if ks < best_ks:
            best_ks, best_alpha = ks, float(alpha)
    return best_alpha


def norm_measures(model: nn.Module, init_model: Optional[nn.Module] = None) -> dict:
    mats = _body_matrices(model)
    log_spec2, log_fro2, fro2, fro_over_spec = [], [], [], []
    for W in mats.values():
        s_max = torch.linalg.matrix_norm(W, ord=2).item()
        f2 = W.square().sum().item()
        log_spec2.append(2 * math.log(s_max))
        log_fro2.append(math.log(f2))
        fro2.append(f2)
        fro_over_spec.append(f2 / s_max ** 2)
    out = {
        "log_prod_spec": sum(log_spec2),
        "log_prod_fro": sum(log_fro2),
        "log_sum_fro": math.log(sum(fro2)),
        "sum_fro_over_spec": sum(fro_over_spec),
    }
    if init_model is not None:
        init = _body_matrices(init_model)
        d2 = sum((W - init[n]).square().sum().item() for n, W in mats.items())
        d2_spec = sum(torch.linalg.matrix_norm(W - init[n], ord=2).item() ** 2
                      for n, W in mats.items())
        out["dist_init_fro"] = math.sqrt(d2)
        out["log_sum_dist_init_spec"] = math.log(d2_spec) if d2_spec > 0 else float("-inf")
    return out


def spectral_measures(model: nn.Module) -> dict:
    stable, eff, alphas = [], [], []
    for W in _body_matrices(model).values():
        s = torch.linalg.svdvals(W).double()
        stable.append((s.square().sum() / s[0] ** 2).item())
        p = s[s > 0] / s.sum()
        eff.append(math.exp(-(p * p.log()).sum().item()))
        alphas.append(powerlaw_alpha(s.square().cpu().numpy()))
    return {
        "stable_rank_mean": float(np.mean(stable)),
        "effective_rank_mean": float(np.mean(eff)),
        "alpha_mean": float(np.nanmean(alphas)) if np.isfinite(alphas).any() else float("nan"),
    }


# ----------------------------------------------------------------------------
# Fisher-Rao and sharpness
# ----------------------------------------------------------------------------

def fisher_rao(model, batches, loss_fn=_default_loss, depth: Optional[int] = None) -> dict:
    """``(depth + 1)^2 * mean_b <w, grad L_b(w)>^2`` over the given batches."""
    params = _params(model)
    vals = []
    for b in batches:
        model.zero_grad(set_to_none=True)
        loss_fn(model, b).backward()
        with torch.no_grad():
            vals.append(sum((p * p.grad).sum().item() for p in params if p.grad is not None) ** 2)
    model.zero_grad(set_to_none=True)
    scale = (depth + 1) ** 2 if depth is not None else 1.0
    return {"fisher_rao": scale * float(np.mean(vals))}


def sharpness(model, batches, loss_fn=_default_loss, rho: float = 0.05,
              rho_adaptive: float = 0.5, eta: float = 0.01) -> dict:
    """Loss increase after one ascent step per batch.

    ``m_sharpness``: step of L2 length ``rho``. ``adaptive_sharpness``: step
    ``rho * T^2 g / ||T g||`` with ``T = |w| + eta``, so each weight moves in
    proportion to its magnitude.
    """
    params = _params(model)
    plain, adaptive = [], []
    for b in batches:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, b)
        loss.backward()
        l0 = float(loss)
        with torch.no_grad():
            g = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            gnorm = math.sqrt(sum(gi.square().sum().item() for gi in g))
            if gnorm < 1e-12:
                continue
            T = [p.abs() + eta for p in params]
            tg = math.sqrt(sum((t * gi).square().sum().item() for t, gi in zip(T, g)))
            for eps, out in (([gi * (rho / gnorm) for gi in g], plain),
                             ([t * t * gi * (rho_adaptive / tg) for t, gi in zip(T, g)], adaptive)):
                torch._foreach_add_(params, eps)
                out.append(float(loss_fn(model, b)) - l0)
                torch._foreach_sub_(params, eps)
    model.zero_grad(set_to_none=True)
    return {"m_sharpness": float(np.mean(plain)), "adaptive_sharpness": float(np.mean(adaptive))}


# ----------------------------------------------------------------------------
# PAC-Bayes
# ----------------------------------------------------------------------------

@torch.no_grad()
def _noisy_loss_increase(model, batches, loss_fn, base: float, sigma: float,
                         magnitude: bool, eps: float, n_samples: int, gen) -> float:
    params = _params(model)
    incs = []
    for _ in range(n_samples):
        noise = []
        for p in params:
            std = torch.sqrt(sigma ** 2 * p.square() + eps ** 2) if magnitude else sigma
            noise.append(torch.randn(p.shape, generator=gen, device="cpu").to(p) * std)
        torch._foreach_add_(params, noise)
        incs.append(_mean_loss(model, batches, loss_fn) - base)
        torch._foreach_sub_(params, noise)
    return float(np.mean(incs))


def _largest_sigma(model, batches, loss_fn, base, magnitude, threshold, eps,
                   n_samples, iters, gen) -> float:
    """Bisection (in log space) for the largest noise scale whose expected loss
    increase stays below ``threshold``."""
    lo, hi = math.log(1e-6), math.log(10.0 if magnitude else 1.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        inc = _noisy_loss_increase(model, batches, loss_fn, base, math.exp(mid),
                                   magnitude, eps, n_samples, gen)
        lo, hi = (mid, hi) if inc <= threshold else (lo, mid)
    return math.exp(lo)


def pac_bayes(model, batches, init_model: Optional[nn.Module] = None,
              loss_fn=_default_loss, n_train: Optional[int] = None,
              threshold: float = 0.1, eps: float = 1e-3, n_samples: int = 3,
              iters: int = 10, seed: int = 0) -> dict:
    """PAC-Bayes flatness and bounds (Jiang et al., Sec. C.2 / measures 47-50).

    ``sigma`` is the largest isotropic Gaussian noise scale whose expected loss
    increase is at most ``threshold``; ``sigma_mag`` the same for magnitude-aware
    noise ``N(0, sigma^2 w_i^2 + eps^2)``. ``n_train`` is the training set size
    ``m`` (defaults to the number of batches).
    """
    gen = torch.Generator().manual_seed(seed)
    base = _mean_loss(model, batches, loss_fn)
    m = n_train or len(batches)
    sigma = _largest_sigma(model, batches, loss_fn, base, False, threshold, eps,
                           n_samples, iters, gen)
    sigma_mag = _largest_sigma(model, batches, loss_fn, base, True, threshold, eps,
                               n_samples, iters, gen)
    out = {"pac_bayes_flatness": 1 / sigma ** 2,
           "pac_bayes_mag_flatness": 1 / sigma_mag ** 2}
    if init_model is not None:
        w = torch.cat([p.detach().flatten().float() for p in _params(model)])
        w0 = torch.cat([p.detach().flatten().float() for p in _params(init_model)])
        d = w - w0
        omega = d.numel()
        out["pac_bayes_init"] = d.square().sum().item() / (4 * sigma ** 2) + math.log(m / sigma) + 10
        num = eps ** 2 + (sigma_mag ** 2 + 1) * d.square().sum() / omega
        den = eps ** 2 + sigma_mag ** 2 * d.square()
        out["pac_bayes_mag_init"] = (0.25 * torch.log(num / den).sum().item()
                                     + math.log(m / sigma_mag) + 10)
    return out


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------

def compute_measures(model: nn.Module, batches: Sequence, init_model: Optional[nn.Module] = None,
                     which: Iterable[str] = ALL_MEASURES,
                     loss_fn: Callable = _default_loss, n_train: Optional[int] = None,
                     depth: Optional[int] = None) -> dict:
    """Compute the selected measure groups and return them as one flat dict.

    ``model`` is not modified (gradient-based measures run on a copy).
    """
    which = set(which)
    unknown = which - set(ALL_MEASURES)
    if unknown:
        raise ValueError(f"unknown measures {sorted(unknown)}; choose from {ALL_MEASURES}")
    batches = list(batches)
    out: dict = {}
    if "norm" in which:
        out.update(norm_measures(model, init_model))
    if "spectral" in which:
        out.update(spectral_measures(model))
    if which & {"fisher_rao", "sharpness", "pac_bayes"}:
        work = copy.deepcopy(model)
        was_training = work.training
        work.eval()
        if "fisher_rao" in which:
            out.update(fisher_rao(work, batches, loss_fn, depth))
        if "sharpness" in which:
            out.update(sharpness(work, batches, loss_fn))
        if "pac_bayes" in which:
            out.update(pac_bayes(work, batches, init_model, loss_fn, n_train))
        work.train(was_training)
    return out
