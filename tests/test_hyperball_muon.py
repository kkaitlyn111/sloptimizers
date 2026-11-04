"""Checks for MuonEq-R, the Hyperball wrappers, AdamHD and SAM/Nexus mechanics."""
import torch
from torch import nn

from sloptimizers import (SAM, AdamHD, AdamHyperball, MuonEqAdamW, MuonEqHyperball,
                           Nexus, equilibrate, zeropower_via_newtonschulz5)


def test_row_equilibration_gives_unit_rows():
    g = torch.randn(8, 5)
    assert torch.allclose(equilibrate(g, "row").norm(dim=1), torch.ones(8), atol=1e-4)


def test_newton_schulz_is_near_orthogonal():
    torch.manual_seed(0)
    o = zeropower_via_newtonschulz5(torch.randn(16, 32)).float()
    # Quintic NS targets singular values in roughly [0.7, 1.2], not exactly 1.
    s = torch.linalg.svdvals(o)
    assert s.min() > 0.5 and s.max() < 1.3


def test_hyperball_variants_conserve_norm():
    for cls, kw in ((AdamHyperball, dict(lr=1e-2)), (MuonEqHyperball, dict(muon_lr=3e-2))):
        torch.manual_seed(0)
        body = [nn.Parameter(torch.randn(16, 24) * 0.1)]
        opt = cls(body, [nn.Parameter(torch.randn(4))], **kw)
        n0 = body[0].detach().norm().item()
        for _ in range(30):
            opt.zero_grad()
            (body[0] * torch.randn_like(body[0])).sum().backward()
            opt.step()
        assert abs(body[0].detach().norm().item() - n0) / n0 < 1e-4, cls.__name__


def test_muoneq_weight_decay_shrinks_with_zero_grad():
    W = nn.Parameter(torch.ones(4, 4))
    opt = MuonEqAdamW([W], [], muon_lr=0.1, weight_decay=0.5)
    W.grad = torch.zeros_like(W)
    opt.step()
    # Zero momentum -> zero NS direction; only decoupled decay applies.
    assert torch.allclose(W.detach(), torch.full((4, 4), 1 - 0.1 * 0.5), atol=1e-6)


def test_adamhd_large_c_matches_adamw():
    """With a huge Huber threshold every weight is in the l2 branch: w <- y / (1 + tau)."""
    torch.manual_seed(0)
    w0 = torch.randn(32)
    p = nn.Parameter(w0.clone())
    p.grad = torch.zeros_like(p)
    AdamHD([p], lr=0.1, weight_decay=0.5, huber_c=1e6, delta_mode="mean").step()
    assert torch.allclose(p.detach(), w0 / (1 + 0.05), atol=1e-6)


def test_sam_restores_weights_after_ascent():
    p = nn.Parameter(torch.randn(10))
    sam = SAM([p], torch.optim.SGD([p], lr=0.0), rho=0.3)
    w0 = p.detach().clone()
    p.grad = torch.randn(10)
    sam.ascent_step()
    assert torch.allclose((p.detach() - w0).norm(), torch.tensor(0.3), atol=1e-5)
    sam.restore()
    assert torch.allclose(p.detach(), w0)


def test_nexus_pseudo_gradient_is_inner_displacement():
    p = nn.Parameter(torch.zeros(3))
    nexus = Nexus([p], torch.optim.SGD([p], lr=1.0), gamma=0.5)
    for g in (torch.tensor([3.0, 4.0, 0.0]), torch.tensor([0.0, 0.0, 2.0])):
        p.grad = g.clone()
        nexus.inner_step()
    inner = p.detach().clone()               # = -0.5*(0.6, 0.8, 0) - 0.5*(0, 0, 1)
    nexus.step()                             # SGD(lr=1) on grad = 0 - inner -> p = inner
    assert torch.allclose(p.detach(), inner)
    assert torch.allclose(inner, torch.tensor([-0.3, -0.4, -0.5]))
