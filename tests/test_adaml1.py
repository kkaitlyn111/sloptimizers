"""Unit tests for AdamL1 (Adam + decoupled L1 proximal step)."""
import torch

from sloptimizers import AdamL1


def _soft_threshold(w, t):
    return torch.sign(w) * (w.abs() - t).clamp_min(0.0)


def test_prox_only_when_grad_zero():
    """With zero gradient a step is exactly the soft-threshold prox by lr * l1_lambda."""
    lr, lam = 0.1, 0.5
    w0 = torch.tensor([0.0, 0.02, 0.05, 0.051, -0.051, 0.2, -0.3])
    p = torch.nn.Parameter(w0.clone())
    p.grad = torch.zeros_like(p)
    AdamL1([dict(params=[p], l1_lambda=lam)], lr=lr).step()
    assert torch.allclose(p.detach(), _soft_threshold(w0, lr * lam), atol=1e-7)
    assert torch.all(p.detach()[w0.abs() <= lr * lam + 1e-12] == 0.0)


def test_no_penalty_group_untouched():
    w0 = torch.tensor([0.001, 0.04, -0.3, 1.0])
    p = torch.nn.Parameter(w0.clone())
    p.grad = torch.zeros_like(p)
    AdamL1([dict(params=[p], l1_lambda=0.0)], lr=0.1).step()
    assert torch.equal(p.detach(), w0)


def test_threshold_scales_with_group_lr():
    lam, w0 = 0.5, torch.tensor([0.3])
    pa, pb = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    pa.grad, pb.grad = torch.zeros_like(pa), torch.zeros_like(pb)
    AdamL1([dict(params=[pa], l1_lambda=lam, lr=0.1),
            dict(params=[pb], l1_lambda=lam, lr=0.01)], lr=0.1).step()
    assert abs((w0 - pa.detach()).item() - 0.1 * lam) < 1e-7
    assert abs((w0 - pb.detach()).item() - 0.01 * lam) < 1e-7


def test_matches_torch_adam_without_l1():
    torch.manual_seed(0)
    betas, eps, lr = (0.9, 0.95), 1e-8, 1e-2
    w_init, target = torch.randn(16), torch.randn(16)
    p_ref, p_ours = torch.nn.Parameter(w_init.clone()), torch.nn.Parameter(w_init.clone())
    ref = torch.optim.Adam([p_ref], lr=lr, betas=betas, eps=eps)
    ours = AdamL1([dict(params=[p_ours], l1_lambda=0.0)], lr=lr, betas=betas, eps=eps)
    for _ in range(25):
        for p, o in ((p_ref, ref), (p_ours, ours)):
            o.zero_grad()
            ((p - target) ** 2).sum().backward()
            o.step()
    assert torch.allclose(p_ref.detach(), p_ours.detach(), atol=1e-6)


def test_l1_produces_exact_zeros_under_noise():
    torch.manual_seed(0)
    p = torch.nn.Parameter(0.05 * torch.randn(4096))
    opt = AdamL1([dict(params=[p], l1_lambda=0.3)], lr=1e-2)
    for _ in range(200):
        p.grad = 0.05 * torch.randn_like(p)
        opt.step()
    assert (p.detach() == 0.0).float().mean().item() > 0.05
