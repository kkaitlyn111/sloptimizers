"""Correctness checks for SOAP and SOAP-H."""
import torch
from torch import nn

from sloptimizers import SOAPAdamW, SOAPHyperball


def _params(shapes, scale=0.1):
    return [nn.Parameter(torch.randn(*s) * scale) for s in shapes]


def test_soaph_conserves_norm():
    torch.manual_seed(0)
    body = _params([(16, 16), (32, 16), (16, 32)])
    opt = SOAPHyperball(body, [nn.Parameter(torch.randn(16))], lr=0.018, q_freq=1)
    init = [p.detach().norm().item() for p in body]
    for _ in range(40):
        opt.zero_grad()
        loss = sum((p * torch.randn_like(p)).sum() + p.square().sum() for p in body)
        loss.backward()
        opt.step()
    for p, n0 in zip(body, init):
        assert abs(p.detach().norm().item() - n0) / n0 < 1e-3


def test_soaph_single_sided_preconditioning():
    torch.manual_seed(0)
    body = _params([(32, 16), (16, 32)])
    opt = SOAPHyperball(body, [nn.Parameter(torch.randn(8))], lr=0.018, q_freq=1,
                        max_precond_dim=24)
    init = [p.detach().norm().item() for p in body]
    for _ in range(20):
        opt.zero_grad()
        sum(p.square().sum() for p in body).backward()
        opt.step()
    s0, s1 = opt.state[body[0]], opt.state[body[1]]
    assert (s0["left"], s0["right"]) == (False, True) and s0["L"] is None
    assert (s1["left"], s1["right"]) == (True, False) and s1["R"] is None
    for p, n0 in zip(body, init):
        assert abs(p.detach().norm().item() - n0) / n0 < 1e-3


def test_soap_reduces_quadratic():
    torch.manual_seed(1)
    target = torch.randn(16, 16)
    W = nn.Parameter(torch.randn(16, 16))
    opt = SOAPAdamW([W], [nn.Parameter(torch.randn(4))], soap_lr=3e-2,
                    precondition_frequency=2)
    first = None
    for _ in range(400):
        opt.zero_grad()
        loss = (W - target).square().sum()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first


def test_soaph_reduces_quadratic_on_sphere():
    torch.manual_seed(2)
    target = torch.randn(16, 16)
    target = target / target.norm() * 4.0
    W = nn.Parameter(torch.randn(16, 16))
    W.data = W.data / W.data.norm() * 4.0
    opt = SOAPHyperball([W], [nn.Parameter(torch.randn(4))], lr=0.05, q_freq=1)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = (W - target).square().sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
