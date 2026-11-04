"""CPU smoke test: every optimizer trains a toy tied-embedding model without error."""
import pytest
import torch
from torch import nn

from sloptimizers import OPTIMIZERS, SAM, Nexus, build_optimizer, split_body_params


class ToyLM(nn.Module):
    """Embedding -> 2 residual MLP blocks -> RMSNorm -> tied head."""

    def __init__(self, vocab=32, dim=16, hidden=48):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                          nn.Linear(hidden, dim))
            for _ in range(2))
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)
        self.output.weight = self.tok_embeddings.weight

    def forward(self, x, y):
        h = self.tok_embeddings(x)
        for blk in self.blocks:
            h = h + blk(h)
        logits = self.output(self.norm(h))
        return nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())


def _batch(gen):
    # Learnable task: predict (token + 1) mod vocab.
    x = torch.randint(0, 32, (16, 8), generator=gen)
    return x, (x + 1) % 32


def test_split_body_params():
    m = ToyLM()
    body, other = split_body_params(m)
    assert len(body) == 4 and all(p.ndim == 2 for p in body)
    assert m.tok_embeddings.weight in other           # tied head appears once
    assert len(body) + len(other) == len(list(m.parameters()))


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizer_trains_toy_model(name):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = ToyLM()
    opt = build_optimizer(name, model)
    eval_x, eval_y = _batch(torch.Generator().manual_seed(123))
    with torch.no_grad():
        before = model(eval_x, eval_y).item()

    for _ in range(60):
        if isinstance(opt, SAM):
            def closure():
                opt.zero_grad()
                loss = model(*_batch(gen))
                loss.backward()
                return loss
            opt.step(closure, max_grad_norm=1.0)
        elif isinstance(opt, Nexus):
            for _ in range(2):  # K = 2 micro-batches per outer step
                model(*_batch(gen)).backward()
                opt.inner_step()
            opt.step(max_grad_norm=1.0)
        else:
            opt.zero_grad()
            model(*_batch(gen)).backward()
            opt.step()

    with torch.no_grad():
        after = model(eval_x, eval_y).item()
    assert all(torch.isfinite(p).all() for p in model.parameters()), name
    assert after < before, f"{name}: loss {before:.3f} -> {after:.3f}"
