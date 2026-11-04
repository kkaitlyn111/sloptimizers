import copy
import math

import numpy as np
import torch
from torch import nn

from sloptimizers.measures import compute_measures, powerlaw_alpha


class ToyLM(nn.Module):
    def __init__(self, vocab=32, dim=24, hidden=64):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)
        self.output.weight = self.tok_embeddings.weight

    def forward(self, x, y):
        h = self.tok_embeddings(x)
        logits = self.output(self.norm(h + self.mlp(h)))
        return nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())


def _batches(n=3):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        x = torch.randint(0, 32, (8, 6), generator=g)
        out.append((x, (x + 1) % 32))
    return out


def test_all_measures_finite_after_training():
    torch.manual_seed(0)
    model = ToyLM()
    init = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for x, y in _batches(20):
        opt.zero_grad()
        model(x, y).backward()
        opt.step()
    before = [p.detach().clone() for p in model.parameters()]
    m = compute_measures(model, _batches(), init_model=init, n_train=10_000)
    expected = {"log_prod_spec", "log_prod_fro", "dist_init_fro", "stable_rank_mean",
                "effective_rank_mean", "fisher_rao", "m_sharpness", "adaptive_sharpness",
                "pac_bayes_flatness", "pac_bayes_mag_flatness", "pac_bayes_init",
                "pac_bayes_mag_init"}
    assert expected <= m.keys()
    for k in expected:
        assert math.isfinite(m[k]), (k, m[k])
    assert m["dist_init_fro"] > 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_distance_from_init_is_zero_for_same_model():
    model = ToyLM()
    m = compute_measures(model, _batches(), init_model=copy.deepcopy(model), which=["norm"])
    assert m["dist_init_fro"] == 0.0


def test_powerlaw_alpha_recovers_exponent():
    rng = np.random.default_rng(0)
    samples = (1 - rng.random(5000)) ** (-1 / 2.0)  # Pareto tail with alpha = 3
    assert abs(powerlaw_alpha(samples) - 3.0) < 0.3
