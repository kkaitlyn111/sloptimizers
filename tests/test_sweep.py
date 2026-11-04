import math

import pytest

from sloptimizers.sweep import extend_axis, locally_optimal_sweep, neighbors


def bowl(opt_lr, opt_wd, opt_ep):
    def f(c):
        return ((math.log2(c["lr"] / opt_lr)) ** 2
                + (math.log2(c["weight_decay"] / opt_wd)) ** 2
                + 0.5 * (math.log2(c["epochs"] / opt_ep)) ** 2)
    return f


GRID = {"lr": [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2],
        "weight_decay": [0.1, 0.2, 0.4, 0.8],
        "epochs": [2, 4, 8, 16]}


def test_neighbors_and_extension():
    nb = neighbors({"lr": 1e-3, "weight_decay": 0.4, "epochs": 4}, GRID, list(GRID))
    assert len(nb) == 1 + 2 + 2
    assert extend_axis([2, 4, 8], 1) == [2, 4, 8, 16]
    assert extend_axis([0.1, 0.2], -1) == pytest.approx([0.05, 0.1, 0.2])


def test_converges_and_certifies_interior_optimum():
    calls = []
    f = bowl(8e-3, 0.2, 8)
    res = locally_optimal_sweep(lambda c: calls.append(c) or f(c), GRID,
                                init={"lr": 1e-3, "weight_decay": 0.8, "epochs": 2},
                                log=lambda s: None)
    assert res.certified
    assert res.best == {"lr": 8e-3, "weight_decay": 0.2, "epochs": 8}
    assert len(calls) == len({str(sorted(c.items())) for c in calls})  # no repeats


def test_extends_grid_when_optimum_is_outside():
    f = bowl(1e-3, 3.2, 8)  # weight decay optimum is two steps past the grid edge
    res = locally_optimal_sweep(f, GRID, init={"lr": 4e-3, "weight_decay": 0.2, "epochs": 4},
                                log=lambda s: None)
    assert res.certified
    assert abs(res.best["weight_decay"] - 3.2) < 1e-9
    assert any(h["event"] == "extend" for h in res.history)
    # lr optimum sits at the low edge: it is extended once, then certified as interior.
    assert res.grid["lr"][0] < 1e-3


def test_cache_skips_finished_runs(tmp_path):
    f = bowl(4e-3, 0.4, 4)
    cache = tmp_path / "sweep.jsonl"
    init = {"lr": 2e-3, "weight_decay": 0.2, "epochs": 4}
    first = locally_optimal_sweep(f, GRID, init, cache=cache, log=lambda s: None)
    calls = []
    second = locally_optimal_sweep(lambda c: calls.append(c) or f(c), GRID, init,
                                   cache=cache, log=lambda s: None)
    assert second.best == first.best and calls == []
