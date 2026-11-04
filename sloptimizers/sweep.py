"""Locally optimal hyperparameter sweep.

Coordinate descent over log-spaced grids, following Kim, Kotha, Liang &
Hashimoto, "Pre-training under Infinite Compute" (2025, arXiv:2509.14786).
From the current point we evaluate every one-step neighbor along each axis
and move to the best one. When no neighbor is better and the point sits on
the edge of a grid, that axis is extended by one step (keeping its spacing)
and the search continues. The point is certified as a local optimum once every
neighbor is strictly worse.

    from sloptimizers.sweep import locally_optimal_sweep
    grid = {"lr": [1e-3, 3e-3, 1e-2], "weight_decay": [0.1, 0.2, 0.4], "epochs": [2, 4, 8]}
    result = locally_optimal_sweep(train_and_eval, grid,
                                   init={"lr": 3e-3, "weight_decay": 0.2, "epochs": 4},
                                   cache="sweep.jsonl")

``objective(config) -> float`` returns the validation loss of one run. Results
are appended to ``cache`` (jsonl) so an interrupted sweep resumes without
re-running finished configs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class SweepResult:
    best: dict
    loss: float
    certified: bool
    evaluated: Dict[str, float] = field(default_factory=dict)
    grid: Dict[str, list] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)


def config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True)


def neighbors(point: dict, grid: Dict[str, list], axes: Sequence[str]) -> List[dict]:
    """All points one grid step away from ``point`` along a single axis."""
    out = []
    for ax in axes:
        vals = grid[ax]
        i = vals.index(point[ax])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**point, ax: vals[j]})
    return out


def extend_axis(vals: list, direction: int) -> list:
    """Add one value past the low (-1) or high (+1) end, keeping the edge ratio."""
    if direction < 0:
        new = vals[0] * vals[0] / vals[1]
    else:
        new = vals[-1] * vals[-1] / vals[-2]
    if all(isinstance(v, int) for v in vals):
        new = int(round(new))
    return [new] + vals if direction < 0 else vals + [new]


def _load_cache(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[config_key(rec["config"])] = rec["loss"]
    return out


def locally_optimal_sweep(
    objective: Callable[[dict], float],
    grid: Dict[str, Sequence],
    init: dict,
    axes: Optional[Sequence[str]] = None,
    cache: Optional[str | Path] = None,
    max_extensions: int = 3,
    log: Callable[[str], None] = print,
) -> SweepResult:
    """Run the sweep and return the certified (or best-found) configuration.

    Args:
        objective: maps a config dict to a validation loss (lower is better).
            ``None`` or NaN counts as a failed run (infinite loss).
        grid: ordered candidate values per hyperparameter.
        init: starting config; may contain fixed keys that are not in ``grid``.
        axes: which grid keys to search (default: all of them).
        cache: jsonl file of finished runs, read on start and appended to.
        max_extensions: maximum number of grid extensions per axis and direction.
    """
    grid = {k: list(v) for k, v in grid.items()}
    axes = list(axes) if axes is not None else list(grid)
    for ax in axes:
        if init[ax] not in grid[ax]:
            raise ValueError(f"init[{ax!r}]={init[ax]!r} is not in grid {grid[ax]}")
    cache_path = Path(cache) if cache is not None else None
    evaluated = _load_cache(cache_path)
    extensions: Dict[tuple, int] = {}
    history: List[dict] = []

    def evaluate(config: dict) -> float:
        key = config_key(config)
        if key not in evaluated:
            loss = objective(dict(config))
            loss = math.inf if loss is None or loss != loss else float(loss)
            evaluated[key] = loss
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a") as f:
                    f.write(json.dumps({"config": config, "loss": loss}) + "\n")
            log(f"  {loss:.4f}  {key}")
        return evaluated[key]

    best = dict(init)
    best_loss = evaluate(best)
    while True:
        nbrs = [(evaluate(nb), nb) for nb in neighbors(best, grid, axes)]
        better = [(l, nb) for l, nb in nbrs if l < best_loss]
        if better:
            new_loss, new = min(better, key=lambda t: t[0])
            history.append(dict(event="move", to=new, loss=new_loss))
            log(f"move -> {config_key(new)} ({best_loss:.4f} -> {new_loss:.4f})")
            best, best_loss = new, new_loss
            continue

        extended = False
        for ax in axes:
            for direction, edge in ((-1, grid[ax][0]), (1, grid[ax][-1])):
                n = extensions.get((ax, direction), 0)
                if best[ax] == edge and len(grid[ax]) > 1 and n < max_extensions:
                    grid[ax] = extend_axis(grid[ax], direction)
                    extensions[(ax, direction)] = n + 1
                    history.append(dict(event="extend", axis=ax, values=grid[ax]))
                    log(f"extend {ax}: {grid[ax]}")
                    extended = True
        if extended:
            continue

        certified = all(l > best_loss for l, _ in nbrs)
        at_limit = any(len(grid[ax]) > 1 and best[ax] in (grid[ax][0], grid[ax][-1])
                       for ax in axes)
        certified = certified and not at_limit
        history.append(dict(event="certified" if certified else "stopped",
                            config=best, loss=best_loss))
        log(f"{'certified' if certified else 'stopped (grid limit or tie)'}: "
            f"{config_key(best)} loss={best_loss:.4f}")
        return SweepResult(best, best_loss, certified, evaluated, grid, history)
