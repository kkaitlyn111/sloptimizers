"""Locally optimal sweep of train_tiny.py over learning rate, weight decay and epochs.

Each config is one ``train_tiny.py`` run; finished runs are cached in
``runs/sweep_<optimizer>.jsonl``, so rerunning resumes the sweep.

    python examples/sweep_tiny.py --optimizer muoneq-r
    python examples/sweep_tiny.py --optimizer adamw --synthetic --steps 5 --dim 64 --n-layers 2

Unrecognized flags are passed through to ``train_tiny.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from sloptimizers import DEFAULTS, OPTIMIZERS
from sloptimizers.sweep import locally_optimal_sweep

TRAIN = Path(__file__).resolve().parent / "train_tiny.py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    ap.add_argument("--cache", default=None)
    args, passthrough = ap.parse_known_args()

    defaults = DEFAULTS[args.optimizer]
    lr_key = next(k for k in ("muon_lr", "soap_lr", "lr") if k in defaults)
    lr0 = defaults[lr_key]
    grid = {"lr": [lr0 / 3, lr0, lr0 * 3], "epochs": [2, 4, 8]}
    init = {"lr": lr0, "epochs": 4}
    if "weight_decay" in defaults:
        grid["weight_decay"] = [0.1, 0.2, 0.4, 0.8, 1.6]
        init["weight_decay"] = 0.4

    def objective(config: dict) -> float:
        with tempfile.NamedTemporaryFile(suffix=".json") as out:
            cmd = [sys.executable, str(TRAIN), "--optimizer", args.optimizer,
                   "--lr", str(config["lr"]), "--epochs", str(config["epochs"]),
                   "--out", out.name, *passthrough]
            if "weight_decay" in config:
                cmd += ["--opt-kw", f"weight_decay={config['weight_decay']}"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            return json.loads(Path(out.name).read_text())["val_loss"]

    cache = args.cache or f"runs/sweep_{args.optimizer}.jsonl"
    result = locally_optimal_sweep(objective, grid, init, cache=cache)
    print(json.dumps(dict(optimizer=args.optimizer, best=result.best, val_loss=result.loss,
                          certified=result.certified, runs=len(result.evaluated)), indent=2))


if __name__ == "__main__":
    main()
