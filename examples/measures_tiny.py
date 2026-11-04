"""Train a tiny model briefly with two optimizers and compare generalization measures.

    python examples/measures_tiny.py --optimizers adamw muoneq-r --steps 200

Uses random tokens by default so it runs anywhere; pass --data-dir (prepared by
prepare_data.py) to use DCLM.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sloptimizers import OPTIMIZERS, SAM, Nexus, build_optimizer
from sloptimizers.measures import compute_measures

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tiny import TinyLlama, TokenWindows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=["adamw", "muoneq-r"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--measure-batches", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data_dir:
        tokens = np.memmap(Path(args.data_dir) / "train.bin", dtype=np.uint16, mode="r")
        vocab = json.loads((Path(args.data_dir) / "meta.json").read_text())["vocab_size"]
    else:
        vocab = 1000
        tokens = np.random.default_rng(0).integers(0, vocab, 500_000, dtype=np.uint16)
    data = TokenWindows(tokens, args.seq_len, device)
    rng = np.random.default_rng(0)
    measure_batches = [data.batch(rng.integers(0, data.n, args.batch_size) * args.seq_len)
                       for _ in range(args.measure_batches)]

    results = {}
    for name in args.optimizers:
        torch.manual_seed(0)
        model = TinyLlama(vocab, args.dim, args.n_layers, args.n_heads, args.hidden,
                          args.seq_len).to(device)
        init = copy.deepcopy(model)
        opt = build_optimizer(name, model)
        for step in range(args.steps):
            x, y = data.batch(rng.integers(0, data.n, args.batch_size) * args.seq_len)
            if isinstance(opt, SAM):
                def closure():
                    opt.zero_grad()
                    loss = model(x, y)
                    loss.backward()
                    return loss
                opt.step(closure)
            elif isinstance(opt, Nexus):
                model(x, y).backward()
                opt.inner_step()
                opt.step()
            else:
                opt.zero_grad()
                model(x, y).backward()
                opt.step()
        results[name] = compute_measures(model, measure_batches, init_model=init,
                                         n_train=args.steps * args.batch_size * args.seq_len,
                                         depth=args.n_layers)

    keys = list(next(iter(results.values())))
    print(f"{'measure':<26}" + "".join(f"{n:>14}" for n in results))
    for k in keys:
        print(f"{k:<26}" + "".join(f"{results[n][k]:>14.4g}" for n in results))


if __name__ == "__main__":
    main()
