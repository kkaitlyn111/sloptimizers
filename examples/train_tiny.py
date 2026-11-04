"""Train a tiny Llama-style model for several epochs on a fixed token budget.

A small decoder-only transformer (RMSNorm, rotary embeddings, SwiGLU MLP, tied
input/output embeddings) is trained for several epochs over the same small
slice of DCLM, and held-out validation loss is reported at the end. If
``--data-dir`` has no data yet, it is downloaded and tokenized first
(see ``prepare_data.py``).

    python examples/train_tiny.py --optimizer adamw
    python examples/train_tiny.py --optimizer muoneq-r
    python examples/train_tiny.py --optimizer soap-h --synthetic --steps 20   # offline check

Any optimizer in ``sloptimizers.OPTIMIZERS`` works. Hyperparameters default to
the values in ``sloptimizers.DEFAULTS`` (tuned for our 40M model); override the
main learning rate with ``--lr`` or any keyword with ``--opt-kw key=value``.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sloptimizers import DEFAULTS, OPTIMIZERS, SAM, Nexus, build_optimizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import prepare  # noqa: E402


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def rope_tables(seq_len: int, head_dim: int, theta: float, device) -> tuple:
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); rotate interleaved pairs.
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    cos, sin = cos[: x.size(2)], sin[: x.size(2)]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int):
        super().__init__()
        self.n_heads = n_heads
        self.attention_norm = nn.RMSNorm(dim, eps=1e-5)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.attention_norm(x)
        q, k, v = (w(h).view(B, T, self.n_heads, -1).transpose(1, 2)
                   for w in (self.wq, self.wk, self.wv))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(att.transpose(1, 2).reshape(B, T, C))
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class TinyLlama(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int, n_heads: int,
                 hidden: int, max_seq: int, rope_theta: float = 10_000.0):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(Block(dim, n_heads, hidden) for _ in range(n_layers))
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.output = nn.Linear(dim, vocab, bias=False)
        self.output.weight = self.tok_embeddings.weight  # tied
        cos, sin = rope_tables(max_seq, dim // n_heads, rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02 / math.sqrt(2 * n_layers) if name.endswith(("wo.weight", "w2.weight")) else 0.02
                nn.init.trunc_normal_(p, std=std, a=-3 * std, b=3 * std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        logits = self.output(self.norm(x)).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


# ----------------------------------------------------------------------------
# Data: non-overlapping windows, reshuffled every epoch
# ----------------------------------------------------------------------------

class TokenWindows:
    def __init__(self, data: np.ndarray, seq_len: int, device):
        self.data = data
        self.seq_len = seq_len
        self.n = (len(self.data) - 1) // seq_len
        self.device = device

    def batch(self, starts: np.ndarray):
        x = np.stack([self.data[s: s + self.seq_len + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def epoch(self, batch_size: int, rng: np.random.Generator):
        order = rng.permutation(self.n) * self.seq_len
        for i in range(0, self.n - batch_size + 1, batch_size):
            yield self.batch(order[i: i + batch_size])


@torch.no_grad()
def evaluate(model, val: TokenWindows, batch_size: int, max_batches: int | None = None) -> float:
    model.eval()
    starts = np.arange(val.n) * val.seq_len
    total, count = 0.0, 0
    for b, i in enumerate(range(0, val.n, batch_size)):
        if max_batches is not None and b >= max_batches:
            break
        x, y = val.batch(starts[i: i + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            total += model(x, y).item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / count


def lr_lambda(step: int, total: int, warmup: int, min_ratio: float = 0.01) -> float:
    """Linear warmup then cosine decay to ``min_ratio`` (the schedule used in the paper)."""
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def parse_kw(items):
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny multi-epoch pretraining run.")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    ap.add_argument("--data-dir", default="data/dclm_tiny")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=None,
                    help="main learning rate (lr / muon_lr / soap_lr, depending on optimizer)")
    ap.add_argument("--opt-kw", nargs="*", default=[], help="extra optimizer kwargs, key=value")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--nexus-k", type=int, default=2, help="Nexus inner steps per outer step")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--steps", type=int, default=None, help="stop after this many steps")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on random tokens instead of DCLM (no download)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON file for the final metrics")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.synthetic:
        vocab = 32000
        rng0 = np.random.default_rng(args.seed)
        train_np = rng0.integers(0, vocab, 200_000, dtype=np.uint16)
        val_np = rng0.integers(0, vocab, 20_000, dtype=np.uint16)
    else:
        data_dir = Path(args.data_dir)
        if not (data_dir / "meta.json").exists():
            prepare(data_dir)
        vocab = json.loads((data_dir / "meta.json").read_text())["vocab_size"]
        train_np = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        val_np = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    train = TokenWindows(train_np, args.seq_len, device)
    val = TokenWindows(val_np, args.seq_len, device)

    model = TinyLlama(vocab, args.dim, args.n_layers, args.n_heads,
                      args.hidden, args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.tok_embeddings.weight.numel()

    kw = parse_kw(args.opt_kw)
    if args.lr is not None:
        main_key = next(k for k in ("muon_lr", "soap_lr", "lr") if k in DEFAULTS[args.optimizer])
        kw[main_key] = args.lr
    opt = build_optimizer(args.optimizer, model, **kw)
    base = opt.base_optimizer if isinstance(opt, (SAM, Nexus)) else opt

    k = args.nexus_k if isinstance(opt, Nexus) else 1
    steps_per_epoch = train.n // (args.batch_size * k)
    total = steps_per_epoch * args.epochs
    if args.steps is not None:
        total = min(total, args.steps)
    # SOAP-H was tuned with a power-law decay and no warmup; others use cosine + 1% warmup.
    if args.optimizer == "soap-h":
        sched = torch.optim.lr_scheduler.LambdaLR(
            base, lambda s: 0.01 + 0.99 * max(0.0, 1 - s / total) ** 1.25)
    else:
        warmup = max(1, total // 100)
        sched = torch.optim.lr_scheduler.LambdaLR(base, lambda s: lr_lambda(s, total, warmup))

    print(f"optimizer={args.optimizer} params={n_params / 1e6:.1f}M "
          f"(non-embedding {(n_params - n_embed) / 1e6:.1f}M) "
          f"train_tokens={train.n * args.seq_len / 1e6:.1f}M epochs={args.epochs} steps={total}")
    print(f"hyperparameters: {dict(DEFAULTS[args.optimizer], **kw)}")

    def loss_fn(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            return model(x, y)

    rng = np.random.default_rng(args.seed)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        if step >= total:
            break
        it = train.epoch(args.batch_size, rng)
        for _ in range(min(steps_per_epoch, total - step)):
            if isinstance(opt, SAM):
                x, y = next(it)

                def closure():
                    opt.zero_grad()
                    loss = loss_fn(x, y)
                    loss.backward()
                    return loss
                loss = opt.step(closure, max_grad_norm=args.clip)
            elif isinstance(opt, Nexus):
                loss = 0.0
                for _ in range(k):
                    x, y = next(it)
                    micro = loss_fn(x, y)
                    micro.backward()
                    opt.inner_step()
                    loss += micro.detach() / k
                opt.step(max_grad_norm=args.clip)
            else:
                x, y = next(it)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
            sched.step()
            step += 1
            if step % args.eval_every == 0 or step == total:
                v = evaluate(model, val, args.batch_size, max_batches=10)
                print(f"epoch {epoch + 1} step {step}/{total} train_loss {float(loss):.4f} "
                      f"val_loss(partial) {v:.4f} elapsed {time.time() - t0:.0f}s", flush=True)

    final = evaluate(model, val, args.batch_size)
    print(f"FINAL optimizer={args.optimizer} val_loss={final:.4f} "
          f"time={time.time() - t0:.0f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(
            optimizer=args.optimizer, val_loss=final, steps=total, epochs=args.epochs,
            params=n_params, seconds=time.time() - t0, hyperparameters={**DEFAULTS[args.optimizer], **kw},
        ), indent=2))


if __name__ == "__main__":
    main()
