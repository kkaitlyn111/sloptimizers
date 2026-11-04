"""Download and tokenize a small slice of DCLM-Baseline for the tiny example.

Streams documents from the Hugging Face dataset
``mlfoundations/dclm-baseline-1.0-parquet`` (the parquet export of
DCLM-Baseline 1.0), tokenizes them with the Llama-2 tokenizer (32k SentencePiece
vocabulary, as in our experiments) and writes flat uint16 token files::

    <out_dir>/train.bin   first --train-tokens tokens
    <out_dir>/val.bin     the next --val-tokens tokens (held out; at most one
                          document straddles the boundary)
    <out_dir>/meta.json

Each document is followed by the EOS token. The tokenizer is loaded from
``hf-internal-testing/llama-tokenizer``, an ungated copy of the Llama-2
tokenizer, so no Hugging Face access token is needed.

Example:
    python examples/prepare_data.py --out-dir data/dclm_tiny --train-tokens 5_000_000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "mlfoundations/dclm-baseline-1.0-parquet"
TOKENIZER = "hf-internal-testing/llama-tokenizer"


def prepare(out_dir, train_tokens: int = 5_000_000, val_tokens: int = 500_000,
            batch_docs: int = 256) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    assert tok.vocab_size <= 2 ** 16, "uint16 storage needs a vocab under 65536"
    ds = load_dataset(DATASET, split="train", streaming=True)

    target = train_tokens + val_tokens
    chunks, n, batch = [], 0, []
    for row in ds:
        batch.append(row["text"])
        if len(batch) < batch_docs:
            continue
        ids = tok(batch, add_special_tokens=False)["input_ids"]
        arr = np.fromiter((t for doc in ids for t in (*doc, tok.eos_token_id)), dtype=np.uint16)
        chunks.append(arr)
        n += arr.size
        batch = []
        print(f"\rtokenized {n / 1e6:.1f}M / {target / 1e6:.1f}M tokens", end="", flush=True)
        if n >= target:
            break
    print()

    tokens = np.concatenate(chunks)
    train, val = tokens[:train_tokens], tokens[train_tokens:target]
    train.tofile(out / "train.bin")
    val.tofile(out / "val.bin")
    meta = dict(dataset=DATASET, tokenizer=TOKENIZER, vocab_size=tok.vocab_size,
                eos_token_id=tok.eos_token_id, train_tokens=int(train.size),
                val_tokens=int(val.size), dtype="uint16")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/train.bin ({train.size:,} tokens) and val.bin ({val.size:,} tokens)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize a small DCLM sample.")
    ap.add_argument("--out-dir", default="data/dclm_tiny")
    ap.add_argument("--train-tokens", type=int, default=5_000_000)
    ap.add_argument("--val-tokens", type=int, default=500_000)
    args = ap.parse_args()
    prepare(args.out_dir, args.train_tokens, args.val_tokens)


if __name__ == "__main__":
    main()
