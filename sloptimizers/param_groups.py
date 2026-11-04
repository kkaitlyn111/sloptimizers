"""Split a model's parameters into "body matrices" and "everything else".

Every matrix-aware optimizer in this package (MuonEq-R, SOAP, SOAP-H, the
Hyperball variants) is only defined for 2D weight matrices. Following common
practice, we route the 2D weight matrices of the transformer *body* (attention
and MLP projections) to the specialized optimizer and keep embeddings, the LM
head, norm gains and biases on AdamW (or plain Adam).

The same split also decides which parameters get weight decay in the AdamW,
SAM, Nexus and AdamHD baselines: decay on body matrices only.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from torch import nn

# Substrings that mark a parameter as an embedding / output head even when it
# is a plain nn.Linear (e.g. an untied `lm_head`).
DEFAULT_HEAD_NAMES: Tuple[str, ...] = ("embed", "lm_head", "output", "wte", "wpe")


def split_body_params(
    model: nn.Module,
    head_names: Iterable[str] = DEFAULT_HEAD_NAMES,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Return ``(body, other)`` parameter lists.

    ``body``: trainable 2D parameters that are neither an ``nn.Embedding``
    weight nor named like an embedding / LM head.
    ``other``: all remaining trainable parameters (embeddings, tied or untied
    LM head, norm gains, biases).

    Tied parameters are emitted once (``named_parameters`` de-duplicates them),
    so a tied embedding/head always lands in ``other``.
    """
    head_names = tuple(head_names)
    embedding_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }
    body: List[nn.Parameter] = []
    other: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in embedding_ids or any(h in name for h in head_names)
        if p.ndim == 2 and not is_head:
            body.append(p)
        else:
            other.append(p)
    return body, other
