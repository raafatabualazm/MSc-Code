"""Shared position-id helpers for graph-prefix causal decoding."""

from __future__ import annotations

import torch


def cumsum_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Assign contiguous positions only to active tokens.

    Graph-prefix causal batches may contain right-padded prompt tokens followed
    by active generated/target tokens. Plain ``arange`` positions count those
    masked holes and therefore score a different positional layout from
    ``generate``. This helper mirrors the Hugging Face cumulative-mask
    convention: active tokens receive 0, 1, 2, ... and masked locations receive
    0 without advancing the counter.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be rank-2 [batch, sequence], got {tuple(attention_mask.shape)}"
        )
    active = attention_mask.to(dtype=torch.long)
    positions = active.cumsum(dim=-1) - 1
    positions.masked_fill_(active == 0, 0)
    return positions.clamp_min_(0)
