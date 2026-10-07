"""Shared RoPE position-id convention for the graph decompiler (torch-only,
no heavy imports, so test harnesses can use the real implementation)."""

from __future__ import annotations

import torch


def cumsum_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids matching HF generate()'s attention-mask-cumsum convention.

    The tokenized text prompt is right-padded to a fixed length, so the
    combined [graph prefix | prompt | target/completion] sequence has masked
    holes in the middle. generate() derives RoPE positions from the
    attention-mask cumsum (padding skipped), while a plain forward without
    position_ids uses arange (padding counted) - target-vs-prompt relative
    distances would then be inflated by the pad-hole size. SFT training,
    GRPO log-prob scoring, and generation all share this one convention.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids
