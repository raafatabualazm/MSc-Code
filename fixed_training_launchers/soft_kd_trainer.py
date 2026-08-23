#!/usr/bin/env python3
"""Fail-closed compatibility stub for the retired legacy soft-KD trainer."""

raise SystemExit(
    "This legacy trainer is disabled: it renormalized rounded API top-k "
    "logprobs, discarded tail mass/EOS, and was not full-distribution KD. "
    "Run `bash /workspace/run_true_kd.sh audit-legacy` for the artifact decision "
    "or `bash /workspace/run_true_kd.sh dense` with an explicit local teacher."
)
