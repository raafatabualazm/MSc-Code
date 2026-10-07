#!/usr/bin/env python3
"""Fail-closed compatibility stub for the retired legacy top-k converter."""

raise SystemExit(
    "This legacy converter is disabled: decoded top-k strings without sealed "
    "token IDs, exact probabilities, explicit tail mass, and EOS cannot be "
    "converted into true KD data. Retain the verified code fields for RS-SFT; "
    "use /workspace/true_kd_patch_v1/examples for any future sparse collection."
)
