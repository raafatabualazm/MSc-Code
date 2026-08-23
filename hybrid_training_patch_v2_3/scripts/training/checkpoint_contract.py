#!/usr/bin/env python3
"""Fail-closed validation for trainable-only Antigravity checkpoints.

Antigravity checkpoints intentionally omit frozen pretrained base weights.  A
large ``missing_keys`` list is therefore expected with ``strict=False``.  What
is *not* safe is silently loading a checkpoint under a different PEFT/prefix/
graph architecture: that leaves current trainable tensors randomly initialised
or ignores checkpoint tensors.  This helper validates exactly those conditions.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence


def validate_trainable_checkpoint_load(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    missing_keys: Sequence[str] = (),
    unexpected_keys: Sequence[str] = (),
    context: str = "checkpoint",
    allow_partial: bool | None = None,
    allowed_absent_trainable_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate a trainable-only checkpoint after ``load_state_dict(strict=False)``.

    Frozen base-model keys may be missing. Every parameter currently marked
    trainable must, however, be present in the checkpoint, and every checkpoint
    tensor must be recognised by the instantiated architecture. A caller may
    explicitly allow a narrow set of newly introduced trainable tensors (for
    example zero-output LoRA modules during a documented adapter expansion). The
    only broad escape hatch is ``GRAPH_ALLOW_PARTIAL_CHECKPOINT=1``.
    """
    if allow_partial is None:
        allow_partial = os.environ.get("GRAPH_ALLOW_PARTIAL_CHECKPOINT", "0") == "1"

    checkpoint_keys = set(state_dict)
    model_state_keys = set(module.state_dict())
    trainable_keys = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    absent_trainables = sorted(trainable_keys - checkpoint_keys)
    allowed_absent_set = set(str(value) for value in allowed_absent_trainable_keys)
    allowed_absent = sorted(set(absent_trainables) & allowed_absent_set)
    unapproved_absent = sorted(set(absent_trainables) - allowed_absent_set)
    missing = sorted(set(str(value) for value in missing_keys))
    unexpected = sorted(set(str(value) for value in unexpected_keys))
    recognised = checkpoint_keys & model_state_keys
    missing_frozen = sorted(set(missing) - set(absent_trainables))

    failures: list[str] = []
    if not checkpoint_keys:
        failures.append("checkpoint contains no tensors")
    if not recognised:
        failures.append("checkpoint has no keys recognised by the instantiated model")
    if unapproved_absent:
        failures.append(
            f"{len(unapproved_absent)} current trainable tensors are absent from the checkpoint "
            "outside the explicitly allowed initialization set"
        )
    if unexpected:
        failures.append(
            f"{len(unexpected)} checkpoint tensors are unexpected under the current architecture"
        )

    report = {
        "context": context,
        "checkpoint_tensor_count": len(checkpoint_keys),
        "model_state_tensor_count": len(model_state_keys),
        "recognised_checkpoint_tensor_count": len(recognised),
        "current_trainable_tensor_count": len(trainable_keys),
        "load_missing_tensor_count": len(missing),
        "missing_frozen_tensor_count": len(missing_frozen),
        "absent_trainable_tensor_count": len(absent_trainables),
        "allowed_absent_trainable_tensor_count": len(allowed_absent),
        "unapproved_absent_trainable_tensor_count": len(unapproved_absent),
        "unexpected_tensor_count": len(unexpected),
        "missing_frozen_examples": missing_frozen[:16],
        "absent_trainable_examples": absent_trainables[:16],
        "allowed_absent_trainable_examples": allowed_absent[:16],
        "unapproved_absent_trainable_examples": unapproved_absent[:16],
        "unexpected_examples": unexpected[:16],
        "allow_partial": bool(allow_partial),
        "status": "passed" if not failures else ("overridden" if allow_partial else "failed"),
        "failures": failures,
    }
    if failures and not allow_partial:
        details = []
        if unapproved_absent:
            details.append("missing trainables: " + ", ".join(unapproved_absent[:8]))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected[:8]))
        suffix = ("; " + "; ".join(details)) if details else ""
        raise RuntimeError(
            f"{context} architecture/configuration mismatch: "
            + "; ".join(failures)
            + suffix
            + ". Supply the exact graph_environment provenance used by the checkpoint. "
              "GRAPH_ALLOW_PARTIAL_CHECKPOINT=1 is a research-only override."
        )
    return report
