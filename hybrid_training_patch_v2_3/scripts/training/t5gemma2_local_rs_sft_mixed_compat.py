#!/usr/bin/env python3
"""Run the sealed local RS-SFT harvester from a mixed RS-SFT checkpoint.

This compatibility entry point changes only checkpoint-contract loading.  The
historical sampler, generation code, task schedule, and compiler/private
scorers remain in ``t5gemma2_local_rs_sft_pilot.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation import t5gemma2_f2_passk_mixed_compat as mixed_compat
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_local_rs_sft_pilot as pilot


COMPAT_SCHEMA = "t5gemma2-mixed-local-rs-sft-loader-compat-v1"
_mixed_checkpoint_record = mixed_compat._mixed_checkpoint_record


def _pop_flag_value(arguments: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"mixed local compatibility requires exactly one {flag}")
    index = positions[0]
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--sft_checkpoint" in arguments:
        raise ValueError("pass --compat_checkpoint, not --sft_checkpoint")
    compat_record = Path(
        _pop_flag_value(arguments, "--compat_record")
    ).expanduser().resolve()
    checkpoint = Path(
        _pop_flag_value(arguments, "--compat_checkpoint")
    ).expanduser().resolve()
    arguments.extend(["--sft_checkpoint", str(checkpoint)])

    contract, _record = _mixed_checkpoint_record(checkpoint, "sft")
    pilot_path = Path(pilot.__file__).resolve()
    inference_path = Path(inference.__file__).resolve()
    mixed_loader_path = Path(mixed_compat.__file__).resolve()
    wrapper_path = Path(__file__).resolve()
    compatibility = {
        "schema": COMPAT_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_run_contract_sha256": canonical_sha256(contract),
        "pilot_core_path": str(pilot_path),
        "pilot_core_sha256": sha256_file(pilot_path),
        "inference_core_path": str(inference_path),
        "inference_core_sha256": sha256_file(inference_path),
        "mixed_loader_path": str(mixed_loader_path),
        "mixed_loader_sha256": sha256_file(mixed_loader_path),
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),
        "scope": "checkpoint_contract_loader_only",
        "schedule_code_changed": False,
        "sampling_code_changed": False,
        "generation_code_changed": False,
        "scoring_code_changed": False,
        "heldout_175_opened": False,
    }
    require_exact_or_write(compat_record, compatibility)

    # The pilot imported both symbols directly. Patch its preflight reference,
    # and patch the defining inference module because load_policy resolves its
    # own module-global loader at call time.
    pilot._checkpoint_record = _mixed_checkpoint_record
    pilot.CHECKPOINT_LOADER_COMPATIBILITY = compatibility
    inference._checkpoint_record = _mixed_checkpoint_record
    print(json.dumps(compatibility, sort_keys=True), flush=True)
    return pilot.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
