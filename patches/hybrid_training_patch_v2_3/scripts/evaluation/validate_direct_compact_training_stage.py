#!/usr/bin/env python3
"""Fail-closed validation for a completed direct-compact training stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.direct_compact_causal import (  # noqa: E402
    DirectCompactContract,
    sha256_artifact,
    sha256_file,
    validate_join_seal,
)
from scripts.training.direct_compact_qwen_decompiler import (  # noqa: E402
    validate_warmstart_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--train-seal", required=True, type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--eval-seal", type=Path)
    parser.add_argument("--expected-train-rows", required=True, type=int)
    parser.add_argument("--expected-eval-rows", type=int)
    parser.add_argument(
        "--no-eval-during-training",
        action="store_true",
        help="Require null eval provenance and no heldout loading.",
    )
    parser.add_argument(
        "--loss-mode",
        required=True,
        choices=("token_mean", "sequence_sum"),
    )
    parser.add_argument("--base-warmstart", required=True, type=Path)
    parser.add_argument("--stage-contract", type=Path)
    parser.add_argument("--expected-stage-contract-sha256", default="")
    return parser.parse_args()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def validate_stage(args: argparse.Namespace) -> dict[str, Any]:
    no_eval_during_training = bool(
        getattr(args, "no_eval_during_training", False)
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    train_file = args.train_file.expanduser().resolve()
    train_seal = args.train_seal.expanduser().resolve()
    if no_eval_during_training:
        if (
            args.eval_file is not None
            or args.eval_seal is not None
            or args.expected_eval_rows is not None
        ):
            raise ValueError(
                "no-eval validation cannot accept eval artifacts or row counts"
            )
        eval_file = None
        eval_seal = None
    else:
        if (
            args.eval_file is None
            or args.eval_seal is None
            or args.expected_eval_rows is None
        ):
            raise ValueError(
                "eval file, seal, and row count are required unless no-eval is set"
            )
        eval_file = args.eval_file.expanduser().resolve()
        eval_seal = args.eval_seal.expanduser().resolve()
    base = args.base_warmstart.expanduser().resolve()
    stage_contract_arg = getattr(args, "stage_contract", None)
    expected_stage_contract_sha256 = str(
        getattr(args, "expected_stage_contract_sha256", "") or ""
    ).strip().lower()
    if bool(stage_contract_arg) != bool(expected_stage_contract_sha256):
        raise ValueError(
            "stage contract and expected stage-contract SHA-256 "
            "must be supplied together"
        )
    stage_contract = (
        None
        if stage_contract_arg is None
        else stage_contract_arg.expanduser().resolve()
    )
    if stage_contract is not None:
        if not stage_contract.is_file():
            raise ValueError(
                f"training-stage contract is missing: {stage_contract}"
            )
        if sha256_file(stage_contract) != expected_stage_contract_sha256:
            raise ValueError(
                "training-stage contract differs from its expected SHA-256"
            )

    contract = DirectCompactContract.load(contract_path)
    del contract
    train_contract = validate_join_seal(
        train_file, train_seal, contract_path, expected_role="fit"
    )
    eval_contract = (
        None
        if eval_file is None or eval_seal is None
        else validate_join_seal(
            eval_file, eval_seal, contract_path, expected_role="measure"
        )
    )
    if int(train_contract.get("rows", -1)) != int(args.expected_train_rows):
        raise ValueError("training-stage train row count mismatch")
    if (
        eval_contract is not None
        and int(eval_contract.get("rows", -1)) != int(args.expected_eval_rows)
    ):
        raise ValueError("training-stage eval row count mismatch")

    checkpoint_paths = validate_warmstart_checkpoint(
        checkpoint, contract_path=contract_path
    )
    base_paths = validate_warmstart_checkpoint(
        base, contract_path=contract_path
    )
    provenance = _json_object(checkpoint_paths["provenance"])
    expected_sequence_sum = args.loss_mode == "sequence_sum"
    loss_contract = provenance.get("loss_contract")
    if (
        not isinstance(loss_contract, dict)
        or loss_contract.get("sequence_distribution_nll")
        is not expected_sequence_sum
        or loss_contract.get("primary_reduction")
        != (
            "equal_weight_mean_of_eos_inclusive_per_sequence_nll_sums"
            if expected_sequence_sum
            else "base_causal_lm_token_mean"
        )
    ):
        raise ValueError("training-stage loss contract mismatch")

    expected_bindings = {
        "contract_sha256": sha256_file(contract_path),
        "train_file_sha256": sha256_file(train_file),
        "eval_file_sha256": (
            None if eval_file is None else sha256_file(eval_file)
        ),
        "train_seal_sha256": sha256_file(train_seal),
        "eval_seal_sha256": (
            None if eval_seal is None else sha256_file(eval_seal)
        ),
        "train_sealed_rows": int(args.expected_train_rows),
        "eval_sealed_rows": (
            None
            if args.expected_eval_rows is None
            else int(args.expected_eval_rows)
        ),
    }
    mismatches = [
        name
        for name, expected in expected_bindings.items()
        if provenance.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "training-stage data/provenance mismatch: " + ", ".join(mismatches)
        )
    observed_stage_contract = provenance.get("stage_contract")
    if stage_contract is None:
        if observed_stage_contract is not None:
            raise ValueError(
                "training-stage provenance unexpectedly carries a stage contract"
            )
    elif (
        not isinstance(observed_stage_contract, dict)
        or observed_stage_contract.get("sha256")
        != expected_stage_contract_sha256
        or int(observed_stage_contract.get("size_bytes", -1))
        != stage_contract.stat().st_size
    ):
        raise ValueError(
            "training-stage provenance is not bound to the exact stage contract"
        )
    if (
        provenance.get("heldout_loaded_during_training")
        is not (not no_eval_during_training)
        or provenance.get("eval_strategy")
        != ("no" if no_eval_during_training else "epoch")
    ):
        raise ValueError("training-stage heldout/eval-mode provenance mismatch")
    warmstart = provenance.get("warmstart_checkpoint")
    expected_warmstart = {
        "decoder_adapter_sha256": sha256_artifact(base_paths["adapter"]),
        "source_overlay_sha256": sha256_file(base_paths["overlay"]),
        "contract_sha256": sha256_file(base_paths["contract"]),
        "provenance_sha256": sha256_file(base_paths["provenance"]),
    }
    if not isinstance(warmstart, dict) or any(
        warmstart.get(name) != expected
        for name, expected in expected_warmstart.items()
    ):
        raise ValueError("training-stage base warm-start binding mismatch")
    if provenance.get("sparse_topk_tail_auxiliary") is not None:
        raise ValueError("gold adaptation unexpectedly enabled a sparse auxiliary")

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_provenance_sha256": sha256_file(
            checkpoint_paths["provenance"]
        ),
        "decoder_adapter_sha256": sha256_artifact(
            checkpoint_paths["adapter"]
        ),
        "source_overlay_sha256": sha256_file(checkpoint_paths["overlay"]),
        "train_file_sha256": expected_bindings["train_file_sha256"],
        "eval_file_sha256": expected_bindings["eval_file_sha256"],
        "train_rows": int(args.expected_train_rows),
        "eval_rows": (
            None
            if args.expected_eval_rows is None
            else int(args.expected_eval_rows)
        ),
        "stage_contract_sha256": (
            None
            if stage_contract is None
            else expected_stage_contract_sha256
        ),
        "loss_mode": args.loss_mode,
        "valid": True,
    }


def main() -> int:
    result = validate_stage(parse_args())
    print(
        "DIRECT_COMPACT_STAGE_VALID "
        f"checkpoint={result['checkpoint']} "
        f"train_rows={result['train_rows']} "
        f"eval_rows={result['eval_rows']} "
        f"loss_mode={result['loss_mode']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
