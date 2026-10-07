#!/usr/bin/env python3
"""Rebind a rich multifunction parent seal to a capacity-only contract.

The normalized Qwen fit seal intentionally contains less provenance than the
rich Phase-0 expansion seal.  An executable-view capacity migration must retain
that rich provenance byte-semantically and change only ``contract_sha256``.
This tool proves the two dataset files are byte-identical, validates both input
join seals, proves the contract change is capacity-only, and emits the required
rich target seal plus an auditable receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from models.direct_compact_causal import sha256_file, validate_join_seal
from scripts.preprocessing.migrate_multifunction_executable_capacity import (
    _validate_seal_rebind,
    validate_capacity_only_contract_change,
)


SCHEMA = "multifunction-parent-capacity-seal-rebind-v1"


class CapacityRebindError(ValueError):
    """Raised when the requested seal rebind is not capacity-only and exact."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CapacityRebindError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise CapacityRebindError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityRebindError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise CapacityRebindError(f"{label} must be a JSON object")
    return value


def _record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise CapacityRebindError(
            f"required regular non-symlink file is absent: {resolved}"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_rich_output_binding(
    seal: Mapping[str, Any],
    dataset_record: Mapping[str, Any],
) -> None:
    output = seal.get("output")
    if not isinstance(output, Mapping):
        raise CapacityRebindError("source rich seal has no output record")
    output_size = output.get("size_bytes", output.get("bytes"))
    if (
        output.get("sha256") != dataset_record["sha256"]
        or seal.get("output_sha256") != dataset_record["sha256"]
        or int(output_size if output_size is not None else -1)
        != dataset_record["size_bytes"]
    ):
        raise CapacityRebindError(
            "source rich seal output record does not bind the source dataset"
        )


def _validate_existing_receipt(
    *,
    receipt: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    target_records: Mapping[str, Mapping[str, Any]],
    compatibility: Mapping[str, Any],
) -> None:
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("source") != source_records
        or receipt.get("target") != target_records
        or receipt.get("contract_compatibility") != compatibility
        or receipt.get("invariants")
        != {
            "all_non_capacity_contract_fields_identical": True,
            "datasets_byte_identical": True,
            "generic_target_seal_validated": True,
            "heldout_data_opened": False,
            "no_generation_or_training_performed": True,
            "rich_target_seal_changed_only_contract_sha256": True,
            "source_artifacts_modified": False,
        }
    ):
        raise CapacityRebindError("existing capacity-rebind receipt differs")


def rebind_parent_capacity(
    *,
    source_rich_seal: str | Path,
    source_dataset: str | Path,
    source_contract: str | Path,
    target_dataset: str | Path,
    target_contract: str | Path,
    generic_target_seal: str | Path,
    output_seal: str | Path,
    output_receipt: str | Path,
) -> dict[str, Any]:
    source_paths = {
        "dataset": Path(source_dataset).expanduser().resolve(),
        "rich_seal": Path(source_rich_seal).expanduser().resolve(),
        "contract": Path(source_contract).expanduser().resolve(),
    }
    target_paths = {
        "dataset": Path(target_dataset).expanduser().resolve(),
        "generic_seal": Path(generic_target_seal).expanduser().resolve(),
        "contract": Path(target_contract).expanduser().resolve(),
    }
    output_seal_path = Path(output_seal).expanduser().resolve()
    output_receipt_path = Path(output_receipt).expanduser().resolve()
    all_inputs = set(source_paths.values()).union(target_paths.values())
    if (
        output_seal_path in all_inputs
        or output_receipt_path in all_inputs
        or output_seal_path == output_receipt_path
    ):
        raise CapacityRebindError("outputs must not overwrite any input")

    source_records = {key: _record(path) for key, path in source_paths.items()}
    target_input_records = {
        key: _record(path) for key, path in target_paths.items()
    }
    if (
        source_records["dataset"]["sha256"]
        != target_input_records["dataset"]["sha256"]
        or source_records["dataset"]["size_bytes"]
        != target_input_records["dataset"]["size_bytes"]
    ):
        raise CapacityRebindError(
            "source and target parent datasets are not byte-identical"
        )

    try:
        validate_join_seal(
            source_paths["dataset"],
            source_paths["rich_seal"],
            source_paths["contract"],
            expected_role="fit",
        )
        validate_join_seal(
            target_paths["dataset"],
            target_paths["generic_seal"],
            target_paths["contract"],
            expected_role="fit",
        )
    except (OSError, ValueError) as error:
        raise CapacityRebindError("an input join seal is invalid") from error

    source_seal_value = _strict_json(
        source_paths["rich_seal"], "source rich seal"
    )
    generic_target_value = _strict_json(
        target_paths["generic_seal"], "generic target seal"
    )
    _validate_rich_output_binding(
        source_seal_value, source_records["dataset"]
    )
    if (
        generic_target_value.get("output_sha256")
        != target_input_records["dataset"]["sha256"]
    ):
        raise CapacityRebindError(
            "generic target seal does not bind the target dataset"
        )

    source_contract_value = _strict_json(
        source_paths["contract"], "source contract"
    )
    target_contract_value = _strict_json(
        target_paths["contract"], "target contract"
    )
    try:
        compatibility = validate_capacity_only_contract_change(
            source_contract_value, target_contract_value
        )
    except ValueError as error:
        raise CapacityRebindError(
            "contracts do not differ only by increased capacity"
        ) from error

    target_seal_value = dict(source_seal_value)
    target_seal_value["contract_sha256"] = target_input_records["contract"][
        "sha256"
    ]
    try:
        _validate_seal_rebind(
            source_seal=source_seal_value,
            target_seal=target_seal_value,
            target_contract_sha256=target_input_records["contract"]["sha256"],
            label="rich target parent train seal",
        )
    except ValueError as error:
        raise CapacityRebindError("rich target seal rebind is invalid") from error

    if output_seal_path.exists():
        existing = _strict_json(output_seal_path, "existing output seal")
        if existing != target_seal_value:
            raise CapacityRebindError("existing output seal differs")
    else:
        _atomic_json(output_seal_path, target_seal_value)
    output_seal_record = _record(output_seal_path)

    source_receipt_records = {
        "contract": source_records["contract"],
        "dataset": source_records["dataset"],
        "rich_seal": source_records["rich_seal"],
    }
    target_receipt_records = {
        "contract": target_input_records["contract"],
        "dataset": target_input_records["dataset"],
        "generic_seal": target_input_records["generic_seal"],
        "rich_rebound_seal": output_seal_record,
    }
    receipt = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "source": source_receipt_records,
        "target": target_receipt_records,
        "contract_compatibility": compatibility,
        "invariants": {
            "all_non_capacity_contract_fields_identical": True,
            "datasets_byte_identical": True,
            "generic_target_seal_validated": True,
            "heldout_data_opened": False,
            "no_generation_or_training_performed": True,
            "rich_target_seal_changed_only_contract_sha256": True,
            "source_artifacts_modified": False,
        },
    }
    if output_receipt_path.exists():
        existing_receipt = _strict_json(
            output_receipt_path, "existing output receipt"
        )
        _validate_existing_receipt(
            receipt=existing_receipt,
            source_records=source_receipt_records,
            target_records=target_receipt_records,
            compatibility=compatibility,
        )
        receipt = existing_receipt
    else:
        _atomic_json(output_receipt_path, receipt)

    print(
        "MULTIFUNCTION_PARENT_CAPACITY_REBOUND "
        f"rows={source_seal_value.get('rows')} "
        f"dataset_sha256={source_records['dataset']['sha256']} "
        f"contract_sha256={target_input_records['contract']['sha256']} "
        f"seal_sha256={output_seal_record['sha256']}",
        flush=True,
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-rich-seal", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--generic-target-seal", required=True)
    parser.add_argument("--output-seal", required=True)
    parser.add_argument("--output-receipt", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rebind_parent_capacity(
        source_rich_seal=args.source_rich_seal,
        source_dataset=args.source_dataset,
        source_contract=args.source_contract,
        target_dataset=args.target_dataset,
        target_contract=args.target_contract,
        generic_target_seal=args.generic_target_seal,
        output_seal=args.output_seal,
        output_receipt=args.output_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
