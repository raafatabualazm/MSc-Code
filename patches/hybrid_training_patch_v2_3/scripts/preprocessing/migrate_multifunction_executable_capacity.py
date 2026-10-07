#!/usr/bin/env python3
"""Re-seal the executable safe1578 view under a capacity-only contract.

The compact/F2/heldout bytes do not change when only the decoder target and
total-context ceilings increase.  This migration validates the original
executable view, proves that the contracts differ only in those two capacity
fields, copies the train-side bytes unchanged, and emits new seals/report
records bound to the expanded contract.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.preprocessing.build_multifunction_executable_view import (
    EXECUTABLE_SCOPE,
    F2_MANIFEST_SCHEMA,
    JOIN_SEAL_SCHEMA,
    SCHEMA,
    ExecutableViewError,
    file_record,
    load_json,
    sha256_file,
    validate_executable_view,
)


MIGRATION_SCHEMA = "binary-multifunction-executable-capacity-migration-v1"
CAPACITY_FIELDS = frozenset({"max_target_tokens", "max_total_tokens"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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
    temporary.replace(path)


def validate_capacity_only_contract_change(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that only max_target_tokens/max_total_tokens increased."""

    source_fixed = {
        key: value for key, value in source.items() if key not in CAPACITY_FIELDS
    }
    target_fixed = {
        key: value for key, value in target.items() if key not in CAPACITY_FIELDS
    }
    if source_fixed != target_fixed:
        changed = sorted(
            key
            for key in set(source_fixed).union(target_fixed)
            if source_fixed.get(key) != target_fixed.get(key)
        )
        raise ExecutableViewError(
            "capacity migration changed non-capacity contract fields: "
            + ", ".join(changed)
        )
    observed = sorted(
        key for key in CAPACITY_FIELDS if source.get(key) != target.get(key)
    )
    if observed != sorted(CAPACITY_FIELDS):
        raise ExecutableViewError(
            "capacity migration must change exactly max_target_tokens and "
            "max_total_tokens"
        )
    values: dict[str, dict[str, int]] = {}
    for field in sorted(CAPACITY_FIELDS):
        old = source.get(field)
        new = target.get(field)
        if (
            isinstance(old, bool)
            or isinstance(new, bool)
            or not isinstance(old, int)
            or not isinstance(new, int)
            or old <= 0
            or new <= old
        ):
            raise ExecutableViewError(
                f"{field} must be a positive strict capacity increase"
            )
        values[field] = {"source": old, "target": new}
    return {
        "schema": MIGRATION_SCHEMA,
        "allowed_changed_fields": sorted(CAPACITY_FIELDS),
        "observed_changed_fields": observed,
        "all_non_capacity_fields_byte_semantically_identical": True,
        "capacity_values": values,
    }


def _validate_seal_rebind(
    *,
    source_seal: Mapping[str, Any],
    target_seal: Mapping[str, Any],
    target_contract_sha256: str,
    label: str,
) -> None:
    source_without_contract = dict(source_seal)
    target_without_contract = dict(target_seal)
    source_without_contract.pop("contract_sha256", None)
    target_without_contract.pop("contract_sha256", None)
    if source_without_contract != target_without_contract:
        raise ExecutableViewError(
            f"{label} differs by more than contract_sha256"
        )
    if target_seal.get("contract_sha256") != target_contract_sha256:
        raise ExecutableViewError(
            f"{label} does not bind the target capacity contract"
        )


def migrate_executable_capacity(
    *,
    source_dir: str | Path,
    expected_source_report_sha256: str,
    target_contract: str | Path,
    target_parent_train_seal: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_dir).expanduser().resolve()
    source_paths = {
        "dataset": source_root / "train_multifunction_binary_executable.jsonl",
        "seal": source_root
        / "train_multifunction_binary_executable.seal.json",
        "f2": source_root / "train_multifunction_binary_executable_f2.jsonl",
        "f2_manifest": source_root
        / "train_multifunction_binary_executable_f2.jsonl.manifest.json",
        "report": source_root / "executable_view.build.json",
        "contract": source_root / "compact_contract.json",
    }
    source_view = validate_executable_view(
        dataset=source_paths["dataset"],
        seal=source_paths["seal"],
        f2=source_paths["f2"],
        f2_manifest=source_paths["f2_manifest"],
        build_report=source_paths["report"],
        expected_build_report_sha256=expected_source_report_sha256,
        contract=source_paths["contract"],
        verify_heldout=True,
    )
    source_report = load_json(source_paths["report"], "source executable report")
    source_contract = load_json(source_paths["contract"], "source contract")
    target_contract_path = Path(target_contract).expanduser().resolve()
    target_contract_value = load_json(target_contract_path, "target contract")
    compatibility = validate_capacity_only_contract_change(
        source_contract, target_contract_value
    )
    target_contract_record = file_record(target_contract_path)

    parent = source_report.get("parent")
    if not isinstance(parent, Mapping):
        raise ExecutableViewError("source executable report lacks parent records")
    source_parent_train_seal_record = parent.get("train_seal")
    if not isinstance(source_parent_train_seal_record, Mapping):
        raise ExecutableViewError("source report lacks its parent train seal")
    source_parent_train_seal_path = Path(
        str(source_parent_train_seal_record.get("path") or "")
    ).expanduser().resolve()
    if (
        not source_parent_train_seal_path.is_file()
        or sha256_file(source_parent_train_seal_path)
        != source_parent_train_seal_record.get("sha256")
    ):
        raise ExecutableViewError("source parent train seal record is invalid")
    target_parent_train_seal_path = (
        Path(target_parent_train_seal).expanduser().resolve()
    )
    source_parent_train_seal = load_json(
        source_parent_train_seal_path, "source parent train seal"
    )
    target_parent_train_seal_value = load_json(
        target_parent_train_seal_path, "target parent train seal"
    )
    _validate_seal_rebind(
        source_seal=source_parent_train_seal,
        target_seal=target_parent_train_seal_value,
        target_contract_sha256=target_contract_record["sha256"],
        label="target parent train seal",
    )

    heldout = source_report.get("heldout_measure_only")
    if not isinstance(heldout, Mapping) or not isinstance(
        heldout.get("seal"), Mapping
    ):
        raise ExecutableViewError("source report lacks heldout seal provenance")
    source_heldout_seal_path = Path(
        str(heldout["seal"].get("path") or "")
    ).expanduser().resolve()
    source_heldout_seal = load_json(
        source_heldout_seal_path, "source heldout seal"
    )
    target_heldout_seal = dict(source_heldout_seal)
    target_heldout_seal["contract_sha256"] = target_contract_record["sha256"]

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite capacity migration output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.capacity.",
            dir=destination.parent,
        )
    )
    try:
        output_paths = {
            "dataset": temporary
            / "train_multifunction_binary_executable.jsonl",
            "seal": temporary
            / "train_multifunction_binary_executable.seal.json",
            "f2": temporary
            / "train_multifunction_binary_executable_f2.jsonl",
            "f2_manifest": temporary
            / "train_multifunction_binary_executable_f2.jsonl.manifest.json",
            "report": temporary / "executable_view.build.json",
            "contract": temporary / "compact_contract.json",
            "heldout_seal": temporary
            / "dev_multifunction_binary_target24k.seal.json",
        }
        shutil.copyfile(source_paths["dataset"], output_paths["dataset"])
        shutil.copyfile(source_paths["f2"], output_paths["f2"])
        shutil.copyfile(target_contract_path, output_paths["contract"])
        if (
            sha256_file(output_paths["dataset"])
            != source_view["dataset"]["sha256"]
            or sha256_file(output_paths["f2"]) != source_view["f2"]["sha256"]
            or sha256_file(output_paths["contract"])
            != target_contract_record["sha256"]
        ):
            raise RuntimeError("capacity migration changed immutable bytes")
        _write_json(output_paths["heldout_seal"], target_heldout_seal)
        _validate_seal_rebind(
            source_seal=source_heldout_seal,
            target_seal=load_json(
                output_paths["heldout_seal"], "migrated heldout seal"
            ),
            target_contract_sha256=target_contract_record["sha256"],
            label="migrated heldout seal",
        )

        migration = compatibility | {
            "created_at": _utc_now(),
            "source_executable_report": file_record(source_paths["report"]),
            "source_contract": file_record(source_paths["contract"]),
            "target_contract": file_record(output_paths["contract"]),
            "source_parent_train_seal": file_record(
                source_parent_train_seal_path
            ),
            "target_parent_train_seal": file_record(
                target_parent_train_seal_path
            ),
            "source_heldout_seal": file_record(source_heldout_seal_path),
            "target_heldout_seal": file_record(output_paths["heldout_seal"]),
            "invariants": {
                "train_dataset_bytes_identical": True,
                "train_f2_bytes_identical": True,
                "heldout_dataset_untouched": True,
                "heldout_seal_changed_only_contract_sha256": True,
                "parent_train_seal_changed_only_contract_sha256": True,
                "no_training_or_generation_performed": True,
            },
        }

        f2_manifest = load_json(
            source_paths["f2_manifest"], "source executable F2 manifest"
        )
        if (
            f2_manifest.get("schema") != F2_MANIFEST_SCHEMA
            or f2_manifest.get("training_objective_scope") != EXECUTABLE_SCOPE
        ):
            raise ExecutableViewError("source executable F2 manifest is invalid")
        f2_manifest["created_at"] = _utc_now()
        f2_manifest["dataset"] = file_record(output_paths["dataset"])
        f2_manifest["output"] = file_record(output_paths["f2"])
        f2_manifest["capacity_contract_migration"] = migration
        _write_json(output_paths["f2_manifest"], f2_manifest)

        source_seal = load_json(source_paths["seal"], "source executable seal")
        if source_seal.get("schema") != JOIN_SEAL_SCHEMA:
            raise ExecutableViewError("source executable seal schema is invalid")
        target_seal = dict(source_seal)
        target_seal["contract_sha256"] = target_contract_record["sha256"]
        target_seal["output"] = file_record(output_paths["dataset"])
        target_seal["f2_output"] = file_record(output_paths["f2"])
        target_seal["f2_manifest"] = file_record(output_paths["f2_manifest"])
        target_seal["capacity_contract_migration"] = migration
        _write_json(output_paths["seal"], target_seal)

        target_report = dict(source_report)
        if target_report.get("schema") != SCHEMA:
            raise ExecutableViewError("source executable report schema is invalid")
        target_report["created_at"] = _utc_now()
        target_report["contract"] = file_record(output_paths["contract"])
        target_report["outputs"] = {
            "dataset": file_record(output_paths["dataset"]),
            "seal": file_record(output_paths["seal"]),
            "f2": file_record(output_paths["f2"]),
            "f2_manifest": file_record(output_paths["f2_manifest"]),
            "contract": file_record(output_paths["contract"]),
        }
        target_report["heldout_measure_only"] = dict(heldout) | {
            "seal": file_record(output_paths["heldout_seal"])
        }
        target_report["capacity_contract_migration"] = migration
        _write_json(output_paths["report"], target_report)

        temporary.replace(destination)

        final_paths = {
            name: destination / path.name for name, path in output_paths.items()
        }
        final_migration = dict(migration)
        final_migration["target_contract"] = file_record(
            final_paths["contract"]
        )
        final_migration["target_heldout_seal"] = file_record(
            final_paths["heldout_seal"]
        )
        final_f2_manifest = load_json(
            final_paths["f2_manifest"], "migrated F2 manifest"
        )
        final_f2_manifest["dataset"] = file_record(final_paths["dataset"])
        final_f2_manifest["output"] = file_record(final_paths["f2"])
        final_f2_manifest["capacity_contract_migration"] = final_migration
        _write_json(final_paths["f2_manifest"], final_f2_manifest)
        final_seal = load_json(final_paths["seal"], "migrated executable seal")
        final_seal["output"] = file_record(final_paths["dataset"])
        final_seal["f2_output"] = file_record(final_paths["f2"])
        final_seal["f2_manifest"] = file_record(final_paths["f2_manifest"])
        final_seal["capacity_contract_migration"] = final_migration
        _write_json(final_paths["seal"], final_seal)
        final_report = load_json(final_paths["report"], "migrated report")
        final_report["contract"] = file_record(final_paths["contract"])
        final_report["outputs"] = {
            "dataset": file_record(final_paths["dataset"]),
            "seal": file_record(final_paths["seal"]),
            "f2": file_record(final_paths["f2"]),
            "f2_manifest": file_record(final_paths["f2_manifest"]),
            "contract": file_record(final_paths["contract"]),
        }
        final_report["heldout_measure_only"] = dict(
            final_report["heldout_measure_only"]
        ) | {"seal": file_record(final_paths["heldout_seal"])}
        final_report["capacity_contract_migration"] = final_migration
        _write_json(final_paths["report"], final_report)

        validated = validate_executable_view(
            dataset=final_paths["dataset"],
            seal=final_paths["seal"],
            f2=final_paths["f2"],
            f2_manifest=final_paths["f2_manifest"],
            build_report=final_paths["report"],
            expected_build_report_sha256=sha256_file(final_paths["report"]),
            contract=final_paths["contract"],
            verify_heldout=True,
        )
        if (
            validated["dataset"]["sha256"]
            != source_view["dataset"]["sha256"]
            or validated["f2"]["sha256"] != source_view["f2"]["sha256"]
            or validated["contract"]["sha256"]
            != target_contract_record["sha256"]
        ):
            raise RuntimeError("migrated executable validation changed bindings")
        print(
            "MULTIFUNCTION_EXECUTABLE_CAPACITY_MIGRATED "
            f"rows={validated['rows']} "
            f"contract_sha256={validated['contract']['sha256']} "
            f"report_sha256={validated['report']['sha256']}",
            flush=True,
        )
        return validated
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--target-parent-train-seal", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    migrate_executable_capacity(
        source_dir=args.source_dir,
        expected_source_report_sha256=args.expected_source_report_sha256,
        target_contract=args.target_contract,
        target_parent_train_seal=args.target_parent_train_seal,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
