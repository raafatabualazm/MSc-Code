#!/usr/bin/env python3
"""Build a matched direct-compact RS-SFT intervention and gold-only control.

The intervention is exactly 50% original gold rows and 50% independently
re-certified teacher repairs.  The control has the identical source-task
sequence and row count, but every target is the original gold implementation.
This isolates the target intervention while keeping warm-start, examples,
steps, seed, and compact conditioning inputs matchable by construction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from models.direct_compact_causal import (
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.preprocessing.build_multifunction_executable_view import (
    REPRESENTATION_SCHEMA,
    validate_executable_view,
)


MODEL_BINDING_FIELDS = (
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
    "binary_multifunction_schema",
    "binary_adapter_contract_sha256",
    "binary_semantic_projection_sha256",
    "binary_source_symbol_attestation_binding_sha256",
)
IMPORTED_REPAIR_ROW_SCHEMA = "direct-compact-rs-hard-target-v1"
IMPORTED_REPAIR_SEAL_SCHEMA = "direct-compact-rs-hard-target-seal-v1"
IMPORTED_REPAIR_MANIFEST_SCHEMA = "direct-compact-rs-hard-target-import-v1"
PROHIBITED_REASONING_FIELDS = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "raw_reasoning_content",
        "chain_of_thought",
        "cot",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_train", required=True)
    parser.add_argument("--base_train_seal", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--executable_view_report", required=True)
    parser.add_argument(
        "--expected_executable_view_report_sha256",
        required=True,
    )
    parser.add_argument(
        "--repairs",
        action="append",
        required=True,
        metavar="PROVIDER=PATH",
        help="Verified-repair JSONL. May be repeated for multiple teachers.",
    )
    parser.add_argument(
        "--repair_report",
        action="append",
        required=True,
        metavar="PROVIDER=PATH",
        help="Collector report paired one-to-one with --repairs.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_arm", type=int, default=3156)
    parser.add_argument(
        "--rows_per_arm_from_sealed_parent",
        action="store_true",
        help=(
            "Set each matched arm to exactly twice the executable-view row "
            "count committed by its sealed build report."
        ),
    )
    parser.add_argument(
        "--expected_parent_fit_rows",
        type=int,
        default=0,
        help=(
            "Optional full fit-universe count checked against the executable "
            "view (expanded production uses 2776)."
        ),
    )
    parser.add_argument("--min_unique_repairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=2)
    parser.add_argument(
        "--allow_low_coverage_smoke",
        action="store_true",
        help=(
            "Testing only: bypass production coverage/shape gates. The build "
            "report records this and production evaluation rejects it."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank rows are forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_repair_specs(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    providers: set[str] = set()
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--repairs must be PROVIDER=PATH, got {raw!r}")
        provider, path = raw.split("=", 1)
        provider = provider.strip().lower()
        if not provider or provider in providers:
            raise ValueError(f"invalid or duplicate repair provider {provider!r}")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"repair artifact does not exist: {resolved}")
        providers.add(provider)
        result.append((provider, resolved))
    return result


def validate_openai_repair_report(
    *,
    provider: str,
    repair_path: Path,
    report_path: Path,
    base_path: Path,
    base_seal_path: Path,
    executable_view_report: Path,
) -> dict[str, Any]:
    """Bind repairs to the official synchronous GPT-5.6-sol collector."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{report_path}: repair report is not an object")
    if provider != "chatgpt":
        raise ValueError(
            "the production post-Qwen RS intervention is sealed to chatgpt"
        )
    inputs = report.get("inputs")
    outputs = report.get("outputs")
    view = inputs.get("executable_view") if isinstance(inputs, Mapping) else None
    view_report = view.get("report") if isinstance(view, Mapping) else None
    if (
        report.get("schema") != "direct-compact-openai-rs-harvest-v2"
        or report.get("status") != "complete"
        or report.get("provider") != "openai"
        or report.get("api") != "responses"
        or report.get("base_url") != "https://api.openai.com/v1"
        or report.get("requested_model") != "gpt-5.6-sol"
        or report.get("production_coverage_met") is not True
        or not isinstance(inputs, Mapping)
        or not isinstance(outputs, Mapping)
        or (inputs.get("train_file") or {}).get("sha256")
        != sha256_file(base_path)
        or (inputs.get("train_seal") or {}).get("sha256")
        != sha256_file(base_seal_path)
        or not isinstance(view_report, Mapping)
        or view_report.get("sha256") != sha256_file(executable_view_report)
        or outputs.get("verified_repairs_sha256")
        != sha256_file(repair_path)
    ):
        raise ValueError(
            "repair artifact is not the sealed official synchronous "
            "gpt-5.6-sol executable-view harvest"
        )
    return report


def validate_file_record(
    value: Any,
    expected_path: Path,
    label: str,
) -> None:
    """Require a manifest record to identify the exact local file."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} file record is missing")
    try:
        recorded_path = Path(str(value.get("path") or "")).expanduser().resolve()
        recorded_size = int(value.get("size_bytes", -1))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} file record is malformed") from exc
    if (
        recorded_path != expected_path.resolve()
        or value.get("sha256") != sha256_file(expected_path)
        or recorded_size != expected_path.stat().st_size
    ):
        raise ValueError(f"{label} file record does not bind the exact file")


def validate_imported_repair_manifest(
    *,
    provider: str,
    repair_path: Path,
    report_path: Path,
    report: Mapping[str, Any],
    base_path: Path,
    base_seal_path: Path,
    contract_path: Path,
    executable_view_report: Path,
    base_seal: Mapping[str, Any],
    executable_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a provider-specific, independently re-certified import."""

    inputs = report.get("inputs")
    outputs = report.get("outputs")
    identity = report.get("provider")
    fit = report.get("fit_universe")
    verifier = report.get("verifier")
    invariants = report.get("invariants")
    if (
        report.get("schema") != IMPORTED_REPAIR_MANIFEST_SCHEMA
        or report.get("status") != "complete"
        or report.get("provider_key") != provider
        or not isinstance(identity, Mapping)
        or identity.get("key") != provider
        or report.get("requested_model") != identity.get("requested_model")
        or report.get("api") != identity.get("api")
        or identity.get("returned_model_must_equal_requested") is not True
        or set(identity.get("returned_models") or [])
        != {str(identity.get("requested_model") or "")}
        or report.get("code_only") is not True
        or report.get("reasoning_is_not_training_target") is not True
        or not isinstance(inputs, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(fit, Mapping)
        or not isinstance(verifier, Mapping)
        or not isinstance(invariants, Mapping)
    ):
        raise ValueError(
            f"{provider}: imported hard-target manifest contract failed"
        )

    validate_file_record(inputs.get("base_train"), base_path, "base train")
    validate_file_record(
        inputs.get("base_train_seal"), base_seal_path, "base train seal"
    )
    validate_file_record(inputs.get("contract"), contract_path, "contract")
    validate_file_record(
        inputs.get("executable_view_report"),
        executable_view_report,
        "executable-view report",
    )
    source_input_paths: dict[str, Path] = {}
    for name in ("source_targets", "source_seal", "source_report", "evaluator"):
        record = inputs.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"{provider}: source input {name!r} is missing")
        source_path = Path(
            str(record.get("path") or "")
        ).expanduser().resolve()
        validate_file_record(record, source_path, f"source input {name}")
        source_input_paths[name] = source_path
    if (
        sha256_file(source_input_paths["evaluator"])
        != base_seal.get("evaluator_sha256")
    ):
        raise ValueError(f"{provider}: source evaluator is not production")
    validate_file_record(
        outputs.get("verified_repairs"), repair_path, "verified repairs"
    )
    seal_record = outputs.get("verified_repairs_seal")
    if not isinstance(seal_record, Mapping):
        raise ValueError(f"{provider}: imported repair seal record is missing")
    seal_path = Path(str(seal_record.get("path") or "")).expanduser().resolve()
    validate_file_record(seal_record, seal_path, "verified repair seal")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{provider}: imported repair seal is invalid") from exc
    if not isinstance(seal, Mapping):
        raise ValueError(f"{provider}: imported repair seal is not an object")

    expected_fit = {
        "base_train_sha256": sha256_file(base_path),
        "base_train_seal_sha256": sha256_file(base_seal_path),
        "contract_sha256": sha256_file(contract_path),
        "executable_view_report_sha256": sha256_file(
            executable_view_report
        ),
        "parent_fit_rows": int(executable_view.get("parent_rows", -1)),
        "executable_rows": int(executable_view.get("rows", base_seal["rows"])),
        "heldout_rows": int(executable_view.get("heldout_rows", -1)),
        "heldout_task_ids_sha256": executable_view.get(
            "heldout_task_ids_sha256"
        ),
        "heldout_intersection_count": 0,
        "heldout_bytes_opened_during_harvest": False,
    }
    for field, expected in expected_fit.items():
        if fit.get(field) != expected:
            raise ValueError(
                f"{provider}: imported fit binding {field!r} differs"
            )
    expected_verifier = {
        "evaluator_sha256": base_seal.get("evaluator_sha256"),
        "completion_attestation_id": base_seal.get(
            "completion_attestation_id"
        ),
        "dart_version": base_seal.get("dart_version"),
        "stability_runs": int(base_seal.get("stability_runs", -1)),
        "compiled": True,
        "passed": True,
        "acceptance_holdback_exposed_to_provider": False,
        "heldout_tests_exposed_to_provider": False,
        "all_candidates_compiled": True,
        "all_candidates_passed": True,
    }
    for field, expected in expected_verifier.items():
        if verifier.get(field) != expected:
            raise ValueError(
                f"{provider}: imported verifier binding {field!r} differs"
            )
    required_invariants = {
        "provider_identity_preserved": True,
        "provider_relabeling_permitted": False,
        "deterministic_exact_code_dedupe": True,
        "all_outputs_independently_recertified": True,
        "reasoning_excluded_from_training_targets": True,
        "fit2776_membership_bound": True,
        "heldout175_intersection_zero": True,
        "heldout_bytes_opened_during_import": False,
        "source_nonexecutable_targets_never_imported": True,
    }
    for field, expected in required_invariants.items():
        if invariants.get(field) is not expected:
            raise ValueError(
                f"{provider}: imported invariant {field!r} is not sealed"
            )

    importer_path = Path(__file__).with_name(
        "import_direct_compact_rs_hard_targets.py"
    )
    imported_rows = read_jsonl(repair_path)
    imported_keys: list[dict[str, str]] = []
    imported_tasks: list[str] = []
    for index, row in enumerate(imported_rows):
        task_id = str(row.get("task_id") or "")
        code = str(row.get("code") or "")
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        if (
            row.get("schema") != IMPORTED_REPAIR_ROW_SCHEMA
            or row.get("ok") is not True
            or row.get("provider_key") != provider
            or dict(row.get("provider") or {}) != dict(identity)
            or not task_id
            or not code.strip()
            or row.get("code_sha256") != code_sha
            or not isinstance(row.get("provider_provenance"), Mapping)
            or PROHIBITED_REASONING_FIELDS.intersection(row)
        ):
            raise ValueError(
                f"{provider}: imported repair row {index} is malformed"
            )
        independent = row.get("independent_recertification")
        if not isinstance(independent, Mapping):
            raise ValueError(
                f"{provider}: imported repair row {index} lacks recertification"
            )
        for field, expected in expected_verifier.items():
            if field.startswith("all_candidates_"):
                continue
            if independent.get(field) != expected:
                raise ValueError(
                    f"{provider}: imported row {index} verifier differs"
                )
        imported_tasks.append(task_id)
        imported_keys.append(
            {"task_id": task_id, "code_sha256": code_sha}
        )

    counts = report.get("counts") or {}
    seal_fit = seal.get("fit_universe")
    seal_verifier = seal.get("verifier")
    if (
        seal.get("schema") != IMPORTED_REPAIR_SEAL_SCHEMA
        or seal.get("selected_role") != "fit"
        or seal.get("training_allowed") is not True
        or seal.get("provider_key") != provider
        or dict(seal.get("provider") or {}) != dict(identity)
        or seal.get("code_only") is not True
        or seal.get("output_sha256") != sha256_file(repair_path)
        or int(seal.get("rows", -1)) != len(imported_rows)
        or int(seal.get("unique_tasks", -1)) != len(set(imported_tasks))
        or dict(seal_fit or {}) != dict(fit)
        or dict(seal_verifier or {}) != dict(verifier)
        or seal.get("ordered_candidate_keys_sha256")
        != stable_sha256(imported_keys)
        or seal.get("task_set_sha256")
        != stable_sha256(sorted(set(imported_tasks)))
        or seal.get("heldout_bytes_opened_during_import") is not False
        or seal.get("source_seal_sha256")
        != sha256_file(source_input_paths["source_seal"])
        or not importer_path.is_file()
        or seal.get("importer_sha256") != sha256_file(importer_path)
        or int(counts.get("output_rows", -1)) != len(imported_rows)
        or int(counts.get("unique_tasks", -1)) != len(set(imported_tasks))
        or int(counts.get("independent_recertification_failures", -1)) != 0
    ):
        raise ValueError(
            f"{provider}: imported repair output seal contract failed"
        )
    return dict(report)


def validate_repair_report(
    *,
    provider: str,
    repair_path: Path,
    report_path: Path,
    base_path: Path,
    base_seal_path: Path,
    contract_path: Path,
    executable_view_report: Path,
    base_seal: Mapping[str, Any],
    executable_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Dispatch without weakening the legacy GPT-5.6 collector seal."""

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{report_path}: invalid repair report") from exc
    if not isinstance(report, Mapping):
        raise ValueError(f"{report_path}: repair report is not an object")
    if report.get("schema") == IMPORTED_REPAIR_MANIFEST_SCHEMA:
        return validate_imported_repair_manifest(
            provider=provider,
            repair_path=repair_path,
            report_path=report_path,
            report=report,
            base_path=base_path,
            base_seal_path=base_seal_path,
            contract_path=contract_path,
            executable_view_report=executable_view_report,
            base_seal=base_seal,
            executable_view=executable_view,
        )
    return validate_openai_repair_report(
        provider=provider,
        repair_path=repair_path,
        report_path=report_path,
        base_path=base_path,
        base_seal_path=base_seal_path,
        executable_view_report=executable_view_report,
    )


def compact_source_fingerprint(row: Mapping[str, Any]) -> str:
    return stable_sha256({field: row.get(field) for field in MODEL_BINDING_FIELDS})


def exact_seal(
    path: Path,
    contract_path: Path,
    rows: int,
    *,
    parent_dataset: Path,
    parent_seal: Path,
    parent_seal_value: Mapping[str, Any],
    executable_view_report: Path,
) -> dict[str, Any]:
    seal = {
        "schema": "compact-public-private-join-seal-v1",
        "selected_role": "fit",
        "training_allowed": True,
        "heldout_measure_only": False,
        "output_sha256": sha256_file(path),
        "contract_sha256": sha256_file(contract_path),
        "rows": rows,
        "representation_schema": REPRESENTATION_SCHEMA,
        "training_objective_scope": "matched_rs_sft_or_gold_control",
        "source_executable_dataset_sha256": sha256_file(parent_dataset),
        "source_executable_seal_sha256": sha256_file(parent_seal),
        "source_executable_view_report_sha256": sha256_file(
            executable_view_report
        ),
        "execution_ineligible_task_ids": [],
    }
    for field in (
        "adapter_contract_sha256",
        "adapter_script_sha256",
        "source_function_bundles_sha256",
        "source_symbol_attestation_used",
        "source_symbol_attestation_is_keyed",
        "source_symbol_attestation_file_sha256",
        "source_symbol_attestation_key_id_sha256",
        "raw_source_names_serialized",
        "sanitation_schema",
        "sanitizer_sha256",
        "evaluator_sha256",
        "completion_attestation_id",
        "dart_version",
        "stability_runs",
        "quarantine_sha256",
    ):
        if field not in parent_seal_value:
            raise ValueError(f"executable parent seal lacks {field!r}")
        seal[field] = parent_seal_value[field]
    return seal


def choose_repairs(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Choose one deterministic implementation without losing provenance."""

    def contribution(row: Mapping[str, Any]) -> dict[str, Any]:
        value = {
            "provider": str(row.get("provider") or ""),
            "provider_identity": dict(row.get("provider_identity") or {}),
            "artifact_sha256": str(row.get("artifact_sha256") or ""),
            "repair_report_sha256": str(
                row.get("repair_report_sha256") or ""
            ),
            "artifact_row": int(row.get("artifact_row", -1)),
            "source_row_sha256": str(row.get("source_row_sha256") or ""),
            "source_schema": str(row.get("source_schema") or ""),
            "provider_provenance": dict(
                row.get("provider_provenance") or {}
            ),
        }
        value["contribution_sha256"] = stable_sha256(value)
        return value

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_task.setdefault(row["task_id"], []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    alternatives: dict[str, int] = {}
    for task_id, rows in by_task.items():
        distinct: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            code_sha = hashlib.sha256(row["code"].encode()).hexdigest()
            distinct.setdefault(code_sha, []).append(row)
        canonical: list[dict[str, Any]] = []
        for code_sha, duplicate_rows in distinct.items():
            ordered_duplicates = sorted(
                duplicate_rows,
                key=lambda row: (
                    stable_sha256(contribution(row)),
                    stable_sha256(row),
                ),
            )
            representative = dict(ordered_duplicates[0])
            contributors = sorted(
                (contribution(row) for row in duplicate_rows),
                key=lambda value: (
                    value["contribution_sha256"],
                    stable_sha256(value),
                ),
            )
            representative["code_sha256"] = code_sha
            representative["dedupe_contributors"] = contributors
            representative["dedupe_contributors_sha256"] = stable_sha256(
                contributors
            )
            canonical.append(representative)
        ordered = sorted(
            canonical,
            key=lambda row: (
                len(row["code"]),
                row["code_sha256"],
                row["provider"],
                row["dedupe_contributors_sha256"],
            ),
        )
        selected[task_id] = ordered[0]
        alternatives[task_id] = len(ordered)
    return selected, alternatives


def main() -> None:
    args = parse_args()
    if (
        not args.rows_per_arm_from_sealed_parent
        and (args.rows_per_arm <= 0 or args.rows_per_arm % 2)
    ):
        raise ValueError("--rows_per_arm must be a positive even integer")
    if args.expected_parent_fit_rows < 0:
        raise ValueError("--expected_parent_fit_rows cannot be negative")
    if args.min_unique_repairs <= 0:
        raise ValueError("--min_unique_repairs must be positive")
    if args.workers <= 0 or args.timeout <= 0 or args.stability_runs <= 0:
        raise ValueError("workers, timeout, and stability_runs must be positive")

    base_path = Path(args.base_train).expanduser().resolve()
    base_seal_path = Path(args.base_train_seal).expanduser().resolve()
    contract_path = Path(args.contract).expanduser().resolve()
    executable_view_report = Path(
        args.executable_view_report
    ).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (
        base_path,
        base_seal_path,
        contract_path,
        executable_view_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = DirectCompactContract.load(contract_path)
    base_seal = validate_join_seal(
        base_path,
        base_seal_path,
        contract_path,
        expected_role="fit",
    )
    executable_report_value = json.loads(
        executable_view_report.read_text(encoding="utf-8")
    )
    if not isinstance(executable_report_value, dict):
        raise ValueError("executable-view build report is not an object")
    executable_outputs = executable_report_value.get("outputs") or {}
    executable_view = validate_executable_view(
        dataset=base_path,
        seal=base_seal_path,
        f2=(executable_outputs.get("f2") or {}).get("path", ""),
        f2_manifest=(executable_outputs.get("f2_manifest") or {}).get(
            "path", ""
        ),
        build_report=executable_view_report,
        expected_build_report_sha256=(
            args.expected_executable_view_report_sha256
        ),
        contract=contract_path,
        verify_heldout=False,
        expected_parent_rows=(
            args.expected_parent_fit_rows or None
        ),
    )
    base_rows = read_jsonl(base_path)
    sealed_executable_rows = int(
        executable_view.get("rows", len(base_rows))
    )
    sealed_parent_rows = int(
        executable_view.get("parent_rows", sealed_executable_rows)
    )
    if len(base_rows) != int(base_seal["rows"]):
        raise RuntimeError("base train row count changed after seal validation")
    if len(base_rows) != sealed_executable_rows:
        raise ValueError(
            "base train rows differ from the sealed executable-view report"
        )
    if args.rows_per_arm_from_sealed_parent:
        args.rows_per_arm = 2 * sealed_executable_rows
    if (
        sealed_parent_rows <= 0
        or executable_view["heldout_rows"] != 175
    ) and not args.allow_low_coverage_smoke:
        raise ValueError(
            "production RS-SFT requires a sealed fit parent and historical "
            "heldout175"
        )
    if (
        args.rows_per_arm != 2 * sealed_executable_rows
        and not args.allow_low_coverage_smoke
    ):
        raise ValueError(
            "--rows_per_arm must be twice the sealed executable row count so "
            "the matched gold half covers every executable task exactly once"
        )

    base: dict[str, dict[str, Any]] = {}
    base_sources: dict[str, str] = {}
    for index, row in enumerate(base_rows):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in base:
            raise ValueError(f"base train has missing/duplicate task_id at row {index}")
        contract.validate_row(row, task_id)
        if (
            row.get("binary_multifunction_schema") != REPRESENTATION_SCHEMA
            and not args.allow_low_coverage_smoke
        ):
            raise ValueError(
                f"{task_id}: base train is not the multi-function representation"
            )
        if not str(row.get("dart_source") or "").strip():
            raise ValueError(f"{task_id}: base train row has no gold target")
        base[task_id] = row
        base_sources[task_id] = compact_source_fingerprint(row)

    repair_specs = parse_repair_specs(args.repairs)
    repair_report_specs = dict(parse_repair_specs(args.repair_report))
    if set(repair_report_specs) != {provider for provider, _ in repair_specs}:
        raise ValueError(
            "--repair_report providers must exactly match --repairs providers"
        )
    repair_reports: dict[str, dict[str, Any]] = {}
    for provider, repair_path in repair_specs:
        repair_reports[provider] = validate_repair_report(
            provider=provider,
            repair_path=repair_path,
            report_path=repair_report_specs[provider],
            base_path=base_path,
            base_seal_path=base_seal_path,
            contract_path=contract_path,
            executable_view_report=executable_view_report,
            base_seal=base_seal,
            executable_view=executable_view,
        )
    raw_candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for provider, path in repair_specs:
        provider_report = repair_reports[provider]
        is_imported = (
            provider_report.get("schema")
            == IMPORTED_REPAIR_MANIFEST_SCHEMA
        )
        if is_imported:
            provider_identity = dict(provider_report["provider"])
        else:
            provider_identity = {
                "key": provider,
                "organization": provider_report.get("provider"),
                "api": provider_report.get("api"),
                "base_url": provider_report.get("base_url"),
                "requested_model": provider_report.get("requested_model"),
                "returned_models": list(
                    provider_report.get("resolved_models") or []
                ),
                "report_schema": provider_report.get("schema"),
            }
        artifact_sha = sha256_file(path)
        repair_report_sha = sha256_file(repair_report_specs[provider])
        for index, row in enumerate(read_jsonl(path)):
            task_id = str(row.get("task_id") or "")
            raw_code = str(row.get("code") or row.get("repair") or "")
            code = raw_code if is_imported else raw_code.strip()
            code_sha = hashlib.sha256(code.encode()).hexdigest()
            if is_imported and (
                row.get("schema") != IMPORTED_REPAIR_ROW_SCHEMA
                or row.get("provider_key") != provider
                or dict(row.get("provider") or {}) != provider_identity
                or row.get("code_sha256") != code_sha
                or not isinstance(row.get("provider_provenance"), Mapping)
            ):
                raise ValueError(
                    f"{provider}: imported row {index} changed after "
                    "manifest validation"
                )
            if row.get("ok") is False:
                rejected.append(
                    {
                        "provider": provider,
                        "row": index,
                        "task_id": task_id,
                        "reason": "artifact_marks_not_ok",
                    }
                )
                continue
            if not task_id or task_id not in base:
                rejected.append(
                    {
                        "provider": provider,
                        "row": index,
                        "task_id": task_id,
                        "reason": "not_in_compact_train_partition",
                    }
                )
                continue
            if not code:
                rejected.append(
                    {
                        "provider": provider,
                        "row": index,
                        "task_id": task_id,
                        "reason": "empty_code",
                    }
                )
                continue
            raw_candidates.append(
                {
                    "provider": provider,
                    "artifact": str(path),
                    "artifact_sha256": artifact_sha,
                    "artifact_row": index,
                    "repair_report_sha256": repair_report_sha,
                    "source_row_sha256": stable_sha256(row),
                    "source_schema": str(row.get("schema") or ""),
                    "provider_identity": provider_identity,
                    "provider_provenance": (
                        dict(row["provider_provenance"])
                        if is_imported
                        else {
                            key: value
                            for key, value in row.items()
                            if key not in {"code", "repair"}
                        }
                    ),
                    "task_id": task_id,
                    "code": code,
                }
            )

    project_root = Path(__file__).resolve().parents[2]
    evaluator_dir = project_root / "scripts" / "evaluation"
    if str(evaluator_dir) not in sys.path:
        sys.path.insert(0, str(evaluator_dir))
    from graph_compile_at_k_antigravity import evaluate_dart_jit_tests_detail

    def recertify(candidate: dict[str, Any]) -> dict[str, Any]:
        task = base[candidate["task_id"]]
        tests = str(task.get("acceptance_tests") or task.get("tests") or "")
        # Fenced input is the evaluator's lossless full-program mode.  Passing
        # raw code can discard a leading enum/class/extension as if it were
        # explanatory prose.
        raw = f"```dart\n{candidate['code'].rstrip()}\n```"
        compiled, passed, diagnostic, _source = evaluate_dart_jit_tests_detail(
            raw,
            tests,
            candidate["task_id"],
            timeout=args.timeout,
            stability_runs=args.stability_runs,
        )
        return {
            **candidate,
            "compiled": bool(compiled),
            "passed": bool(passed),
            "diagnostic": str(diagnostic)[:1000],
            "code_sha256": hashlib.sha256(candidate["code"].encode()).hexdigest(),
        }

    recertified: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(recertify, candidate) for candidate in raw_candidates]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["passed"]:
                recertified.append(result)
            else:
                rejected.append(
                    {
                        key: result.get(key)
                        for key in (
                            "provider",
                            "artifact",
                            "artifact_row",
                            "task_id",
                            "compiled",
                            "diagnostic",
                            "code_sha256",
                        )
                    }
                    | {"reason": "independent_recertification_failed"}
                )

    selected, alternatives = choose_repairs(recertified)
    unique_repairs = len(selected)
    if unique_repairs == 0:
        raise RuntimeError("no repair passed independent recertification")
    if (
        unique_repairs < args.min_unique_repairs
        and not args.allow_low_coverage_smoke
    ):
        raise RuntimeError(
            f"only {unique_repairs} unique repairs passed recertification; "
            f"production floor is {args.min_unique_repairs}"
        )

    rng = random.Random(args.seed)
    half = args.rows_per_arm // 2
    base_ids = sorted(base)
    rng.shuffle(base_ids)
    gold_ids = [base_ids[index % len(base_ids)] for index in range(half)]

    repair_ids = sorted(selected)
    rng.shuffle(repair_ids)
    repair_sequence = [
        repair_ids[index % len(repair_ids)] for index in range(half)
    ]
    # Avoid grouping the intervention in one contiguous half.  Each tuple is
    # (kind, task_id); the same order drives both experimental and control arms.
    schedule = [("gold", task_id) for task_id in gold_ids] + [
        ("repair", task_id) for task_id in repair_sequence
    ]
    rng.shuffle(schedule)

    intervention: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    schedule_report: list[dict[str, Any]] = []
    for position, (kind, task_id) in enumerate(schedule):
        gold_row = dict(base[task_id])
        experimental_row = dict(gold_row)
        if kind == "repair":
            repair = selected[task_id]
            experimental_row["dart_source"] = repair["code"]
            target_sha = repair["code_sha256"]
            provider = repair["provider"]
            provider_identity = repair["provider_identity"]
            provider_provenance = repair["provider_provenance"]
            dedupe_contributors = repair["dedupe_contributors"]
        else:
            target_sha = hashlib.sha256(
                str(gold_row["dart_source"]).encode()
            ).hexdigest()
            provider = "gold"
            provider_identity = None
            provider_provenance = None
            dedupe_contributors = []
        intervention.append(experimental_row)
        control.append(gold_row)
        schedule_report.append(
            {
                "position": position,
                "kind": kind,
                "task_id": task_id,
                "provider": provider,
                "provider_identity": provider_identity,
                "provider_provenance": provider_provenance,
                "dedupe_contributors": dedupe_contributors,
                "source_fingerprint": base_sources[task_id],
                "experimental_target_sha256": target_sha,
                "control_target_sha256": hashlib.sha256(
                    str(gold_row["dart_source"]).encode()
                ).hexdigest(),
            }
        )

    if len(intervention) != len(control) or len(intervention) != args.rows_per_arm:
        raise AssertionError("matched arm row counts drifted")
    kinds = Counter(row["kind"] for row in schedule_report)
    if kinds != {"gold": half, "repair": half}:
        raise AssertionError(f"intervention is not exact 50/50: {kinds}")
    for experimental_row, control_row in zip(intervention, control, strict=True):
        if compact_source_fingerprint(experimental_row) != compact_source_fingerprint(
            control_row
        ):
            raise AssertionError("matched arm compact inputs differ")

    intervention_path = output_dir / "rs_sft_50_50.jsonl"
    control_path = output_dir / "gold_only_matched.jsonl"
    write_jsonl(intervention_path, intervention)
    write_jsonl(control_path, control)
    intervention_seal = exact_seal(
        intervention_path,
        contract_path,
        len(intervention),
        parent_dataset=base_path,
        parent_seal=base_seal_path,
        parent_seal_value=base_seal,
        executable_view_report=executable_view_report,
    )
    control_seal = exact_seal(
        control_path,
        contract_path,
        len(control),
        parent_dataset=base_path,
        parent_seal=base_seal_path,
        parent_seal_value=base_seal,
        executable_view_report=executable_view_report,
    )
    (output_dir / "rs_sft_50_50.seal.json").write_text(
        json.dumps(intervention_seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gold_only_matched.seal.json").write_text(
        json.dumps(control_seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "schedule.jsonl", schedule_report)
    write_jsonl(output_dir / "rejected_repairs.jsonl", rejected)

    per_provider = Counter(row["provider"] for row in selected.values())
    contributor_per_provider: Counter[str] = Counter()
    for repair in selected.values():
        contributor_per_provider.update(
            str(value.get("provider") or "")
            for value in repair["dedupe_contributors"]
        )
    repair_repetitions = Counter(repair_sequence)
    report = {
        "schema": "direct-compact-rs-sft-matched-build-v1",
        "seed": args.seed,
        "base_train": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "seal_path": str(base_seal_path),
            "seal_sha256": sha256_file(base_seal_path),
            "rows": len(base_rows),
        },
        "heldout": {
            "dataset": executable_view["heldout"],
            "seal": executable_view["heldout_seal"],
            "rows": executable_view["heldout_rows"],
            "task_set_sha256": executable_view[
                "heldout_task_ids_sha256"
            ],
            "measure_only": True,
            "bytes_opened_during_build": False,
            "used_for_stage_selection_or_launch": False,
        },
        "executable_view": executable_view,
        "sealed_fit_accounting": {
            "parent_fit_rows": sealed_parent_rows,
            "executable_rows": sealed_executable_rows,
            "rows_per_matched_arm": args.rows_per_arm,
            "rows_per_arm_derived_from_seal": bool(
                args.rows_per_arm_from_sealed_parent
            ),
        },
        "contract_sha256": sha256_file(contract_path),
        "repair_artifacts": [
            {
                "provider": provider,
                "path": str(path),
                "sha256": sha256_file(path),
                "collector_report": {
                    "path": str(repair_report_specs[provider]),
                    "sha256": sha256_file(repair_report_specs[provider]),
                    "schema": repair_reports[provider]["schema"],
                    "status": repair_reports[provider]["status"],
                    "requested_model": repair_reports[provider][
                        "requested_model"
                    ],
                    "api": repair_reports[provider]["api"],
                    "provider_identity": (
                        repair_reports[provider]["provider"]
                        if repair_reports[provider]["schema"]
                        == IMPORTED_REPAIR_MANIFEST_SCHEMA
                        else {
                            "key": provider,
                            "organization": repair_reports[provider][
                                "provider"
                            ],
                            "api": repair_reports[provider]["api"],
                            "base_url": repair_reports[provider][
                                "base_url"
                            ],
                            "requested_model": repair_reports[provider][
                                "requested_model"
                            ],
                            "returned_models": repair_reports[provider].get(
                                "resolved_models", []
                            ),
                        }
                    ),
                    "verifier": repair_reports[provider].get("verifier"),
                },
            }
            for provider, path in repair_specs
        ],
        "repair_candidates": len(raw_candidates),
        "recertified_candidates": len(recertified),
        "unique_recertified_tasks": unique_repairs,
        "unique_repair_floor": args.min_unique_repairs,
        "low_coverage_smoke_override": bool(args.allow_low_coverage_smoke),
        "selected_by_provider": dict(sorted(per_provider.items())),
        "selected_code_contributors_by_provider": dict(
            sorted(contributor_per_provider.items())
        ),
        "provenance_invariants": {
            "provider_identity_preserved": True,
            "provider_relabeling_permitted": False,
            "exact_code_dedupe_is_input_order_independent": True,
            "all_exact_duplicate_contributors_retained": True,
        },
        "distinct_passing_alternatives": {
            "min": min(alternatives.values()),
            "max": max(alternatives.values()),
            "total": sum(alternatives.values()),
        },
        "arms": {
            "rows_each": args.rows_per_arm,
            "experimental_gold_rows": half,
            "experimental_repair_rows": half,
            "control_gold_rows": args.rows_per_arm,
            "source_sequence_exactly_matched": True,
            "repair_repetitions_per_task_min": min(repair_repetitions.values()),
            "repair_repetitions_per_task_max": max(repair_repetitions.values()),
        },
        "outputs": {
            "intervention": {
                "path": str(intervention_path),
                "sha256": sha256_file(intervention_path),
                "seal_sha256": sha256_file(
                    output_dir / "rs_sft_50_50.seal.json"
                ),
            },
            "control": {
                "path": str(control_path),
                "sha256": sha256_file(control_path),
                "seal_sha256": sha256_file(
                    output_dir / "gold_only_matched.seal.json"
                ),
            },
            "schedule_sha256": sha256_file(output_dir / "schedule.jsonl"),
            "rejections_sha256": sha256_file(
                output_dir / "rejected_repairs.jsonl"
            ),
        },
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
