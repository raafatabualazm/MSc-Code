#!/usr/bin/env python3
"""Pre-registered typed direct-only RS-SFT pass 3.

Pass 3 deliberately branches from the pass-1 update-58 checkpoint.  Its
training set is only the new, direct, privately verified Kimi continuation
targets: cohort 1, the completed cohort-2 tail, and the separately verified
three-call cohort-2 prefix.  It never replays pass-1/pass-2 rows or gold.

The cohort-2 row count is late-bound.  All three source triplets are SHA-pinned
by the launcher and audited before their row count is used.  Complete TRAIN
tests are used only for local re-verification and never serialized into a
model-visible row.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_contract_sft as typed_sft
from scripts.training import t5gemma2_typed_direct_rs_sft as pass1
from scripts.training import t5gemma2_typed_direct_rs_sft_pass2 as pass2
from scripts.training import t5gemma2_typed_dual_api_orchestrator as dual
from scripts.training import t5gemma2_typed_kimi_continuation as c001
from scripts.training import t5gemma2_typed_kimi_continuation_c002 as c002
from scripts.training import t5gemma2_typed_kimi_c002_resume47 as resume47


RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-checkpoint-v1"
DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-dataset-v1"
PREFIX_REPORT_SCHEMA = "t5gemma2-typed-c002-prefix3-verification-report-v1"
PREFIX_MANIFEST_SCHEMA = "t5gemma2-typed-c002-prefix3-direct-manifest-v1"
PREFIX_TARGET_SCHEMA = "t5gemma2-typed-c002-prefix3-direct-target-v1"

EXPECTED_C001_ROWS = 12
EXPECTED_PREFIX_ROWS = 1
EXPECTED_PREFIX_TASK_ID = "fresh-eval-dba1fc9af285"
MINIMUM_ROWS = EXPECTED_C001_ROWS + EXPECTED_PREFIX_ROWS
MAX_C002_ROWS = 47

SOURCE_BASENAMES = (
    "continuation_report.json",
    "direct_manifest.json",
    "direct_targets.jsonl",
    "resume_report.json",
    "direct_manifest.json",
    "direct_targets.jsonl",
    "prefix_verification_report.json",
    "direct_manifest.json",
    "direct_targets.jsonl",
)

_MIXED_RUNTIME_CONTRACT = mixed._runtime_contract  # noqa: SLF001
_MIXED_VALIDATE_WARMSTART = mixed.validate_warmstart


def _read_object(path: Path, label: str) -> dict[str, Any]:
    return pass2._read_object(path, label)  # noqa: SLF001


def _read_jsonl(
    path: Path, label: str, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    return pass2._read_jsonl(path, label, allow_empty=allow_empty)  # noqa: SLF001


def _require_sources(
    specs: Sequence[tuple[Path, str]],
) -> tuple[tuple[Path, str], ...]:
    if len(specs) != len(SOURCE_BASENAMES):
        raise ValueError("pass-3 requires exactly three pinned source triplets")
    result = tuple(specs)
    for position, ((path, digest), basename) in enumerate(
        zip(result, SOURCE_BASENAMES, strict=True)
    ):
        if path.name != basename or sha256_file(path) != digest:
            raise ValueError(f"pass-3 source artifact {position} binding differs")
    for offset in (0, 3, 6):
        if len({path.parent for path, _digest in result[offset : offset + 3]}) != 1:
            raise ValueError("pass-3 source triplet artifacts are not siblings")
    return result


def _audit_direct_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_schema: str,
    expected_ids: set[str] | None = None,
) -> list[str]:
    task_ids: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        code = str(row.get("dart_source") or "")
        if (
            row.get("schema") != expected_schema
            or not task_id
            or not code.strip()
            or row.get("visible_train_passed") is not True
            or row.get("private_full_acceptance_passed") is not True
            or row.get("stability_runs") != 2
            or row.get("reasoning_present") is not False
            or row.get("repair_conditioned_training_source_present") is not False
            or row.get("gold_replay") is not False
            or mixed.sha256_text(code) != row.get("dart_source_sha256")
        ):
            raise ValueError("pass-3 source contains a non-direct or unsafe row")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("pass-3 source contains duplicate task identities")
    if expected_ids is not None and set(task_ids) != expected_ids:
        raise ValueError("pass-3 source target identities differ")
    return task_ids


def _audit_c001(
    report_spec: tuple[Path, str],
    manifest_spec: tuple[Path, str],
    targets_spec: tuple[Path, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path, report_sha = report_spec
    records, _remaining, predecessor = c002.load_completed_c001(
        report_path=report_path,
        expected_report_sha256=report_sha,
    )
    report = _read_object(report_path, "cohort-1 continuation report")
    manifest_path, manifest_sha = manifest_spec
    targets_path, targets_sha = targets_spec
    manifest = _read_object(manifest_path, "cohort-1 direct manifest")
    rows = _read_jsonl(targets_path, "cohort-1 direct targets")
    task_ids = _audit_direct_rows(rows, expected_schema=dual.cascade.DIRECT_TARGET_SCHEMA)
    embedded = report.get("new_direct_manifest")
    if (
        manifest != embedded
        or manifest.get("schema") != dual.AGGREGATE_SCHEMA
        or manifest.get("rows") != EXPECTED_C001_ROWS
        or len(rows) != EXPECTED_C001_ROWS
        or manifest.get("targets", {}).get("sha256") != targets_sha
        or manifest.get("task_ids_sha256") != canonical_sha256(task_ids)
        or manifest.get("direct_only") is not True
        or manifest.get("visible_and_private_verified") is not True
        or manifest.get("reasoning_rows") != 0
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
        or manifest.get("tests_in_training_output") is not False
        or manifest.get("diagnostics_in_training_output") is not False
    ):
        raise ValueError("cohort-1 aggregate binding differs")
    return rows, {
        "schema": c001.REPORT_SCHEMA,
        "report": {"path": str(report_path), "sha256": report_sha},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "targets": {"path": str(targets_path), "sha256": targets_sha, "rows": len(rows)},
        "validated_phase_reports": len(records),
        "predecessor": predecessor,
        "task_ids_sha256": canonical_sha256(task_ids),
        "heldout_175_model_visible": False,
    }


def _audit_resume47(
    report_spec: tuple[Path, str],
    manifest_spec: tuple[Path, str],
    targets_spec: tuple[Path, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path, report_sha = report_spec
    manifest_path, manifest_sha = manifest_spec
    targets_path, targets_sha = targets_spec
    report = _read_object(report_path, "cohort-2 resume report")
    manifest = _read_object(manifest_path, "cohort-2 resume direct manifest")
    rows = _read_jsonl(targets_path, "cohort-2 resume direct targets", allow_empty=True)
    task_ids = _audit_direct_rows(rows, expected_schema=dual.cascade.DIRECT_TARGET_SCHEMA)
    phases = report.get("tail_phases")
    retry = report.get("retry")
    prefix = report.get("prefix_disposition")
    budget = report.get("budget")
    if (
        report.get("schema") != resume47.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("heldout_175_opened") is not False
        or not isinstance(phases, list)
        or not phases
        or len(phases) > 2
        or not isinstance(retry, Mapping)
        or retry.get("partial_retry_executed") is not False
        or not (
            retry.get("complete_exact_set_executed") is True
            or retry.get("not_needed") is True
            or retry.get("budget_skipped_entire_set") is True
        )
        or not isinstance(prefix, Mapping)
        or prefix.get("paid_results") != 3
        or prefix.get("private_verified_in_this_stage") is not False
        or prefix.get("training_used_in_this_stage") is not False
        or not isinstance(budget, Mapping)
        or budget.get("within_contract") is not True
        or manifest != report.get("new_tail_direct_manifest")
        or manifest.get("schema") != dual.AGGREGATE_SCHEMA
        or manifest.get("rows") != len(rows)
        or len(rows) > MAX_C002_ROWS
        or manifest.get("targets", {}).get("sha256") != targets_sha
        or manifest.get("task_ids_sha256") != canonical_sha256(task_ids)
        or manifest.get("direct_only") is not True
        or manifest.get("visible_and_private_verified") is not True
        or manifest.get("reasoning_rows") != 0
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
        or manifest.get("tests_in_training_output") is not False
        or manifest.get("diagnostics_in_training_output") is not False
    ):
        raise ValueError("cohort-2 resume aggregate binding differs")
    inspected: list[dict[str, Any]] = []
    for position, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise ValueError("cohort-2 resume phase record is malformed")
        phase_path = report_path.parent / (
            "kimi_initial_tail47" if position == 0 else "kimi_retry_tail47"
        ) / "typed_api_rescue_report.json"
        record = c001.inspect_phase_report(
            phase_path,
            expected_phase=str(phase.get("phase") or ""),
            expected_cohort=2,
        )
        if any(
            record[key] != phase.get(key)
            for key in ("report_sha256", "journal_sha256", "targets_sha256")
        ):
            raise ValueError("cohort-2 phase evidence differs")
        inspected.append(record)
    phase_ids = [task_id for record in inspected for task_id in record["verified_task_ids"]]
    if task_ids != phase_ids:
        raise ValueError("cohort-2 aggregate order differs from its phase evidence")
    return rows, {
        "schema": resume47.REPORT_SCHEMA,
        "report": {"path": str(report_path), "sha256": report_sha},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "targets": {"path": str(targets_path), "sha256": targets_sha, "rows": len(rows)},
        "validated_phase_reports": len(inspected),
        "source_partial": report.get("source_partial"),
        "task_ids_sha256": canonical_sha256(task_ids),
        "heldout_175_model_visible": False,
    }


def _audit_prefix(
    report_spec: tuple[Path, str],
    manifest_spec: tuple[Path, str],
    targets_spec: tuple[Path, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path, report_sha = report_spec
    manifest_path, manifest_sha = manifest_spec
    targets_path, targets_sha = targets_spec
    report = _read_object(report_path, "prefix verification report")
    manifest = _read_object(manifest_path, "prefix direct manifest")
    rows = _read_jsonl(targets_path, "prefix direct targets")
    task_ids = _audit_direct_rows(
        rows,
        expected_schema=PREFIX_TARGET_SCHEMA,
        expected_ids={EXPECTED_PREFIX_TASK_ID},
    )
    actual_journal = journal_record(report_path.parent / "prefix_verification.journal.jsonl")
    embedded_journal = report.get("journal")
    if (
        report.get("schema") != PREFIX_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("paid_prefix_tasks") != 3
        or report.get("provider_calls") != 0
        or report.get("provider_credentials_read") is not False
        or report.get("heldout_175_opened") is not False
        or report.get("verified_task_ids") != [EXPECTED_PREFIX_TASK_ID]
        or report.get("private_diagnostics_persisted") is not False
        or report.get("tests_in_training_output") is not False
        or report.get("direct_manifest") != manifest
        or not isinstance(embedded_journal, Mapping)
        or any(
            embedded_journal.get(key) != actual_journal.get(key)
            for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256")
        )
        or manifest.get("schema") != PREFIX_MANIFEST_SCHEMA
        or manifest.get("rows") != EXPECTED_PREFIX_ROWS
        or manifest.get("targets", {}).get("sha256") != targets_sha
        or manifest.get("task_ids_sha256") != canonical_sha256(task_ids)
        or manifest.get("direct_only") is not True
        or manifest.get("visible_and_private_verified") is not True
        or manifest.get("reasoning_rows") != 0
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
        or manifest.get("tests_in_training_output") is not False
    ):
        raise ValueError("prefix verification aggregate binding differs")
    return rows, {
        "schema": PREFIX_REPORT_SCHEMA,
        "report": {"path": str(report_path), "sha256": report_sha},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "targets": {"path": str(targets_path), "sha256": targets_sha, "rows": len(rows)},
        "task_ids_sha256": canonical_sha256(task_ids),
        "provider_calls": 0,
        "heldout_175_model_visible": False,
    }


def validate_pass3_warmstart(
    checkpoint: Path,
    *,
    expected_update: int,
    expected_run_contract_sha256: str,
    expected_adapter_weights_sha256: str,
    expected_adapter_config_sha256: str,
    model: str,
    model_revision: str,
) -> tuple[mixed.WarmstartIdentity, dict[str, Any]]:
    return pass2.validate_pass2_warmstart(
        checkpoint,
        expected_update=expected_update,
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_adapter_weights_sha256=expected_adapter_weights_sha256,
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        model=model,
        model_revision=model_revision,
    )


def build_typed_direct_pass3_pairs(
    *,
    gold_train_jsonl: Path,
    gold_f2_jsonl: Path,
    expected_gold_train_sha256: str,
    expected_gold_f2_sha256: str,
    expected_gold_rows: int,
    heldout_jsonl: Path,
    expected_heldout_sha256: str,
    expected_heldout_rows: int,
    local_reports: Sequence[tuple[Path, str]],
    api_reports: Sequence[tuple[Path, str]],
    warmstart: mixed.WarmstartIdentity,
    gold_replay_ratio: float,
    gold_replay_rows: int,
    min_verified_direct_targets: int,
    min_repair_conditioned_targets: int,
    allow_exploratory_inputs: bool,
    require_local_production_floor: bool,
    seed: int,
) -> tuple[list[mixed.MixedPair], dict[str, Any]]:
    del warmstart
    if (
        gold_replay_ratio != 0.0
        or gold_replay_rows != 0
        or min_verified_direct_targets != MINIMUM_ROWS
        or min_repair_conditioned_targets != 0
        or allow_exploratory_inputs
        or require_local_production_floor
        or len(local_reports) != 1
    ):
        raise ValueError("pass-3 accepts only its sealed new-target/no-replay profile")
    prior_ids, prior_record = pass2._load_prior_225_manifest(*local_reports[0])  # noqa: SLF001
    sources = _require_sources(api_reports)
    c1_rows, c1_audit = _audit_c001(*sources[0:3])
    c2_rows, c2_audit = _audit_resume47(*sources[3:6])
    prefix_rows, prefix_audit = _audit_prefix(*sources[6:9])

    typed_pairs, typed_manifest = typed_sft.load_typed_text_pairs(
        gold_train_jsonl,
        gold_f2_jsonl,
        expected_dataset_sha256=expected_gold_train_sha256,
        expected_f2_sha256=expected_gold_f2_sha256,
        expected_rows=expected_gold_rows,
        heldout_path=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        exclude_train_task_ids=[pass1.CONTAMINATED_TRAIN_TASK_ID],
        allow_unpinned_inputs=False,
    )
    typed_by_id = {pair.task_id: pair for pair in typed_pairs}
    if len(typed_by_id) != len(typed_pairs):
        raise ValueError("clean typed TRAIN universe contains duplicate task IDs")

    collected = [
        *(('kimi_c001', row) for row in c1_rows),
        *(('kimi_c002_tail', row) for row in c2_rows),
        *(('kimi_c002_prefix', row) for row in prefix_rows),
    ]
    task_ids = [str(row["task_id"]) for _category, row in collected]
    if len(task_ids) < MINIMUM_ROWS or len(task_ids) != len(set(task_ids)):
        raise ValueError("pass-3 target count/identity accounting differs")
    if set(task_ids) & prior_ids:
        raise ValueError("pass-3 attempted to replay a pass-1 task")
    if pass1.CONTAMINATED_TRAIN_TASK_ID in task_ids:
        raise ValueError("known train/heldout contaminant entered pass-3")
    unknown = sorted(set(task_ids) - set(typed_by_id))
    if unknown:
        raise ValueError("pass-3 target is outside clean typed TRAIN: " + unknown[0])

    pairs: list[mixed.MixedPair] = []
    category_by_id: dict[str, str] = {}
    source_digest_by_category = {
        "kimi_c001": sources[0][1],
        "kimi_c002_tail": sources[3][1],
        "kimi_c002_prefix": sources[6][1],
    }
    for category, row in collected:
        task_id = str(row["task_id"])
        canonical = typed_by_id[task_id]
        if row.get("source_sha256") != canonical.source_sha256:
            raise ValueError("pass-3 target is bound to a different typed source")
        category_by_id[task_id] = category
        pairs.append(
            mixed._make_pair(  # noqa: SLF001
                pair_id=f"{task_id}::typed-direct-pass3::{category}",
                source_task_id=task_id,
                kind="verified_direct",
                source=canonical.source,
                target=str(row["dart_source"]),
                provenance=(
                    ("dataset_schema", DATASET_SCHEMA),
                    ("source_category", category),
                    ("source_report_sha256", source_digest_by_category[category]),
                    ("typed_source_sha256", canonical.source_sha256),
                ),
            )
        )

    train_rows = base_sft._read_jsonl(gold_train_jsonl)  # noqa: SLF001
    tests_by_id: dict[str, str] = {}
    gold_target_by_id: dict[str, str] = {}
    for index, row in enumerate(train_rows):
        task_id = base_sft._identity(row, index)  # noqa: SLF001
        if task_id == pass1.CONTAMINATED_TRAIN_TASK_ID:
            continue
        tests_by_id[task_id] = pass1._complete_tests(row, task_id)  # noqa: SLF001
        gold_target_by_id[task_id] = base_sft._target_source(row, task_id)  # noqa: SLF001
    verification = pass1._verify_all(  # noqa: SLF001
        pairs,
        tests_by_id=tests_by_id,
        verify=pass1._runtime_verify,  # noqa: SLF001
        workers=pass1.FULL_VERIFY_WORKERS,
    )
    pairs.sort(
        key=lambda pair: canonical_sha256(
            {
                "schema": DATASET_SCHEMA,
                "seed": seed,
                "pair_id": pair.pair_id,
                "source_sha256": pair.source_sha256,
                "target_sha256": pair.target_sha256,
            }
        )
    )
    exact_gold = sum(
        pair.target.strip() == gold_target_by_id[pair.source_task_id].strip()
        for pair in pairs
    )
    schedule = [
        {
            "position": position,
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "kind": "verified_direct",
            "source_category": category_by_id[pair.source_task_id],
            "source_sha256": pair.source_sha256,
            "target_sha256": pair.target_sha256,
            "provenance": dict(pair.provenance),
        }
        for position, pair in enumerate(pairs)
    ]
    manifest = {
        "schema": DATASET_SCHEMA,
        "rows": len(pairs),
        "architecture": "native_encoder_decoder",
        "composition": {
            "verified_direct": len(pairs),
            "kimi_c001": len(c1_rows),
            "kimi_c002_tail": len(c2_rows),
            "kimi_c002_prefix": len(prefix_rows),
            "prior_225_replay": 0,
            "pass2_209_replay": 0,
            "gold_replay": 0,
            "repair_conditioned": 0,
            "reasoning_rows": 0,
            "independently_generated_exact_gold_matches": exact_gold,
        },
        "row_count_policy": "late_bound_after_all_source_journals_complete_and_pinned",
        "sources": {
            "kimi_c001": c1_audit,
            "kimi_c002_tail": c2_audit,
            "kimi_c002_prefix": prefix_audit,
        },
        "typed_train": typed_manifest,
        "prior_225_exclusion": prior_record,
        "heldout_overlap": 0,
        "known_contaminant_excluded": pass1.CONTAMINATED_TRAIN_TASK_ID,
        "task_id_deduplication": "reject_any_cross_source_or_prior_overlap",
        "all_targets_bound_to_provider_or_zero_api_verification_journals": True,
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "heldout_175_model_visible": False,
        "tests_model_visible": False,
        "private_feedback_model_visible": False,
        "repair_conditioned_prefixes_visible": False,
        "reasoning_model_visible": False,
        "full_acceptance_reverification": verification,
        "schedule": schedule,
        "schedule_sha256": canonical_sha256(schedule),
        "task_ids_sha256": canonical_sha256([pair.source_task_id for pair in pairs]),
        "source_sha256s_sha256": canonical_sha256([pair.source_sha256 for pair in pairs]),
        "target_sha256s_sha256": canonical_sha256([pair.target_sha256 for pair in pairs]),
        "production_floor_eligible": True,
    }
    return pairs, manifest


def _profile_runtime_contract() -> dict[str, str]:
    record = dict(_MIXED_RUNTIME_CONTRACT())
    record["mixed_training_engine_sha256"] = record["trainer_sha256"]
    record["trainer_sha256"] = base_sft.sha256_file(Path(__file__).resolve())
    record["pass1_profile_sha256"] = base_sft.sha256_file(Path(pass1.__file__).resolve())
    record["pass2_validator_sha256"] = base_sft.sha256_file(Path(pass2.__file__).resolve())
    record["c001_validator_sha256"] = base_sft.sha256_file(Path(c001.__file__).resolve())
    record["c002_validator_sha256"] = base_sft.sha256_file(Path(c002.__file__).resolve())
    record["resume47_validator_sha256"] = base_sft.sha256_file(Path(resume47.__file__).resolve())
    record["trainer_profile"] = "typed_direct_only_pass3_new_kimi_continuations"
    return record


def _validate_profile_args(args: argparse.Namespace) -> None:
    expected = {
        "gold_replay_ratio": 0.0,
        "gold_replay_rows": 0,
        "min_verified_direct_targets": MINIMUM_ROWS,
        "min_repair_conditioned_targets": 0,
        "expected_warmstart_update": 58,
        "epochs": 2,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_updates": 0,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.0,
        "seed": 42,
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        matches = (
            math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float)
            else observed == wanted
        )
        if not matches:
            raise ValueError(f"typed direct pass-3 fixes --{name}={wanted}, observed={observed}")
    if args.allow_exploratory_inputs or args.require_local_production_floor:
        raise ValueError("typed direct pass-3 requires sealed aggregate inputs")
    if len(args.local_report) != 1 or len(args.api_report) != len(SOURCE_BASENAMES):
        raise ValueError("typed direct pass-3 requires one prior manifest and nine source artifacts")


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_profile_args(args)
    validate_dart_binary()
    originals = {
        "run_schema": mixed.RUN_SCHEMA,
        "checkpoint_schema": mixed.CHECKPOINT_SCHEMA,
        "dataset_schema": mixed.DATASET_SCHEMA,
        "builder": mixed.build_mixed_pairs,
        "warmstart": mixed.validate_warmstart,
        "runtime": mixed._runtime_contract,  # noqa: SLF001
    }
    mixed.RUN_SCHEMA = RUN_SCHEMA
    mixed.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    mixed.DATASET_SCHEMA = DATASET_SCHEMA
    mixed.build_mixed_pairs = build_typed_direct_pass3_pairs
    mixed.validate_warmstart = validate_pass3_warmstart
    mixed._runtime_contract = _profile_runtime_contract  # noqa: SLF001
    try:
        return mixed.train(args)
    finally:
        mixed.RUN_SCHEMA = originals["run_schema"]
        mixed.CHECKPOINT_SCHEMA = originals["checkpoint_schema"]
        mixed.DATASET_SCHEMA = originals["dataset_schema"]
        mixed.build_mixed_pairs = originals["builder"]
        mixed.validate_warmstart = originals["warmstart"]
        mixed._runtime_contract = originals["runtime"]  # noqa: SLF001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = mixed.parse_args(argv)
    try:
        _validate_profile_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(__import__("json").dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
