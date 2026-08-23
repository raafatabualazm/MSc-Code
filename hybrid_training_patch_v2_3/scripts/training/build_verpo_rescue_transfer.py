#!/usr/bin/env python3
"""Build matched direct-compact transfer arms from verified VeRPO rescues.

This builder performs no inference and no executable scoring.  It accepts the
strict artifacts emitted by ``direct_compact_verpo_rescue.py`` and only admits
student repairs that already carry full visible and development-holdback pass
attestations.  For every admitted task it emits:

* intervention: one verified student repair and one original-gold replay;
* control: the same task twice, with original gold in both positions.

The task sequence, compact conditioning, and row multiplicity are therefore
identical between arms.  Feedback, tests, private holdback material, diagnoses,
judge text, and reasoning never enter either SFT JSONL.  Partial improvements
are validated separately and exported only as off-policy preference pairs.

Join-seal v1 is supported.  Contract v3 is deliberately rejected: its v2 join
seal contains only aggregate pool-use metadata, which is insufficient to
derive the selected-and-duplicated subset's aggregate without a per-row pool
sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PATCH_ROOT = Path(__file__).resolve().parents[2]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V3,
    JOIN_SEAL_SCHEMA_V1,
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)


FULL_REPAIR_SCHEMA = "direct-compact-verpo-rescue-rs-sft-target-v1"
PARTIAL_PREFERENCE_SCHEMA = (
    "direct-compact-verpo-rescue-preference-pair-v1"
)
OUTPUT_PREFERENCE_SCHEMA = (
    "direct-compact-verpo-rescue-off-policy-preference-v1"
)
PREFERENCE_SEAL_SCHEMA = (
    "direct-compact-verpo-rescue-off-policy-preference-seal-v1"
)
BUILD_REPORT_SCHEMA = "direct-compact-verpo-rescue-transfer-build-v1"
SCORE_REPORT_SCHEMA = "direct-compact-verpo-rescue-score-v1"
SCORE_REPORT_HASH_FIELD = "score_artifact_sha256"

INTERVENTION_FILENAME = "rescue_sft_50_50.jsonl"
INTERVENTION_SEAL_FILENAME = "rescue_sft_50_50.seal.json"
CONTROL_FILENAME = "gold_replay_control.jsonl"
CONTROL_SEAL_FILENAME = "gold_replay_control.seal.json"
PREFERENCE_FILENAME = "partial_off_policy_preferences.jsonl"
PREFERENCE_SEAL_FILENAME = "partial_off_policy_preferences.seal.json"
SCHEDULE_FILENAME = "matched_schedule.jsonl"
REPORT_FILENAME = "build_report.json"

FULL_REPAIR_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "code",
        "code_sha256",
        "target_mode",
        "reasoning_in_target",
        "student_checkpoint_sha256",
        "source_plan_sha256",
        "visible_full_pass",
        "development_reward_holdback_full_pass",
        "development_reward_holdback_tests_sha256",
        "development_holdback_consumed_for_transfer_selection",
        "final_175_holdout_touched",
        "provider_saw_development_holdback",
        "contributors",
    }
)
CONTRIBUTOR_FIELDS = frozenset(
    {"arm", "base_candidate_rank", "repair_rank"}
)
ALLOWED_RESCUE_ARMS = frozenset(
    {
        "plain_resample",
        "compiler_only",
        "diagnosis_only",
        "diagnosis_and_steps",
    }
)
PARTIAL_PREFERENCE_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "chosen",
        "chosen_sha256",
        "rejected",
        "rejected_sha256",
        "chosen_visible_passed_tests",
        "rejected_visible_passed_tests",
        "chosen_holdback_delta_passed_tests",
        "off_policy",
        "different_conditioning_prefixes",
        "eligible_for_on_policy_verpo_update",
        "kept_separate_from_rs_sft_targets",
        "source_plan_sha256",
    }
)

CORE_COMPACT_FIELDS = (
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
)
OPTIONAL_BINARY_FIELDS = (
    "binary_multifunction_schema",
    "binary_adapter_contract_sha256",
    "binary_semantic_projection_sha256",
    "binary_source_symbol_attestation_binding_sha256",
)
TARGET_FIELDS = ("supervised_target", "dart_source", "source")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_KEY_TOKENS_RE = re.compile(r"[^a-z0-9]+")
_BASE_FORBIDDEN_TOKENS = frozenset(
    {
        "acceptance",
        "hidden",
        "heldout",
        "private",
        "judge",
        "diagnostic",
        "critique",
        "reasoning",
        "cot",
    }
)


class RescueTransferError(ValueError):
    """An input artifact or matched-transfer invariant failed closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise RescueTransferError("source text is not valid UTF-8") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RescueTransferError(
            f"{label} must be a canonical lowercase SHA-256"
        )
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RescueTransferError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _read_jsonl(
    path: str | Path,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RescueTransferError(
                    f"{source}:{line_number}: blank rows are forbidden"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RescueTransferError(
                    f"{source}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise RescueTransferError(
                    f"{source}:{line_number}: row is not an object"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise RescueTransferError(f"{label} contains zero rows")
    return rows


def _write_jsonl_new(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_file_record(
    value: Any, expected_path: Path, label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise RescueTransferError(f"{label} file record is malformed")
    try:
        recorded_path = Path(str(value.get("path") or "")).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise RescueTransferError(f"{label} path is invalid") from exc
    if (
        recorded_path != expected_path.resolve()
        or value.get("sha256") != sha256_file(expected_path)
        or type(value.get("size_bytes")) is not int
        or value.get("size_bytes") != expected_path.stat().st_size
    ):
        raise RescueTransferError(
            f"{label} does not bind the exact scorer export"
        )


def _validate_score_report(
    path: Path,
    *,
    repairs_path: Path,
    preferences_path: Path,
    repair_rows: int,
    preference_rows: int,
) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueTransferError("rescue score report is invalid JSON") from exc
    if not isinstance(report, dict):
        raise RescueTransferError("rescue score report is not an object")
    if report.get("schema") != SCORE_REPORT_SCHEMA:
        raise RescueTransferError("rescue score report schema mismatch")
    expected_digest = _require_sha256(
        report.get(SCORE_REPORT_HASH_FIELD),
        f"score_report.{SCORE_REPORT_HASH_FIELD}",
    )
    observed_digest = canonical_sha256(
        {
            key: value
            for key, value in report.items()
            if key != SCORE_REPORT_HASH_FIELD
        }
    )
    if expected_digest != observed_digest:
        raise RescueTransferError("rescue score report self-digest mismatch")
    exports = report.get("export_artifacts")
    if not isinstance(exports, Mapping) or set(exports) != {
        "rs_sft_targets",
        "preference_pairs",
    }:
        raise RescueTransferError(
            "rescue score report export_artifacts are incomplete"
        )
    _validate_file_record(
        exports["rs_sft_targets"], repairs_path, "RS-SFT targets"
    )
    _validate_file_record(
        exports["preference_pairs"],
        preferences_path,
        "partial preferences",
    )
    export_counts = report.get("exports")
    privacy = report.get("privacy")
    if (
        report.get("status") != "complete"
        or not isinstance(export_counts, Mapping)
        or export_counts.get("rs_sft_rows") != repair_rows
        or export_counts.get("partial_preference_rows") != preference_rows
        or export_counts.get("rs_sft_requires_full_visible_and_holdback")
        is not True
        or export_counts.get("preference_pairs_are_separate_off_policy")
        is not True
        or not isinstance(privacy, Mapping)
        or privacy.get("holdback_test_source_persisted") is not False
        or privacy.get("holdback_diagnostic_persisted") is not False
        or privacy.get("holdback_exposed_to_provider") is not False
        or privacy.get("reference_dart_exposed_to_provider") is not False
        or privacy.get("final_175_holdout_touched") is not False
        or privacy.get(
            "development_holdback_is_now_consumed_for_transfer_selection"
        )
        is not True
    ):
        raise RescueTransferError(
            "rescue score report completion/privacy contract failed"
        )
    report["source_plan_sha256"] = _require_sha256(
        report.get("source_plan_sha256"),
        "score_report.source_plan_sha256",
    )
    report["student_checkpoint_sha256"] = _require_sha256(
        report.get("student_checkpoint_sha256"),
        "score_report.student_checkpoint_sha256",
    )
    return report


def _task_id(row: Mapping[str, Any], label: str) -> str:
    value = row.get("task_id")
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RescueTransferError(
            f"{label} must have one canonical non-empty task_id"
        )
    return value


def _field_tokens(key: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _KEY_TOKENS_RE.split(key.lower())
        if token
    )


def _validate_base_field_boundary(row: Mapping[str, Any], label: str) -> None:
    """Allow the explicit fit-only feedback tests, reject other oracle fields."""

    def visit(value: Any, path: str, *, top_level: bool) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                if top_level and key == "feedback_tests":
                    if not isinstance(child, str) or not child.strip():
                        raise RescueTransferError(
                            f"{label}.feedback_tests must be non-empty text"
                        )
                    continue
                tokens = _field_tokens(key)
                if (
                    tokens.intersection(_BASE_FORBIDDEN_TOKENS)
                    or "test" in tokens
                    or "tests" in tokens
                    or key.lower() in {
                        "chain_of_thought",
                        "reasoning_content",
                        "raw_reasoning_content",
                    }
                ):
                    raise RescueTransferError(
                        f"{label}: forbidden oracle/reasoning field at "
                        f"{path}.{key}"
                    )
                visit(child, f"{path}.{key}", top_level=False)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", top_level=False)

    visit(row, label, top_level=True)


def _gold_target(row: Mapping[str, Any], task_id: str) -> str:
    observed: list[tuple[str, str]] = []
    for field in TARGET_FIELDS:
        if field not in row or row[field] is None:
            continue
        if not isinstance(row[field], str):
            raise RescueTransferError(
                f"{task_id}: target field {field!r} must be text"
            )
        value = row[field]
        if value.strip():
            observed.append((field, value))
    if not observed:
        raise RescueTransferError(
            f"{task_id}: sealed rollout has zero original-gold target"
        )
    unique = {value for _field, value in observed}
    if len(unique) != 1:
        raise RescueTransferError(
            f"{task_id}: conflicting original-gold target aliases"
        )
    target = observed[0][1]
    if "\x00" in target:
        raise RescueTransferError(f"{task_id}: gold target contains NUL")
    sha256_text(target)
    return target


def _normalize_base_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: DirectCompactContract,
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    order: list[str] = []
    bases: dict[str, dict[str, Any]] = {}
    gold: dict[str, str] = {}
    optional_shape: frozenset[str] | None = None
    for index, raw in enumerate(rows):
        label = f"sealed rollout row {index + 1}"
        task_id = _task_id(raw, label)
        if task_id in bases:
            raise RescueTransferError(
                f"sealed rollout has duplicate task {task_id!r}"
            )
        _validate_base_field_boundary(raw, label)
        compact_ids = contract.validate_row(raw, task_id)
        row_function = raw.get("function")
        if row_function not in (None, "", contract.target_function):
            raise RescueTransferError(
                f"{task_id}: function differs from compact contract"
            )
        row_language = raw.get("language", raw.get("lang"))
        if (
            row_language not in (None, "")
            and str(row_language).lower()
            != contract.target_language.lower()
        ):
            raise RescueTransferError(
                f"{task_id}: language differs from compact contract"
            )

        present_optional = frozenset(
            field for field in OPTIONAL_BINARY_FIELDS if field in raw
        )
        if present_optional and present_optional != frozenset(
            OPTIONAL_BINARY_FIELDS
        ):
            raise RescueTransferError(
                f"{task_id}: incomplete binary provenance field group"
            )
        if optional_shape is None:
            optional_shape = present_optional
        elif present_optional != optional_shape:
            raise RescueTransferError(
                "sealed rollout rows have inconsistent binary provenance shape"
            )

        base: dict[str, Any] = {
            "task_id": task_id,
            "lang": contract.target_language,
            "function": contract.target_function,
            "compact_input_ids": list(compact_ids),
            "compact_codec_sha256": contract.codec_sha256,
            "compact_codebook_sha256": contract.codebook_sha256,
            "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
        }
        for field in present_optional:
            value = raw[field]
            if field.endswith("_sha256"):
                value = _require_sha256(value, f"{task_id}.{field}")
            elif not isinstance(value, str) or not value:
                raise RescueTransferError(
                    f"{task_id}.{field} must be non-empty text"
                )
            base[field] = value

        order.append(task_id)
        bases[task_id] = base
        gold[task_id] = _gold_target(raw, task_id)
    if not order:
        raise RescueTransferError("sealed rollout contains zero targets")
    return order, bases, gold


def _normalize_contributors(
    value: Any, label: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RescueTransferError(f"{label}.contributors must be non-empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != CONTRIBUTOR_FIELDS:
            raise RescueTransferError(
                f"{label}.contributors[{index}] has an invalid field set"
            )
        arm = raw.get("arm")
        if arm not in ALLOWED_RESCUE_ARMS:
            raise RescueTransferError(
                f"{label}.contributors[{index}] has unknown arm {arm!r}"
            )
        contributor = {
            "arm": arm,
            "base_candidate_rank": _plain_int(
                raw.get("base_candidate_rank"),
                f"{label}.contributors[{index}].base_candidate_rank",
            ),
            "repair_rank": _plain_int(
                raw.get("repair_rank"),
                f"{label}.contributors[{index}].repair_rank",
            ),
        }
        digest = canonical_sha256(contributor)
        if digest in seen:
            raise RescueTransferError(
                f"{label}.contributors contains an exact duplicate"
            )
        seen.add(digest)
        normalized.append(contributor)
    return sorted(
        normalized,
        key=lambda item: (
            item["arm"],
            item["base_candidate_rank"],
            item["repair_rank"],
        ),
    )


def _normalize_full_repair(
    raw: Mapping[str, Any],
    *,
    index: int,
    task_ids: frozenset[str],
) -> dict[str, Any]:
    label = f"full repair row {index + 1}"
    if set(raw) != FULL_REPAIR_FIELDS:
        raise RescueTransferError(
            f"{label} field set differs from {FULL_REPAIR_SCHEMA}"
        )
    if raw.get("schema") != FULL_REPAIR_SCHEMA:
        raise RescueTransferError(f"{label} has the wrong schema")
    task_id = _task_id(raw, label)
    if task_id not in task_ids:
        raise RescueTransferError(
            f"{label} task {task_id!r} is outside the sealed fit rollout"
        )
    code = raw.get("code")
    if not isinstance(code, str) or not code.strip() or "\x00" in code:
        raise RescueTransferError(f"{label}.code is empty or invalid")
    code_sha = _require_sha256(raw.get("code_sha256"), f"{label}.code_sha256")
    if sha256_text(code) != code_sha:
        raise RescueTransferError(f"{label} code SHA-256 mismatch")
    if (
        raw.get("target_mode") != "final_dart_code_only"
        or raw.get("reasoning_in_target") is not False
        or raw.get("visible_full_pass") is not True
        or raw.get("development_reward_holdback_full_pass") is not True
        or raw.get(
            "development_holdback_consumed_for_transfer_selection"
        )
        is not True
        or raw.get("final_175_holdout_touched") is not False
        or raw.get("provider_saw_development_holdback") is not False
    ):
        raise RescueTransferError(
            f"{label} lacks full visible/private pass and leakage attestations"
        )
    normalized = dict(raw)
    normalized["code_sha256"] = code_sha
    normalized["student_checkpoint_sha256"] = _require_sha256(
        raw.get("student_checkpoint_sha256"),
        f"{label}.student_checkpoint_sha256",
    )
    normalized["source_plan_sha256"] = _require_sha256(
        raw.get("source_plan_sha256"), f"{label}.source_plan_sha256"
    )
    normalized["development_reward_holdback_tests_sha256"] = (
        _require_sha256(
            raw.get("development_reward_holdback_tests_sha256"),
            f"{label}.development_reward_holdback_tests_sha256",
        )
    )
    normalized["contributors"] = _normalize_contributors(raw["contributors"], label)
    normalized["stable_key"] = canonical_sha256(normalized)
    return normalized


def _normalize_partial_preference(
    raw: Mapping[str, Any],
    *,
    index: int,
    task_ids: frozenset[str],
) -> dict[str, Any]:
    label = f"partial preference row {index + 1}"
    if set(raw) != PARTIAL_PREFERENCE_FIELDS:
        raise RescueTransferError(
            f"{label} field set differs from {PARTIAL_PREFERENCE_SCHEMA}"
        )
    if raw.get("schema") != PARTIAL_PREFERENCE_SCHEMA:
        raise RescueTransferError(f"{label} has the wrong schema")
    task_id = _task_id(raw, label)
    if task_id not in task_ids:
        raise RescueTransferError(
            f"{label} task {task_id!r} is outside the sealed fit rollout"
        )
    chosen = raw.get("chosen")
    rejected = raw.get("rejected")
    if (
        not isinstance(chosen, str)
        or not chosen.strip()
        or "\x00" in chosen
        or not isinstance(rejected, str)
        or not rejected.strip()
        or "\x00" in rejected
    ):
        raise RescueTransferError(
            f"{label} must contain non-empty chosen/rejected source"
        )
    chosen_sha = _require_sha256(
        raw.get("chosen_sha256"), f"{label}.chosen_sha256"
    )
    rejected_sha = _require_sha256(
        raw.get("rejected_sha256"), f"{label}.rejected_sha256"
    )
    if sha256_text(chosen) != chosen_sha:
        raise RescueTransferError(f"{label} chosen SHA-256 mismatch")
    if sha256_text(rejected) != rejected_sha:
        raise RescueTransferError(f"{label} rejected SHA-256 mismatch")
    if chosen_sha == rejected_sha or chosen == rejected:
        raise RescueTransferError(f"{label} chosen and rejected are identical")
    chosen_visible = _plain_int(
        raw.get("chosen_visible_passed_tests"),
        f"{label}.chosen_visible_passed_tests",
    )
    rejected_visible = _plain_int(
        raw.get("rejected_visible_passed_tests"),
        f"{label}.rejected_visible_passed_tests",
    )
    holdback_delta = raw.get("chosen_holdback_delta_passed_tests")
    if type(holdback_delta) is not int:
        raise RescueTransferError(
            f"{label}.chosen_holdback_delta_passed_tests must be an integer"
        )
    if chosen_visible <= rejected_visible or rejected_visible != 0:
        raise RescueTransferError(
            f"{label} is not a strict visible-test improvement"
        )
    if (
        raw.get("off_policy") is not True
        or raw.get("different_conditioning_prefixes") is not True
        or raw.get("eligible_for_on_policy_verpo_update") is not False
        or raw.get("kept_separate_from_rs_sft_targets") is not True
    ):
        raise RescueTransferError(
            f"{label} lacks the required off-policy separation attestations"
        )
    normalized = dict(raw)
    normalized["source_plan_sha256"] = _require_sha256(
        raw.get("source_plan_sha256"), f"{label}.source_plan_sha256"
    )
    normalized["chosen_sha256"] = chosen_sha
    normalized["rejected_sha256"] = rejected_sha
    normalized["stable_key"] = canonical_sha256(normalized)
    normalized["visible_and_private_improved"] = holdback_delta > 0
    return normalized


def _select_full_repairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_ids: frozenset[str],
    gold_by_task: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    exact_seen: set[str] = set()
    code_seen: dict[tuple[str, str], str] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    exact_duplicates = 0
    gold_identical_rows = 0
    gold_identical_keys: set[tuple[str, str]] = set()
    tasks_with_any_input: set[str] = set()
    for index, raw in enumerate(rows):
        normalized = _normalize_full_repair(
            raw, index=index, task_ids=task_ids
        )
        task_id = str(normalized["task_id"])
        tasks_with_any_input.add(task_id)
        gold = gold_by_task.get(task_id)
        if not isinstance(gold, str):
            raise RescueTransferError(
                f"{task_id}: no sealed original-gold comparator"
            )
        is_gold_identical = str(normalized["code"]) == gold
        if is_gold_identical:
            gold_identical_rows += 1
            gold_identical_keys.add(
                (task_id, str(normalized["code_sha256"]))
            )
        stable_key = str(normalized["stable_key"])
        if stable_key in exact_seen:
            exact_duplicates += 1
            continue
        exact_seen.add(stable_key)
        if is_gold_identical:
            continue
        code_key = (
            task_id,
            str(normalized["code_sha256"]),
        )
        prior = code_seen.get(code_key)
        if prior is not None and prior != stable_key:
            raise RescueTransferError(
                "the same task/code repair has conflicting provenance"
            )
        code_seen[code_key] = stable_key
        by_task.setdefault(task_id, []).append(normalized)

    selected: dict[str, dict[str, Any]] = {}
    alternatives: dict[str, int] = {}
    for task_id, candidates in by_task.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                row["stable_key"],
                row["code_sha256"],
            ),
        )
        selected[task_id] = ordered[0]
        alternatives[task_id] = len(ordered)
    if not selected:
        raise RescueTransferError(
            "zero genuine non-gold fully verified student repair targets "
            "were selected"
        )
    checkpoint_hashes = {
        row["student_checkpoint_sha256"] for row in selected.values()
    }
    source_plan_hashes = {
        row["source_plan_sha256"] for row in selected.values()
    }
    if len(checkpoint_hashes) != 1:
        raise RescueTransferError(
            "full repairs bind more than one student checkpoint"
        )
    if len(source_plan_hashes) != 1:
        raise RescueTransferError(
            "full repairs bind more than one source plan"
        )
    return selected, {
        "input_rows": len(rows),
        "exact_duplicates_removed": exact_duplicates,
        "gold_identical_input_rows_excluded": gold_identical_rows,
        "gold_identical_unique_task_code_excluded": len(
            gold_identical_keys
        ),
        "tasks_with_only_gold_identical_repairs": len(
            tasks_with_any_input - set(by_task)
        ),
        "selected_tasks": len(selected),
        "alternatives_by_task": dict(sorted(alternatives.items())),
        "student_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "source_plan_sha256": next(iter(source_plan_hashes)),
    }


def _select_partial_preferences(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_ids: frozenset[str],
    source_plan_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    exact_seen: set[str] = set()
    pair_seen: dict[tuple[str, str, str], str] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    exact_duplicates = 0
    excluded_no_private_gain = 0
    for index, raw in enumerate(rows):
        normalized = _normalize_partial_preference(
            raw, index=index, task_ids=task_ids
        )
        if normalized["source_plan_sha256"] != source_plan_sha256:
            raise RescueTransferError(
                "partial preference source plan differs from full repairs"
            )
        stable_key = str(normalized["stable_key"])
        if stable_key in exact_seen:
            exact_duplicates += 1
            continue
        exact_seen.add(stable_key)
        pair_key = (
            str(normalized["task_id"]),
            str(normalized["chosen_sha256"]),
            str(normalized["rejected_sha256"]),
        )
        prior = pair_seen.get(pair_key)
        if prior is not None and prior != stable_key:
            raise RescueTransferError(
                "the same partial preference pair has conflicting metadata"
            )
        pair_seen[pair_key] = stable_key
        if not normalized["visible_and_private_improved"]:
            excluded_no_private_gain += 1
            continue
        by_task.setdefault(str(normalized["task_id"]), []).append(normalized)

    selected: dict[str, dict[str, Any]] = {}
    alternatives: dict[str, int] = {}
    for task_id, candidates in by_task.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                row["stable_key"],
                row["chosen_sha256"],
                row["rejected_sha256"],
            ),
        )
        selected[task_id] = ordered[0]
        alternatives[task_id] = len(ordered)
    return selected, {
        "input_rows": len(rows),
        "exact_duplicates_removed": exact_duplicates,
        "excluded_without_private_improvement": excluded_no_private_gain,
        "selected_tasks": len(selected),
        "alternatives_by_task": dict(sorted(alternatives.items())),
    }


def _sft_row(base: Mapping[str, Any], target: str) -> dict[str, Any]:
    row = dict(base)
    row["supervised_target"] = target
    allowed = {
        "task_id",
        "lang",
        "function",
        *CORE_COMPACT_FIELDS,
        *OPTIONAL_BINARY_FIELDS,
        "supervised_target",
    }
    if not set(row).issubset(allowed):
        raise AssertionError("an unconditioned SFT row gained a private field")
    return row


def _preference_row(
    base: Mapping[str, Any], preference: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "schema": OUTPUT_PREFERENCE_SCHEMA,
            "chosen": preference["chosen"],
            "chosen_sha256": preference["chosen_sha256"],
            "rejected": preference["rejected"],
            "rejected_sha256": preference["rejected_sha256"],
            "off_policy": True,
            "different_conditioning_prefixes": True,
            "eligible_for_on_policy_verpo_update": False,
            "kept_separate_from_rs_sft_targets": True,
            "source_plan_sha256": preference["source_plan_sha256"],
        }
    )
    return row


def _join_seal_v1(
    *,
    output_path: Path,
    rows: int,
    contract_path: Path,
    rollout_path: Path,
    rollout_seal_path: Path,
    task_sequence_sha256: str,
    source_plan_sha256: str,
    student_checkpoint_sha256: str,
    arm: str,
    training_allowed: bool,
) -> dict[str, Any]:
    return {
        "schema": JOIN_SEAL_SCHEMA_V1,
        "selected_role": "fit",
        "training_allowed": training_allowed,
        "heldout_measure_only": False,
        "output_sha256": sha256_file(output_path),
        "contract_sha256": sha256_file(contract_path),
        "rows": rows,
        "training_objective_scope": (
            "matched_verpo_rescue_sft"
            if arm == "intervention"
            else "matched_gold_replay_control"
        ),
        "matched_arm": arm,
        "source_rollout_sha256": sha256_file(rollout_path),
        "source_rollout_seal_sha256": sha256_file(rollout_seal_path),
        "source_plan_sha256": source_plan_sha256,
        "student_checkpoint_sha256": student_checkpoint_sha256,
        "task_sequence_sha256": task_sequence_sha256,
        "rows_per_selected_task": 2,
        "feedback_tests_serialized": False,
        "private_holdback_serialized": False,
        "judge_text_serialized": False,
        "reasoning_serialized": False,
        "final_175_holdout_touched": False,
    }


def _prepare_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RescueTransferError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_rescue_transfer(
    rollout_path: str | Path,
    rollout_seal_path: str | Path,
    contract_path: str | Path,
    repairs_path: str | Path,
    score_report_path: str | Path,
    partial_preferences_path: str | Path,
    output_dir: str | Path,
    *,
    min_unique_repairs: int = 400,
    allow_low_coverage_smoke: bool = False,
) -> dict[str, Any]:
    """Build the matched SFT arms and separate off-policy preference artifact."""

    rollout = Path(rollout_path).expanduser().resolve()
    rollout_seal = Path(rollout_seal_path).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    repairs = Path(repairs_path).expanduser().resolve()
    score_report_file = Path(score_report_path).expanduser().resolve()
    partials = Path(partial_preferences_path).expanduser().resolve()
    for path in (
        rollout,
        rollout_seal,
        contract_file,
        repairs,
        score_report_file,
        partials,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min_unique_repairs <= 0:
        raise RescueTransferError("min_unique_repairs must be positive")

    contract = DirectCompactContract.load(contract_file)
    if contract.schema == CONTRACT_SCHEMA_V3:
        raise RescueTransferError(
            "direct-compact v3 rescue transfer is unsupported: the aggregate-"
            "only v2 join seal cannot safely derive selected-task pool-use "
            "aggregates after matched duplication"
        )
    input_seal = validate_join_seal(
        rollout,
        rollout_seal,
        contract_file,
        expected_role="fit",
    )
    if input_seal.get("schema") != JOIN_SEAL_SCHEMA_V1:
        raise RescueTransferError(
            "rescue transfer currently requires a v1 fit join seal"
        )
    if input_seal.get("training_allowed") is False:
        raise RescueTransferError("sealed rollout is not training-allowed")
    if input_seal.get("heldout_measure_only") is True:
        raise RescueTransferError("measure-only rows cannot enter transfer")

    rollout_rows = _read_jsonl(
        rollout, label="sealed fit rollout", allow_empty=False
    )
    if len(rollout_rows) != int(input_seal["rows"]):
        raise RescueTransferError(
            "sealed rollout row count changed after join-seal validation"
        )
    task_order, bases, gold = _normalize_base_rows(rollout_rows, contract)
    task_ids = frozenset(task_order)

    full_rows = _read_jsonl(
        repairs, label="verified student repairs", allow_empty=False
    )
    partial_rows = _read_jsonl(
        partials,
        label="partial preference pairs",
        allow_empty=True,
    )
    score_report = _validate_score_report(
        score_report_file,
        repairs_path=repairs,
        preferences_path=partials,
        repair_rows=len(full_rows),
        preference_rows=len(partial_rows),
    )
    selected_repairs, full_stats = _select_full_repairs(
        full_rows,
        task_ids=task_ids,
        gold_by_task=gold,
    )
    source_plan_sha = str(full_stats["source_plan_sha256"])
    checkpoint_sha = str(full_stats["student_checkpoint_sha256"])
    if (
        score_report["source_plan_sha256"] != source_plan_sha
        or score_report["student_checkpoint_sha256"] != checkpoint_sha
    ):
        raise RescueTransferError(
            "score report source-plan/checkpoint bindings differ from repair rows"
        )
    production_coverage_met = len(selected_repairs) >= min_unique_repairs
    if not production_coverage_met and not allow_low_coverage_smoke:
        raise RescueTransferError(
            f"only {len(selected_repairs)} unique fully verified repairs; "
            f"production minimum is {min_unique_repairs}"
        )
    selected_preferences, preference_stats = (
        _select_partial_preferences(
            partial_rows,
            task_ids=task_ids,
            source_plan_sha256=source_plan_sha,
        )
    )

    intervention_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    selected_task_order = [
        task_id for task_id in task_order if task_id in selected_repairs
    ]
    for task_id in selected_task_order:
        repair = selected_repairs[task_id]
        gold_target = gold[task_id]
        repair_target = str(repair["code"])
        pair_targets = (("repair", repair_target), ("gold", gold_target))
        for slot, (kind, intervention_target) in enumerate(pair_targets):
            intervention_rows.append(
                _sft_row(bases[task_id], intervention_target)
            )
            control_rows.append(_sft_row(bases[task_id], gold_target))
            schedule.append(
                {
                    "position": len(schedule),
                    "pair_slot": slot,
                    "task_id": task_id,
                    "intervention_kind": kind,
                    "repair_stable_key": repair["stable_key"],
                    "intervention_target_sha256": sha256_text(
                        intervention_target
                    ),
                    "control_target_sha256": sha256_text(gold_target),
                }
            )

    if not intervention_rows or not control_rows:
        raise RescueTransferError("matched SFT arms contain zero targets")
    if len(intervention_rows) != 2 * len(selected_task_order):
        raise AssertionError("intervention is not exactly two rows per task")
    if len(control_rows) != len(intervention_rows):
        raise AssertionError("matched arm row counts differ")
    if Counter(row["intervention_kind"] for row in schedule) != {
        "repair": len(selected_task_order),
        "gold": len(selected_task_order),
    }:
        raise AssertionError("intervention is not exact 50/50")
    if [row["task_id"] for row in intervention_rows] != [
        row["task_id"] for row in control_rows
    ]:
        raise AssertionError("matched arms have different task order")
    for intervention, control in zip(
        intervention_rows, control_rows, strict=True
    ):
        intervention_conditioning = {
            key: value
            for key, value in intervention.items()
            if key != "supervised_target"
        }
        control_conditioning = {
            key: value
            for key, value in control.items()
            if key != "supervised_target"
        }
        if intervention_conditioning != control_conditioning:
            raise AssertionError("matched compact conditioning differs")

    full_target_keys = {
        (task_id, row["code_sha256"])
        for task_id, row in selected_repairs.items()
    }
    preference_rows: list[dict[str, Any]] = []
    for task_id in task_order:
        preference = selected_preferences.get(task_id)
        if preference is None:
            continue
        if (task_id, preference["chosen_sha256"]) in full_target_keys:
            raise RescueTransferError(
                f"{task_id}: a partial chosen target also entered full SFT"
            )
        preference_rows.append(_preference_row(bases[task_id], preference))

    output = _prepare_output_dir(output_dir)
    intervention_path = output / INTERVENTION_FILENAME
    intervention_seal_path = output / INTERVENTION_SEAL_FILENAME
    control_path = output / CONTROL_FILENAME
    control_seal_path = output / CONTROL_SEAL_FILENAME
    preference_path = output / PREFERENCE_FILENAME
    preference_seal_path = output / PREFERENCE_SEAL_FILENAME
    schedule_path = output / SCHEDULE_FILENAME
    report_path = output / REPORT_FILENAME

    _write_jsonl_new(intervention_path, intervention_rows)
    _write_jsonl_new(control_path, control_rows)
    _write_jsonl_new(preference_path, preference_rows)
    _write_jsonl_new(schedule_path, schedule)

    task_sequence = [row["task_id"] for row in intervention_rows]
    task_sequence_sha = canonical_sha256(task_sequence)
    intervention_seal_value = _join_seal_v1(
        output_path=intervention_path,
        rows=len(intervention_rows),
        contract_path=contract_file,
        rollout_path=rollout,
        rollout_seal_path=rollout_seal,
        task_sequence_sha256=task_sequence_sha,
        source_plan_sha256=source_plan_sha,
        student_checkpoint_sha256=checkpoint_sha,
        arm="intervention",
        training_allowed=production_coverage_met,
    )
    control_seal_value = _join_seal_v1(
        output_path=control_path,
        rows=len(control_rows),
        contract_path=contract_file,
        rollout_path=rollout,
        rollout_seal_path=rollout_seal,
        task_sequence_sha256=task_sequence_sha,
        source_plan_sha256=source_plan_sha,
        student_checkpoint_sha256=checkpoint_sha,
        arm="control",
        training_allowed=production_coverage_met,
    )
    _write_json_new(intervention_seal_path, intervention_seal_value)
    _write_json_new(control_seal_path, control_seal_value)
    # Machine-check that both emitted seals are accepted by the production
    # join-seal verifier, rather than merely resembling a valid seal.
    validate_join_seal(
        intervention_path,
        intervention_seal_path,
        contract_file,
        expected_role="fit",
    )
    validate_join_seal(
        control_path,
        control_seal_path,
        contract_file,
        expected_role="fit",
    )

    preference_seal_value = {
        "schema": PREFERENCE_SEAL_SCHEMA,
        "output_schema": OUTPUT_PREFERENCE_SCHEMA,
        "output_sha256": sha256_file(preference_path),
        "rows": len(preference_rows),
        "source_preferences_sha256": (
            sha256_file(partials)
        ),
        "source_rollout_sha256": sha256_file(rollout),
        "source_rollout_seal_sha256": sha256_file(rollout_seal),
        "contract_sha256": sha256_file(contract_file),
        "source_plan_sha256": source_plan_sha,
        "off_policy": True,
        "eligible_for_sft": False,
        "eligible_for_on_policy_verpo_update": False,
        "full_repairs_mixed_into_preferences": False,
    }
    _write_json_new(preference_seal_path, preference_seal_value)

    report = {
        "schema": BUILD_REPORT_SCHEMA,
        "status": "complete",
        "inputs": {
            "rollout_sha256": sha256_file(rollout),
            "rollout_seal_sha256": sha256_file(rollout_seal),
            "contract_sha256": sha256_file(contract_file),
            "repairs_sha256": sha256_file(repairs),
            "score_report_sha256": sha256_file(score_report_file),
            "score_report_self_digest": score_report[
                SCORE_REPORT_HASH_FIELD
            ],
            "partial_preferences_sha256": sha256_file(partials),
        },
        "bindings": {
            "source_plan_sha256": source_plan_sha,
            "student_checkpoint_sha256": checkpoint_sha,
            "sealed_fit_tasks": len(task_ids),
            "selected_task_ids_sha256": canonical_sha256(
                selected_task_order
            ),
            "matched_task_sequence_sha256": task_sequence_sha,
            "final_175_holdout_touched": False,
        },
        "coverage_gate": {
            "minimum_unique_repairs": min_unique_repairs,
            "observed_unique_repairs": len(selected_repairs),
            "production_coverage_met": production_coverage_met,
            "allow_low_coverage_smoke": bool(allow_low_coverage_smoke),
            "low_coverage_smoke_bypass_used": (
                not production_coverage_met
                and bool(allow_low_coverage_smoke)
            ),
            "training_use_permitted": production_coverage_met,
        },
        "counts": {
            "selected_full_repair_tasks": len(selected_task_order),
            "intervention_rows": len(intervention_rows),
            "intervention_repair_rows": len(selected_task_order),
            "intervention_gold_replay_rows": len(selected_task_order),
            "control_gold_rows": len(control_rows),
            "partial_preference_rows": len(preference_rows),
        },
        "full_repairs": full_stats,
        "partial_preferences": preference_stats,
        "invariants": {
            "intervention_is_exact_50_50": True,
            "control_has_identical_task_order_and_multiplicity": True,
            "all_sft_repairs_visible_full_pass": True,
            "all_sft_repairs_development_holdback_full_pass": True,
            "all_sft_repairs_byte_differ_from_original_gold": True,
            "gold_identical_repairs_excluded_from_coverage": True,
            "all_training_targets_use_supervised_target": True,
            "feedback_tests_serialized": False,
            "private_holdback_serialized": False,
            "judge_text_serialized": False,
            "reasoning_serialized": False,
            "partial_repairs_mixed_into_sft": False,
            "preferences_are_off_policy_only": True,
            "scorer_report_self_digest_verified": True,
            "scorer_export_paths_and_hashes_verified": True,
        },
        "outputs": {
            "intervention": {
                "path": str(intervention_path),
                "sha256": sha256_file(intervention_path),
                "seal_path": str(intervention_seal_path),
                "seal_sha256": sha256_file(intervention_seal_path),
            },
            "control": {
                "path": str(control_path),
                "sha256": sha256_file(control_path),
                "seal_path": str(control_seal_path),
                "seal_sha256": sha256_file(control_seal_path),
            },
            "partial_preferences": {
                "path": str(preference_path),
                "sha256": sha256_file(preference_path),
                "seal_path": str(preference_seal_path),
                "seal_sha256": sha256_file(preference_seal_path),
            },
            "schedule": {
                "path": str(schedule_path),
                "sha256": sha256_file(schedule_path),
            },
        },
    }
    _write_json_new(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched unconditioned direct-compact SFT transfer arms "
            "from independently verified VeRPO student rescues"
        )
    )
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--rollout-seal", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--repairs", required=True)
    parser.add_argument("--score-report", required=True)
    parser.add_argument(
        "--partial-preferences",
        required=True,
        help=(
            "Scorer-emitted partial preference JSONL; only pairs "
            "improving both visible and development-holdback tests are copied"
        ),
    )
    parser.add_argument("--min-unique-repairs", type=int, default=400)
    parser.add_argument(
        "--allow-low-coverage-smoke",
        action="store_true",
        help=(
            "Testing only: permit fewer than --min-unique-repairs. The "
            "resulting report marks training use as not permitted."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_rescue_transfer(
        args.rollout,
        args.rollout_seal,
        args.contract,
        args.repairs,
        args.score_report,
        args.partial_preferences,
        args.output_dir,
        min_unique_repairs=args.min_unique_repairs,
        allow_low_coverage_smoke=args.allow_low_coverage_smoke,
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "counts": report["counts"],
                "outputs": report["outputs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
