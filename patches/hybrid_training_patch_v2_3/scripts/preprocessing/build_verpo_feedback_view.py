#!/usr/bin/env python3
"""Build an attested TRAIN-only VeRPO feedback/holdback view.

Only the fit row's ``tests`` field is read.  ``acceptance_tests`` is never used.
For harnesses with at least two balanced ``expect(...)`` statements, a stable
task-bound ordering assigns floor(N/2) cases to visible feedback and the
remainder to a private reward holdback.  Single/no-case or malformed harnesses
are excluded rather than guessed.  The holdback artifact is never a trainer or
API input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.preprocessing.build_multifunction_executable_view import (
    EXPECTED_EXECUTABLE_ROWS,
    F2_REPRESENTATION_SCHEMA,
    REPRESENTATION_SCHEMA,
    file_record,
    load_json,
    load_jsonl,
    sha256_file,
    stable_sha256,
    validate_executable_view,
)


SCHEMA = "verpo-train-feedback-view-v1"
PUBLIC_SCHEMA = "verpo-train-feedback-public-manifest-v1"
SPLIT_SCHEMA = "task-bound-expect-half-split-v1"
ROLLOUT_SCOPE = "verpo_train_visible_feedback_only"
PRODUCTION_EXPECTED_ACCOUNTING = {
    "parent_rows": EXPECTED_EXECUTABLE_ROWS,
    "eligible_rows": 1232,
    "excluded_rows": 346,
    "source_expect_cases": 9929,
    "visible_expect_cases": 4621,
    "holdback_expect_cases": 5308,
    "odd_case_tasks": 687,
}
PRODUCTION_ELIGIBLE_TASK_IDS_SHA256 = (
    "2b08563df4bc091e6e8d2599c7f390ef7358185e40b5c56cdf21272e0193767a"
)
PRODUCTION_EXCLUDED_TASK_IDS_SHA256 = (
    "4595be226b3e7c226031df2e996dbf5ef7701585d5cc3af4f93efdcb5ca6af43"
)
FORBIDDEN_ROLLOUT_FIELDS = frozenset(
    {
        "tests",
        "acceptance_tests",
        "hidden_tests",
        "reward_holdback_tests",
    }
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
F2_REQUIRED_INVARIANTS = (
    "all_artifact_hashes_verified",
    "all_row_contract_hashes_verified",
    "all_codec_roundtrips_verified",
    "all_student_constant_prefixes_verified",
    "all_f2_semantic_roundtrips_verified",
    "f2_system_prompt_self_contained_and_hashed",
    "all_complete_prompts_within_limit",
    "opaque_source_ids_expanded",
    "cfg_explicit",
    "all_user_functions_retained",
    "all_external_symbols_retained",
    "transfer_table_redundancy_proven",
    "train_dev_representation_contract_identical",
)
_MAIN_RE = re.compile(
    r"(?m)^[ \t]*(?:Future(?:<void>)?[ \t]+|void[ \t]+)?"
    r"main[ \t]*\([^)]*\)[ \t]*(?:async[ \t]*)?\{"
)


class FeedbackViewError(ValueError):
    pass


def normalize_expected_accounting(
    value: Mapping[str, Any],
) -> dict[str, int]:
    required = tuple(PRODUCTION_EXPECTED_ACCOUNTING)
    if set(value) != set(required):
        raise FeedbackViewError(
            "feedback accounting expectation must contain exactly "
            + ", ".join(required)
        )
    normalized: dict[str, int] = {}
    for key in required:
        observed = value.get(key)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise FeedbackViewError(
                f"feedback accounting expectation {key} is not an integer"
            )
        if observed < 0:
            raise FeedbackViewError(
                f"feedback accounting expectation {key} is negative"
            )
        normalized[key] = observed
    if (
        normalized["parent_rows"] <= 0
        or normalized["eligible_rows"] + normalized["excluded_rows"]
        != normalized["parent_rows"]
        or normalized["visible_expect_cases"]
        + normalized["holdback_expect_cases"]
        != normalized["source_expect_cases"]
        or normalized["eligible_rows"] <= 0
    ):
        raise FeedbackViewError("feedback accounting expectation is incoherent")
    return normalized


def normalize_task_digest(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise FeedbackViewError(f"{label} must be one lowercase SHA-256")
    return normalized


def resolve_feedback_expectation(
    *,
    expected_accounting: Mapping[str, Any] | None,
    expected_eligible_task_ids_sha256: str | None,
    expected_excluded_task_ids_sha256: str | None,
    attested: Any,
    label: str,
) -> tuple[dict[str, int], str, str]:
    """Resolve either an explicit legacy contract or a hash-pinned attestation."""

    supplied = (
        expected_accounting is not None,
        expected_eligible_task_ids_sha256 is not None,
        expected_excluded_task_ids_sha256 is not None,
    )
    if any(supplied) and not all(supplied):
        raise FeedbackViewError(
            "feedback expectation must provide accounting and both task "
            "digests together"
        )
    if all(supplied):
        assert expected_accounting is not None
        assert expected_eligible_task_ids_sha256 is not None
        assert expected_excluded_task_ids_sha256 is not None
        return (
            normalize_expected_accounting(expected_accounting),
            normalize_task_digest(
                expected_eligible_task_ids_sha256,
                "expected eligible task IDs digest",
            ),
            normalize_task_digest(
                expected_excluded_task_ids_sha256,
                "expected excluded task IDs digest",
            ),
        )
    if not isinstance(attested, Mapping):
        raise FeedbackViewError(f"{label} has no sealed feedback expectation")
    accounting = attested.get("accounting")
    return (
        normalize_expected_accounting(
            accounting if isinstance(accounting, Mapping) else {}
        ),
        normalize_task_digest(
            str(attested.get("eligible_task_ids_sha256") or ""),
            f"{label} eligible task IDs digest",
        ),
        normalize_task_digest(
            str(attested.get("excluded_task_ids_sha256") or ""),
            f"{label} excluded task IDs digest",
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl_allow_empty(path: str | Path, label: str) -> list[dict[str, Any]]:
    """Load an audit JSONL whose valid cardinality may be zero.

    The shared executable-view loader intentionally rejects empty datasets.
    That is the correct contract for training inputs, but an exclusion audit is
    legitimately empty when every synthetic fixture row is eligible.
    """

    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise FeedbackViewError(
                    f"{label}:{line_number}: blank rows are forbidden"
                )
            try:
                row = json.loads(line)
            except Exception as exc:
                raise FeedbackViewError(
                    f"{label}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise FeedbackViewError(
                    f"{label}:{line_number}: row is not an object"
                )
            rows.append(row)
    return rows


def assert_only_allowed_path_keys(
    value: Any,
    *,
    allowed: set[tuple[str, ...]],
    label: str,
    prefix: tuple[str, ...] = (),
) -> None:
    """Reject accidental parent/private artifact path exposure recursively."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            current = prefix + (str(key),)
            if str(key) == "path" and current not in allowed:
                raise FeedbackViewError(
                    f"{label}: forbidden artifact path at {'.'.join(current)}"
                )
            assert_only_allowed_path_keys(
                item,
                allowed=allowed,
                label=label,
                prefix=current,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_only_allowed_path_keys(
                item,
                allowed=allowed,
                label=label,
                prefix=prefix + (str(index),),
            )


def _skip_quoted_or_comment(source: str, index: int) -> int | None:
    """Return the first index after a string/comment, or None for ordinary code."""

    if source.startswith("//", index):
        end = source.find("\n", index + 2)
        return len(source) if end < 0 else end
    if source.startswith("/*", index):
        end = source.find("*/", index + 2)
        if end < 0:
            raise FeedbackViewError("unterminated block comment")
        return end + 2
    quote = source[index : index + 1]
    if quote not in {"'", '"'}:
        return None
    triple = source.startswith(quote * 3, index)
    delimiter = quote * (3 if triple else 1)
    cursor = index + len(delimiter)
    while cursor < len(source):
        if source[cursor] == "\\":
            cursor += 2
            continue
        if source.startswith(delimiter, cursor):
            return cursor + len(delimiter)
        cursor += 1
    raise FeedbackViewError("unterminated Dart string")


def _matching_brace(source: str, opening: int) -> int:
    depth = 0
    cursor = opening
    while cursor < len(source):
        skipped = _skip_quoted_or_comment(source, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return cursor
            if depth < 0:
                break
        cursor += 1
    raise FeedbackViewError("unbalanced Dart main braces")


def extract_expect_spans(test_code: str) -> list[tuple[int, int]]:
    """Return balanced standalone expect-call spans, including multiline calls."""

    source = str(test_code)
    matches = list(_MAIN_RE.finditer(source))
    if len(matches) != 1:
        raise FeedbackViewError("test harness must contain exactly one main")
    opening = source.find("{", matches[0].start(), matches[0].end())
    closing = _matching_brace(source, opening)
    spans: list[tuple[int, int]] = []
    cursor = opening + 1
    while cursor < closing:
        skipped = _skip_quoted_or_comment(source, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if (
            source.startswith("expect", cursor)
            and (cursor == 0 or not (source[cursor - 1].isalnum() or source[cursor - 1] == "_"))
            and (
                cursor + 6 >= len(source)
                or not (
                    source[cursor + 6].isalnum()
                    or source[cursor + 6] == "_"
                )
            )
        ):
            paren = cursor + 6
            while paren < closing and source[paren].isspace():
                paren += 1
            if paren >= closing or source[paren] != "(":
                cursor += 6
                continue
            depth = 0
            end = paren
            while end < closing:
                skipped = _skip_quoted_or_comment(source, end)
                if skipped is not None:
                    end = skipped
                    continue
                if source[end] == "(":
                    depth += 1
                elif source[end] == ")":
                    depth -= 1
                    if depth == 0:
                        end += 1
                        break
                    if depth < 0:
                        raise FeedbackViewError("unbalanced expect parentheses")
                end += 1
            if depth != 0:
                raise FeedbackViewError("unterminated expect call")
            while end < closing and source[end].isspace():
                end += 1
            if end >= closing or source[end] != ";":
                raise FeedbackViewError("expect call is not a standalone statement")
            spans.append((cursor, end + 1))
            cursor = end + 1
            continue
        cursor += 1
    return spans


def harness_with_cases(
    source: str,
    spans: list[tuple[int, int]],
    selected: set[int],
) -> str:
    chars = list(source)
    for index, (start, end) in enumerate(spans):
        if index in selected:
            continue
        for position in range(start, end):
            if chars[position] not in {"\r", "\n"}:
                chars[position] = " "
    return "".join(chars)


def split_train_harness(
    *,
    task_id: str,
    tests: str,
    seed: int,
) -> dict[str, Any]:
    spans = extract_expect_spans(tests)
    if len(spans) < 2:
        raise FeedbackViewError(
            "fewer than two independently splittable expect cases"
        )
    tests_sha = hashlib.sha256(tests.encode("utf-8")).hexdigest()
    ranked = sorted(
        range(len(spans)),
        key=lambda index: stable_sha256(
            {
                "schema": SPLIT_SCHEMA,
                "seed": seed,
                "task_id": task_id,
                "tests_sha256": tests_sha,
                "case_index": index,
            }
        ),
    )
    # Odd policy: expose floor(N/2), retain ceil(N/2).  No more than half of
    # the train behavioral cases can cross the teacher boundary.
    visible_count = len(spans) // 2
    visible = set(ranked[:visible_count])
    holdback = set(range(len(spans))) - visible
    if not visible or not holdback:
        raise FeedbackViewError("feedback split did not produce two nonempty halves")
    feedback = harness_with_cases(tests, spans, visible)
    reward_holdback = harness_with_cases(tests, spans, holdback)
    if len(extract_expect_spans(feedback)) != len(visible):
        raise FeedbackViewError("visible feedback reparse count mismatch")
    if len(extract_expect_spans(reward_holdback)) != len(holdback):
        raise FeedbackViewError("reward holdback reparse count mismatch")
    return {
        "tests_sha256": tests_sha,
        "case_count": len(spans),
        "visible_count": len(visible),
        "holdback_count": len(holdback),
        "visible_case_indices": sorted(visible),
        "holdback_case_indices": sorted(holdback),
        "feedback_tests": feedback,
        "reward_holdback_tests": reward_holdback,
    }


def _derive(
    parent_rows: list[dict[str, Any]],
    parent_f2_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    f2_by_id = {str(row.get("task_id") or ""): row for row in parent_f2_rows}
    rollout: list[dict[str, Any]] = []
    filtered_f2: list[dict[str, Any]] = []
    holdbacks: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    accounting = {
        "parent_rows": len(parent_rows),
        "eligible_rows": 0,
        "excluded_rows": 0,
        "source_expect_cases": 0,
        "visible_expect_cases": 0,
        "holdback_expect_cases": 0,
        "odd_case_tasks": 0,
    }
    for row in parent_rows:
        task_id = str(row.get("task_id") or "")
        tests = row.get("tests")
        if not task_id or not isinstance(tests, str) or not tests.strip():
            excluded.append(
                {"task_id": task_id, "reason": "missing_train_tests"}
            )
            continue
        try:
            split = split_train_harness(task_id=task_id, tests=tests, seed=seed)
        except FeedbackViewError as exc:
            case_count = 0
            try:
                case_count = len(extract_expect_spans(tests))
            except FeedbackViewError:
                pass
            excluded.append(
                {
                    "task_id": task_id,
                    "reason": str(exc),
                    "detected_expect_cases": case_count,
                    "tests_sha256": hashlib.sha256(
                        tests.encode("utf-8")
                    ).hexdigest(),
                }
            )
            continue
        f2_row = f2_by_id.get(task_id)
        if f2_row is None:
            raise FeedbackViewError(f"{task_id}: missing parent F2 row")
        safe_row = {
            key: value
            for key, value in row.items()
            if key not in FORBIDDEN_ROLLOUT_FIELDS
            and key != "feedback_tests"
        }
        safe_row["feedback_tests"] = split["feedback_tests"]
        safe_row["verpo_feedback_split_schema"] = SPLIT_SCHEMA
        safe_row["verpo_feedback_split_binding_sha256"] = stable_sha256(
            {
                key: split[key]
                for key in (
                    "tests_sha256",
                    "case_count",
                    "visible_count",
                    "holdback_count",
                    "visible_case_indices",
                    "holdback_case_indices",
                )
            }
        )
        rollout.append(safe_row)
        filtered_f2.append(f2_row)
        holdbacks.append(
            {
                "task_id": task_id,
                "schema": SPLIT_SCHEMA,
                **split,
            }
        )
        accounting["source_expect_cases"] += int(split["case_count"])
        accounting["visible_expect_cases"] += int(split["visible_count"])
        accounting["holdback_expect_cases"] += int(split["holdback_count"])
        accounting["odd_case_tasks"] += int(split["case_count"] % 2 == 1)
    accounting["eligible_rows"] = len(rollout)
    accounting["excluded_rows"] = len(excluded)
    if (
        len(rollout) + len(excluded) != len(parent_rows)
        or not rollout
        or len(filtered_f2) != len(rollout)
        or len(holdbacks) != len(rollout)
    ):
        raise FeedbackViewError("feedback derivation accounting failed")
    return rollout, filtered_f2, holdbacks, excluded, accounting


def build_feedback_view(
    *,
    executable_dataset: str | Path,
    executable_seal: str | Path,
    executable_f2: str | Path,
    executable_f2_manifest: str | Path,
    executable_view_report: str | Path,
    expected_executable_view_report_sha256: str,
    contract: str | Path,
    output_dir: str | Path,
    seed: int,
    expected_accounting: Mapping[str, Any] | None,
    expected_eligible_task_ids_sha256: str | None,
    expected_excluded_task_ids_sha256: str | None,
    expected_parent_fit_rows: int | None = None,
) -> dict[str, Any]:
    explicit_contract = (
        expected_accounting is not None,
        expected_eligible_task_ids_sha256 is not None,
        expected_excluded_task_ids_sha256 is not None,
    )
    if any(explicit_contract) and not all(explicit_contract):
        raise FeedbackViewError(
            "feedback expectation must provide accounting and both task digests "
            "together, or derive all three from the sealed parent"
        )
    parent = validate_executable_view(
        dataset=executable_dataset,
        seal=executable_seal,
        f2=executable_f2,
        f2_manifest=executable_f2_manifest,
        build_report=executable_view_report,
        expected_build_report_sha256=expected_executable_view_report_sha256,
        contract=contract,
        verify_heldout=False,
        expected_parent_rows=expected_parent_fit_rows,
    )
    parent_rows = load_jsonl(Path(executable_dataset), "parent executable train")
    parent_f2_rows = load_jsonl(Path(executable_f2), "parent executable F2")
    rollout, filtered_f2, holdbacks, excluded, accounting = _derive(
        parent_rows, parent_f2_rows, seed=seed
    )
    eligible_digest = stable_sha256(
        [str(row["task_id"]) for row in rollout]
    )
    excluded_digest = stable_sha256(
        [str(row["task_id"]) for row in excluded]
    )
    if all(explicit_contract):
        assert expected_accounting is not None
        assert expected_eligible_task_ids_sha256 is not None
        assert expected_excluded_task_ids_sha256 is not None
        pinned_accounting = normalize_expected_accounting(
            expected_accounting
        )
        pinned_eligible_digest = normalize_task_digest(
            expected_eligible_task_ids_sha256,
            "expected eligible task IDs digest",
        )
        pinned_excluded_digest = normalize_task_digest(
            expected_excluded_task_ids_sha256,
            "expected excluded task IDs digest",
        )
    else:
        # The output report/public manifest become the immutable predeclaration;
        # all downstream consumers pin one or both file hashes.
        pinned_accounting = dict(accounting)
        pinned_eligible_digest = eligible_digest
        pinned_excluded_digest = excluded_digest
    if (
        accounting["parent_rows"] != parent["rows"]
        or len(parent_rows) != parent["rows"]
        or accounting != pinned_accounting
        or eligible_digest != pinned_eligible_digest
        or excluded_digest != pinned_excluded_digest
    ):
        raise FeedbackViewError(
            "feedback split differs from the independently pinned production "
            "membership/accounting contract: "
            f"observed={accounting} eligible_ids={eligible_digest} "
            f"excluded_ids={excluded_digest}"
        )

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    rollout_path = destination / "verpo_rollout_feedback.jsonl"
    seal_path = destination / "verpo_rollout_feedback.seal.json"
    f2_path = destination / "verpo_teacher_f2.jsonl"
    f2_manifest_path = destination / "verpo_teacher_f2.jsonl.manifest.json"
    holdback_path = destination / "reward_holdback.private.jsonl"
    excluded_path = destination / "excluded_feedback_tasks.jsonl"
    report_path = destination / "verpo_feedback_view.build.json"
    public_manifest_path = destination / "verpo_feedback_view.public.json"
    _write_jsonl(rollout_path, rollout)
    _write_jsonl(f2_path, filtered_f2)
    _write_jsonl(holdback_path, holdbacks)
    _write_jsonl(excluded_path, excluded)

    parent_manifest = load_json(
        Path(executable_f2_manifest), "parent executable F2 manifest"
    )
    parent_invariants = parent_manifest.get("invariants") or {}
    if any(parent_invariants.get(name) is not True for name in F2_REQUIRED_INVARIANTS):
        raise FeedbackViewError(
            "parent F2 manifest lacks required public representation invariants"
        )
    f2_manifest = {
        "schema": parent_manifest.get("schema"),
        "rows": len(filtered_f2),
        "output": file_record(f2_path),
        "dataset": file_record(rollout_path),
        "f2_prompt_contract": dict(
            parent_manifest.get("f2_prompt_contract") or {}
        ),
        "invariants": {
            name: True for name in F2_REQUIRED_INVARIANTS
        },
        "verpo_feedback_derivation": {
            "schema": SPLIT_SCHEMA,
            "parent_manifest_sha256": sha256_file(executable_f2_manifest),
            "task_ids_sha256": stable_sha256(
                [str(row["task_id"]) for row in rollout]
            ),
            "tests_or_holdback_exposed_in_f2": False,
            "parent_or_private_artifact_paths_exposed": False,
        },
    }
    assert_only_allowed_path_keys(
        f2_manifest,
        allowed={("output", "path"), ("dataset", "path")},
        label="trainer-safe F2 manifest",
    )
    _write_json(f2_manifest_path, f2_manifest)

    parent_seal = load_json(Path(executable_seal), "parent executable seal")
    seal = {
        "schema": parent_seal.get("schema"),
        "selected_role": "fit",
        "rows": len(rollout),
        "output_sha256": sha256_file(rollout_path),
        "contract_sha256": parent_seal.get("contract_sha256"),
        "representation_schema": parent_seal.get("representation_schema"),
        "training_objective_scope": ROLLOUT_SCOPE,
        "parent_executable_rows": parent["rows"],
        "parent_executable_dataset_sha256": sha256_file(executable_dataset),
        "feedback_split_schema": SPLIT_SCHEMA,
        "feedback_split_seed": seed,
        "feedback_tests_train_visible": True,
        "acceptance_tests_in_rollout": False,
        "reward_holdback_in_rollout": False,
        "heldout_measure_only": False,
        "training_allowed": True,
        "execution_ineligible_task_ids": [],
        "parent_or_private_artifact_paths_exposed": False,
    }
    # V3 join seals require source-blind pool metadata.  A production parent
    # executable view currently uses the v1 seal; fail closed rather than copy
    # a parent aggregate whose row count would be wrong for this subset.
    if "contract_schema" in parent_seal or "pool_metadata" in parent_seal:
        raise FeedbackViewError(
            "feedback subset builder needs an explicit filtered v3 pool seal"
        )
    assert_only_allowed_path_keys(
        seal, allowed=set(), label="trainer-safe rollout seal"
    )
    _write_json(seal_path, seal)

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "split_policy": {
            "schema": SPLIT_SCHEMA,
            "seed": seed,
            "source_field": "tests",
            "acceptance_tests_used": False,
            "minimum_cases": 2,
            "even_policy": "N/2 visible; N/2 holdback",
            "odd_policy": "floor(N/2) visible; ceil(N/2) holdback",
            "single_case_policy": "exclude",
            "no_expect_policy": "exclude",
            "malformed_policy": "exclude",
            "selection": "lowest task-bound SHA-256 ranks become visible",
        },
        "predeclared_expectation": {
            "accounting": pinned_accounting,
            "eligible_task_ids_sha256": pinned_eligible_digest,
            "excluded_task_ids_sha256": pinned_excluded_digest,
        },
        "inputs": {
            "executable_view_report": file_record(executable_view_report),
            "executable_dataset": file_record(executable_dataset),
            "executable_seal": file_record(executable_seal),
            "executable_f2": file_record(executable_f2),
            "executable_f2_manifest": file_record(executable_f2_manifest),
            "contract": file_record(contract),
        },
        "outputs": {
            "rollout": file_record(rollout_path),
            "seal": file_record(seal_path),
            "f2": file_record(f2_path),
            "f2_manifest": file_record(f2_manifest_path),
            "reward_holdback_private": file_record(holdback_path),
            "excluded": file_record(excluded_path),
        },
        "accounting": accounting,
        "digests": {
            "eligible_task_ids_sha256": eligible_digest,
            "excluded_task_ids_sha256": excluded_digest,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "invariants": {
            "parent_is_sealed_executable_view": True,
            "parent_is_exact_safe1578": (
                parent["rows"] == EXPECTED_EXECUTABLE_ROWS
            ),
            "dev175_bytes_opened": False,
            "acceptance_tests_read_or_used": False,
            "rollout_contains_no_acceptance_or_holdback_fields": True,
            "deepseek_f2_contains_no_tests": True,
            "visible_and_holdback_nonempty_for_every_eligible_task": True,
            "all_expect_cases_accounted_exactly_once": True,
            "compact_model_binding_fields_unchanged": True,
            "holdback_is_not_a_trainer_input": True,
        },
    }
    _write_json(report_path, report)
    public_manifest = {
        "schema": PUBLIC_SCHEMA,
        "status": "complete",
        "full_preprocessing_report_sha256": sha256_file(report_path),
        "split_policy": dict(report["split_policy"]),
        "predeclared_expectation": dict(report["predeclared_expectation"]),
        "artifacts": {
            key: dict(report["outputs"][key])
            for key in ("rollout", "seal", "f2", "f2_manifest")
        },
        "accounting": dict(accounting),
        "digests": {
            "eligible_task_ids_sha256": eligible_digest,
            "excluded_task_ids_sha256": excluded_digest,
        },
        "contract_sha256": sha256_file(contract),
        "invariants": {
            "dev175_bytes_opened": False,
            "acceptance_tests_read_or_used": False,
            "rollout_contains_no_acceptance_or_holdback_fields": True,
            "holdback_or_exclusion_artifact_record_exposed": False,
            "parent_artifact_record_exposed": False,
            "deepseek_f2_contains_no_tests": True,
            "full_private_report_is_not_a_trainer_input": True,
        },
    }
    _write_json(public_manifest_path, public_manifest)
    report["public_manifest"] = file_record(public_manifest_path)
    return report


def validate_feedback_view(
    *,
    rollout: str | Path,
    seal: str | Path,
    f2: str | Path,
    f2_manifest: str | Path,
    build_report: str | Path,
    expected_build_report_sha256: str,
    public_manifest: str | Path,
    expected_public_manifest_sha256: str,
    executable_dataset: str | Path,
    executable_seal: str | Path,
    executable_f2: str | Path,
    executable_f2_manifest: str | Path,
    executable_view_report: str | Path,
    expected_executable_view_report_sha256: str,
    contract: str | Path,
    expected_accounting: Mapping[str, Any] | None = None,
    expected_eligible_task_ids_sha256: str | None = None,
    expected_excluded_task_ids_sha256: str | None = None,
    expected_parent_fit_rows: int | None = None,
) -> dict[str, Any]:
    report_path = Path(build_report).expanduser().resolve()
    if sha256_file(report_path) != expected_build_report_sha256.strip().lower():
        raise FeedbackViewError("feedback-view report hash mismatch")
    report = load_json(report_path, "VeRPO feedback-view report")
    if report.get("schema") != SCHEMA or report.get("status") != "complete":
        raise FeedbackViewError("feedback-view report schema/status failed")
    (
        pinned_accounting,
        pinned_eligible_digest,
        pinned_excluded_digest,
    ) = resolve_feedback_expectation(
        expected_accounting=expected_accounting,
        expected_eligible_task_ids_sha256=(
            expected_eligible_task_ids_sha256
        ),
        expected_excluded_task_ids_sha256=(
            expected_excluded_task_ids_sha256
        ),
        attested=report.get("predeclared_expectation"),
        label="feedback-view report",
    )
    if report.get("predeclared_expectation") != {
        "accounting": pinned_accounting,
        "eligible_task_ids_sha256": pinned_eligible_digest,
        "excluded_task_ids_sha256": pinned_excluded_digest,
    }:
        raise FeedbackViewError(
            "feedback-view report differs from predeclared membership contract"
        )
    public_path = Path(public_manifest).expanduser().resolve()
    if sha256_file(public_path) != expected_public_manifest_sha256.strip().lower():
        raise FeedbackViewError("feedback-view public manifest hash mismatch")
    parent = validate_executable_view(
        dataset=executable_dataset,
        seal=executable_seal,
        f2=executable_f2,
        f2_manifest=executable_f2_manifest,
        build_report=executable_view_report,
        expected_build_report_sha256=expected_executable_view_report_sha256,
        contract=contract,
        verify_heldout=False,
        expected_parent_rows=expected_parent_fit_rows,
    )
    paths = {
        "rollout": Path(rollout).expanduser().resolve(),
        "seal": Path(seal).expanduser().resolve(),
        "f2": Path(f2).expanduser().resolve(),
        "f2_manifest": Path(f2_manifest).expanduser().resolve(),
    }
    for name, path in paths.items():
        expected = (report.get("outputs") or {}).get(name) or {}
        if (
            str(path) != str(Path(str(expected.get("path") or "")).resolve())
            or sha256_file(path) != expected.get("sha256")
        ):
            raise FeedbackViewError(f"feedback-view {name} differs from report")
    seed = int((report.get("split_policy") or {}).get("seed", -1))
    parent_rows = load_jsonl(Path(executable_dataset), "parent executable train")
    parent_f2_rows = load_jsonl(Path(executable_f2), "parent executable F2")
    expected_rows, expected_f2, expected_holdbacks, excluded, accounting = _derive(
        parent_rows, parent_f2_rows, seed=seed
    )
    observed_rows = load_jsonl(paths["rollout"], "VeRPO rollout")
    observed_f2 = load_jsonl(paths["f2"], "VeRPO F2")
    observed_f2_manifest = load_json(
        paths["f2_manifest"], "VeRPO filtered F2 manifest"
    )
    assert_only_allowed_path_keys(
        observed_f2_manifest,
        allowed={("output", "path"), ("dataset", "path")},
        label="trainer-safe F2 manifest",
    )
    if observed_rows != expected_rows or observed_f2 != expected_f2:
        raise FeedbackViewError("feedback-view rows do not reproduce exactly")
    observed_seal = load_json(paths["seal"], "VeRPO rollout seal")
    assert_only_allowed_path_keys(
        observed_seal,
        allowed=set(),
        label="trainer-safe rollout seal",
    )
    if (
        observed_seal.get("selected_role") != "fit"
        or observed_seal.get("training_allowed") is not True
        or observed_seal.get("heldout_measure_only") is not False
        or observed_seal.get("training_objective_scope") != ROLLOUT_SCOPE
        or int(observed_seal.get("rows", -1)) != len(observed_rows)
        or int(observed_seal.get("parent_executable_rows", -1))
        != parent["rows"]
        or observed_seal.get("output_sha256") != sha256_file(paths["rollout"])
        or observed_seal.get("contract_sha256") != sha256_file(contract)
        or observed_seal.get("acceptance_tests_in_rollout") is not False
        or observed_seal.get("reward_holdback_in_rollout") is not False
    ):
        raise FeedbackViewError("VeRPO rollout seal contract failed")
    for parent_row, task_id in zip(
        [row for row in parent_rows if str(row.get("task_id")) in {
            str(item.get("task_id")) for item in observed_rows
        }],
        [str(row.get("task_id")) for row in observed_rows],
        strict=True,
    ):
        if str(parent_row.get("task_id")) != task_id:
            raise FeedbackViewError(
                "eligible task order differs from sealed executable parent"
            )
    parent_by_id = {
        str(row.get("task_id") or ""): row for row in parent_rows
    }
    for row in observed_rows:
        task_id = str(row.get("task_id") or "")
        if any(field in row for field in FORBIDDEN_ROLLOUT_FIELDS):
            raise FeedbackViewError(f"{task_id}: forbidden tests leaked to rollout")
        if len(extract_expect_spans(str(row.get("feedback_tests") or ""))) < 1:
            raise FeedbackViewError(f"{task_id}: no visible reward component")
        original = parent_by_id[task_id]
        if any(row.get(field) != original.get(field) for field in MODEL_BINDING_FIELDS):
            raise FeedbackViewError(f"{task_id}: compact model binding changed")
    if (
        accounting.get("parent_rows") != parent["rows"]
        or accounting != report.get("accounting")
        or accounting != pinned_accounting
    ):
        raise FeedbackViewError("feedback-view accounting differs from report")
    observed_digests = report.get("digests") or {}
    if (
        stable_sha256([str(row["task_id"]) for row in observed_rows])
        != pinned_eligible_digest
        or stable_sha256([str(row["task_id"]) for row in excluded])
        != pinned_excluded_digest
        or observed_digests.get("eligible_task_ids_sha256")
        != pinned_eligible_digest
        or observed_digests.get("excluded_task_ids_sha256")
        != pinned_excluded_digest
    ):
        raise FeedbackViewError(
            "feedback-view membership differs from predeclared digests"
        )
    holdback_record = (report.get("outputs") or {}).get(
        "reward_holdback_private"
    ) or {}
    excluded_record = (report.get("outputs") or {}).get("excluded") or {}
    if (
        sha256_file(holdback_record["path"]) != holdback_record.get("sha256")
        or sha256_file(excluded_record["path"]) != excluded_record.get("sha256")
        or load_jsonl(
            Path(holdback_record["path"]), "private reward holdback"
        )
        != expected_holdbacks
        or _load_jsonl_allow_empty(
            Path(excluded_record["path"]), "excluded feedback tasks"
        )
        != excluded
    ):
        raise FeedbackViewError("private split attestations differ")
    expected_public = {
        "schema": PUBLIC_SCHEMA,
        "status": "complete",
        "full_preprocessing_report_sha256": sha256_file(report_path),
        "split_policy": dict(report["split_policy"]),
        "predeclared_expectation": dict(report["predeclared_expectation"]),
        "artifacts": {
            key: dict(report["outputs"][key])
            for key in ("rollout", "seal", "f2", "f2_manifest")
        },
        "accounting": dict(accounting),
        "digests": {
            "eligible_task_ids_sha256": pinned_eligible_digest,
            "excluded_task_ids_sha256": pinned_excluded_digest,
        },
        "contract_sha256": sha256_file(contract),
        "invariants": {
            "dev175_bytes_opened": False,
            "acceptance_tests_read_or_used": False,
            "rollout_contains_no_acceptance_or_holdback_fields": True,
            "holdback_or_exclusion_artifact_record_exposed": False,
            "parent_artifact_record_exposed": False,
            "deepseek_f2_contains_no_tests": True,
            "full_private_report_is_not_a_trainer_input": True,
        },
    }
    observed_public = load_json(
        public_path, "VeRPO feedback-view public manifest"
    )
    if observed_public != expected_public:
        raise FeedbackViewError(
            "feedback-view public manifest differs from full validation"
        )
    return {
        "schema": SCHEMA,
        "report": file_record(report_path),
        "public_manifest": file_record(public_path),
        "rollout": file_record(paths["rollout"]),
        "seal": file_record(paths["seal"]),
        "f2": file_record(paths["f2"]),
        "f2_manifest": file_record(paths["f2_manifest"]),
        "rows": len(observed_rows),
        "parent_rows": parent["rows"],
        "task_ids_sha256": stable_sha256(
            [str(row["task_id"]) for row in observed_rows]
        ),
        "excluded_task_ids_sha256": pinned_excluded_digest,
        "accounting": accounting,
        "parent_executable_view": parent,
        "heldout_bytes_opened_during_validation": False,
        "acceptance_tests_exposed": False,
        "reward_holdback_exposed": False,
    }


def validate_feedback_training_boundary(
    *,
    rollout: str | Path,
    seal: str | Path,
    f2: str | Path,
    f2_manifest: str | Path,
    public_manifest: str | Path,
    expected_public_manifest_sha256: str,
    contract: str | Path,
    expected_accounting: Mapping[str, Any] | None = None,
    expected_eligible_task_ids_sha256: str | None = None,
    expected_excluded_task_ids_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate only artifacts permitted inside the GPU/API training process.

    This deliberately never resolves or opens the parent dataset, its
    acceptance tests, the excluded-task audit, or the private holdback artifact.
    Full split reproduction belongs to preprocessing and chain sealing.
    """

    public_path = Path(public_manifest).expanduser().resolve()
    if sha256_file(public_path) != expected_public_manifest_sha256.strip().lower():
        raise FeedbackViewError("feedback-view public manifest hash mismatch")
    public = load_json(public_path, "VeRPO feedback-view public manifest")
    (
        pinned_accounting,
        pinned_eligible_digest,
        pinned_excluded_digest,
    ) = resolve_feedback_expectation(
        expected_accounting=expected_accounting,
        expected_eligible_task_ids_sha256=(
            expected_eligible_task_ids_sha256
        ),
        expected_excluded_task_ids_sha256=(
            expected_excluded_task_ids_sha256
        ),
        attested=public.get("predeclared_expectation"),
        label="feedback-view public manifest",
    )
    invariants = public.get("invariants") or {}
    accounting = public.get("accounting") or {}
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or public.get("status") != "complete"
        or invariants.get("dev175_bytes_opened") is not False
        or invariants.get("acceptance_tests_read_or_used") is not False
        or invariants.get(
            "rollout_contains_no_acceptance_or_holdback_fields"
        )
        is not True
        or invariants.get(
            "holdback_or_exclusion_artifact_record_exposed"
        )
        is not False
        or invariants.get("parent_artifact_record_exposed") is not False
        or invariants.get("full_private_report_is_not_a_trainer_input")
        is not True
        or int(accounting.get("eligible_rows", 0)) <= 0
        or int(accounting.get("parent_rows", -1)) <= 0
        or dict(accounting) != pinned_accounting
        or public.get("predeclared_expectation")
        != {
            "accounting": pinned_accounting,
            "eligible_task_ids_sha256": pinned_eligible_digest,
            "excluded_task_ids_sha256": pinned_excluded_digest,
        }
        or (public.get("digests") or {}).get("eligible_task_ids_sha256")
        != pinned_eligible_digest
        or (public.get("digests") or {}).get("excluded_task_ids_sha256")
        != pinned_excluded_digest
    ):
        raise FeedbackViewError("feedback-view training boundary attestation failed")
    paths = {
        "rollout": Path(rollout).expanduser().resolve(),
        "seal": Path(seal).expanduser().resolve(),
        "f2": Path(f2).expanduser().resolve(),
        "f2_manifest": Path(f2_manifest).expanduser().resolve(),
    }
    for name, path in paths.items():
        expected = (public.get("artifacts") or {}).get(name) or {}
        if (
            str(path) != str(Path(str(expected.get("path") or "")).resolve())
            or sha256_file(path) != expected.get("sha256")
        ):
            raise FeedbackViewError(f"training-boundary {name} differs from report")
    rollout_rows = load_jsonl(paths["rollout"], "VeRPO rollout")
    f2_rows = load_jsonl(paths["f2"], "VeRPO filtered F2")
    rollout_ids = [str(row.get("task_id") or "") for row in rollout_rows]
    f2_ids = [str(row.get("task_id") or "") for row in f2_rows]
    if (
        len(rollout_rows) != int(accounting["eligible_rows"])
        or rollout_ids != f2_ids
        or len(set(rollout_ids)) != len(rollout_ids)
        or stable_sha256(rollout_ids)
        != pinned_eligible_digest
    ):
        raise FeedbackViewError("training-boundary task set differs")
    for row in rollout_rows:
        task_id = str(row.get("task_id") or "")
        if any(field in row for field in FORBIDDEN_ROLLOUT_FIELDS):
            raise FeedbackViewError(f"{task_id}: private tests leaked to rollout")
        if len(extract_expect_spans(str(row.get("feedback_tests") or ""))) < 1:
            raise FeedbackViewError(f"{task_id}: no visible reward component")
    manifest = load_json(paths["f2_manifest"], "VeRPO filtered F2 manifest")
    assert_only_allowed_path_keys(
        manifest,
        allowed={("output", "path"), ("dataset", "path")},
        label="trainer-safe F2 manifest",
    )
    if (
        int(manifest.get("rows", -1)) != len(f2_rows)
        or (manifest.get("output") or {}).get("sha256")
        != sha256_file(paths["f2"])
        or (manifest.get("dataset") or {}).get("sha256")
        != sha256_file(paths["rollout"])
        or (
            manifest.get("verpo_feedback_derivation") or {}
        ).get("tests_or_holdback_exposed_in_f2")
        is not False
        or (
            manifest.get("verpo_feedback_derivation") or {}
        ).get("parent_or_private_artifact_paths_exposed")
        is not False
    ):
        raise FeedbackViewError("filtered F2 training boundary failed")
    observed_seal = load_json(paths["seal"], "VeRPO rollout seal")
    assert_only_allowed_path_keys(
        observed_seal,
        allowed=set(),
        label="trainer-safe rollout seal",
    )
    if (
        observed_seal.get("selected_role") != "fit"
        or observed_seal.get("training_objective_scope") != ROLLOUT_SCOPE
        or int(observed_seal.get("rows", -1)) != len(rollout_rows)
        or int(observed_seal.get("parent_executable_rows", -1))
        != int(accounting["parent_rows"])
        or observed_seal.get("output_sha256") != sha256_file(paths["rollout"])
        or observed_seal.get("contract_sha256") != sha256_file(contract)
        or observed_seal.get("acceptance_tests_in_rollout") is not False
        or observed_seal.get("reward_holdback_in_rollout") is not False
        or observed_seal.get("parent_or_private_artifact_paths_exposed")
        is not False
    ):
        raise FeedbackViewError("training-boundary rollout seal failed")
    return {
        "schema": SCHEMA,
        "public_manifest": file_record(public_path),
        "full_preprocessing_report_sha256": public.get(
            "full_preprocessing_report_sha256"
        ),
        "rollout": file_record(paths["rollout"]),
        "seal": file_record(paths["seal"]),
        "f2": file_record(paths["f2"]),
        "f2_manifest": file_record(paths["f2_manifest"]),
        "rows": len(rollout_rows),
        "parent_rows": int(accounting["parent_rows"]),
        "task_ids_sha256": stable_sha256(rollout_ids),
        "excluded_task_ids_sha256": pinned_excluded_digest,
        "accounting": dict(accounting),
        "heldout_bytes_opened_during_validation": False,
        "parent_or_private_bytes_opened_during_validation": False,
        "acceptance_tests_exposed": False,
        "reward_holdback_exposed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--executable-dataset", required=True)
    parser.add_argument("--executable-seal", required=True)
    parser.add_argument("--executable-f2", required=True)
    parser.add_argument("--executable-f2-manifest", required=True)
    parser.add_argument("--executable-view-report", required=True)
    parser.add_argument(
        "--expected-executable-view-report-sha256", required=True
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-parent-fit-rows",
        type=int,
        default=1580,
        help=(
            "Sealed full fit-universe row count. Expanded production passes "
            "2776; the default preserves the legacy build-v2 contract."
        ),
    )
    parser.add_argument(
        "--derive-sealed-accounting",
        action="store_true",
        help=(
            "Derive feedback counts and task-ID digests from the hash-pinned "
            "executable view, then seal them in the output manifests."
        ),
    )
    parser.add_argument(
        "--expected-eligible-rows",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["eligible_rows"],
    )
    parser.add_argument(
        "--expected-excluded-rows",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["excluded_rows"],
    )
    parser.add_argument(
        "--expected-source-expect-cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["source_expect_cases"],
    )
    parser.add_argument(
        "--expected-visible-expect-cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["visible_expect_cases"],
    )
    parser.add_argument(
        "--expected-holdback-expect-cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["holdback_expect_cases"],
    )
    parser.add_argument(
        "--expected-odd-case-tasks",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["odd_case_tasks"],
    )
    parser.add_argument(
        "--expected-eligible-task-ids-sha256",
        default=PRODUCTION_ELIGIBLE_TASK_IDS_SHA256,
    )
    parser.add_argument(
        "--expected-excluded-task-ids-sha256",
        default=PRODUCTION_EXCLUDED_TASK_IDS_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_accounting = None
    expected_eligible_digest = None
    expected_excluded_digest = None
    if not args.derive_sealed_accounting:
        expected_accounting = {
            "parent_rows": EXPECTED_EXECUTABLE_ROWS,
            "eligible_rows": args.expected_eligible_rows,
            "excluded_rows": args.expected_excluded_rows,
            "source_expect_cases": args.expected_source_expect_cases,
            "visible_expect_cases": args.expected_visible_expect_cases,
            "holdback_expect_cases": args.expected_holdback_expect_cases,
            "odd_case_tasks": args.expected_odd_case_tasks,
        }
        expected_eligible_digest = args.expected_eligible_task_ids_sha256
        expected_excluded_digest = args.expected_excluded_task_ids_sha256
    report = build_feedback_view(
        executable_dataset=args.executable_dataset,
        executable_seal=args.executable_seal,
        executable_f2=args.executable_f2,
        executable_f2_manifest=args.executable_f2_manifest,
        executable_view_report=args.executable_view_report,
        expected_executable_view_report_sha256=(
            args.expected_executable_view_report_sha256
        ),
        contract=args.contract,
        output_dir=args.output_dir,
        seed=args.seed,
        expected_accounting=expected_accounting,
        expected_eligible_task_ids_sha256=expected_eligible_digest,
        expected_excluded_task_ids_sha256=expected_excluded_digest,
        expected_parent_fit_rows=args.expected_parent_fit_rows,
    )
    print(
        "VERPO_FEEDBACK_VIEW_BUILT "
        f"eligible={report['accounting']['eligible_rows']} "
        f"excluded={report['accounting']['excluded_rows']} "
        f"visible_cases={report['accounting']['visible_expect_cases']} "
        f"holdback_cases={report['accounting']['holdback_expect_cases']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
