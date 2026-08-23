#!/usr/bin/env python3
"""Phased three-arm semantic-rescue ablation for native T5Gemma 2.

The experiment is deliberately evaluation-only.  It compares, on the same
predeclared all-zero TRAIN groups and under one frozen SFT checkpoint:

* ``plain_resample`` -- four fresh samples from the original F2 source;
* ``compiler_only`` -- two samples for each of two diverse failed parents;
* ``semantic_judge`` -- the same parent/slot schedule with grounded judge
  diagnosis and repair steps appended to the compiler observation.

The phases are separate commands:

``project``
    Validate completed local-harvest journals and emit only their visible-safe
    base candidates.  Holdback-derived journal fields are never copied.
``plan``
    Re-score projected candidates per visible case, select exactly the first
    hash-ranked all-zero groups, and freeze two diverse parents per group.
``diagnose``
    Make exactly one visible-only, grounded judge call per group.  A durable
    start marker prevents an ambiguous paid call from being repeated.
``generate``
    Materialize twelve fixed ITT slots per group and generate with the frozen
    native T5 policy.  Rejected diagnoses reserve failures and make no call.
``score``
    Score all generated slots visibly before opening the complementary
    development holdback.  The same visibly selected repair is then checked
    on holdback.  No final-175 artifact is accepted by any command.

No optimizer is constructed here.  Judge-conditioned outputs are therefore
not mislabeled as on-policy updates.  Full visible+holdback student repairs
are exported only as possible off-line RS-SFT targets; partial preferences are
explicitly marked off-policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


PATCH_ROOT = Path(__file__).resolve().parents[2]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation.durable_evaluation_journal import (
    GENESIS_SHA256,
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.t5gemma2_f2_passk_inference import (
    _checkpoint_record,
    generate_candidate_batch,
    load_policy,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    PUBLIC_SCHEMA,
    SPLIT_SCHEMA,
    extract_expect_spans,
)
from scripts.training.seq2seq_verpo_core import (
    max_min_diverse_indices,
    sanitize_compiler_diagnostic,
    sha256_text,
)
from scripts.training.t5gemma2_compiler_feedback_verpo import (
    _decoder_special_ids,
    _encode_source,
    score_dart_candidate,
)
from scripts.training.t5gemma2_enriched_sft import build_encoder_source
from scripts.training.t5gemma2_local_rs_sft_pilot import (
    JOURNAL_SCHEMA as LOCAL_JOURNAL_SCHEMA,
    RUN_SCHEMA as LOCAL_RUN_SCHEMA,
    _split_binding,
)
from scripts.training.verpo_judge_antigravity import (
    DIAGNOSE_RESULT_SCHEMA,
    DIAGNOSE_VALIDATOR_SCHEMA_VERSION,
    RESPONSE_RECEIPT_ATTESTATION_SCHEMA,
    RESPONSE_RECEIPT_SCHEMA,
    VerpoJudge,
)
from scripts.training.verpo_rescue_grounding import (
    GROUNDING_SCHEMA,
    GroundingCatalog,
    build_grounding_catalog,
    validate_diagnosis_item,
)


PROJECT_SCHEMA = "t5gemma2-semantic-rescue-base-projection-v1"
PLAN_SCHEMA = "t5gemma2-semantic-rescue-plan-v1"
DIAGNOSIS_SCHEMA = "t5gemma2-semantic-rescue-diagnoses-v1"
DIAGNOSIS_JOURNAL_SCHEMA = "t5gemma2-semantic-rescue-diagnosis-journal-v1"
GENERATION_SCHEMA = "t5gemma2-semantic-rescue-generation-v1"
GENERATION_JOURNAL_SCHEMA = "t5gemma2-semantic-rescue-generation-journal-v1"
VISIBLE_SCORE_SCHEMA = "t5gemma2-semantic-rescue-visible-selection-v1"
SCORE_SCHEMA = "t5gemma2-semantic-rescue-score-v1"
CONDITIONING_SCHEMA = "t5gemma2-semantic-rescue-conditioning-v1"
TRANSFER_SCHEMA = "t5gemma2-semantic-rescue-rs-sft-target-v1"
PREFERENCE_SCHEMA = "t5gemma2-semantic-rescue-off-policy-preference-v1"
RUNTIME_SCHEMA = "t5gemma2-semantic-rescue-runtime-v1"

PROJECT_HASH = "projection_sha256"
PLAN_HASH = "plan_sha256"
DIAGNOSIS_HASH = "diagnoses_sha256"
GENERATION_HASH = "generation_sha256"
VISIBLE_SCORE_HASH = "visible_selection_sha256"
SCORE_HASH = "score_sha256"

ARM_ORDER = ("plain_resample", "compiler_only", "semantic_judge")
SELECT_K = 2
REPAIRS_PER_PARENT = 2
SLOTS_PER_ARM = SELECT_K * REPAIRS_PER_PARENT
BASE_DRAWS = 4
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_GENERATION_RECEIPT_KEYS = (
    "action_tokens",
    "batch_position",
    "encoder_tokens",
    "eos_observed",
    "group_sample_index",
    "max_token_completion",
    "seed",
    "text_sha256",
)

_PRIVATE_KEYS = frozenset(
    {
        "acceptance_tests",
        "hidden_tests",
        "holdback_tests",
        "reward_holdback_tests",
        "private_tests",
        "private_gate_results",
        "private_gate_passed",
        "selected_target",
        "gold",
        "gold_code",
        "gold_target",
        "reference",
        "reference_code",
        "reference_dart",
    }
)
_PROVIDER_FORBIDDEN_KEYS = _PRIVATE_KEYS | frozenset(
    {
        "target",
        "target_code",
        "solution",
        "dart_source",
        "supervised_target",
    }
)
_FORBIDDEN_ROLLOUT_FIELDS = frozenset(
    {
        "tests",
        "acceptance_tests",
        "hidden_tests",
        "holdback_tests",
        "reward_holdback_tests",
    }
)


class RescueAblationError(ValueError):
    """An experiment artifact or phase transition failed closed."""


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueAblationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RescueAblationError(f"{label} is not a JSON object")
    return value


def _read_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise RescueAblationError(
                        f"{label}:{line_number}: blank JSONL row"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RescueAblationError(
                        f"{label}:{line_number}: row is not an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueAblationError(f"cannot read {label} {path}: {exc}") from exc
    return rows


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise RescueAblationError(f"caller supplied reserved digest {field}")
    body = dict(value)
    return {**body, field: canonical_sha256(body)}


def _require_seal(
    value: Mapping[str, Any], *, schema: str, field: str, label: str
) -> str:
    if value.get("schema") != schema or value.get("status") != "complete":
        raise RescueAblationError(f"{label} schema/status is invalid")
    observed = str(value.get(field) or "")
    body = {key: item for key, item in value.items() if key != field}
    if not _SHA_RE.fullmatch(observed) or canonical_sha256(body) != observed:
        raise RescueAblationError(f"{label} self digest is invalid")
    return observed


def _file_record(path: str | Path, *, expose_path: bool = False) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    result = {
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        "name": source.name,
    }
    if expose_path:
        result["path"] = str(source)
    return result


def _task_id(value: Mapping[str, Any], label: str) -> str:
    result = str(value.get("task_id") or "").strip()
    if not result:
        raise RescueAblationError(f"{label} has no task_id")
    return result


def _index_unique(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = _task_id(row, label)
        if task_id in result:
            raise RescueAblationError(f"{label} duplicates task {task_id}")
        result[task_id] = row
    return result


def _assert_no_keys(value: Any, forbidden: frozenset[str], label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in forbidden or (
                "holdback" in normalized and normalized not in {"holdback_exposed"}
            ):
                raise RescueAblationError(
                    f"{label} contains forbidden key {key!r}"
                )
            _assert_no_keys(child, forbidden, label)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            _assert_no_keys(child, forbidden, label)


def _runtime_provenance() -> dict[str, Any]:
    relative = {
        "runner": Path("scripts/training/t5gemma2_semantic_rescue_ablation.py"),
        "judge": Path("scripts/training/verpo_judge_antigravity.py"),
        "grounding": Path("scripts/training/verpo_rescue_grounding.py"),
        "native_scorer": Path(
            "scripts/training/t5gemma2_compiler_feedback_verpo.py"
        ),
        "seq2seq_core": Path("scripts/training/seq2seq_verpo_core.py"),
        "dart_evaluator": Path(
            "scripts/evaluation/graph_compile_at_k_antigravity.py"
        ),
        "feedback_builder": Path(
            "scripts/preprocessing/build_verpo_feedback_view.py"
        ),
        "inference": Path(
            "scripts/evaluation/t5gemma2_f2_passk_inference.py"
        ),
        "local_harvest": Path(
            "scripts/training/t5gemma2_local_rs_sft_pilot.py"
        ),
        "durable_journal": Path(
            "scripts/evaluation/durable_evaluation_journal.py"
        ),
        "f2_decoder": Path("../frontier_ceiling_patch_v1/frontier_f2.py"),
    }
    code: dict[str, dict[str, Any]] = {}
    for key, path in relative.items():
        resolved = PATCH_ROOT / path
        if not resolved.is_file():
            raise RescueAblationError(f"runtime dependency is missing: {path}")
        code[key] = {
            "path": path.as_posix(),
            "sha256": sha256_file(resolved),
        }
    return {
        "schema": RUNTIME_SCHEMA,
        "code": code,
        "code_bundle_sha256": canonical_sha256(code),
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda or ""),
        "cuda_device_count": torch.cuda.device_count(),
    }


def _validate_local_source(
    journal_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    report = _read_json(report_path, "local harvest report")
    if (
        report.get("schema") != "t5gemma2-local-rs-sft-pilot-report-v1"
        or report.get("status") != "complete"
    ):
        raise RescueAblationError("local harvest report is not complete")
    events = load_journal(journal_path)
    actual = journal_record(journal_path)
    declared = report.get("journal")
    if not isinstance(declared, Mapping):
        raise RescueAblationError("local harvest report has no journal binding")
    for field in (
        "sha256",
        "chain_head_sha256",
        "event_count",
        "head_event_sha256",
    ):
        if declared.get(field) != actual.get(field):
            raise RescueAblationError(
                f"local harvest report/journal differs at {field}"
            )
    if len(events) < 3:
        raise RescueAblationError("local harvest journal is incomplete")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("event") != "header"
        or header.get("schema") != LOCAL_JOURNAL_SCHEMA
        or not isinstance(contract, Mapping)
        or contract.get("schema") != LOCAL_RUN_SCHEMA
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise RescueAblationError("local harvest journal header is invalid")
    checkpoint = contract.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or report.get("checkpoint") != checkpoint
        or checkpoint.get("arm") != "sft"
    ):
        raise RescueAblationError("local harvest checkpoint binding differs")
    sampling = contract.get("sampling")
    if (
        not isinstance(sampling, Mapping)
        or sampling.get("base_samples") != BASE_DRAWS
        or sampling.get("max_source_tokens") != 32768
        or sampling.get("max_new_tokens") != 4096
    ):
        raise RescueAblationError("local harvest base sampling contract differs")
    terminals = events[1:-1]
    complete = events[-1]
    if (
        complete.get("event") != "complete"
        or complete.get("schema") != LOCAL_JOURNAL_SCHEMA
        or complete.get("tasks") != len(terminals)
    ):
        raise RescueAblationError("local harvest completion marker is invalid")
    ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for position, event in enumerate(terminals):
        task_id = _task_id(event, "local terminal")
        candidates = event.get("base_candidates")
        if (
            event.get("event") != "task_terminal"
            or event.get("schema") != LOCAL_JOURNAL_SCHEMA
            or event.get("task_position") != position
            or not isinstance(candidates, list)
            or len(candidates) != BASE_DRAWS
            or event.get("source_sha256") is None
            or not _SHA_RE.fullmatch(str(event.get("source_sha256")))
        ):
            raise RescueAblationError(
                f"{task_id}: local terminal/base group is malformed"
            )
        split_binding = str(event.get("split_binding_sha256") or "")
        if not _SHA_RE.fullmatch(split_binding):
            raise RescueAblationError(
                f"{task_id}: local terminal split binding is malformed"
            )
        safe_candidates: list[dict[str, Any]] = []
        for sample_index, raw in enumerate(candidates):
            if not isinstance(raw, Mapping):
                raise RescueAblationError(f"{task_id}: base candidate is malformed")
            code = str(raw.get("code") or "")
            digest = str(raw.get("code_sha256") or "")
            generation = raw.get("generation")
            if (
                raw.get("origin") != "base"
                or raw.get("sample_index") != sample_index
                or digest != sha256_text(code)
                or not isinstance(generation, Mapping)
                or type(generation.get("seed")) is not int
                or type(generation.get("action_tokens")) is not int
                or generation.get("action_tokens") < 0
                or type(generation.get("batch_position")) is not int
                or generation.get("batch_position") < 0
                or type(generation.get("encoder_tokens")) is not int
                or generation.get("encoder_tokens") <= 0
                or type(generation.get("group_sample_index")) is not int
                or generation.get("group_sample_index") != sample_index
                or type(generation.get("eos_observed")) is not bool
                or type(generation.get("max_token_completion")) is not bool
                or generation.get("text_sha256") != digest
            ):
                raise RescueAblationError(
                    f"{task_id}: base candidate receipt is malformed"
                )
            safe_candidates.append(
                {
                    "sample_index": sample_index,
                    "code": code,
                    "code_sha256": digest,
                    "generation": {
                        key: generation[key]
                        for key in _SAFE_GENERATION_RECEIPT_KEYS
                    },
                }
            )
        normalized.append(
            {
                "task_id": task_id,
                "source_sha256": str(event["source_sha256"]),
                "split_binding_sha256": split_binding,
                "source_terminal_event_sha256": str(
                    event["journal_event_sha256"]
                ),
                "base_candidates": safe_candidates,
            }
        )
        ids.append(task_id)
    if (
        len(ids) != len(set(ids))
        or complete.get("terminal_task_ids_sha256") != canonical_sha256(ids)
        or (contract.get("schedule") or {}).get("task_ids_sha256")
        != canonical_sha256(ids)
    ):
        raise RescueAblationError("local harvest task schedule binding differs")
    source_record = {
        "journal": {
            key: actual[key]
            for key in (
                "sha256",
                "chain_head_sha256",
                "event_count",
                "head_event_sha256",
            )
        },
        "report": _file_record(report_path),
        "contract_sha256": str(header["contract_sha256"]),
        "terminal_task_ids_sha256": canonical_sha256(ids),
        "task_split_bindings_sha256": canonical_sha256(
            [
                {
                    "task_id": row["task_id"],
                    "split_binding_sha256": row["split_binding_sha256"],
                }
                for row in normalized
            ]
        ),
        "tasks": len(ids),
    }
    return dict(checkpoint), normalized, source_record


def project(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.base_journal) != len(args.base_report) or not args.base_journal:
        raise RescueAblationError(
            "project requires paired --base-journal/--base-report values"
        )
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    expected_checkpoint: dict[str, Any] | None = None
    seen: set[str] = set()
    for journal_value, report_value in zip(
        args.base_journal, args.base_report, strict=True
    ):
        checkpoint, rows, record = _validate_local_source(
            Path(journal_value).expanduser().resolve(),
            Path(report_value).expanduser().resolve(),
        )
        if expected_checkpoint is None:
            expected_checkpoint = checkpoint
        elif checkpoint != expected_checkpoint:
            raise RescueAblationError(
                "base projection journals use different checkpoints"
            )
        for row in rows:
            task_id = str(row["task_id"])
            if task_id in seen:
                raise RescueAblationError(
                    f"base projection journals overlap at {task_id}"
                )
            seen.add(task_id)
            all_rows.append(row)
        sources.append(record)
    assert expected_checkpoint is not None
    body = {
        "schema": PROJECT_SCHEMA,
        "status": "complete",
        "runtime_provenance": _runtime_provenance(),
        "checkpoint": expected_checkpoint,
        "sources": sources,
        "tasks": all_rows,
        "accounting": {
            "source_journals": len(sources),
            "tasks": len(all_rows),
            "base_draws": len(all_rows) * BASE_DRAWS,
            "task_ids_sha256": canonical_sha256(
                [row["task_id"] for row in all_rows]
            ),
            "task_split_bindings_sha256": canonical_sha256(
                [
                    {
                        "task_id": row["task_id"],
                        "split_binding_sha256": row[
                            "split_binding_sha256"
                        ],
                    }
                    for row in all_rows
                ]
            ),
        },
        "privacy": {
            "projection_whitelist_only": True,
            "private_gate_results_copied": False,
            "selected_targets_copied": False,
            "holdback_bytes_copied": False,
            "final_175_opened": False,
        },
    }
    result = _seal(body, PROJECT_HASH)
    _assert_no_keys(result["tasks"], _PRIVATE_KEYS, "base projection")
    require_exact_or_write(args.output, result)
    return result


def _f2_prompt_contract(manifest: Mapping[str, Any]) -> tuple[str, str]:
    contract = manifest.get("f2_prompt_contract")
    if not isinstance(contract, Mapping):
        raise RescueAblationError("F2 manifest has no prompt contract")
    guide = contract.get("system_prompt")
    digest = str(contract.get("system_prompt_sha256") or "")
    if (
        not isinstance(guide, str)
        or not guide.strip()
        or not _SHA_RE.fullmatch(digest)
        or sha256_text(guide) != digest
        or contract.get("representation_schema") != "lossless-semantic-f2"
    ):
        raise RescueAblationError("F2 prompt contract is invalid")
    return guide, digest


def _visible_detail(
    candidate: str,
    tests: str,
    identity: str,
    *,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    raw = score_dart_candidate(
        candidate,
        tests,
        identity,
        timeout=timeout,
        stability_runs=stability_runs,
    )
    passes = raw.get("test_passes")
    compiled = raw.get("compiled")
    full_pass = raw.get("full_pass")
    if (
        not isinstance(passes, list)
        or not passes
        or any(type(value) is not bool for value in passes)
        or type(compiled) is not bool
        or type(full_pass) is not bool
        or (full_pass and (not compiled or not all(passes)))
    ):
        raise RescueAblationError("native visible scorer returned invalid detail")
    safe = sanitize_compiler_diagnostic(str(raw.get("diagnostic") or ""))
    return {
        "compiled": compiled,
        "full_pass": full_pass,
        "test_passes": list(passes),
        "passed_tests": sum(passes),
        "total_tests": len(passes),
        "diagnostic": safe,
        "diagnostic_sha256": sha256_text(safe),
    }


def _stable_task_rank(task_id: str, seed: int) -> str:
    return canonical_sha256(
        {
            "schema": "t5gemma2-semantic-rescue-task-order-v1",
            "seed": seed,
            "task_id": task_id,
        }
    )


def plan(args: argparse.Namespace) -> dict[str, Any]:
    projection = _read_json(args.projection, "base projection")
    projection_sha = _require_seal(
        projection,
        schema=PROJECT_SCHEMA,
        field=PROJECT_HASH,
        label="base projection",
    )
    rollout_rows = _read_jsonl(args.rollout_file, "visible rollout")
    f2_rows = _read_jsonl(args.f2_jsonl, "F2 source")
    manifest = _read_json(args.f2_manifest, "F2 manifest")
    public_manifest = _read_json(
        args.public_manifest, "feedback-view public manifest"
    )
    rollout_by_id = _index_unique(rollout_rows, "visible rollout")
    f2_by_id = _index_unique(f2_rows, "F2 source")
    if set(rollout_by_id) != set(f2_by_id):
        raise RescueAblationError("visible rollout and F2 task sets differ")
    rollout_sha = sha256_file(args.rollout_file)
    f2_sha = sha256_file(args.f2_jsonl)
    f2_manifest_sha = sha256_file(args.f2_manifest)
    public_artifacts = public_manifest.get("artifacts")
    public_digests = public_manifest.get("digests")
    public_split = public_manifest.get("split_policy")
    ordered_task_ids = [_task_id(row, "visible rollout") for row in rollout_rows]
    if (
        public_manifest.get("schema") != PUBLIC_SCHEMA
        or public_manifest.get("status") != "complete"
        or not isinstance(public_artifacts, Mapping)
        or not isinstance(public_digests, Mapping)
        or not isinstance(public_split, Mapping)
        or public_split.get("schema") != SPLIT_SCHEMA
        or not isinstance(public_artifacts.get("rollout"), Mapping)
        or public_artifacts["rollout"].get("sha256") != rollout_sha
        or not isinstance(public_artifacts.get("f2"), Mapping)
        or public_artifacts["f2"].get("sha256") != f2_sha
        or not isinstance(public_artifacts.get("f2_manifest"), Mapping)
        or public_artifacts["f2_manifest"].get("sha256") != f2_manifest_sha
        or public_digests.get("eligible_task_ids_sha256")
        != canonical_sha256(ordered_task_ids)
    ):
        raise RescueAblationError(
            "feedback-view public manifest binding differs"
        )
    output_record = manifest.get("output")
    dataset_record = manifest.get("dataset")
    derivation = manifest.get("verpo_feedback_derivation")
    if (
        not isinstance(output_record, Mapping)
        or output_record.get("sha256") != f2_sha
        or output_record.get("rows") not in {None, len(f2_rows)}
        or not isinstance(dataset_record, Mapping)
        or dataset_record.get("sha256") != rollout_sha
        or not isinstance(derivation, Mapping)
        or derivation.get("schema") != SPLIT_SCHEMA
        or derivation.get("task_ids_sha256")
        != canonical_sha256(ordered_task_ids)
    ):
        raise RescueAblationError("F2 manifest output binding differs")
    guide, guide_sha = _f2_prompt_contract(manifest)
    projected = projection.get("tasks")
    if not isinstance(projected, list):
        raise RescueAblationError("base projection task rows are invalid")
    projected_by_id = _index_unique(projected, "base projection")
    if not set(projected_by_id).issubset(rollout_by_id):
        raise RescueAblationError("base projection contains unknown tasks")
    ranked = sorted(
        projected_by_id,
        key=lambda task_id: (_stable_task_rank(task_id, args.seed), task_id),
    )
    groups: list[dict[str, Any]] = []
    scanned = 0
    excluded_nonzero = 0
    excluded_sparse = 0
    for task_id in ranked:
        if len(groups) >= args.groups:
            break
        scanned += 1
        projected_row = projected_by_id[task_id]
        rollout = rollout_by_id[task_id]
        f2 = f2_by_id[task_id]
        leaked = sorted(
            field for field in _FORBIDDEN_ROLLOUT_FIELDS if field in rollout
        )
        tests = rollout.get("feedback_tests")
        if leaked or not isinstance(tests, str) or not tests.strip():
            raise RescueAblationError(
                f"{task_id}: rollout leaks private fields or lacks visible tests"
            )
        if not extract_expect_spans(tests):
            raise RescueAblationError(f"{task_id}: no visible expect cases")
        split_binding = str(
            rollout.get("verpo_feedback_split_binding_sha256") or ""
        )
        if (
            rollout.get("verpo_feedback_split_schema") != SPLIT_SCHEMA
            or not _SHA_RE.fullmatch(split_binding)
            or projected_row.get("split_binding_sha256") != split_binding
        ):
            raise RescueAblationError(
                f"{task_id}: projected/public split binding differs"
            )
        encoder_source = build_encoder_source(f2, task_id)
        if sha256_text(encoder_source) != projected_row.get("source_sha256"):
            raise RescueAblationError(
                f"{task_id}: projected/native encoder source differs"
            )
        raw_f2 = f2.get("text")
        if (
            not isinstance(raw_f2, str)
            or not raw_f2.startswith("F2\n")
            or f2.get("text_sha256") != sha256_text(raw_f2)
            or f2.get("system_prompt_sha256") != guide_sha
        ):
            raise RescueAblationError(f"{task_id}: exact F2 binding is invalid")
        base_candidates = projected_row.get("base_candidates")
        if not isinstance(base_candidates, list) or len(base_candidates) != BASE_DRAWS:
            raise RescueAblationError(f"{task_id}: projected base group differs")
        details: list[dict[str, Any]] = []
        for index, candidate in enumerate(base_candidates):
            code = str(candidate.get("code") or "")
            details.append(
                _visible_detail(
                    code,
                    tests,
                    f"semantic-rescue-plan-{task_id}-{index}",
                    timeout=args.reward_timeout,
                    stability_runs=args.stability_runs,
                )
            )
        if any(detail["passed_tests"] > 0 for detail in details):
            excluded_nonzero += 1
            continue
        eligible_indices = [
            index
            for index, candidate in enumerate(base_candidates)
            if str(candidate.get("code") or "").strip()
        ]
        if len(eligible_indices) < SELECT_K:
            excluded_sparse += 1
            continue
        subset = [str(base_candidates[index]["code"]) for index in eligible_indices]
        selected_local = max_min_diverse_indices(subset, SELECT_K)
        selected_indices = [eligible_indices[index] for index in selected_local]
        parents: list[dict[str, Any]] = []
        for rank, base_index in enumerate(selected_indices):
            candidate = base_candidates[base_index]
            detail = details[base_index]
            parents.append(
                {
                    "parent_rank": rank,
                    "base_candidate_index": base_index,
                    "candidate": str(candidate["code"]),
                    "candidate_sha256": str(candidate["code_sha256"]),
                    "visible_detail": detail,
                }
            )
        catalog = build_grounding_catalog(
            raw_f2,
            [parent["candidate"] for parent in parents],
            diagnostics=[
                parent["visible_detail"]["diagnostic"] for parent in parents
            ],
        )
        groups.append(
            {
                "task_id": task_id,
                "task_rank_sha256": _stable_task_rank(task_id, args.seed),
                "raw_f2": raw_f2,
                "raw_f2_sha256": sha256_text(raw_f2),
                "f2_row_sha256": canonical_sha256(dict(f2)),
                "source_format_guide": guide,
                "source_format_guide_sha256": guide_sha,
                "encoder_source": encoder_source,
                "encoder_source_sha256": sha256_text(encoder_source),
                "visible_tests": tests,
                "visible_tests_sha256": sha256_text(tests),
                "split_binding_sha256": split_binding,
                "parents": parents,
                "grounding_catalog_sha256": catalog.catalog_sha256,
            }
        )
    if len(groups) != args.groups:
        raise RescueAblationError(
            f"only {len(groups)} all-zero groups found; required {args.groups}"
        )
    body = {
        "schema": PLAN_SCHEMA,
        "status": "complete",
        "runtime_provenance": _runtime_provenance(),
        "base_projection_sha256": projection_sha,
        "checkpoint": dict(projection["checkpoint"]),
        "inputs": {
            "projection": _file_record(args.projection),
            "rollout": _file_record(args.rollout_file),
            "f2": _file_record(args.f2_jsonl),
            "f2_manifest": _file_record(args.f2_manifest),
            "public_manifest": _file_record(args.public_manifest),
        },
        "policy": {
            "groups": args.groups,
            "base_draws_per_task": BASE_DRAWS,
            "eligibility": "every base draw passes zero isolated visible cases",
            "task_order": "task-bound SHA-256 rank",
            "seed": args.seed,
            "parents_per_group": SELECT_K,
            "parent_selection": "normalized-token-trigram greedy max-min",
            "repairs_per_parent": REPAIRS_PER_PARENT,
            "slots_per_arm": SLOTS_PER_ARM,
            "arms": list(ARM_ORDER),
            "itt_missing_or_rejected_slots_are_failures": True,
        },
        "groups": groups,
        "funnel": {
            "projected_tasks": len(projected_by_id),
            "tasks_scanned_in_hash_order": scanned,
            "excluded_nonzero_visible": excluded_nonzero,
            "excluded_fewer_than_two_nonempty_parents": excluded_sparse,
            "selected_all_zero_groups": len(groups),
            "planned_student_slots": len(groups)
            * len(ARM_ORDER)
            * SLOTS_PER_ARM,
            "planned_judge_calls": len(groups),
        },
        "privacy": {
            "visible_train_tests_only": True,
            "base_projection_has_no_private_outcomes": True,
            "holdback_argument_exists": False,
            "final_175_opened": False,
        },
    }
    result = _seal(body, PLAN_HASH)
    require_exact_or_write(args.output, result)
    return result


def _assert_provider_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "source",
        "source_sha256",
        "source_format_guide",
        "tests",
        "candidates",
        "reference_catalog",
        "reference_catalog_sha256",
        "guidance_mode",
    }
    if set(payload) != expected:
        raise RescueAblationError("judge payload differs from visible whitelist")
    _assert_no_keys(payload, _PROVIDER_FORBIDDEN_KEYS, "judge payload")


def _catalog_for_group(group: Mapping[str, Any]) -> GroundingCatalog:
    parents = group.get("parents")
    if not isinstance(parents, list) or len(parents) != SELECT_K:
        raise RescueAblationError("plan parent group is malformed")
    catalog = build_grounding_catalog(
        str(group["raw_f2"]),
        [str(parent["candidate"]) for parent in parents],
        diagnostics=[
            str(parent["visible_detail"]["diagnostic"]) for parent in parents
        ],
    )
    if catalog.catalog_sha256 != group.get("grounding_catalog_sha256"):
        raise RescueAblationError("grounding catalogue changed after planning")
    return catalog


def _judge_payload(
    group: Mapping[str, Any], catalog: GroundingCatalog
) -> dict[str, Any]:
    payload = {
        "source": str(group["raw_f2"]),
        "source_sha256": str(group["raw_f2_sha256"]),
        "source_format_guide": str(group["source_format_guide"]),
        "tests": str(group["visible_tests"]),
        "candidates": [
            {
                "group_index": int(parent["base_candidate_index"]),
                "candidate": str(parent["candidate"]),
                "diagnostic": str(parent["visible_detail"]["diagnostic"]),
                "compiled": bool(parent["visible_detail"]["compiled"]),
                "full_pass": False,
            }
            for parent in group["parents"]
        ],
        "reference_catalog": catalog,
        "reference_catalog_sha256": catalog.catalog_sha256,
        "guidance_mode": "diagnosis_and_steps",
    }
    _assert_provider_payload(
        {**payload, "reference_catalog": catalog.to_prompt_dict()}
    )
    return payload


def _judge_contract(judge: Any, total_calls: int) -> dict[str, Any]:
    telemetry = judge.telemetry()
    result = {
        "model": judge.model,
        "base_url": judge.base_url,
        "api_style": judge.api_style,
        "max_tokens": judge.max_tokens,
        "timeout_seconds": judge.timeout_seconds,
        "max_retries": judge.max_retries,
        "completion_retries": judge.completion_retries,
        "retry_max_tokens": judge.retry_max_tokens,
        "thinking_mode": judge.thinking_mode,
        "reasoning_effort": judge.reasoning_effort,
        "reasoning_mode": judge.reasoning_mode,
        "chat_json_schema": judge.chat_json_schema,
        "fail_closed": judge.fail_closed,
        "total_calls": total_calls,
        "guidance_mode": "diagnosis_and_steps",
        "prompt_schema": telemetry.get("diagnose_prompt_schema_version"),
        "response_schema": telemetry.get("diagnose_response_schema"),
        "grounding_validator_schema": GROUNDING_SCHEMA,
        "one_call_per_group": True,
        "sdk_retries": 0,
        "completion_retries_allowed": 0,
    }
    if (
        result["max_retries"] != 0
        or result["completion_retries"] != 0
        or result["retry_max_tokens"] != result["max_tokens"]
        or result["fail_closed"] is not True
    ):
        raise RescueAblationError("judge violates the one-attempt contract")
    return result


def _validate_receipt_slice(
    value: Mapping[str, Any],
    *,
    expected_before: int,
    expected_previous: str,
) -> tuple[int, str, list[str]]:
    if value.get("schema") != RESPONSE_RECEIPT_ATTESTATION_SCHEMA:
        raise RescueAblationError("judge receipt slice schema is invalid")
    before = value.get("receipt_count_before_step")
    count = value.get("receipt_count_this_step")
    cumulative = value.get("cumulative_receipt_count")
    receipts = value.get("receipts")
    previous = str(value.get("previous_receipt_chain_sha256") or "")
    head = str(value.get("cumulative_receipt_chain_sha256") or "")
    if (
        before != expected_before
        or type(count) is not int
        or count < 0
        or cumulative != expected_before + count
        or previous != expected_previous
        or not _SHA_RE.fullmatch(head)
        or not isinstance(receipts, list)
        or len(receipts) != count
        or value.get("plaintext_prompts_persisted") is not False
        or value.get("plaintext_reasoning_persisted") is not False
    ):
        raise RescueAblationError("judge receipt slice counters differ")
    cursor = previous
    accepted_ids: list[str] = []
    for offset, receipt in enumerate(receipts, 1):
        if not isinstance(receipt, Mapping):
            raise RescueAblationError("judge receipt is not an object")
        body = dict(receipt)
        digest = str(body.pop("receipt_sha256", ""))
        if (
            receipt.get("schema") != RESPONSE_RECEIPT_SCHEMA
            or receipt.get("receipt_index") != expected_before + offset
            or receipt.get("previous_receipt_sha256") != cursor
            or digest != canonical_sha256(body)
            or receipt.get("plaintext_prompt_persisted") is not False
            or receipt.get("plaintext_reasoning_persisted") is not False
        ):
            raise RescueAblationError("judge receipt chain is invalid")
        validation = receipt.get("validation")
        response = receipt.get("response")
        if (
            isinstance(validation, Mapping)
            and validation.get("accepted") is True
            and validation.get("rejection_reasons") == []
            and isinstance(response, Mapping)
            and isinstance(response.get("id"), str)
            and response["id"]
        ):
            accepted_ids.append(sha256_text(str(response["id"])))
        cursor = digest
    if cursor != head:
        raise RescueAblationError("judge receipt slice head differs")
    return int(cumulative), head, accepted_ids


def _diagnosis_journal_state(
    path: Path,
    *,
    contract: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    events = load_journal(path)
    if not events:
        return {
            "events": [],
            "completed": [],
            "receipt_count": 0,
            "receipt_head": GENESIS_SHA256,
            "response_ids": [],
            "complete": False,
        }
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != DIAGNOSIS_JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise RescueAblationError("diagnosis journal header differs")
    cursor = 1
    completed: list[dict[str, Any]] = []
    receipt_count = 0
    receipt_head = GENESIS_SHA256
    response_ids: list[str] = []
    for group in groups:
        if cursor >= len(events) or events[cursor].get("event") == "complete":
            break
        started = events[cursor]
        task_id = str(group["task_id"])
        if (
            started.get("event") != "diagnosis_started"
            or started.get("schema") != DIAGNOSIS_JOURNAL_SCHEMA
            or started.get("task_id") != task_id
        ):
            raise RescueAblationError("diagnosis journal start order differs")
        cursor += 1
        if cursor >= len(events):
            raise RescueAblationError(
                f"{task_id}: ambiguous paid call has no terminal; refusing retry"
            )
        terminal = events[cursor]
        if (
            terminal.get("event") != "diagnosis_terminal"
            or terminal.get("schema") != DIAGNOSIS_JOURNAL_SCHEMA
            or terminal.get("task_id") != task_id
            or terminal.get("started_event_sha256")
            != started.get("journal_event_sha256")
            or terminal.get("result_sha256")
            != canonical_sha256(terminal.get("result"))
            or not isinstance(terminal.get("receipt_attestation"), Mapping)
        ):
            raise RescueAblationError("diagnosis terminal binding differs")
        receipt_count, receipt_head, ids = _validate_receipt_slice(
            terminal["receipt_attestation"],
            expected_before=receipt_count,
            expected_previous=receipt_head,
        )
        response_ids.extend(ids)
        if len(response_ids) != len(set(response_ids)):
            raise RescueAblationError("judge response ID repeats across resume")
        completed.append(terminal)
        cursor += 1
    complete = False
    if cursor < len(events):
        marker = events[cursor]
        if (
            marker.get("event") != "complete"
            or marker.get("schema") != DIAGNOSIS_JOURNAL_SCHEMA
            or len(completed) != len(groups)
            or marker.get("task_ids_sha256")
            != canonical_sha256([group["task_id"] for group in groups])
            or marker.get("result_sha256s_sha256")
            != canonical_sha256([row["result_sha256"] for row in completed])
        ):
            raise RescueAblationError("diagnosis completion marker differs")
        cursor += 1
        complete = True
    if cursor != len(events):
        raise RescueAblationError("diagnosis journal has an unexpected tail")
    return {
        "events": events,
        "completed": completed,
        "receipt_count": receipt_count,
        "receipt_head": receipt_head,
        "response_ids": response_ids,
        "complete": complete,
    }


def _failure_diagnosis(
    group: Mapping[str, Any], exc: Exception
) -> dict[str, Any]:
    return {
        "schema": DIAGNOSE_RESULT_SCHEMA,
        "guidance_mode": "diagnosis_and_steps",
        "validator_schema_version": DIAGNOSE_VALIDATOR_SCHEMA_VERSION,
        "diagnoses": [
            {
                "group_index": int(parent["base_candidate_index"]),
                "accepted": False,
                "rejection_reasons": ["judge_call_failure"],
                "fault_class": None,
                "edit_location": None,
                "evidence": [],
                "explanation": "",
                "repair_steps": [],
            }
            for parent in group["parents"]
        ],
        "failure_class": type(exc).__name__,
        "failure_message_sha256": sha256_text(str(exc)),
    }


def _validate_diagnosis_result(
    group: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    rows = result.get("diagnoses")
    if not isinstance(rows, list) or len(rows) != SELECT_K:
        raise RescueAblationError("judge diagnosis coverage differs")
    for parent, row in zip(group["parents"], rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("group_index") != parent["base_candidate_index"]
            or type(row.get("accepted")) is not bool
        ):
            raise RescueAblationError("judge diagnosis identity differs")


def _build_diagnosis_artifact(
    plan_value: Mapping[str, Any],
    plan_sha: str,
    contract: Mapping[str, Any],
    journal_path: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    for group, terminal in zip(
        plan_value["groups"], state["completed"], strict=True
    ):
        result = terminal["result"]
        _validate_diagnosis_result(group, result)
        for diagnosis in result["diagnoses"]:
            accepted += int(diagnosis["accepted"])
            rejected += int(not diagnosis["accepted"])
        rows.append(
            {
                "task_id": group["task_id"],
                "grounding_catalog_sha256": group[
                    "grounding_catalog_sha256"
                ],
                "result": result,
                "result_sha256": terminal["result_sha256"],
                "receipt_attestation_sha256": canonical_sha256(
                    terminal["receipt_attestation"]
                ),
            }
        )
    body = {
        "schema": DIAGNOSIS_SCHEMA,
        "status": "complete",
        "runtime_provenance": _runtime_provenance(),
        "source_plan_sha256": plan_sha,
        "judge_contract": dict(contract["judge"]),
        "rows": rows,
        "funnel": {
            "groups": len(rows),
            "calls": len(rows),
            "accepted_parent_diagnoses": accepted,
            "rejected_parent_diagnoses": rejected,
            "reserved_semantic_slots": len(rows) * SLOTS_PER_ARM,
            "generatable_semantic_slots": accepted * REPAIRS_PER_PARENT,
            "rejected_semantic_slots": rejected * REPAIRS_PER_PARENT,
        },
        "receipt_chain": {
            "count": state["receipt_count"],
            "head_sha256": state["receipt_head"],
            "unique_accepted_response_ids": len(state["response_ids"]),
            "plaintext_prompts_persisted": False,
            "plaintext_reasoning_persisted": False,
        },
        "journal": journal_record(journal_path),
        "privacy": {
            "provider_visible_tests_only": True,
            "provider_saw_holdback": False,
            "provider_saw_reference_dart": False,
            "holdback_argument_exists": False,
            "final_175_opened": False,
        },
    }
    return _seal(body, DIAGNOSIS_HASH)


def _construct_judge(
    args: argparse.Namespace,
    *,
    max_calls: int,
    receipt_chain_seed: str = GENESIS_SHA256,
    receipt_index_offset: int = 0,
    prior_response_id_sha256s: Sequence[str] = (),
    connect: bool,
) -> Any:
    # Construction is intentionally connection-free; ``connect`` documents
    # whether the caller is making a contract probe or will subsequently call
    # the provider.  VerpoJudge opens its SDK client lazily.
    _ = connect
    base_url = args.base_url or (
        "https://api.anthropic.com"
        if args.api_style == "anthropic_messages"
        else None
    )
    return VerpoJudge(
        model=args.model or None,
        base_url=base_url,
        api_style=args.api_style or None,
        concurrency=1,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=0,
        completion_retries=0,
        retry_max_tokens=args.max_tokens,
        thinking_mode=args.thinking_mode or None,
        reasoning_effort=args.reasoning_effort or None,
        reasoning_mode=args.reasoning_mode or None,
        chat_json_schema=args.chat_json_schema,
        max_calls=max_calls,
        fail_closed=True,
        receipt_chain_seed=receipt_chain_seed,
        receipt_index_offset=receipt_index_offset,
        prior_response_id_sha256s=list(prior_response_id_sha256s),
    )


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    plan_value = _read_json(args.plan, "semantic rescue plan")
    plan_sha = _require_seal(
        plan_value, schema=PLAN_SCHEMA, field=PLAN_HASH, label="rescue plan"
    )
    groups = plan_value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RescueAblationError("rescue plan has no groups")
    probe = _construct_judge(
        args,
        max_calls=len(groups),
        connect=False,
    )
    judge_contract = _judge_contract(probe, len(groups))
    contract = {
        "schema": DIAGNOSIS_JOURNAL_SCHEMA,
        "source_plan_sha256": plan_sha,
        "task_ids_sha256": canonical_sha256(
            [group["task_id"] for group in groups]
        ),
        "judge": judge_contract,
        "runtime_code_bundle_sha256": _runtime_provenance()[
            "code_bundle_sha256"
        ],
    }
    journal_path = Path(args.journal).expanduser().resolve()
    state = _diagnosis_journal_state(
        journal_path, contract=contract, groups=groups
    )
    if not state["events"]:
        append_event(
            journal_path,
            {
                "event": "header",
                "schema": DIAGNOSIS_JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        state = _diagnosis_journal_state(
            journal_path, contract=contract, groups=groups
        )
    if state["complete"]:
        result = _build_diagnosis_artifact(
            plan_value, plan_sha, contract, journal_path, state
        )
        require_exact_or_write(args.output, result)
        return result
    remaining = len(groups) - len(state["completed"])
    judge = _construct_judge(
        args,
        max_calls=remaining,
        receipt_chain_seed=state["receipt_head"],
        receipt_index_offset=state["receipt_count"],
        prior_response_id_sha256s=state["response_ids"],
        connect=True,
    )
    for group in groups[len(state["completed"]) :]:
        catalog = _catalog_for_group(group)
        payload = _judge_payload(group, catalog)
        safe_payload = {
            **payload,
            "reference_catalog": catalog.to_prompt_dict(),
        }
        started = append_event(
            journal_path,
            {
                "event": "diagnosis_started",
                "schema": DIAGNOSIS_JOURNAL_SCHEMA,
                "task_id": group["task_id"],
                "provider_payload_sha256": canonical_sha256(safe_payload),
                "grounding_catalog_sha256": catalog.catalog_sha256,
            },
        )
        cursor = state["receipt_count"]
        try:
            result = judge.diagnose_group(
                payload,
                guidance_mode="diagnosis_and_steps",
                item_validator=validate_diagnosis_item,
                validator_schema_version=GROUNDING_SCHEMA,
            )
            if not isinstance(result, Mapping):
                raise RescueAblationError("judge returned a non-object")
            result = dict(result)
        except Exception as exc:
            result = _failure_diagnosis(group, exc)
        _validate_diagnosis_result(group, result)
        receipt = judge.receipt_attestation_since(cursor)
        append_event(
            journal_path,
            {
                "event": "diagnosis_terminal",
                "schema": DIAGNOSIS_JOURNAL_SCHEMA,
                "task_id": group["task_id"],
                "started_event_sha256": started["journal_event_sha256"],
                "result": result,
                "result_sha256": canonical_sha256(result),
                "receipt_attestation": receipt,
            },
        )
        state = _diagnosis_journal_state(
            journal_path, contract=contract, groups=groups
        )
    append_event(
        journal_path,
        {
            "event": "complete",
            "schema": DIAGNOSIS_JOURNAL_SCHEMA,
            "task_ids_sha256": canonical_sha256(
                [group["task_id"] for group in groups]
            ),
            "result_sha256s_sha256": canonical_sha256(
                [row["result_sha256"] for row in state["completed"]]
            ),
        },
    )
    state = _diagnosis_journal_state(
        journal_path, contract=contract, groups=groups
    )
    if not state["complete"]:
        raise RescueAblationError("diagnosis journal did not complete")
    result = _build_diagnosis_artifact(
        plan_value, plan_sha, contract, journal_path, state
    )
    require_exact_or_write(args.output, result)
    return result


def _diagnosis_lookup(
    diagnoses: Mapping[str, Any],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    rows = diagnoses.get("rows")
    if not isinstance(rows, list):
        raise RescueAblationError("diagnosis artifact rows are invalid")
    result: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        task_id = _task_id(row, "diagnosis row")
        raw_result = row.get("result")
        raw_diagnoses = (
            raw_result.get("diagnoses")
            if isinstance(raw_result, Mapping)
            else None
        )
        if not isinstance(raw_diagnoses, list):
            raise RescueAblationError(f"{task_id}: diagnosis list is invalid")
        lookup: dict[int, Mapping[str, Any]] = {}
        for diagnosis in raw_diagnoses:
            if not isinstance(diagnosis, Mapping):
                raise RescueAblationError(f"{task_id}: diagnosis is malformed")
            index = diagnosis.get("group_index")
            if type(index) is not int or index in lookup:
                raise RescueAblationError(f"{task_id}: diagnosis identity repeats")
            lookup[index] = diagnosis
        result[task_id] = lookup
    return result


def _conditioning_text(
    *,
    arm: str,
    group: Mapping[str, Any],
    parent: Mapping[str, Any],
    diagnosis: Mapping[str, Any] | None,
) -> str:
    if arm == "plain_resample":
        return ""
    context: dict[str, Any] = {
        "schema": CONDITIONING_SCHEMA,
        "arm": arm,
        "task_id": group["task_id"],
        "source_sha256": group["encoder_source_sha256"],
        "base_candidate": parent["candidate"],
        "base_candidate_sha256": parent["candidate_sha256"],
        "base_candidate_compiled": parent["visible_detail"]["compiled"],
        "visible_execution_diagnostic": parent["visible_detail"]["diagnostic"],
        "visible_execution_diagnostic_sha256": parent["visible_detail"][
            "diagnostic_sha256"
        ],
        "instruction": (
            "Return only a complete corrected Dart compilation unit. "
            "Do not return prose or Markdown fences."
        ),
    }
    if arm == "semantic_judge":
        if not isinstance(diagnosis, Mapping) or diagnosis.get("accepted") is not True:
            raise RescueAblationError("cannot condition on a rejected diagnosis")
        context["judge_feedback"] = {
            "fault_class": diagnosis.get("fault_class"),
            "edit_location": diagnosis.get("edit_location"),
            "evidence": diagnosis.get("evidence"),
            "explanation": diagnosis.get("explanation"),
            "repair_steps": diagnosis.get("repair_steps"),
        }
    return (
        "\n\nSEMANTIC_RESCUE_CONTEXT_JSON\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _slot_seed(
    *,
    base_seed: int,
    plan_sha: str,
    task_id: str,
    parent_rank: int,
    repair_rank: int,
) -> int:
    digest = canonical_sha256(
        {
            "schema": "t5gemma2-semantic-rescue-slot-seed-v1",
            "base_seed": base_seed,
            "source_plan_sha256": plan_sha,
            "task_id": task_id,
            "parent_rank": parent_rank,
            "repair_rank": repair_rank,
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)


def _materialize_slots(
    plan_value: Mapping[str, Any],
    plan_sha: str,
    diagnoses: Mapping[str, Any],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = _diagnosis_lookup(diagnoses)
    slots: list[dict[str, Any]] = []
    for group in plan_value["groups"]:
        task_id = str(group["task_id"])
        task_diagnoses = lookup.get(task_id)
        if task_diagnoses is None:
            raise RescueAblationError(f"{task_id}: diagnosis row is missing")
        for arm in ARM_ORDER:
            for parent_rank, parent in enumerate(group["parents"]):
                diagnosis = task_diagnoses.get(parent["base_candidate_index"])
                for repair_rank in range(REPAIRS_PER_PARENT):
                    generate = not (
                        arm == "semantic_judge"
                        and (
                            diagnosis is None
                            or diagnosis.get("accepted") is not True
                        )
                    )
                    rejection_reasons: list[str] = []
                    if not generate:
                        raw = (
                            diagnosis.get("rejection_reasons")
                            if isinstance(diagnosis, Mapping)
                            else ["diagnosis_missing"]
                        )
                        rejection_reasons = sorted(
                            {str(item) for item in raw if str(item)}
                        ) or ["diagnosis_rejected"]
                    conditioning = (
                        _conditioning_text(
                            arm=arm,
                            group=group,
                            parent=parent,
                            diagnosis=diagnosis,
                        )
                        if generate
                        else ""
                    )
                    source = str(group["encoder_source"]) + conditioning
                    slot_id = (
                        f"{task_id}:{arm}:p{parent_rank}:r{repair_rank}"
                    )
                    slots.append(
                        {
                            "slot_id": slot_id,
                            "task_id": task_id,
                            "arm": arm,
                            "parent_rank": parent_rank,
                            "base_candidate_index": parent[
                                "base_candidate_index"
                            ],
                            "base_candidate_sha256": parent[
                                "candidate_sha256"
                            ],
                            "repair_rank": repair_rank,
                            "generate": generate,
                            "rejection_reasons": rejection_reasons,
                            "conditioning": conditioning,
                            "conditioning_sha256": sha256_text(conditioning),
                            "source": source,
                            "source_sha256": sha256_text(source),
                            "seed": _slot_seed(
                                base_seed=seed,
                                plan_sha=plan_sha,
                                task_id=task_id,
                                parent_rank=parent_rank,
                                repair_rank=repair_rank,
                            ),
                        }
                    )
    return slots


def _slot_public_binding(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: slot[key]
        for key in (
            "slot_id",
            "task_id",
            "arm",
            "parent_rank",
            "base_candidate_index",
            "base_candidate_sha256",
            "repair_rank",
            "generate",
            "rejection_reasons",
            "conditioning_sha256",
            "source_sha256",
            "seed",
        )
    }


def _generation_journal_state(
    path: Path,
    *,
    contract: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    events = load_journal(path)
    if not events:
        return {"events": [], "terminals": [], "complete": False}
    header = events[0]
    preflight = header.get("source_preflight")
    if (
        header.get("event") != "header"
        or header.get("schema") != GENERATION_JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
        or not isinstance(preflight, Mapping)
        or preflight.get("slot_sources_checked") != len(slots)
        or preflight.get("unique_sources_checked")
        != len({str(slot["source_sha256"]) for slot in slots})
        or type(preflight.get("max_observed_source_tokens")) is not int
        or preflight["max_observed_source_tokens"] <= 0
        or preflight["max_observed_source_tokens"]
        > int(contract["sampling"]["max_source_tokens"])
        or not _SHA_RE.fullmatch(
            str(preflight.get("source_token_counts_sha256") or "")
        )
        or preflight.get("truncation_used") is not False
    ):
        raise RescueAblationError("generation journal header differs")
    terminals: list[dict[str, Any]] = []
    cursor = 1
    for slot in slots:
        if cursor >= len(events) or events[cursor].get("event") == "complete":
            break
        event = events[cursor]
        expected_binding = _slot_public_binding(slot)
        if (
            event.get("event") != "slot_terminal"
            or event.get("schema") != GENERATION_JOURNAL_SCHEMA
            or event.get("slot_id") != slot["slot_id"]
            or event.get("slot_binding_sha256")
            != canonical_sha256(expected_binding)
            or event.get("status")
            not in {"generated", "diagnosis_rejected"}
        ):
            raise RescueAblationError("generation terminal order/binding differs")
        if slot["generate"]:
            candidate = event.get("candidate")
            if (
                event.get("status") != "generated"
                or not isinstance(candidate, Mapping)
                or candidate.get("text_sha256")
                != sha256_text(str(candidate.get("text") or ""))
                or candidate.get("seed") != slot["seed"]
            ):
                raise RescueAblationError("generated slot receipt differs")
        elif (
            event.get("status") != "diagnosis_rejected"
            or event.get("candidate") is not None
        ):
            raise RescueAblationError("rejected ITT slot was generated")
        terminals.append(event)
        cursor += 1
    complete = False
    if cursor < len(events):
        marker = events[cursor]
        if (
            marker.get("event") != "complete"
            or marker.get("schema") != GENERATION_JOURNAL_SCHEMA
            or len(terminals) != len(slots)
            or marker.get("slot_ids_sha256")
            != canonical_sha256([slot["slot_id"] for slot in slots])
            or marker.get("terminal_event_sha256s_sha256")
            != canonical_sha256(
                [row["journal_event_sha256"] for row in terminals]
            )
        ):
            raise RescueAblationError("generation completion marker differs")
        cursor += 1
        complete = True
    if cursor != len(events):
        raise RescueAblationError("generation journal has an unexpected tail")
    return {"events": events, "terminals": terminals, "complete": complete}


def _build_generation_artifact(
    *,
    plan_sha: str,
    diagnoses_sha: str,
    contract: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    journal_path: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts = Counter()
    by_arm: dict[str, Counter[str]] = {
        arm: Counter() for arm in ARM_ORDER
    }
    for slot, terminal in zip(slots, state["terminals"], strict=True):
        status = str(terminal["status"])
        counts[status] += 1
        by_arm[str(slot["arm"])]["planned"] += 1
        by_arm[str(slot["arm"])][status] += 1
        records.append(
            {
                **_slot_public_binding(slot),
                "status": status,
                "candidate": terminal.get("candidate"),
                "terminal_event_sha256": terminal[
                    "journal_event_sha256"
                ],
            }
        )
    body = {
        "schema": GENERATION_SCHEMA,
        "status": "complete",
        "runtime_provenance": _runtime_provenance(),
        "source_plan_sha256": plan_sha,
        "source_diagnoses_sha256": diagnoses_sha,
        "run_contract": dict(contract),
        "run_contract_sha256": canonical_sha256(contract),
        "records": records,
        "accounting": {
            "planned_slots": len(slots),
            "generated_slots": counts["generated"],
            "diagnosis_rejected_slots": counts["diagnosis_rejected"],
            "by_arm": {
                arm: dict(by_arm[arm]) for arm in ARM_ORDER
            },
            "itt_denominator_includes_rejected_slots": True,
        },
        "journal": journal_record(journal_path),
        "privacy": {
            "holdback_opened": False,
            "reference_dart_opened": False,
            "final_175_opened": False,
        },
    }
    return _seal(body, GENERATION_HASH)


def _cached_runtime_generator(
    *,
    model: Any,
    tokenizer: Any,
    max_source_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Callable[[str, int], dict[str, Any]]:
    """Generate one independently seeded slot with a one-source encoder cache.

    Slots are sealed in task/arm/parent/repair order, so this reduces a normal
    task from twelve encoder passes to five without coupling decoder sampling:
    each slot still calls ``generate_candidate_batch`` with its own sealed
    arm-independent seed and ``count=1``.
    """

    decoder_start, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
    device = torch.device("cuda")
    cached_source: str | None = None
    cached_source_sha256: str | None = None
    cached_input_ids: torch.Tensor | None = None
    cached_attention_mask: torch.Tensor | None = None
    cached_encoder_outputs: Any = None

    def generate_one(source: str, seed: int) -> dict[str, Any]:
        nonlocal cached_source
        nonlocal cached_source_sha256
        nonlocal cached_input_ids
        nonlocal cached_attention_mask
        nonlocal cached_encoder_outputs

        source_sha = sha256_text(source)
        if cached_source_sha256 != source_sha:
            input_ids, attention_mask = _encode_source(
                tokenizer,
                source,
                max_source_tokens=max_source_tokens,
                device=device,
            )
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            cached_source = source
            cached_source_sha256 = source_sha
            cached_input_ids = input_ids
            cached_attention_mask = attention_mask
            cached_encoder_outputs = encoder_outputs
        elif cached_source != source:
            raise RescueAblationError("encoder-source SHA-256 collision")
        if (
            cached_input_ids is None
            or cached_attention_mask is None
            or cached_encoder_outputs is None
        ):
            raise RescueAblationError("encoder cache is unexpectedly empty")
        batch = generate_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            encoder_outputs=cached_encoder_outputs,
            attention_mask=cached_attention_mask,
            decoder_start=decoder_start,
            pad_id=pad_id,
            eos_ids=eos_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            count=1,
        )
        if len(batch) != 1:
            raise RescueAblationError("native generator returned wrong count")
        return {
            **batch[0],
            "group_sample_index": 0,
            "encoder_tokens": int(cached_input_ids.size(1)),
        }

    return generate_one


def _preflight_slot_sources(
    tokenizer: Any,
    slots: Sequence[Mapping[str, Any]],
    *,
    max_source_tokens: int,
) -> dict[str, Any]:
    """Tokenize every distinct fixed source before a journal can be created."""

    unique: dict[str, str] = {}
    for slot in slots:
        source = str(slot["source"])
        source_sha = str(slot["source_sha256"])
        if source_sha != sha256_text(source):
            raise RescueAblationError("slot source digest differs")
        prior = unique.setdefault(source_sha, source)
        if prior != source:
            raise RescueAblationError("slot source SHA-256 collision")
    token_counts: list[dict[str, Any]] = []
    for source_sha, source in sorted(unique.items()):
        input_ids, _attention_mask = _encode_source(
            tokenizer,
            source,
            max_source_tokens=max_source_tokens,
            device=torch.device("cpu"),
        )
        token_counts.append(
            {
                "source_sha256": source_sha,
                "tokens": int(input_ids.size(1)),
            }
        )
    if not token_counts:
        raise RescueAblationError("generation has no slot sources")
    return {
        "slot_sources_checked": len(slots),
        "unique_sources_checked": len(token_counts),
        "max_observed_source_tokens": max(
            int(row["tokens"]) for row in token_counts
        ),
        "source_token_counts_sha256": canonical_sha256(token_counts),
        "truncation_used": False,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("native T5 semantic rescue generation requires CUDA")
    plan_value = _read_json(args.plan, "semantic rescue plan")
    plan_sha = _require_seal(
        plan_value, schema=PLAN_SCHEMA, field=PLAN_HASH, label="rescue plan"
    )
    diagnoses = _read_json(args.diagnoses, "semantic diagnoses")
    diagnoses_sha = _require_seal(
        diagnoses,
        schema=DIAGNOSIS_SCHEMA,
        field=DIAGNOSIS_HASH,
        label="semantic diagnoses",
    )
    if diagnoses.get("source_plan_sha256") != plan_sha:
        raise RescueAblationError("diagnoses belong to another plan")
    slots = _materialize_slots(
        plan_value, plan_sha, diagnoses, seed=args.seed
    )
    expected_slots = len(plan_value["groups"]) * len(ARM_ORDER) * SLOTS_PER_ARM
    if len(slots) != expected_slots:
        raise RescueAblationError("materialized ITT slot count differs")
    checkpoint_path = Path(args.sft_checkpoint).expanduser().resolve()
    checkpoint_contract, checkpoint_record = _checkpoint_record(
        checkpoint_path, "sft"
    )
    if checkpoint_record != plan_value.get("checkpoint"):
        raise RescueAblationError("generation checkpoint differs from plan")
    contract = {
        "schema": GENERATION_JOURNAL_SCHEMA,
        "source_plan_sha256": plan_sha,
        "source_diagnoses_sha256": diagnoses_sha,
        "checkpoint": checkpoint_record,
        "checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
        "slot_manifest_sha256": canonical_sha256(
            [_slot_public_binding(slot) for slot in slots]
        ),
        "sampling": {
            "seed": args.seed,
            "seed_policy": (
                "sha256(base_seed,plan,task,parent_rank,repair_rank); "
                "arm-independent"
            ),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_source_tokens": args.max_source_tokens,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": 1,
            "bf16": args.bf16,
            "attn_implementation": args.attn_implementation,
            "checkpoint_frozen": True,
            "optimizer_constructed": False,
        },
        "arms": list(ARM_ORDER),
        "slots_per_arm_task": SLOTS_PER_ARM,
        "itt_rejected_diagnoses_reserved_as_failures": True,
        "source_preflight_policy": {
            "all_fixed_slot_sources_before_header": True,
            "tokenizer_truncation_forbidden": True,
        },
        "runtime_code_bundle_sha256": _runtime_provenance()[
            "code_bundle_sha256"
        ],
    }
    journal_path = Path(args.journal).expanduser().resolve()
    state = _generation_journal_state(
        journal_path, contract=contract, slots=slots
    )
    if state["complete"]:
        result = _build_generation_artifact(
            plan_sha=plan_sha,
            diagnoses_sha=diagnoses_sha,
            contract=contract,
            slots=slots,
            state=state,
            journal_path=journal_path,
        )
        require_exact_or_write(args.output, result)
        return result
    model, tokenizer, loaded_record = load_policy(
        checkpoint=checkpoint_path,
        arm="sft",
        bf16=args.bf16,
        attn_implementation=args.attn_implementation,
    )
    if loaded_record != checkpoint_record:
        raise RescueAblationError("loaded generation checkpoint differs")
    source_preflight = _preflight_slot_sources(
        tokenizer,
        slots,
        max_source_tokens=args.max_source_tokens,
    )
    if not state["events"]:
        append_event(
            journal_path,
            {
                "event": "header",
                "schema": GENERATION_JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
                "source_preflight": source_preflight,
            },
        )
        state = _generation_journal_state(
            journal_path, contract=contract, slots=slots
        )
    elif state["events"][0].get("source_preflight") != source_preflight:
        raise RescueAblationError(
            "resumed generation source-token preflight differs"
        )
    runtime_generate = _cached_runtime_generator(
        model=model,
        tokenizer=tokenizer,
        max_source_tokens=args.max_source_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    for slot in slots[len(state["terminals"]) :]:
        binding = _slot_public_binding(slot)
        if slot["generate"]:
            raw = dict(
                runtime_generate(str(slot["source"]), int(slot["seed"]))
            )
            text = str(raw.pop("text", "")).strip()
            candidate = {
                **raw,
                "text": text,
                "text_sha256": sha256_text(text),
            }
            status = "generated"
        else:
            candidate = None
            status = "diagnosis_rejected"
        append_event(
            journal_path,
            {
                "event": "slot_terminal",
                "schema": GENERATION_JOURNAL_SCHEMA,
                "slot_id": slot["slot_id"],
                "slot_binding_sha256": canonical_sha256(binding),
                "status": status,
                "candidate": candidate,
            },
        )
        state = _generation_journal_state(
            journal_path, contract=contract, slots=slots
        )
        print(
            json.dumps(
                {
                    "completed_slots": len(state["terminals"]),
                    "planned_slots": len(slots),
                    "slot_id": slot["slot_id"],
                    "status": status,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    append_event(
        journal_path,
        {
            "event": "complete",
            "schema": GENERATION_JOURNAL_SCHEMA,
            "slot_ids_sha256": canonical_sha256(
                [slot["slot_id"] for slot in slots]
            ),
            "terminal_event_sha256s_sha256": canonical_sha256(
                [
                    row["journal_event_sha256"]
                    for row in state["terminals"]
                ]
            ),
        },
    )
    state = _generation_journal_state(
        journal_path, contract=contract, slots=slots
    )
    if not state["complete"]:
        raise RescueAblationError("generation journal did not complete")
    result = _build_generation_artifact(
        plan_sha=plan_sha,
        diagnoses_sha=diagnoses_sha,
        contract=contract,
        slots=slots,
        state=state,
        journal_path=journal_path,
    )
    require_exact_or_write(args.output, result)
    return result


def _score_many(
    jobs: Sequence[tuple[str, str, str]],
    *,
    timeout: int,
    stability_runs: int,
    workers: int,
) -> list[dict[str, Any]]:
    def one(job: tuple[str, str, str]) -> dict[str, Any]:
        code, tests, identity = job
        return _visible_detail(
            code,
            tests,
            identity,
            timeout=timeout,
            stability_runs=stability_runs,
        )

    if workers <= 1:
        return [one(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs) or 1)) as pool:
        return list(pool.map(one, jobs))


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    detail = row["visible_detail"]
    return (
        -int(detail["passed_tests"]),
        -int(detail["full_pass"]),
        -int(detail["compiled"]),
        str(row["candidate_sha256"]),
        str(row["slot_id"]),
    )


def _visible_selections(
    scored_records: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in scored_records:
        by_cell.setdefault(
            (str(row.get("task_id") or ""), str(row.get("arm") or "")), []
        ).append(row)
    selections: list[dict[str, Any]] = []
    for group in groups:
        task_id = str(group["task_id"])
        for arm in ARM_ORDER:
            rows = by_cell.get((task_id, arm), [])
            if len(rows) != SLOTS_PER_ARM:
                raise RescueAblationError(
                    f"{task_id}/{arm}: visible ITT slot coverage differs"
                )
            generated: list[Mapping[str, Any]] = []
            for row in rows:
                status = row.get("status")
                detail = row.get("visible_detail")
                if status == "generated":
                    candidate = row.get("candidate")
                    if (
                        not isinstance(candidate, Mapping)
                        or not isinstance(detail, Mapping)
                        or row.get("candidate_sha256")
                        != sha256_text(str(candidate.get("text") or ""))
                        or detail.get("diagnostic_sha256")
                        != sha256_text(str(detail.get("diagnostic") or ""))
                        or detail.get("passed_tests")
                        != sum(detail.get("test_passes") or [])
                        or detail.get("total_tests")
                        != len(detail.get("test_passes") or [])
                    ):
                        raise RescueAblationError(
                            f"{task_id}/{arm}: visible score is malformed"
                        )
                    generated.append(row)
                elif status == "diagnosis_rejected":
                    if detail is not None or row.get("candidate") is not None:
                        raise RescueAblationError(
                            f"{task_id}/{arm}: rejected slot has a score"
                        )
                else:
                    raise RescueAblationError(
                        f"{task_id}/{arm}: unknown visible slot status"
                    )
            selected = min(generated, key=_selection_key) if generated else None
            selections.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "generated_slots": len(generated),
                    "selected_slot_id": (
                        None if selected is None else selected["slot_id"]
                    ),
                    "selected_candidate_sha256": (
                        None
                        if selected is None
                        else selected["candidate_sha256"]
                    ),
                    "selection_rule": (
                        "max visible passed cases, then visible full/compile, "
                        "then candidate SHA-256 and sealed slot ID"
                    ),
                }
            )
    return selections


def _visible_artifact_records(
    visible: Mapping[str, Any],
    *,
    plan_sha: str,
    generation_sha: str,
    source_records: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    scoring_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    visible_sha = _require_seal(
        visible,
        schema=VISIBLE_SCORE_SCHEMA,
        field=VISIBLE_SCORE_HASH,
        label="visible selection",
    )
    records = visible.get("records")
    selections = visible.get("selections")
    if (
        visible.get("source_plan_sha256") != plan_sha
        or visible.get("source_generation_sha256") != generation_sha
        or visible.get("scoring_contract") != scoring_contract
        or visible.get("holdback_evaluation_contract")
        != {
            "baseline": (
                "maximum private passed-case count across the same two "
                "frozen parents; common to every arm"
            ),
            "improvement": "selected repair passed cases strictly exceed baseline",
            "holdback_does_not_choose_repair": True,
        }
        or not isinstance(records, list)
        or not isinstance(selections, list)
        or len(records) != len(source_records)
    ):
        raise RescueAblationError("visible selection binding differs")
    normalized_records = [dict(row) for row in records]
    for source, scored in zip(source_records, normalized_records, strict=True):
        base = dict(scored)
        base.pop("candidate_sha256", None)
        base.pop("visible_detail", None)
        if base != source:
            raise RescueAblationError(
                "visible record differs from sealed generation"
            )
    expected_selections = _visible_selections(normalized_records, groups)
    if selections != expected_selections:
        raise RescueAblationError("visible selections are not reproducible")
    return (
        normalized_records,
        [dict(row) for row in selections],
        visible_sha,
    )


def _private_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compiled": bool(detail["compiled"]),
        "full_pass": bool(detail["full_pass"]),
        "passed_tests": int(detail["passed_tests"]),
        "total_tests": int(detail["total_tests"]),
        "diagnostic_persisted": False,
    }


def _exact_mcnemar(
    rows: Sequence[Mapping[str, Any]],
    *,
    control: str,
    treatment: str,
    field: str,
) -> dict[str, Any]:
    by_task: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), {})[str(row["arm"])] = bool(
            row[field]
        )
    control_only = treatment_only = both = neither = 0
    for task_id, values in by_task.items():
        if control not in values or treatment not in values:
            raise RescueAblationError(
                f"{task_id}: paired contrast arm coverage differs"
            )
        left, right = values[control], values[treatment]
        if left and right:
            both += 1
        elif left:
            control_only += 1
        elif right:
            treatment_only += 1
        else:
            neither += 1
    discordant = control_only + treatment_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(control_only, treatment_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    total = len(by_task)
    return {
        "control": control,
        "treatment": treatment,
        "outcome": field,
        "tasks": total,
        "control_only": control_only,
        "treatment_only": treatment_only,
        "both": both,
        "neither": neither,
        "risk_difference": (
            (treatment_only - control_only) / total if total else 0.0
        ),
        "two_sided_exact_mcnemar_p": p_value,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    plan_value = _read_json(args.plan, "semantic rescue plan")
    plan_sha = _require_seal(
        plan_value, schema=PLAN_SCHEMA, field=PLAN_HASH, label="rescue plan"
    )
    generation = _read_json(args.generation, "semantic rescue generation")
    generation_sha = _require_seal(
        generation,
        schema=GENERATION_SCHEMA,
        field=GENERATION_HASH,
        label="semantic generation",
    )
    if generation.get("source_plan_sha256") != plan_sha:
        raise RescueAblationError("generation belongs to another plan")
    records = generation.get("records")
    if not isinstance(records, list):
        raise RescueAblationError("generation records are invalid")
    groups = plan_value["groups"]
    group_by_id = {str(group["task_id"]): group for group in groups}
    expected = len(groups) * len(ARM_ORDER) * SLOTS_PER_ARM
    if len(records) != expected:
        raise RescueAblationError("generation ITT denominator differs")
    scoring_contract = {
        "native_scorer_sha256": _runtime_provenance()["code"][
            "native_scorer"
        ]["sha256"],
        "reward_timeout": args.reward_timeout,
        "stability_runs": args.stability_runs,
        "isolated_expect_case_scoring": True,
    }
    visible_path = Path(args.visible_output).expanduser().resolve()
    if not visible_path.exists():
        visible_jobs: list[tuple[str, str, str]] = []
        visible_coordinates: list[int] = []
        for index, record in enumerate(records):
            if record.get("status") != "generated":
                continue
            candidate = record.get("candidate")
            task_id = str(record.get("task_id") or "")
            if not isinstance(candidate, Mapping) or task_id not in group_by_id:
                raise RescueAblationError("generated score record is malformed")
            visible_jobs.append(
                (
                    str(candidate.get("text") or ""),
                    str(group_by_id[task_id]["visible_tests"]),
                    f"semantic-rescue-visible-{record['slot_id']}",
                )
            )
            visible_coordinates.append(index)
        visible_details = _score_many(
            visible_jobs,
            timeout=args.reward_timeout,
            stability_runs=args.stability_runs,
            workers=args.workers,
        )
        fresh_records: list[dict[str, Any]] = [
            dict(row) for row in records
        ]
        for index, detail in zip(
            visible_coordinates, visible_details, strict=True
        ):
            candidate = fresh_records[index]["candidate"]
            fresh_records[index]["candidate_sha256"] = sha256_text(
                str(candidate.get("text") or "")
            )
            fresh_records[index]["visible_detail"] = detail
        fresh_selections = _visible_selections(fresh_records, groups)
        visible_body = {
            "schema": VISIBLE_SCORE_SCHEMA,
            "status": "complete",
            "runtime_provenance": _runtime_provenance(),
            "source_plan_sha256": plan_sha,
            "source_generation_sha256": generation_sha,
            "scoring_contract": scoring_contract,
            "records": fresh_records,
            "selections": fresh_selections,
            "holdback_evaluation_contract": {
                "baseline": (
                    "maximum private passed-case count across the same two "
                    "frozen parents; common to every arm"
                ),
                "improvement": (
                    "selected repair passed cases strictly exceed baseline"
                ),
                "holdback_does_not_choose_repair": True,
            },
            "privacy": {
                "visible_only": True,
                "private_holdback_path_opened": False,
                "private_holdback_sha256_computed": False,
                "final_175_opened": False,
            },
        }
        require_exact_or_write(
            visible_path, _seal(visible_body, VISIBLE_SCORE_HASH)
        )
    # Consume the exact durable visible-only artifact.  The private path is not
    # even hashed until this self-seal, source binding, and selection replay
    # have all succeeded.
    visible_value = _read_json(visible_path, "visible selection")
    scored_records, visible_selections, visible_sha = (
        _visible_artifact_records(
            visible_value,
            plan_sha=plan_sha,
            generation_sha=generation_sha,
            source_records=records,
            groups=groups,
            scoring_contract=scoring_contract,
        )
    )
    selection_by_cell = {
        (str(row["task_id"]), str(row["arm"])): row
        for row in visible_selections
    }

    # Privacy boundary: no private file is opened or hashed until every
    # generated slot and every visible selection is durably sealed above.
    observed_holdback_sha = sha256_file(args.private_holdback)
    if observed_holdback_sha != args.expected_private_holdback_sha256:
        raise RescueAblationError("private holdback SHA-256 differs")
    holdback_rows = _read_jsonl(args.private_holdback, "private holdback")
    holdback_by_id = _index_unique(holdback_rows, "private holdback")
    if set(group_by_id) - set(holdback_by_id):
        raise RescueAblationError("selected task lacks private holdback")
    for task_id, row in holdback_by_id.items():
        if task_id not in group_by_id:
            continue
        tests = row.get("reward_holdback_tests")
        visible_tests = row.get("feedback_tests")
        group = group_by_id[task_id]
        if (
            not isinstance(tests, str)
            or not tests.strip()
            or visible_tests != group["visible_tests"]
            or _split_binding(row, task_id)
            != group["split_binding_sha256"]
            or len(extract_expect_spans(str(visible_tests)))
            != int(row["visible_count"])
            or len(extract_expect_spans(tests))
            != int(row["holdback_count"])
        ):
            raise RescueAblationError(f"{task_id}: private holdback is malformed")

    records_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored_records:
        records_by_cell.setdefault(
            (str(row["task_id"]), str(row["arm"])), []
        ).append(row)
    private_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def private_score(code: str, tests: str, identity: str) -> dict[str, Any]:
        key = (sha256_text(code), sha256_text(tests))
        if key not in private_cache:
            private_cache[key] = _visible_detail(
                code,
                tests,
                identity,
                timeout=args.reward_timeout,
                stability_runs=args.stability_runs,
            )
        return dict(private_cache[key])

    parent_private_by_task: dict[str, list[dict[str, Any]]] = {}
    common_baseline_by_task: dict[str, dict[str, Any]] = {}
    for group in groups:
        task_id = str(group["task_id"])
        private_tests = str(
            holdback_by_id[task_id]["reward_holdback_tests"]
        )
        parent_details = [
            private_score(
                str(parent["candidate"]),
                private_tests,
                f"semantic-rescue-private-common-{task_id}-p{rank}",
            )
            for rank, parent in enumerate(group["parents"])
        ]
        parent_private_by_task[task_id] = parent_details
        best_passed = max(
            int(detail["passed_tests"]) for detail in parent_details
        )
        best_ranks = [
            rank
            for rank, detail in enumerate(parent_details)
            if int(detail["passed_tests"]) == best_passed
        ]
        common_baseline_by_task[task_id] = {
            "rule": "max holdback passed cases across both frozen parents",
            "passed_tests": best_passed,
            "total_tests": int(parent_details[0]["total_tests"]),
            "parent_ranks_at_max": best_ranks,
            "any_parent_full_pass": any(
                bool(detail["full_pass"]) for detail in parent_details
            ),
            "diagnostic_persisted": False,
        }

    task_arm_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    transfer: dict[tuple[str, str], dict[str, Any]] = {}
    preferences: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm in ARM_ORDER:
        counts = Counter()
        cells = Counter()
        for group in groups:
            task_id = str(group["task_id"])
            rows = records_by_cell.get((task_id, arm), [])
            if len(rows) != SLOTS_PER_ARM:
                raise RescueAblationError(
                    f"{task_id}/{arm}: ITT slot coverage differs"
                )
            counts["planned_slots"] += len(rows)
            generated = [
                row
                for row in rows
                if row.get("status") == "generated"
                and isinstance(row.get("visible_detail"), Mapping)
            ]
            counts["generated_slots"] += len(generated)
            counts["missing_or_rejected_slots"] += len(rows) - len(generated)
            for row in generated:
                detail = row["visible_detail"]
                counts["visible_compiled_slots"] += int(detail["compiled"])
                counts["visible_any_pass_slots"] += int(
                    detail["passed_tests"] > 0
                )
                counts["visible_full_pass_slots"] += int(detail["full_pass"])
            sealed_selection = selection_by_cell.get((task_id, arm))
            if sealed_selection is None:
                raise RescueAblationError(
                    f"{task_id}/{arm}: sealed visible selection is missing"
                )
            selected_slot_id = sealed_selection["selected_slot_id"]
            selected = next(
                (
                    row
                    for row in generated
                    if row["slot_id"] == selected_slot_id
                ),
                None,
            )
            if (selected is None) != (selected_slot_id is None):
                raise RescueAblationError(
                    f"{task_id}/{arm}: sealed visible selection differs"
                )
            counts["groups"] += 1
            if selected is None:
                cells["no_effect"] += 1
                task_arm_rows.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "selected": None,
                        "visible_rescued": False,
                        "holdback_improved": False,
                        "genuine_rescue": False,
                        "cell": "no_effect",
                    }
                )
                continue
            visible = selected["visible_detail"]
            visible_up = int(visible["passed_tests"]) > 0
            parent = group["parents"][int(selected["parent_rank"])]
            private_tests = str(
                holdback_by_id[task_id]["reward_holdback_tests"]
            )
            repair_code = str(selected["candidate"]["text"])
            repair_private = private_score(
                repair_code,
                private_tests,
                f"semantic-rescue-private-{arm}-{task_id}-repair",
            )
            origin_parent_private = parent_private_by_task[task_id][
                int(selected["parent_rank"])
            ]
            common_baseline = common_baseline_by_task[task_id]
            holdback_delta = int(repair_private["passed_tests"]) - int(
                common_baseline["passed_tests"]
            )
            origin_parent_delta = int(
                repair_private["passed_tests"]
            ) - int(origin_parent_private["passed_tests"])
            holdback_up = holdback_delta > 0
            if visible_up and holdback_up:
                cell = "genuine_rescue"
            elif visible_up:
                cell = "visible_only_overfit"
            elif holdback_up:
                cell = "holdback_only_check"
            else:
                cell = "no_effect"
            cells[cell] += 1
            counts["groups_visible_rescued"] += int(visible_up)
            counts["groups_holdback_improved"] += int(holdback_up)
            counts["groups_genuine_rescue"] += int(visible_up and holdback_up)
            both_full = bool(visible["full_pass"] and repair_private["full_pass"])
            counts["groups_full_visible_and_holdback"] += int(both_full)
            selected_summary = {
                "slot_id": selected["slot_id"],
                "parent_rank": selected["parent_rank"],
                "repair_rank": selected["repair_rank"],
                "candidate_sha256": selected["candidate_sha256"],
                "selection_rule": (
                    "max visible passed cases, then visible full/compile, "
                    "then candidate SHA-256 and sealed slot ID"
                ),
                "visible_detail": {
                    key: visible[key]
                    for key in (
                        "compiled",
                        "full_pass",
                        "test_passes",
                        "passed_tests",
                        "total_tests",
                    )
                },
                "holdback_detail": _private_summary(repair_private),
                "common_baseline_holdback": dict(common_baseline),
                "origin_parent_holdback_detail": _private_summary(
                    origin_parent_private
                ),
                "visible_delta_passed_tests": int(visible["passed_tests"]),
                "holdback_delta_passed_tests": holdback_delta,
                "origin_parent_holdback_delta_passed_tests": (
                    origin_parent_delta
                ),
            }
            task_arm_rows.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "selected": selected_summary,
                    "visible_rescued": visible_up,
                    "holdback_improved": holdback_up,
                    "genuine_rescue": visible_up and holdback_up,
                    "cell": cell,
                }
            )
            code_sha = str(selected["candidate_sha256"])
            if both_full and holdback_up:
                key = (task_id, code_sha)
                contributor = {
                    "arm": arm,
                    "slot_id": selected["slot_id"],
                }
                if key not in transfer:
                    transfer[key] = {
                        "schema": TRANSFER_SCHEMA,
                        "task_id": task_id,
                        "code": repair_code,
                        "code_sha256": code_sha,
                        "target_mode": "final_dart_code_only",
                        "reasoning_in_target": False,
                        "student_checkpoint": plan_value["checkpoint"],
                        "source_plan_sha256": plan_sha,
                        "visible_full_pass": True,
                        "development_holdback_full_pass": True,
                        "common_baseline_holdback_improved": True,
                        "common_baseline_holdback_passed_tests": int(
                            common_baseline["passed_tests"]
                        ),
                        "development_holdback_tests_sha256": sha256_text(
                            private_tests
                        ),
                        "development_holdback_consumed_for_transfer_selection": True,
                        "provider_saw_development_holdback": False,
                        "final_175_holdout_touched": False,
                        "contributors": [contributor],
                    }
                else:
                    transfer[key]["contributors"].append(contributor)
            elif visible_up and holdback_up:
                base_code = str(parent["candidate"])
                base_sha = str(parent["candidate_sha256"])
                key = (task_id, code_sha, base_sha)
                preferences.setdefault(
                    key,
                    {
                        "schema": PREFERENCE_SCHEMA,
                        "task_id": task_id,
                        "chosen": repair_code,
                        "chosen_sha256": code_sha,
                        "rejected": base_code,
                        "rejected_sha256": base_sha,
                        "chosen_visible_passed_tests": int(
                            visible["passed_tests"]
                        ),
                        "rejected_visible_passed_tests": 0,
                        "chosen_holdback_delta_passed_tests": holdback_delta,
                        "off_policy": True,
                        "different_conditioning_prefixes": arm
                        != "plain_resample",
                        "eligible_for_on_policy_verpo_update": False,
                        "source_plan_sha256": plan_sha,
                    },
                )
        total_groups = counts["groups"]
        planned_slots = counts["planned_slots"]
        metrics[arm] = {
            **dict(counts),
            "visible_holdback_cells": {
                key: cells[key]
                for key in (
                    "genuine_rescue",
                    "visible_only_overfit",
                    "holdback_only_check",
                    "no_effect",
                )
            },
            "group_visible_rescue_rate_itt": (
                counts["groups_visible_rescued"] / total_groups
                if total_groups
                else 0.0
            ),
            "group_genuine_rescue_rate_itt": (
                counts["groups_genuine_rescue"] / total_groups
                if total_groups
                else 0.0
            ),
            "slot_visible_any_pass_rate_itt": (
                counts["visible_any_pass_slots"] / planned_slots
                if planned_slots
                else 0.0
            ),
            "slot_visible_full_pass_rate_itt": (
                counts["visible_full_pass_slots"] / planned_slots
                if planned_slots
                else 0.0
            ),
            "itt_denominator_includes_rejected_and_missing": True,
        }
    contrasts = []
    for control, treatment in (
        ("plain_resample", "compiler_only"),
        ("compiler_only", "semantic_judge"),
        ("plain_resample", "semantic_judge"),
    ):
        for field in ("visible_rescued", "genuine_rescue"):
            contrasts.append(
                _exact_mcnemar(
                    task_arm_rows,
                    control=control,
                    treatment=treatment,
                    field=field,
                )
            )
    transfer_rows = sorted(
        transfer.values(), key=lambda row: (row["task_id"], row["code_sha256"])
    )
    preference_rows = sorted(
        preferences.values(),
        key=lambda row: (
            row["task_id"],
            row["chosen_sha256"],
            row["rejected_sha256"],
        ),
    )
    body = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "runtime_provenance": _runtime_provenance(),
        "source_plan_sha256": plan_sha,
        "source_generation_sha256": generation_sha,
        "source_visible_selection_sha256": visible_sha,
        "visible_selection_artifact": _file_record(visible_path),
        "private_holdback": {
            "sha256": observed_holdback_sha,
            "rows": len(holdback_rows),
            "path_persisted": False,
        },
        "metrics_by_arm": metrics,
        "paired_contrasts": contrasts,
        "task_arm_results": task_arm_rows,
        "exports": {
            "rs_sft_targets": transfer_rows,
            "off_policy_preferences": preference_rows,
            "rs_sft_requires_full_visible_and_holdback": True,
            "rs_sft_requires_common_holdback_baseline_improvement": True,
            "partial_preferences_eligible_for_on_policy_update": False,
        },
        "selection_policy": {
            "all_visible_scores_complete_before_holdback_open": True,
            "visible_selection_artifact_sealed_before_private_hash": True,
            "holdback_used_to_choose_slot": False,
            "same_selected_repair_checked_on_holdback": True,
            "transfer_gated_on_visible_and_holdback": True,
            "common_holdback_baseline": (
                "max passed cases across the same two frozen parents"
            ),
            "common_baseline_shared_by_all_arms": True,
        },
        "privacy": {
            "holdback_test_source_persisted": False,
            "holdback_diagnostic_persisted": False,
            "holdback_exposed_to_provider": False,
            "reference_dart_exposed_to_provider": False,
            "final_175_opened": False,
            "development_holdback_consumed_for_transfer_selection": True,
        },
        "optimization": {
            "optimizer_constructed": False,
            "policy_updated": False,
            "artifact_is_offline_ablation": True,
        },
    }
    result = _seal(body, SCORE_HASH)
    require_exact_or_write(args.output, result)
    return result


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise argparse.ArgumentTypeError(f"{label} must be positive")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)

    project_parser = sub.add_parser("project", allow_abbrev=False)
    project_parser.add_argument("--base-journal", action="append", default=[])
    project_parser.add_argument("--base-report", action="append", default=[])
    project_parser.add_argument("--output", required=True)

    plan_parser = sub.add_parser("plan", allow_abbrev=False)
    plan_parser.add_argument("--projection", required=True)
    plan_parser.add_argument("--rollout-file", required=True)
    plan_parser.add_argument("--f2-jsonl", required=True)
    plan_parser.add_argument("--f2-manifest", required=True)
    plan_parser.add_argument("--public-manifest", required=True)
    plan_parser.add_argument("--output", required=True)
    plan_parser.add_argument("--groups", type=int, default=100)
    plan_parser.add_argument("--seed", type=int, default=260103525)
    plan_parser.add_argument("--reward-timeout", type=int, default=30)
    plan_parser.add_argument("--stability-runs", type=int, default=2)

    diagnose_parser = sub.add_parser("diagnose", allow_abbrev=False)
    diagnose_parser.add_argument("--plan", required=True)
    diagnose_parser.add_argument("--output", required=True)
    diagnose_parser.add_argument("--journal", required=True)
    diagnose_parser.add_argument("--model", default="")
    diagnose_parser.add_argument("--base-url", default="")
    diagnose_parser.add_argument(
        "--api-style",
        choices=[
            "anthropic_messages",
            "openai_responses",
            "openai_compatible_chat",
        ],
        default="openai_responses",
    )
    diagnose_parser.add_argument("--max-tokens", type=int, default=4096)
    diagnose_parser.add_argument("--timeout-seconds", type=float, default=300.0)
    diagnose_parser.add_argument(
        "--thinking-mode",
        choices=["adaptive", "disabled", "enabled", "provider_default"],
        default="provider_default",
    )
    diagnose_parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    diagnose_parser.add_argument(
        "--reasoning-mode", choices=["standard", "pro"], default="standard"
    )
    diagnose_parser.add_argument(
        "--chat-json-schema",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    generate_parser = sub.add_parser("generate", allow_abbrev=False)
    generate_parser.add_argument("--plan", required=True)
    generate_parser.add_argument("--diagnoses", required=True)
    generate_parser.add_argument("--sft-checkpoint", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--journal", required=True)
    generate_parser.add_argument("--seed", type=int, default=9102026)
    generate_parser.add_argument("--max-source-tokens", type=int, default=32768)
    generate_parser.add_argument("--max-new-tokens", type=int, default=4096)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument(
        "--attn-implementation", choices=["eager", "sdpa"], default="sdpa"
    )
    generate_parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )

    score_parser = sub.add_parser("score", allow_abbrev=False)
    score_parser.add_argument("--plan", required=True)
    score_parser.add_argument("--generation", required=True)
    score_parser.add_argument("--private-holdback", required=True)
    score_parser.add_argument(
        "--expected-private-holdback-sha256", required=True
    )
    score_parser.add_argument("--visible-output", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--reward-timeout", type=int, default=30)
    score_parser.add_argument("--stability-runs", type=int, default=2)
    score_parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "plan":
        for name in ("groups", "reward_timeout", "stability_runs"):
            _positive(getattr(args, name), f"--{name.replace('_', '-')}")
        if args.seed < 0:
            parser.error("--seed must be non-negative")
    elif args.command == "diagnose":
        if args.max_tokens <= 0 or args.timeout_seconds <= 0:
            parser.error("diagnosis token/time limits must be positive")
    elif args.command == "generate":
        for name in ("max_source_tokens", "max_new_tokens"):
            _positive(getattr(args, name), f"--{name.replace('_', '-')}")
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            parser.error("--temperature must be finite and positive")
        if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
            parser.error("--top-p must be in (0,1]")
    elif args.command == "score":
        for name in ("reward_timeout", "stability_runs", "workers"):
            _positive(getattr(args, name), f"--{name.replace('_', '-')}")
        if not _SHA_RE.fullmatch(args.expected_private_holdback_sha256):
            parser.error("--expected-private-holdback-sha256 is invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "project": project,
        "plan": plan,
        "diagnose": diagnose,
        "generate": generate,
        "score": score,
    }
    result = handlers[args.command](args)
    print(
        json.dumps(
            {
                "command": args.command,
                "schema": result["schema"],
                "status": result["status"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
