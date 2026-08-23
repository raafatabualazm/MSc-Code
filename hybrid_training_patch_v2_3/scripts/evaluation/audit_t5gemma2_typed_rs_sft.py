#!/usr/bin/env python3
"""Fail-closed matched audit for typed SFT versus typed-direct RS-SFT.

The audit consumes already-published evaluation artifacts.  It never runs a
model or a scorer.  Success means that both arms have complete hash-chained
generation and scoring journals, bind the same sealed 175-task evaluation,
use the same typed input/sampling/scoring contracts, and differ only in the
validated checkpoint lineage.  The resulting report includes full-175 and
known-contaminant-clean-174 paired binary comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.score_direct_compact_passk import extract_scored_code


AUDIT_SCHEMA = "t5gemma2-typed-rs-sft-matched-eval-audit-v1"
BASELINE_SEAL_SCHEMA = "t5gemma2-typed-contract-eval-baseline-seal-v1"
PROVENANCE_SCHEMA = "t5gemma2-f2-measurement-ablation-provenance-v1"
INFERENCE_SCHEMA = "t5gemma2-f2-measurement-ablation-inference-v1"
GENERATION_JOURNAL_SCHEMA = "t5gemma2-f2-heldout-generation-journal-v1"
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
SCORE_JOURNAL_SCHEMA = "direct-compact-attested-passk-journal-v1"
CLEAN_SCORE_SCHEMA = "direct-compact-passk-exclusion-sensitivity-v1"
BASELINE_CHECKPOINT_SCHEMA = "t5gemma2-typed-opaque-contract-sft-run-v1"
UPDATE_CHECKPOINT_SCHEMA = "t5gemma2-typed-direct-rs-sft-run-v1"
UPDATE_DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-dataset-v1"

EXPECTED_ROWS = 175
EXPECTED_K = 10
EXPECTED_CLEAN_ROWS = 174
EXPECTED_SCORE_WORKERS = 32
EXPECTED_SCORE_TIMEOUT = 30
EXPECTED_STABILITY_RUNS = 2
EXPECTED_EXCLUDED_TASK_IDS = ("sigless_8bf7f40ca356",)
EXPECTED_TRAIN_EXCLUSION = "sigless_6b1dd0c6b6fc"
EXPECTED_INPUT_VIEW = "typed_opaque_contract"
EXPECTED_SAMPLING = {
    "decoder_prefix_is_not_output": True,
    "fabricated_eos": False,
    "generation_batch_size": 10,
    "max_new_tokens": 4096,
    "max_source_tokens": 32768,
    "num_samples": 10,
    "sampled_eos_retained": True,
    "seed": 42,
    "seed_policy": "seed+task_index*100003+batch_start",
    "temperature": 0.8,
    "top_k": 0,
    "top_p": 0.95,
}
EXPECTED_COMPLETION_ATTESTATION = "per-run-256-bit-marker-exactly-once-v1"
EXPECTED_RETRY_POLICY = (
    "retry_identical_sealed_batch_with_hash_chained_receipt"
)

HASH_RE = re.compile(r"[0-9a-f]{64}")
JOURNAL_RECORD_KEYS = {
    "path",
    "sha256",
    "chain_head_path",
    "chain_head_sha256",
    "event_count",
    "head_event_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HASH_RE.fullmatch(value) is not None


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _read_array(path: Path, label: str) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a JSON array: {path}")
    return value


def _read_evaluation_task_ids(path: Path, expected_rows: int) -> list[str]:
    rows: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(
                        f"evaluation file has blank row at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"evaluation row {line_number} is not an object"
                    )
                task_id = value.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(
                        f"evaluation row {line_number} lacks task_id"
                    )
                rows.append(task_id)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"evaluation file is not readable JSONL: {path}") from exc
    _require(len(rows) == expected_rows, "evaluation row count is not exact")
    _require(len(set(rows)) == expected_rows, "evaluation task IDs are not unique")
    return rows


def _record_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "sha256",
            "chain_head_sha256",
            "event_count",
            "head_event_sha256",
        )
    }


def _validate_journal_binding(
    *,
    stored: Any,
    actual_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    _require(isinstance(stored, Mapping), f"{label} record is not an object")
    _require(set(stored) == JOURNAL_RECORD_KEYS, f"{label} record fields differ")
    events = load_journal(actual_path)
    actual = journal_record(actual_path)
    _require(
        _record_projection(stored) == _record_projection(actual),
        f"{label} content/head seal differs",
    )
    _require(
        Path(str(stored["path"])).name == actual_path.name,
        f"{label} recorded journal filename differs",
    )
    _require(
        Path(str(stored["chain_head_path"])).name
        == Path(str(actual_path) + ".chain-head.json").name,
        f"{label} recorded chain-head filename differs",
    )
    return events


def _validate_file_record(
    record: Any,
    *,
    actual_path: Path,
    expected_keys: set[str],
    label: str,
) -> None:
    _require(isinstance(record, Mapping), f"{label} record is not an object")
    _require(set(record) == expected_keys, f"{label} record fields differ")
    _require(record.get("sha256") == sha256_file(actual_path), f"{label} hash differs")
    _require(
        Path(str(record.get("path") or "")).name == actual_path.name,
        f"{label} filename differs",
    )


def _metric(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "rate": count / total}


def _validate_metric(value: Any, *, count: int, total: int, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(set(value) == {"count", "rate"}, f"{label} fields differ")
    _require(_is_int(value.get("count")), f"{label} count is not an integer")
    rate = value.get("rate")
    _require(
        isinstance(rate, (int, float)) and not isinstance(rate, bool),
        f"{label} rate is not numeric",
    )
    _require(int(value["count"]) == count, f"{label} count is inconsistent")
    _require(
        math.isfinite(float(rate))
        and abs(float(rate) - count / total) <= 1e-15,
        f"{label} rate is inconsistent",
    )


def _checkpoint_paths_record(path: Path, result_path: Path) -> dict[str, Any]:
    contract = _read_object(path, "checkpoint run contract")
    result = _read_object(result_path, "training result")
    return {
        "path": path,
        "result_path": result_path,
        "contract": contract,
        "result": result,
        "file_sha256": sha256_file(path),
        "result_sha256": sha256_file(result_path),
        "canonical_sha256": canonical_sha256(contract),
    }


def _validate_checkpoint_contracts(
    *,
    baseline_contract_path: Path,
    baseline_result_path: Path,
    update_contract_path: Path,
    update_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = _checkpoint_paths_record(
        baseline_contract_path, baseline_result_path
    )
    update = _checkpoint_paths_record(update_contract_path, update_result_path)
    b = baseline["contract"]
    u = update["contract"]
    bd = b.get("dataset")
    bh = bd.get("heldout") if isinstance(bd, Mapping) else None
    bx = bd.get("training_exclusions") if isinstance(bd, Mapping) else None
    bo = b.get("optimization")
    _require(
        b.get("schema") == BASELINE_CHECKPOINT_SCHEMA
        and b.get("status") == "training"
        and b.get("architecture") == "native_encoder_decoder"
        and isinstance(bo, Mapping)
        and bo.get("epochs") == 2
        and bo.get("planned_updates") == 348
        and bo.get("seed") == 42
        and isinstance(bd, Mapping)
        and bd.get("schema") == BASELINE_CHECKPOINT_SCHEMA
        and bd.get("rows") == 2775
        and bd.get("input_rows") == 2776
        and bd.get("model_visible_fields")
        == ["opaque_typed_contract", "F2.text"]
        and isinstance(bh, Mapping)
        and bh.get("rows") == 175
        and bh.get("model_visible") is False
        and bh.get("task_id_overlap") == 0
        and bh.get("exact_gold_source_overlap") == 0
        and bh.get("exact_acceptance_test_overlap") == 0
        and isinstance(bx, Mapping)
        and bx.get("count") == 1
        and bx.get("task_ids") == [EXPECTED_TRAIN_EXCLUSION],
        "baseline checkpoint schema/privacy/schedule contract differs",
    )
    b_result = baseline["result"]
    _require(
        b_result.get("schema") == BASELINE_CHECKPOINT_SCHEMA
        and b_result.get("status") == "complete"
        and b_result.get("updates") == 348
        and b_result.get("planned_updates") == 348
        and b_result.get("rows") == 2775
        and b_result.get("latest_checkpoint") == "checkpoint-optstep-000348",
        "baseline training result contract differs",
    )

    ud = u.get("dataset")
    uc = ud.get("composition") if isinstance(ud, Mapping) else None
    uv = (
        ud.get("full_acceptance_reverification")
        if isinstance(ud, Mapping)
        else None
    )
    ut = ud.get("typed_train") if isinstance(ud, Mapping) else None
    ux = ut.get("training_exclusions") if isinstance(ut, Mapping) else None
    up = u.get("privacy")
    uo = u.get("optimization")
    uw = u.get("warmstart")
    ul = u.get("lora")
    _require(
        u.get("schema") == UPDATE_CHECKPOINT_SCHEMA
        and u.get("status") == "training"
        and u.get("architecture") == "native_encoder_decoder"
        and isinstance(uo, Mapping)
        and uo.get("epochs") == 2
        and uo.get("planned_updates") == 58
        and uo.get("gradient_accumulation") == 8
        and uo.get("learning_rate") == 0.00002
        and uo.get("warmup_updates") == 0
        and uo.get("seed") == 42
        and isinstance(ud, Mapping)
        and ud.get("schema") == UPDATE_DATASET_SCHEMA
        and ud.get("rows") == 225
        and ud.get("architecture") == "native_encoder_decoder"
        and ud.get("heldout_overlap") == 0
        and ud.get("known_contaminant_excluded") == EXPECTED_TRAIN_EXCLUSION
        and ud.get("model_visible_fields")
        == ["opaque_typed_contract", "F2.text"]
        and ud.get("tests_model_visible") is False
        and ud.get("private_feedback_model_visible") is False
        and ud.get("repair_conditioned_prefixes_visible") is False
        and isinstance(uc, Mapping)
        and uc.get("verified_direct") == 225
        and uc.get("local_student_direct") == 141
        and uc.get("external_teacher_direct") == 84
        and uc.get("repair_conditioned") == 0
        and uc.get("gold_replay") == 0
        and isinstance(uv, Mapping)
        and uv.get("rows") == 225
        and uv.get("passed") == 225
        and uv.get("tests_model_visible") is False
        and uv.get("diagnostics_persisted") is False
        and isinstance(ux, Mapping)
        and ux.get("count") == 1
        and ux.get("task_ids") == [EXPECTED_TRAIN_EXCLUSION]
        and isinstance(up, Mapping)
        and up.get("heldout_overlap") == 0
        and up.get("heldout_content_model_visible") is False
        and up.get("tests_model_visible") is False
        and up.get("private_feedback_model_visible") is False,
        "typed-direct RS-SFT checkpoint schema/privacy/schedule contract differs",
    )
    b_lora = b.get("lora")
    _require(
        isinstance(uw, Mapping)
        and uw.get("checkpoint_name") == "checkpoint-optstep-000348"
        and uw.get("update") == 348
        and uw.get("run_contract_sha256") == baseline["canonical_sha256"]
        and isinstance(b_lora, Mapping)
        and isinstance(ul, Mapping)
        and ul.get("targets") == b_lora.get("targets")
        and ul.get("new_adapter_attached") is False
        and ul.get("warmstart_weights_continued") is True,
        "typed-direct RS-SFT checkpoint is not the baseline continuation",
    )
    u_result = update["result"]
    _require(
        u_result.get("schema") == UPDATE_CHECKPOINT_SCHEMA
        and u_result.get("status") == "complete"
        and u_result.get("updates") == 58
        and u_result.get("planned_updates") == 58
        and u_result.get("rows") == 225
        and u_result.get("latest_checkpoint") == "checkpoint-optstep-000058",
        "typed-direct RS-SFT training result contract differs",
    )
    _require(
        b.get("base_model") == u.get("base_model")
        and b.get("model") == u.get("model")
        and b.get("model_revision") == u.get("model_revision"),
        "baseline/update base-model identity differs",
    )
    return baseline, update


def _validate_generation(
    *,
    label: str,
    prediction_path: Path,
    provenance: Mapping[str, Any],
    predictions: list[Any],
    journal_path: Path,
    checkpoint: Mapping[str, Any],
    evaluation_sha256: str,
    expected_rows: int,
    expected_k: int,
    expected_sampling: Mapping[str, Any],
) -> dict[str, Any]:
    events = _validate_journal_binding(
        stored=provenance.get("generation_journal"),
        actual_path=journal_path,
        label=f"{label} generation journal",
    )
    _require(len(events) == expected_rows + 2, f"{label} journal event count differs")
    header = events[0]
    contract = header.get("contract")
    _require(
        header.get("event") == "header"
        and header.get("schema") == GENERATION_JOURNAL_SCHEMA
        and isinstance(contract, Mapping)
        and header.get("contract_sha256") == canonical_sha256(contract),
        f"{label} generation header is not sealed",
    )
    heldout = provenance.get("heldout")
    model = provenance.get("model")
    sampling = provenance.get("sampling")
    input_view = heldout.get("input_view") if isinstance(heldout, Mapping) else None
    _require(
        provenance.get("schema") == PROVENANCE_SCHEMA
        and provenance.get("architecture") == "native_t5gemma2_encoder_decoder"
        and provenance.get("arm") == "sft"
        and provenance.get("input_view") == EXPECTED_INPUT_VIEW
        and provenance.get("num_rows") == expected_rows
        and provenance.get("num_samples") == expected_k
        and provenance.get("output_sha256") == sha256_file(prediction_path)
        and provenance.get("no_frontier_api") is True
        and provenance.get("tests_exposed_to_model") is False
        and provenance.get("full_gold_targets_exposed_to_model") is False
        and provenance.get("gold_interface_types_and_arity_exposed_to_model")
        is True
        and sampling == expected_sampling
        and contract.get("schema") == INFERENCE_SCHEMA
        and contract.get("sampling") == expected_sampling
        and contract.get("arm") == "sft"
        and contract.get("heldout") == heldout
        and contract.get("model") == model
        and contract.get("source_truncation") is False
        and contract.get("no_frontier_api") is True
        and contract.get("tests_exposed_to_model") is False
        and contract.get("full_gold_targets_exposed_to_model") is False
        and isinstance(contract.get("runtime"), Mapping)
        and contract["runtime"].get("attn_implementation") == "sdpa"
        and contract["runtime"].get("bf16") is True
        and _is_sha256(contract.get("script_sha256"))
        and _is_sha256(contract.get("base_inference_script_sha256")),
        f"{label} provenance/inference contract differs",
    )
    _require(
        isinstance(heldout, Mapping)
        and heldout.get("dataset")
        == {"sha256": evaluation_sha256, "rows": expected_rows}
        and heldout.get("selected_rows") == expected_rows
        and heldout.get("tests_serialized_to_model") is False
        and heldout.get("full_gold_targets_serialized_to_model") is False
        and heldout.get("gold_targets_serialized_to_model") is False
        and heldout.get("gold_interface_types_and_arity_serialized_to_model")
        is True
        and heldout.get("model_visible_fields")
        == ["transformed_F2.text", "gold_derived_types_and_arity"]
        and isinstance(input_view, Mapping)
        and input_view.get("schema")
        == "t5gemma2-f2-measurement-input-view-v1"
        and input_view.get("view") == EXPECTED_INPUT_VIEW
        and input_view.get("rows") == expected_rows
        and input_view.get("tests_exposed_to_model") is False
        and input_view.get("full_gold_targets_exposed_to_model") is False,
        f"{label} typed held-out input-view contract differs",
    )
    summary = input_view.get("summary")
    _require(
        isinstance(summary, Mapping)
        and summary.get("intervention") == "gold_derived_types_and_arity_only"
        and summary.get("gold_implementation_body_exposed_to_model") is False
        and summary.get("gold_semantic_parameter_names_exposed_to_model") is False,
        f"{label} typed input-view privacy summary differs",
    )
    checkpoint_digest = str(checkpoint["canonical_sha256"])
    adapter = model.get("adapter") if isinstance(model, Mapping) else None
    _require(
        isinstance(model, Mapping)
        and model.get("training_stage_schema")
        == checkpoint["contract"].get("schema")
        and model.get("warmstart_contract_sha256") == checkpoint_digest
        and isinstance(adapter, Mapping)
        and adapter.get("run_contract_sha256") == checkpoint_digest
        and provenance.get("sft_checkpoint_contract_sha256")
        == checkpoint_digest
        and _is_sha256(model.get("tokenizer_sha256"))
        and _is_sha256(adapter.get("adapter_weights_sha256"))
        and _is_sha256(adapter.get("adapter_config_sha256")),
        f"{label} checkpoint-to-provenance binding differs",
    )

    task_ids: list[str] = []
    source_sha256s: list[str] = []
    encoder_tokens: list[int] = []
    built_predictions: list[dict[str, Any]] = []
    max_token_completions = 0
    max_source = int(expected_sampling["max_source_tokens"])
    max_new = int(expected_sampling["max_new_tokens"])
    batch_size = int(expected_sampling["generation_batch_size"])
    seed0 = int(expected_sampling["seed"])
    for task_index, terminal in enumerate(events[1:-1]):
        candidates = terminal.get("candidates")
        task_id = terminal.get("task_id")
        source_sha = terminal.get("source_sha256")
        tokens = terminal.get("encoder_tokens")
        _require(
            terminal.get("event") == "task_terminal"
            and terminal.get("schema") == GENERATION_JOURNAL_SCHEMA
            and terminal.get("task_index") == task_index
            and isinstance(task_id, str)
            and bool(task_id)
            and _is_sha256(source_sha)
            and _is_int(tokens)
            and 0 < int(tokens) <= max_source
            and isinstance(candidates, list)
            and len(candidates) == expected_k,
            f"{label} generation terminal {task_index} differs",
        )
        texts: list[str] = []
        for sample_index, candidate in enumerate(candidates):
            batch_start = (sample_index // batch_size) * batch_size
            expected_seed = seed0 + task_index * 100_003 + batch_start
            action_tokens = candidate.get("action_tokens") if isinstance(candidate, Mapping) else None
            eos = candidate.get("eos_observed") if isinstance(candidate, Mapping) else None
            capped = (
                candidate.get("max_token_completion")
                if isinstance(candidate, Mapping)
                else None
            )
            text = candidate.get("text") if isinstance(candidate, Mapping) else None
            _require(
                isinstance(candidate, Mapping)
                and set(candidate)
                == {
                    "sample_index",
                    "seed",
                    "batch_position",
                    "text",
                    "text_sha256",
                    "action_tokens",
                    "eos_observed",
                    "max_token_completion",
                }
                and candidate.get("sample_index") == sample_index
                and candidate.get("seed") == expected_seed
                and candidate.get("batch_position") == sample_index - batch_start
                and isinstance(text, str)
                and candidate.get("text_sha256")
                == hashlib.sha256(text.encode("utf-8")).hexdigest()
                and _is_int(action_tokens)
                and 0 < int(action_tokens) <= max_new
                and type(eos) is bool
                and type(capped) is bool
                and capped == (not eos and int(action_tokens) >= max_new),
                f"{label} candidate coordinate/hash differs at "
                f"{task_id}:{sample_index}",
            )
            max_token_completions += int(capped)
            texts.append(text)
        task_ids.append(str(task_id))
        source_sha256s.append(str(source_sha))
        encoder_tokens.append(int(tokens))
        built_predictions.append({"id": task_id, "predictions": texts})
    _require(len(set(task_ids)) == expected_rows, f"{label} task IDs are not unique")
    _require(
        predictions == built_predictions,
        f"{label} predictions do not exactly match generation terminals",
    )
    complete = events[-1]
    _require(
        complete.get("event") == "complete"
        and complete.get("schema") == GENERATION_JOURNAL_SCHEMA
        and complete.get("rows") == expected_rows
        and complete.get("predictions_canonical_sha256")
        == canonical_sha256(built_predictions),
        f"{label} generation completion seal differs",
    )
    task_digest = canonical_sha256(task_ids)
    source_digest = canonical_sha256(source_sha256s)
    _require(
        task_digest == heldout.get("task_set_sha256")
        == heldout.get("selected_ordered_task_ids_sha256")
        == input_view.get("ordered_task_ids_sha256")
        and source_digest
        == heldout.get("selected_ordered_source_sha256s_sha256")
        == input_view.get("ordered_source_sha256s_sha256")
        and provenance.get("max_token_completions") == max_token_completions,
        f"{label} ordered task/source/capped-completion accounting differs",
    )
    return {
        "events": events,
        "header_contract": dict(contract),
        "task_ids": task_ids,
        "source_sha256s": source_sha256s,
        "encoder_tokens": encoder_tokens,
        "max_token_completions": max_token_completions,
        "task_ids_sha256": task_digest,
        "source_sha256s_sha256": source_digest,
    }


def _score_jobs(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in predictions:
        task_id = str(row["id"])
        for sample_index, raw in enumerate(row["predictions"]):
            raw_text = str(raw)
            code = extract_scored_code(raw_text)
            jobs.append(
                {
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                }
            )
    return jobs


def _validate_score_journal(
    *,
    label: str,
    events: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    prediction_sha256: str,
    provenance_sha256: str,
    evaluation_sha256: str,
    evaluator_sha256: str,
    expected_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(bool(events), f"{label} score journal is empty")
    header = events[0]
    contract = header.get("contract")
    slot_ids = [f"{job['task_id']}:{job['sample_index']}" for job in jobs]
    _require(
        header.get("event") == "score_header"
        and header.get("schema") == SCORE_JOURNAL_SCHEMA
        and isinstance(contract, Mapping)
        and header.get("contract_sha256") == canonical_sha256(contract)
        and contract.get("schema") == SCORE_JOURNAL_SCHEMA
        and contract.get("predictions_sha256") == prediction_sha256
        and contract.get("prediction_provenance_sha256") == provenance_sha256
        and contract.get("evaluation_sha256") == evaluation_sha256
        and contract.get("evaluator_sha256") == evaluator_sha256
        and contract.get("completion_attestation")
        == EXPECTED_COMPLETION_ATTESTATION
        and contract.get("k") == expected_k
        and contract.get("workers") == EXPECTED_SCORE_WORKERS
        and contract.get("batch_size") == EXPECTED_SCORE_WORKERS
        and contract.get("timeout") == EXPECTED_SCORE_TIMEOUT
        and contract.get("stability_runs") == EXPECTED_STABILITY_RUNS
        and contract.get("ordered_slot_ids_sha256") == canonical_sha256(slot_ids)
        and contract.get("slots") == len(jobs)
        and contract.get("started_without_terminal_policy")
        == EXPECTED_RETRY_POLICY,
        f"{label} score journal header contract differs",
    )
    batch_size = EXPECTED_SCORE_WORKERS
    batch_count = (len(jobs) + batch_size - 1) // batch_size
    cursor = 1
    candidate_results: list[dict[str, Any]] = []
    retry_events = 0
    retry_slots = 0
    for batch_index in range(batch_count):
        batch_jobs = list(jobs[batch_index * batch_size : (batch_index + 1) * batch_size])
        expected_slots = [
            f"{job['task_id']}:{job['sample_index']}" for job in batch_jobs
        ]
        expected_jobs_sha = canonical_sha256(batch_jobs)
        _require(cursor < len(events), f"{label} score journal ended before batch start")
        started = events[cursor]
        _require(
            started.get("event") == "score_batch_started"
            and started.get("schema") == SCORE_JOURNAL_SCHEMA
            and started.get("batch_index") == batch_index
            and started.get("slot_ids") == expected_slots
            and started.get("jobs_canonical_sha256") == expected_jobs_sha,
            f"{label} score batch {batch_index} start seal differs",
        )
        cursor += 1
        retries: list[Mapping[str, Any]] = []
        previous_attempt = started.get("journal_event_sha256")
        batch_seal = {
            "schema": "direct-compact-score-sealed-batch-v1",
            "batch_index": batch_index,
            "slot_ids": expected_slots,
            "jobs_canonical_sha256": expected_jobs_sha,
        }
        while cursor < len(events) and events[cursor].get("event") == "score_batch_orphan_retry":
            retry = events[cursor]
            _require(
                retry.get("schema") == SCORE_JOURNAL_SCHEMA
                and retry.get("batch_index") == batch_index
                and retry.get("retry_index") == len(retries) + 1
                and retry.get("started_event_sha256")
                == started.get("journal_event_sha256")
                and retry.get("previous_attempt_event_sha256") == previous_attempt
                and retry.get("sealed_batch") == batch_seal
                and retry.get("sealed_batch_sha256") == canonical_sha256(batch_seal)
                and retry.get("completed_terminal_batches_preserved") == batch_index
                and retry.get("recovery_reason")
                == "process_interrupted_after_durable_batch_start"
                and retry.get("rerun_identical_sealed_batch") is True
                and retry.get("change_evaluation_slots") is False,
                f"{label} score batch {batch_index} retry receipt differs",
            )
            previous_attempt = retry.get("journal_event_sha256")
            retries.append(retry)
            retry_events += 1
            retry_slots += len(expected_slots)
            cursor += 1
        _require(cursor < len(events), f"{label} score journal lacks batch terminal")
        terminal = events[cursor]
        results = terminal.get("candidate_results")
        _require(
            terminal.get("event") == "score_batch_terminal"
            and terminal.get("schema") == SCORE_JOURNAL_SCHEMA
            and terminal.get("batch_index") == batch_index
            and terminal.get("started_event_sha256")
            == started.get("journal_event_sha256")
            and terminal.get("retry_count") == len(retries)
            and terminal.get("latest_retry_event_sha256")
            == (retries[-1].get("journal_event_sha256") if retries else None)
            and isinstance(results, list)
            and [
                f"{row.get('task_id')}:{row.get('sample_index')}"
                for row in results
            ]
            == expected_slots
            and terminal.get("candidate_results_canonical_sha256")
            == canonical_sha256(results),
            f"{label} score batch {batch_index} terminal seal differs",
        )
        for result, job in zip(results, batch_jobs, strict=True):
            _require(
                isinstance(result, Mapping)
                and set(result)
                == {
                    "task_id",
                    "sample_index",
                    "raw_sha256",
                    "code_sha256",
                    "compiled",
                    "passed",
                    "diagnostic",
                }
                and all(result.get(key) == job[key] for key in job)
                and type(result.get("compiled")) is bool
                and type(result.get("passed")) is bool
                and (not result["passed"] or result["compiled"])
                and isinstance(result.get("diagnostic"), str),
                f"{label} scored result does not bind prediction slot "
                f"{job['task_id']}:{job['sample_index']}",
            )
            candidate_results.append(dict(result))
        cursor += 1
    _require(cursor == len(events) - 1, f"{label} score journal has extra events")
    complete = events[cursor]
    _require(
        complete.get("event") == "score_complete"
        and complete.get("schema") == SCORE_JOURNAL_SCHEMA
        and complete.get("slots") == len(jobs)
        and complete.get("candidate_results_canonical_sha256")
        == canonical_sha256(candidate_results)
        and complete.get("rerun_slots") == 0
        and complete.get("orphan_retry_events") == retry_events
        and complete.get("orphan_rerun_slots") == retry_slots,
        f"{label} score completion seal differs",
    )
    return candidate_results, dict(contract)


def _validate_full_score(
    *,
    label: str,
    score_path: Path,
    score: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    prediction_path: Path,
    provenance_path: Path,
    evaluation_path: Path,
    evaluator_path: Path,
    expected_rows: int,
    expected_k: int,
) -> dict[str, Any]:
    score_journal_path = Path(str(score_path) + ".evaluation.journal.jsonl")
    events = _validate_journal_binding(
        stored=score.get("evaluation_journal"),
        actual_path=score_journal_path,
        label=f"{label} score journal",
    )
    prediction_sha = sha256_file(prediction_path)
    provenance_sha = sha256_file(provenance_path)
    evaluation_sha = sha256_file(evaluation_path)
    evaluator_sha = sha256_file(evaluator_path)
    _validate_file_record(
        score.get("predictions"),
        actual_path=prediction_path,
        expected_keys={"path", "sha256", "provenance_sha256"},
        label=f"{label} score prediction",
    )
    _require(
        score["predictions"].get("provenance_sha256") == provenance_sha,
        f"{label} score provenance hash differs",
    )
    _validate_file_record(
        score.get("evaluation"),
        actual_path=evaluation_path,
        expected_keys={"path", "sha256"},
        label=f"{label} score evaluation",
    )
    _validate_file_record(
        score.get("evaluator"),
        actual_path=evaluator_path,
        expected_keys={"path", "sha256", "completion_attestation"},
        label=f"{label} evaluator",
    )
    _require(
        score["evaluator"].get("completion_attestation")
        == EXPECTED_COMPLETION_ATTESTATION
        and score.get("schema") == SCORE_SCHEMA
        and score.get("tasks") == expected_rows
        and score.get("k") == expected_k
        and score.get("timeout") == EXPECTED_SCORE_TIMEOUT
        and score.get("stability_runs") == EXPECTED_STABILITY_RUNS
        and score.get("started_without_terminal_policy") == EXPECTED_RETRY_POLICY
        and score.get("rerun_slots") == 0,
        f"{label} score top-level contract differs",
    )
    jobs = _score_jobs(predictions)
    journal_results, score_contract = _validate_score_journal(
        label=label,
        events=events,
        jobs=jobs,
        prediction_sha256=prediction_sha,
        provenance_sha256=provenance_sha,
        evaluation_sha256=evaluation_sha,
        evaluator_sha256=evaluator_sha,
        expected_k=expected_k,
    )
    candidate_results = score.get("candidate_results")
    _require(
        candidate_results == journal_results,
        f"{label} score candidates differ from journal terminals",
    )
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in journal_results:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    expected_task_results: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        rows = by_task[task_id]
        _require(
            [row["sample_index"] for row in rows] == list(range(expected_k)),
            f"{label} score sample coverage differs for {task_id}",
        )
        expected_task_results.append(
            {
                "task_id": task_id,
                "pass_at_1": bool(rows[0]["passed"]),
                "pass_at_k": any(row["passed"] for row in rows),
                "compile_at_k": any(row["compiled"] for row in rows),
                "passing_samples": sum(row["passed"] for row in rows),
                "compiling_samples": sum(row["compiled"] for row in rows),
            }
        )
    _require(
        score.get("task_results") == expected_task_results,
        f"{label} task aggregates are inconsistent",
    )
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in expected_task_results),
        "pass_at_k": sum(row["pass_at_k"] for row in expected_task_results),
        "compile_at_k": sum(row["compile_at_k"] for row in expected_task_results),
    }
    for metric_name, count in counts.items():
        _validate_metric(
            score.get(metric_name),
            count=count,
            total=expected_rows,
            label=f"{label} {metric_name}",
        )
    _require(
        score.get("orphan_retry_events")
        == events[-1].get("orphan_retry_events")
        and score.get("orphan_rerun_slots")
        == events[-1].get("orphan_rerun_slots"),
        f"{label} score retry accounting differs",
    )
    return {
        "journal_events": events,
        "journal_contract": score_contract,
        "candidate_results": journal_results,
        "task_results": expected_task_results,
        "by_task": {row["task_id"]: row for row in expected_task_results},
        "counts": counts,
        "task_order": [row["task_id"] for row in expected_task_results],
        "candidate_slots": [
            (row["task_id"], row["sample_index"]) for row in journal_results
        ],
    }


def _validate_clean_score(
    *,
    label: str,
    clean_path: Path,
    clean: Mapping[str, Any],
    full_path: Path,
    full: Mapping[str, Any],
    expected_excluded: Sequence[str],
    expected_rows: int,
    expected_k: int,
) -> dict[str, Any]:
    excluded = list(expected_excluded)
    excluded_set = set(excluded)
    expected_tasks = [
        row for row in full["task_results"] if row["task_id"] not in excluded_set
    ]
    expected_candidates = [
        row
        for row in full["candidate_results"]
        if row["task_id"] not in excluded_set
    ]
    clean_rows = expected_rows - len(excluded)
    _require(
        clean.get("schema") == CLEAN_SCORE_SCHEMA
        and clean.get("tasks") == clean_rows
        and clean.get("k") == expected_k
        and clean.get("excluded_task_ids") == excluded
        and clean.get("excluded_task_ids_sha256") == canonical_sha256(excluded)
        and clean.get("exclusion_reason")
        == "known train/heldout exact acceptance-test duplicate in comparator training set"
        and clean.get("source_score_schema") == SCORE_SCHEMA
        and clean.get("task_results") == expected_tasks
        and clean.get("candidate_results") == expected_candidates,
        f"{label} clean-score projection differs",
    )
    _validate_file_record(
        clean.get("source_score"),
        actual_path=full_path,
        expected_keys={"path", "sha256"},
        label=f"{label} clean source score",
    )
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in expected_tasks),
        "pass_at_k": sum(row["pass_at_k"] for row in expected_tasks),
        "compile_at_k": sum(row["compile_at_k"] for row in expected_tasks),
    }
    for metric_name, count in counts.items():
        _validate_metric(
            clean.get(metric_name),
            count=count,
            total=clean_rows,
            label=f"{label} clean {metric_name}",
        )
    return {
        "counts": counts,
        "task_results": expected_tasks,
        "by_task": {row["task_id"]: row for row in expected_tasks},
    }


def _validate_arm(
    *,
    label: str,
    prediction_path: Path,
    full_score_path: Path,
    clean_score_path: Path,
    checkpoint: Mapping[str, Any],
    evaluation_path: Path,
    evaluator_path: Path,
    expected_rows: int,
    expected_k: int,
    expected_sampling: Mapping[str, Any],
    expected_excluded: Sequence[str],
) -> dict[str, Any]:
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    generation_journal_path = Path(
        str(prediction_path) + ".generation.journal.jsonl"
    )
    score_journal_path = Path(
        str(full_score_path) + ".evaluation.journal.jsonl"
    )
    required = (
        prediction_path,
        provenance_path,
        generation_journal_path,
        Path(str(generation_journal_path) + ".chain-head.json"),
        full_score_path,
        score_journal_path,
        Path(str(score_journal_path) + ".chain-head.json"),
        clean_score_path,
    )
    for path in required:
        _require(path.is_file() and path.stat().st_size > 0, f"{label} missing {path}")
    predictions = _read_array(prediction_path, f"{label} predictions")
    provenance = _read_object(provenance_path, f"{label} provenance")
    full_score = _read_object(full_score_path, f"{label} full score")
    clean_score = _read_object(clean_score_path, f"{label} clean score")
    generation = _validate_generation(
        label=label,
        prediction_path=prediction_path,
        provenance=provenance,
        predictions=predictions,
        journal_path=generation_journal_path,
        checkpoint=checkpoint,
        evaluation_sha256=sha256_file(evaluation_path),
        expected_rows=expected_rows,
        expected_k=expected_k,
        expected_sampling=expected_sampling,
    )
    full = _validate_full_score(
        label=label,
        score_path=full_score_path,
        score=full_score,
        predictions=predictions,
        prediction_path=prediction_path,
        provenance_path=provenance_path,
        evaluation_path=evaluation_path,
        evaluator_path=evaluator_path,
        expected_rows=expected_rows,
        expected_k=expected_k,
    )
    clean = _validate_clean_score(
        label=label,
        clean_path=clean_score_path,
        clean=clean_score,
        full_path=full_score_path,
        full=full_score,
        expected_excluded=expected_excluded,
        expected_rows=expected_rows,
        expected_k=expected_k,
    )
    return {
        "label": label,
        "prediction_path": prediction_path,
        "provenance_path": provenance_path,
        "generation_journal_path": generation_journal_path,
        "full_score_path": full_score_path,
        "score_journal_path": score_journal_path,
        "clean_score_path": clean_score_path,
        "predictions": predictions,
        "provenance": provenance,
        "full_score": full_score,
        "clean_score": clean_score,
        "generation": generation,
        "full": full,
        "clean": clean,
    }


def _hash_list(values: Sequence[str]) -> dict[str, Any]:
    return {
        "count": len(values),
        "task_ids": list(values),
        "task_ids_sha256": canonical_sha256(list(values)),
    }


def _exact_two_sided_sign_mcnemar(gains: int, losses: int) -> dict[str, Any]:
    discordant = gains + losses
    if discordant == 0:
        fraction = Fraction(1, 1)
    else:
        tail = sum(comb(discordant, index) for index in range(min(gains, losses) + 1))
        fraction = Fraction(min(2**discordant, 2 * tail), 2**discordant)
    with localcontext() as context:
        context.prec = 80
        decimal = Decimal(fraction.numerator) / Decimal(fraction.denominator)
    return {
        "test": "exact_two_sided_sign_test_equivalent_to_exact_mcnemar",
        "null_discordant_probability": "1/2",
        "numerator": fraction.numerator,
        "denominator": fraction.denominator,
        "fraction": f"{fraction.numerator}/{fraction.denominator}",
        "decimal": format(decimal, "f"),
        "value": float(fraction),
    }


def _paired_metric(
    *,
    task_order: Sequence[str],
    baseline_by_task: Mapping[str, Mapping[str, Any]],
    update_by_task: Mapping[str, Mapping[str, Any]],
    source_metric: str,
) -> dict[str, Any]:
    gains: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    baseline_positive: list[str] = []
    update_positive: list[str] = []
    for task_id in task_order:
        before = bool(baseline_by_task[task_id][source_metric])
        after = bool(update_by_task[task_id][source_metric])
        if before:
            baseline_positive.append(task_id)
        if after:
            update_positive.append(task_id)
        if after and not before:
            gains.append(task_id)
        elif before and not after:
            losses.append(task_id)
        else:
            ties.append(task_id)
    _require(
        len(gains) + len(losses) + len(ties) == len(task_order),
        "paired outcome partition is inconsistent",
    )
    return {
        "source_metric": source_metric,
        "tasks": len(task_order),
        "baseline": _hash_list(baseline_positive),
        "update58": _hash_list(update_positive),
        "absolute_count_delta": len(update_positive) - len(baseline_positive),
        "rate_delta": (
            len(update_positive) - len(baseline_positive)
        )
        / len(task_order),
        "gains": _hash_list(gains),
        "losses": _hash_list(losses),
        "ties": _hash_list(ties),
        "discordant_tasks": len(gains) + len(losses),
        "exact_two_sided_sign_mcnemar_p": _exact_two_sided_sign_mcnemar(
            len(gains), len(losses)
        ),
    }


def _comparison_block(
    *,
    task_order: Sequence[str],
    baseline_by_task: Mapping[str, Mapping[str, Any]],
    update_by_task: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "tasks": len(task_order),
        "ordered_task_ids": list(task_order),
        "ordered_task_ids_sha256": canonical_sha256(list(task_order)),
        "metrics": {
            "pass_at_1": _paired_metric(
                task_order=task_order,
                baseline_by_task=baseline_by_task,
                update_by_task=update_by_task,
                source_metric="pass_at_1",
            ),
            "pass_at_10": _paired_metric(
                task_order=task_order,
                baseline_by_task=baseline_by_task,
                update_by_task=update_by_task,
                source_metric="pass_at_k",
            ),
            "compile_at_10": _paired_metric(
                task_order=task_order,
                baseline_by_task=baseline_by_task,
                update_by_task=update_by_task,
                source_metric="compile_at_k",
            ),
        },
    }


def _artifact_hashes(arm: Mapping[str, Any]) -> dict[str, str]:
    generation_head = Path(str(arm["generation_journal_path"]) + ".chain-head.json")
    score_head = Path(str(arm["score_journal_path"]) + ".chain-head.json")
    return {
        "predictions_sha256": sha256_file(arm["prediction_path"]),
        "provenance_sha256": sha256_file(arm["provenance_path"]),
        "generation_journal_sha256": sha256_file(arm["generation_journal_path"]),
        "generation_journal_chain_head_sha256": sha256_file(generation_head),
        "full_score_sha256": sha256_file(arm["full_score_path"]),
        "clean_score_sha256": sha256_file(arm["clean_score_path"]),
        "evaluation_journal_sha256": sha256_file(arm["score_journal_path"]),
        "evaluation_journal_chain_head_sha256": sha256_file(score_head),
    }


def _validate_baseline_seal(
    *,
    seal_path: Path,
    seal: Mapping[str, Any],
    baseline: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    evaluation_sha256: str,
    task_ids_sha256: str,
) -> None:
    expected_artifacts = dict(_artifact_hashes(baseline))
    expected_artifacts.update(
        {
            "checkpoint_contract_sha256": checkpoint["file_sha256"],
            "training_result_sha256": checkpoint["result_sha256"],
        }
    )
    _require(
        seal.get("schema") == BASELINE_SEAL_SCHEMA
        and seal.get("status") == "sealed"
        and seal.get("evaluation_sha256") == evaluation_sha256
        and seal.get("ordered_task_ids_sha256") == task_ids_sha256
        and seal.get("checkpoint_contract_canonical_sha256")
        == checkpoint["canonical_sha256"]
        and seal.get("artifacts") == expected_artifacts,
        "baseline external seal differs from supplied artifacts",
    )
    metrics = seal.get("metrics")
    _require(
        isinstance(metrics, Mapping)
        and metrics.get("full175") == baseline["full"]["counts"]
        and metrics.get("clean174") == baseline["clean"]["counts"],
        "baseline external seal metric roots differ",
    )
    _require(seal_path.is_file(), "baseline seal path disappeared")


def _arm_report(
    arm: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "checkpoint": {
            "schema": checkpoint["contract"]["schema"],
            "contract_path": str(checkpoint["path"]),
            "contract_file_sha256": checkpoint["file_sha256"],
            "contract_canonical_sha256": checkpoint["canonical_sha256"],
            "result_path": str(checkpoint["result_path"]),
            "result_sha256": checkpoint["result_sha256"],
        },
        "artifacts": _artifact_hashes(arm),
        "paths": {
            "predictions": str(arm["prediction_path"]),
            "provenance": str(arm["provenance_path"]),
            "generation_journal": str(arm["generation_journal_path"]),
            "full_score": str(arm["full_score_path"]),
            "evaluation_journal": str(arm["score_journal_path"]),
            "clean_score": str(arm["clean_score_path"]),
        },
        "metrics": {
            "full175": {
                name: _metric(count, len(arm["generation"]["task_ids"]))
                for name, count in arm["full"]["counts"].items()
            },
            "clean174": {
                name: _metric(
                    count,
                    len(arm["generation"]["task_ids"])
                    - len(EXPECTED_EXCLUDED_TASK_IDS),
                )
                for name, count in arm["clean"]["counts"].items()
            },
        },
        "max_token_completions": arm["generation"]["max_token_completions"],
        "base_inference_script_sha256": arm["generation"]["header_contract"][
            "base_inference_script_sha256"
        ],
    }


def audit(
    *,
    baseline_predictions: Path,
    baseline_full_score: Path,
    baseline_clean_score: Path,
    baseline_checkpoint_contract: Path,
    baseline_training_result: Path,
    baseline_seal: Path,
    update_predictions: Path,
    update_full_score: Path,
    update_clean_score: Path,
    update_checkpoint_contract: Path,
    update_training_result: Path,
    evaluation_file: Path,
    evaluator_file: Path,
    output: Path,
    expected_rows: int = EXPECTED_ROWS,
    expected_k: int = EXPECTED_K,
    expected_sampling: Mapping[str, Any] = EXPECTED_SAMPLING,
    expected_excluded_task_ids: Sequence[str] = EXPECTED_EXCLUDED_TASK_IDS,
) -> dict[str, Any]:
    """Validate both arms and publish a deterministic matched-audit report."""

    paths = [
        baseline_predictions,
        baseline_full_score,
        baseline_clean_score,
        baseline_checkpoint_contract,
        baseline_training_result,
        baseline_seal,
        update_predictions,
        update_full_score,
        update_clean_score,
        update_checkpoint_contract,
        update_training_result,
        evaluation_file,
        evaluator_file,
    ]
    paths = [path.expanduser().resolve() for path in paths]
    (
        baseline_predictions,
        baseline_full_score,
        baseline_clean_score,
        baseline_checkpoint_contract,
        baseline_training_result,
        baseline_seal,
        update_predictions,
        update_full_score,
        update_clean_score,
        update_checkpoint_contract,
        update_training_result,
        evaluation_file,
        evaluator_file,
    ) = paths
    output = output.expanduser().resolve()
    for required in paths:
        _require(required.is_file() and required.stat().st_size > 0, f"missing {required}")
    _require(expected_k == 10, "matched audit is defined for K=10")
    _require(
        expected_rows - len(expected_excluded_task_ids) == EXPECTED_CLEAN_ROWS
        or expected_rows != EXPECTED_ROWS,
        "production clean comparison must contain exactly 174 tasks",
    )
    evaluation_task_ids = _read_evaluation_task_ids(evaluation_file, expected_rows)
    evaluation_sha = sha256_file(evaluation_file)
    baseline_checkpoint, update_checkpoint = _validate_checkpoint_contracts(
        baseline_contract_path=baseline_checkpoint_contract,
        baseline_result_path=baseline_training_result,
        update_contract_path=update_checkpoint_contract,
        update_result_path=update_training_result,
    )
    baseline = _validate_arm(
        label="typed_contract_sft_baseline",
        prediction_path=baseline_predictions,
        full_score_path=baseline_full_score,
        clean_score_path=baseline_clean_score,
        checkpoint=baseline_checkpoint,
        evaluation_path=evaluation_file,
        evaluator_path=evaluator_file,
        expected_rows=expected_rows,
        expected_k=expected_k,
        expected_sampling=expected_sampling,
        expected_excluded=expected_excluded_task_ids,
    )
    update = _validate_arm(
        label="typed_direct_rs_sft_update58",
        prediction_path=update_predictions,
        full_score_path=update_full_score,
        clean_score_path=update_clean_score,
        checkpoint=update_checkpoint,
        evaluation_path=evaluation_file,
        evaluator_path=evaluator_file,
        expected_rows=expected_rows,
        expected_k=expected_k,
        expected_sampling=expected_sampling,
        expected_excluded=expected_excluded_task_ids,
    )
    bgen = baseline["generation"]
    ugen = update["generation"]
    bprov = baseline["provenance"]
    uprov = update["provenance"]
    bscore = baseline["full"]
    uscore = update["full"]
    _require(
        bgen["task_ids"] == ugen["task_ids"] == evaluation_task_ids
        and bgen["source_sha256s"] == ugen["source_sha256s"]
        and bgen["encoder_tokens"] == ugen["encoder_tokens"]
        and bgen["header_contract"]["sampling"]
        == ugen["header_contract"]["sampling"]
        and bgen["header_contract"]["script_sha256"]
        == ugen["header_contract"]["script_sha256"]
        and bgen["header_contract"]["runtime"]
        == ugen["header_contract"]["runtime"]
        and bprov["heldout"] == uprov["heldout"]
        and bprov["model"]["tokenizer_sha256"]
        == uprov["model"]["tokenizer_sha256"]
        and bscore["task_order"] == uscore["task_order"]
        and bscore["candidate_slots"] == uscore["candidate_slots"],
        "baseline/update generation or slot pairing is not exact",
    )
    score_contract_fields = (
        "schema",
        "evaluation_sha256",
        "evaluator_sha256",
        "completion_attestation",
        "k",
        "workers",
        "batch_size",
        "timeout",
        "stability_runs",
        "ordered_slot_ids_sha256",
        "slots",
        "started_without_terminal_policy",
    )
    _require(
        all(
            bscore["journal_contract"].get(key)
            == uscore["journal_contract"].get(key)
            for key in score_contract_fields
        ),
        "baseline/update scoring contracts differ",
    )
    seal = _read_object(baseline_seal, "baseline external seal")
    _validate_baseline_seal(
        seal_path=baseline_seal,
        seal=seal,
        baseline=baseline,
        checkpoint=baseline_checkpoint,
        evaluation_sha256=evaluation_sha,
        task_ids_sha256=bgen["task_ids_sha256"],
    )
    excluded = list(expected_excluded_task_ids)
    _require(
        all(task_id in set(bgen["task_ids"]) for task_id in excluded),
        "clean exclusion task is absent from the matched set",
    )
    clean_order = [task for task in bgen["task_ids"] if task not in set(excluded)]
    report = {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "exact_pairing_validated": True,
        "contract": {
            "tasks": expected_rows,
            "k": expected_k,
            "clean_tasks": len(clean_order),
            "input_view": EXPECTED_INPUT_VIEW,
            "sampling": dict(expected_sampling),
            "scoring": {
                "workers": EXPECTED_SCORE_WORKERS,
                "timeout": EXPECTED_SCORE_TIMEOUT,
                "stability_runs": EXPECTED_STABILITY_RUNS,
                "completion_attestation": EXPECTED_COMPLETION_ATTESTATION,
            },
            "excluded_task_ids": excluded,
            "excluded_task_ids_sha256": canonical_sha256(excluded),
        },
        "checks": {
            "baseline_external_seal_validated": True,
            "prediction_provenance_hashes_validated": True,
            "generation_hash_chains_validated": True,
            "score_hash_chains_validated": True,
            "checkpoint_schemas_and_lineage_validated": True,
            "exact_175_task_order_validated": expected_rows == 175,
            "exact_k10_slots_validated": expected_k == 10,
            "same_task_order_and_sources": True,
            "same_seed_coordinates_and_sampling": True,
            "same_typed_input_view": True,
            "same_tokenizer_and_encoder_lengths": True,
            "no_source_truncation": True,
            "same_scorer_and_scoring_settings": True,
            "tests_exposed_to_model": False,
            "full_gold_targets_exposed_to_model": False,
        },
        "evaluation": {
            "path": str(evaluation_file),
            "sha256": evaluation_sha,
            "ordered_task_ids": list(evaluation_task_ids),
            "ordered_task_ids_sha256": canonical_sha256(evaluation_task_ids),
            "ordered_source_sha256s_sha256": bgen["source_sha256s_sha256"],
        },
        "baseline_seal": {
            "path": str(baseline_seal),
            "sha256": sha256_file(baseline_seal),
            "schema": seal["schema"],
        },
        "arms": {
            "typed_contract_sft_baseline": _arm_report(
                baseline, baseline_checkpoint
            ),
            "typed_direct_rs_sft_update58": _arm_report(
                update, update_checkpoint
            ),
        },
        "paired": {
            "full175": _comparison_block(
                task_order=bgen["task_ids"],
                baseline_by_task=bscore["by_task"],
                update_by_task=uscore["by_task"],
            ),
            "clean174": _comparison_block(
                task_order=clean_order,
                baseline_by_task=baseline["clean"]["by_task"],
                update_by_task=update["clean"]["by_task"],
            ),
        },
    }
    require_exact_or_write(output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--baseline-full-score", type=Path, required=True)
    parser.add_argument("--baseline-clean-score", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-contract", type=Path, required=True)
    parser.add_argument("--baseline-training-result", type=Path, required=True)
    parser.add_argument("--baseline-seal", type=Path, required=True)
    parser.add_argument("--update-predictions", type=Path, required=True)
    parser.add_argument("--update-full-score", type=Path, required=True)
    parser.add_argument("--update-clean-score", type=Path, required=True)
    parser.add_argument("--update-checkpoint-contract", type=Path, required=True)
    parser.add_argument("--update-training-result", type=Path, required=True)
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument(
        "--evaluator-file",
        type=Path,
        default=Path(__file__).with_name("graph_compile_at_k_antigravity.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(
            baseline_predictions=args.baseline_predictions,
            baseline_full_score=args.baseline_full_score,
            baseline_clean_score=args.baseline_clean_score,
            baseline_checkpoint_contract=args.baseline_checkpoint_contract,
            baseline_training_result=args.baseline_training_result,
            baseline_seal=args.baseline_seal,
            update_predictions=args.update_predictions,
            update_full_score=args.update_full_score,
            update_clean_score=args.update_clean_score,
            update_checkpoint_contract=args.update_checkpoint_contract,
            update_training_result=args.update_training_result,
            evaluation_file=args.evaluation_file,
            evaluator_file=args.evaluator_file,
            output=args.output,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_RS_SFT_MATCHED_AUDIT_FAILED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_RS_SFT_MATCHED_AUDIT_PASS "
        + json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "full_pass_at_10_delta": report["paired"]["full175"]["metrics"]
                ["pass_at_10"]["absolute_count_delta"],
                "clean_pass_at_10_delta": report["paired"]["clean174"]["metrics"]
                ["pass_at_10"]["absolute_count_delta"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
