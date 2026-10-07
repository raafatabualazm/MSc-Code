#!/usr/bin/env python3
"""Seal a matched four-arm heldout175 evaluation after training is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from models.direct_compact_causal import sha256_artifact, sha256_file
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
)
from scripts.preprocessing.build_multifunction_executable_view import (
    EXPECTED_HELDOUT_ROWS,
    stable_sha256,
)


SCHEMA = "post-qwen-heldout175-evaluation-suite-v1"
CHAIN_SCHEMA = "post-qwen-predeclared-training-chain-v1"
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
INFERENCE_SCHEMA = "direct-compact-inference-v1"
EVAL_VIEW_SCHEMA = "direct-compact-eval-views-v1"
PUBLIC_FIELDS = (
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
)


def load_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{resolved}: expected one JSON object")
    return value


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def completed_evaluation_journal(
    record: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    if kind == "generation":
        (
            header,
            started,
            retry,
            terminal,
            complete,
            zero_field,
            retry_count_field,
            retry_slots_field,
        ) = (
            "inference_header",
            "inference_batch_started",
            "inference_batch_orphan_retry",
            "inference_batch_terminal",
            "inference_complete",
            "resampled_slots",
            "orphan_retry_events",
            "orphan_recomputed_slots",
        )
    elif kind == "score":
        (
            header,
            started,
            retry,
            terminal,
            complete,
            zero_field,
            retry_count_field,
            retry_slots_field,
        ) = (
            "score_header",
            "score_batch_started",
            "score_batch_orphan_retry",
            "score_batch_terminal",
            "score_complete",
            "rerun_slots",
            "orphan_retry_events",
            "orphan_rerun_slots",
        )
    else:
        raise ValueError(f"unknown evaluation journal kind {kind!r}")
    path = str(record.get("path") or "")
    observed_record = journal_record(path)
    events = load_journal(path)
    if (
        observed_record != dict(record)
        or len(events) < 2
        or events[0].get("event") != header
        or events[-1].get("event") != complete
        or int(events[-1].get(zero_field, -1)) != 0
    ):
        raise ValueError(f"{kind} journal is not terminal and exact-recovery")
    cursor = 1
    retry_count = 0
    retry_slots = 0
    while cursor < len(events) - 1:
        start_event = events[cursor]
        if start_event.get("event") != started:
            raise ValueError(f"{kind} journal batch start order differs")
        cursor += 1
        retries: list[dict[str, Any]] = []
        previous_attempt_sha256 = start_event.get("journal_event_sha256")
        while (
            cursor < len(events) - 1
            and events[cursor].get("event") == retry
        ):
            retry_event = events[cursor]
            sealed_batch = retry_event.get("sealed_batch")
            if (
                retry_event.get("started_event_sha256")
                != start_event.get("journal_event_sha256")
                or retry_event.get("previous_attempt_event_sha256")
                != previous_attempt_sha256
                or int(retry_event.get("retry_index", -1))
                != len(retries) + 1
                or not isinstance(sealed_batch, Mapping)
                or retry_event.get("sealed_batch_sha256")
                != canonical_sha256(sealed_batch)
                or int(retry_event.get("batch_index", -1))
                != int(start_event.get("batch_index", -2))
                or int(sealed_batch.get("batch_index", -1))
                != int(start_event.get("batch_index", -2))
                or sealed_batch.get("slot_ids")
                != start_event.get("slot_ids")
                or (
                    kind == "generation"
                    and (
                        sealed_batch.get("task_ids")
                        != start_event.get("task_ids")
                        or sealed_batch.get("batch_seed")
                        != start_event.get("batch_seed")
                    )
                )
                or (
                    kind == "score"
                    and sealed_batch.get("jobs_canonical_sha256")
                    != start_event.get("jobs_canonical_sha256")
                )
            ):
                raise ValueError(
                    f"{kind} journal orphan-retry binding differs"
                )
            previous_attempt_sha256 = retry_event.get(
                "journal_event_sha256"
            )
            retries.append(retry_event)
            retry_count += 1
            retry_slots += len(sealed_batch.get("slot_ids") or [])
            cursor += 1
        if cursor >= len(events) - 1:
            raise ValueError(
                f"{kind} journal contains an indeterminate started batch"
            )
        terminal_event = events[cursor]
        expected_latest_retry = (
            retries[-1].get("journal_event_sha256") if retries else None
        )
        if (
            terminal_event.get("event") != terminal
            or terminal_event.get("started_event_sha256")
            != start_event.get("journal_event_sha256")
            or int(terminal_event.get("retry_count", 0)) != len(retries)
            or terminal_event.get("latest_retry_event_sha256")
            != expected_latest_retry
        ):
            raise ValueError(
                f"{kind} journal contains an indeterminate started batch"
            )
        cursor += 1
    if (
        cursor != len(events) - 1
        or int(events[-1].get(retry_count_field, -1)) != retry_count
        or int(events[-1].get(retry_slots_field, -1)) != retry_slots
    ):
        raise ValueError(f"{kind} journal recovery accounting differs")
    return observed_record


def read_task_ids(path: Path) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            row = json.loads(line)
            task_id = str(row.get("task_id") or "") if isinstance(row, dict) else ""
            if not task_id or task_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: missing/duplicate task_id"
                )
            seen.add(task_id)
            result.append(task_id)
    if len(result) != EXPECTED_HELDOUT_ROWS:
        raise ValueError(
            f"heldout has {len(result)} rows, expected {EXPECTED_HELDOUT_ROWS}"
        )
    return result


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def checkpoint_record(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    required = {
        "adapter": root / "decoder_adapter",
        "overlay": root / "source_embedding_overlay.pt",
        "contract": root / "compact_contract.json",
        "provenance": root / "run_provenance.json",
    }
    for item in required.values():
        if not item.exists():
            raise FileNotFoundError(item)
    provenance = load_object(required["provenance"])
    adapter_sha = sha256_artifact(required["adapter"])
    overlay_sha = sha256_file(required["overlay"])
    contract_sha = sha256_file(required["contract"])
    if (
        provenance.get("schema") != "direct-compact-run-provenance-v1"
        or provenance.get("architecture")
        != "qwen-causal-compact-tokens-no-encoder"
        or provenance.get("decoder_adapter_sha256") != adapter_sha
        or provenance.get("source_overlay_sha256") != overlay_sha
        or provenance.get("contract_sha256") != contract_sha
    ):
        raise ValueError(f"{root}: checkpoint provenance is stale")
    return {
        "path": str(root),
        "decoder_adapter_sha256": adapter_sha,
        "source_overlay_sha256": overlay_sha,
        "contract_sha256": contract_sha,
        "run_provenance": file_record(required["provenance"]),
        "provenance": provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--chain-contract", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--heldout-seal", required=True)
    parser.add_argument("--eval-views-report", required=True)
    parser.add_argument("--eval-public", required=True)
    parser.add_argument("--eval-alignment", required=True)
    parser.add_argument("--eval-tests", required=True)
    parser.add_argument("--qwen-score", required=True)
    parser.add_argument("--qwen-checkpoint", required=True)
    parser.add_argument("--control-score", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--rs-score", required=True)
    parser.add_argument("--rs-checkpoint", required=True)
    parser.add_argument("--verpo-score", required=True)
    parser.add_argument("--verpo-checkpoint", required=True)
    parser.add_argument("--verpo-completed", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()

    chain_path = Path(args.chain_contract).expanduser().resolve()
    chain = load_object(chain_path)
    payload = chain.get("payload")
    if (
        chain.get("schema") != CHAIN_SCHEMA
        or not isinstance(payload, Mapping)
        or chain.get("payload_sha256") != stable_sha256(payload)
        or payload.get("stage_order_predeclared") is not True
        or payload.get("checkpoint_selection_from_heldout") is not False
        or payload.get("early_stopping_from_heldout") is not False
        or payload.get("launch_decisions_from_heldout") is not False
        or (payload.get("evaluation") or {}).get(
            "after_all_training_stages"
        )
        is not True
    ):
        raise ValueError("predeclared chain contract is invalid")

    heldout = Path(args.heldout).expanduser().resolve()
    heldout_seal_path = Path(args.heldout_seal).expanduser().resolve()
    heldout_record = file_record(heldout)
    heldout_seal_record = file_record(heldout_seal_path)
    expected_eval = payload["evaluation"]
    if (
        (expected_eval.get("heldout") or {}).get("sha256")
        != heldout_record["sha256"]
        or (expected_eval.get("heldout_seal") or {}).get("sha256")
        != heldout_seal_record["sha256"]
        or int(expected_eval.get("heldout_rows", -1))
        != EXPECTED_HELDOUT_ROWS
    ):
        raise ValueError("heldout files differ from the predeclared chain")
    heldout_seal = load_object(heldout_seal_path)
    if (
        heldout_seal.get("selected_role") != "measure"
        or heldout_seal.get("training_allowed") is not False
        or heldout_seal.get("heldout_measure_only") is not True
        or int(heldout_seal.get("rows", -1)) != EXPECTED_HELDOUT_ROWS
        or heldout_seal.get("output_sha256") != heldout_record["sha256"]
    ):
        raise ValueError("heldout175 seal is not measure-only")
    heldout_ids = read_task_ids(heldout)
    heldout_task_sha = stable_sha256(heldout_ids)
    if heldout_task_sha != expected_eval.get("heldout_task_ids_sha256"):
        raise ValueError("heldout175 task order differs from predeclaration")
    heldout_rows = read_jsonl_objects(heldout)
    view_report_path = Path(args.eval_views_report).expanduser().resolve()
    public_path = Path(args.eval_public).expanduser().resolve()
    alignment_path = Path(args.eval_alignment).expanduser().resolve()
    tests_path = Path(args.eval_tests).expanduser().resolve()
    view_report = load_object(view_report_path)
    public_rows = read_jsonl_objects(public_path)
    alignment_rows = read_jsonl_objects(alignment_path)
    tests_rows = read_jsonl_objects(tests_path)
    expected_public = [
        {field: row[field] for field in PUBLIC_FIELDS}
        for row in heldout_rows
    ]
    expected_alignment = [
        {"model_row": index, "role": "measure", "task_id": task_id}
        for index, task_id in enumerate(heldout_ids)
    ]
    expected_tests = [
        {
            "task_id": task_id,
            "tests": str(
                row.get("acceptance_tests")
                or row.get("tests")
                or row.get("feedback_tests")
                or ""
            ),
        }
        for task_id, row in zip(heldout_ids, heldout_rows, strict=True)
    ]
    view_records = {
        "public": file_record(public_path),
        "alignment": file_record(alignment_path),
        "tests": file_record(tests_path),
    }
    if (
        any(not row["tests"] for row in expected_tests)
        or public_rows != expected_public
        or alignment_rows != expected_alignment
        or tests_rows != expected_tests
        or view_report.get("schema") != EVAL_VIEW_SCHEMA
        or view_report.get("role") != "measure"
        or (view_report.get("input") or {}).get("sha256")
        != heldout_record["sha256"]
        or int((view_report.get("input") or {}).get("rows", -1))
        != EXPECTED_HELDOUT_ROWS
        or any(
            (view_report.get("outputs") or {}).get(name, {}).get("sha256")
            != record["sha256"]
            or Path(
                str(
                    (view_report.get("outputs") or {})
                    .get(name, {})
                    .get("path", "")
                )
            ).expanduser().resolve()
            != Path(record["path"])
            for name, record in view_records.items()
        )
        or view_report.get("task_ids_sha256")
        != hashlib.sha256(
            json.dumps(
                sorted(heldout_ids), separators=(",", ":")
            ).encode()
        ).hexdigest()
    ):
        raise ValueError(
            "prepared inference/test views are not the deterministic heldout175 split"
        )

    arms = {
        "qwen_sequence_kd": (
            Path(args.qwen_score).expanduser().resolve(),
            checkpoint_record(args.qwen_checkpoint),
        ),
        "matched_gold_control": (
            Path(args.control_score).expanduser().resolve(),
            checkpoint_record(args.control_checkpoint),
        ),
        "gpt_5_6_sol_rs_sft": (
            Path(args.rs_score).expanduser().resolve(),
            checkpoint_record(args.rs_checkpoint),
        ),
        "sparse_teacher_verpo": (
            Path(args.verpo_score).expanduser().resolve(),
            checkpoint_record(args.verpo_checkpoint),
        ),
    }
    if list(arms) != list(expected_eval.get("arms") or []):
        raise ValueError("evaluation arms differ from predeclaration")

    verpo_completed_path = Path(
        args.verpo_completed
    ).expanduser().resolve()
    verpo_completed = load_object(verpo_completed_path)
    expected_verpo_checkpoint = Path(
        str(verpo_completed.get("latest_checkpoint") or "")
    ).expanduser().resolve()
    declared_verpo = payload.get("verpo") or {}
    declared_verpo_root = Path(
        str(declared_verpo.get("output") or "")
    ).expanduser().resolve()
    verpo_run_contract_path = declared_verpo_root / "run_contract.json"
    verpo_run_contract = load_object(verpo_run_contract_path)
    canonical_verpo_run_contract_sha256 = stable_sha256(verpo_run_contract)
    final_checkpoint_provenance_path = (
        expected_verpo_checkpoint / "checkpoint_provenance.json"
    )
    final_checkpoint_provenance = load_object(
        final_checkpoint_provenance_path
    )
    final_run_provenance = load_object(
        expected_verpo_checkpoint / "run_provenance.json"
    )
    expected_final_step = int(declared_verpo.get("max_updates", -1))
    if (
        verpo_completed.get("schema") != "direct-compact-verpo-completed-v1"
        or expected_verpo_checkpoint
        != Path(args.verpo_checkpoint).expanduser().resolve()
        or expected_verpo_checkpoint
        != declared_verpo_root
        / f"checkpoint-optstep-{expected_final_step:06d}"
        or verpo_completed.get("latest_checkpoint_provenance_sha256")
        != sha256_file(final_checkpoint_provenance_path)
        or int(verpo_completed.get("optimizer_steps", -1))
        != expected_final_step
        or int(verpo_completed.get("checkpoint_interval", -1))
        != int(declared_verpo.get("checkpoint_interval", -2))
        or verpo_completed.get("run_contract_sha256")
        != canonical_verpo_run_contract_sha256
        or final_checkpoint_provenance.get("schema")
        != "direct-compact-verpo-checkpoint-v1"
        or int(final_checkpoint_provenance.get("optimizer_step", -1))
        != expected_final_step
        or final_checkpoint_provenance.get("run_contract_sha256")
        != canonical_verpo_run_contract_sha256
        or int(final_run_provenance.get("optimizer_step", -1))
        != expected_final_step
        or final_run_provenance.get("run_contract_sha256")
        != canonical_verpo_run_contract_sha256
        or verpo_completed.get("rollout_journal_chain_sha256")
        != final_checkpoint_provenance.get(
            "rollout_journal_chain_sha256"
        )
        or verpo_completed.get("latest_step_journal_sha256")
        != final_checkpoint_provenance.get(
            "latest_step_journal_sha256"
        )
    ):
        raise ValueError("VeRPO completion does not bind the evaluated checkpoint")

    common_generation: dict[str, Any] | None = None
    common_score: dict[str, Any] | None = None
    arm_reports: dict[str, Any] = {}
    expected_task_set = set(heldout_ids)
    expected_prompt_modes = expected_eval.get("direct_prompt_modes")
    if expected_prompt_modes != {
        "qwen_sequence_kd": "qwen_cot_v1",
        "matched_gold_control": "code_only_v1",
        "gpt_5_6_sol_rs_sft": "code_only_v1",
        "sparse_teacher_verpo": "code_only_v1",
    }:
        raise ValueError(
            "predeclared per-arm direct prompt modes are missing or invalid"
        )
    for name, (score_path, checkpoint) in arms.items():
        score = load_object(score_path)
        score_journal = score.get("evaluation_journal")
        if (
            score.get("schema") != SCORE_SCHEMA
            or int(score.get("tasks", -1)) != EXPECTED_HELDOUT_ROWS
            or int(score.get("k", -1)) != int(expected_eval["k"])
            or (score.get("evaluation") or {}).get("sha256")
            != view_records["tests"]["sha256"]
            or (score.get("evaluator") or {}).get("completion_attestation")
            != "per-run-256-bit-marker-exactly-once-v1"
            or score.get("started_without_terminal_policy")
            != "retry_identical_sealed_batch_with_hash_chained_receipt"
            or int(score.get("rerun_slots", -1)) != 0
            or int(score.get("orphan_retry_events", -1)) < 0
            or int(score.get("orphan_rerun_slots", -1)) < 0
            or not isinstance(score_journal, Mapping)
            or score_journal
            != completed_evaluation_journal(score_journal, kind="score")
        ):
            raise ValueError(f"{name}: score contract differs from heldout175")
        score_completion = load_journal(str(score_journal["path"]))[-1]
        if (
            int(score.get("orphan_retry_events", -1))
            != int(score_completion.get("orphan_retry_events", -2))
            or int(score.get("orphan_rerun_slots", -1))
            != int(score_completion.get("orphan_rerun_slots", -2))
        ):
            raise ValueError(f"{name}: score recovery accounting differs")
        task_rows = score.get("task_results")
        task_ids = {
            str(row.get("task_id") or "")
            for row in task_rows
            if isinstance(row, Mapping)
        } if isinstance(task_rows, list) else set()
        if len(task_rows or []) != EXPECTED_HELDOUT_ROWS or task_ids != expected_task_set:
            raise ValueError(f"{name}: score task set differs from heldout175")

        prediction_path = Path(
            str((score.get("predictions") or {}).get("path") or "")
        ).expanduser().resolve()
        provenance_path = Path(str(prediction_path) + ".provenance.json")
        prediction_record = file_record(prediction_path)
        provenance_record = file_record(provenance_path)
        provenance = load_object(provenance_path)
        generation_journal = provenance.get("generation_journal")
        if (
            provenance.get("schema") != INFERENCE_SCHEMA
            or provenance.get("selected_role") != "measure"
            or provenance.get("dataset_sha256")
            != view_records["public"]["sha256"]
            or provenance.get("alignment_sha256")
            != view_records["alignment"]["sha256"]
            or int(provenance.get("num_rows", -1)) != EXPECTED_HELDOUT_ROWS
            or int(provenance.get("num_samples", -1))
            != int(expected_eval["k"])
            or provenance.get("direct_prompt_mode")
            != expected_prompt_modes[name]
            or provenance.get("output_sha256") != prediction_record["sha256"]
            or provenance.get("decoder_adapter_sha256")
            != checkpoint["decoder_adapter_sha256"]
            or provenance.get("source_overlay_sha256")
            != checkpoint["source_overlay_sha256"]
            or (score.get("predictions") or {}).get("sha256")
            != prediction_record["sha256"]
            or (score.get("predictions") or {}).get("provenance_sha256")
            != provenance_record["sha256"]
            or provenance.get("started_without_terminal_policy")
            != "retry_identical_seeded_batch_with_hash_chained_receipt"
            or int(provenance.get("resampled_slots", -1)) != 0
            or int(provenance.get("orphan_retry_events", -1)) < 0
            or int(provenance.get("orphan_recomputed_slots", -1)) < 0
            or not isinstance(generation_journal, Mapping)
            or generation_journal
            != completed_evaluation_journal(
                generation_journal, kind="generation"
            )
        ):
            raise ValueError(f"{name}: inference/checkpoint binding failed")
        generation_completion = load_journal(
            str(generation_journal["path"])
        )[-1]
        if (
            int(provenance.get("orphan_retry_events", -1))
            != int(generation_completion.get("orphan_retry_events", -2))
            or int(provenance.get("orphan_recomputed_slots", -1))
            != int(
                generation_completion.get("orphan_recomputed_slots", -2)
            )
        ):
            raise ValueError(
                f"{name}: inference recovery accounting differs"
            )
        generation = {
            field: provenance.get(field)
            for field in (
                "dataset_sha256",
                "alignment_sha256",
                "selected_role",
                "contract_sha256",
                "codebook_sha256",
                "codec_sha256",
                "tokenizer_json_sha256",
                "decoder_model",
                "decoder_revision",
                "model_config_sha256",
                "attn_implementation",
                "num_rows",
                "num_samples",
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "batch_size",
                "limit",
                "seed",
                "bf16",
                "fp16",
            )
        }
        score_settings = {
            field: score.get(field)
            for field in ("k", "timeout", "stability_runs")
        }
        if common_generation is None:
            common_generation = generation
            common_score = score_settings
        elif generation != common_generation or score_settings != common_score:
            raise ValueError(
                f"{name}: generation/scoring differs from the other arms"
            )
        arm_reports[name] = {
            "checkpoint": {
                key: value
                for key, value in checkpoint.items()
                if key != "provenance"
            },
            "score": file_record(score_path),
            "predictions": prediction_record,
            "prediction_provenance": provenance_record,
            "direct_prompt_mode": provenance["direct_prompt_mode"],
            "generation_journal": dict(generation_journal),
            "evaluation_journal": dict(score_journal),
            "pass_at_1": score["pass_at_1"],
            "pass_at_k": score["pass_at_k"],
            "compile_at_k": score["compile_at_k"],
        }

    qwen_checkpoint = arms["qwen_sequence_kd"][1]
    control_checkpoint = arms["matched_gold_control"][1]
    rs_checkpoint = arms["gpt_5_6_sol_rs_sft"][1]
    verpo_checkpoint = arms["sparse_teacher_verpo"][1]
    expected_qwen_path = Path(
        str(
            (payload.get("qwen_stage") or {})
            .get("checkpoint", {})
            .get("path", "")
        )
    ).expanduser().resolve()
    expected_qwen_provenance_sha = (
        ((payload.get("qwen_stage") or {}).get("checkpoint") or {})
        .get("run_provenance", {})
        .get("sha256")
    )
    if (
        Path(qwen_checkpoint["path"]) != expected_qwen_path
        or qwen_checkpoint["run_provenance"]["sha256"]
        != expected_qwen_provenance_sha
    ):
        raise ValueError("evaluated Qwen checkpoint differs from predeclaration")
    rs_decl = payload["rs_sft"]
    if (
        Path(control_checkpoint["path"])
        != Path(rs_decl["matched_control_output"])
        or Path(rs_checkpoint["path"]) != Path(rs_decl["intervention_output"])
    ):
        raise ValueError("evaluated RS/control paths differ from predeclaration")
    rs_root = Path(rs_decl["intervention_output"]).parent
    if Path(rs_decl["matched_control_output"]).parent != rs_root:
        raise ValueError("RS/control outputs do not share their sealed stage root")
    rs_handoff_path = rs_root / "train_side_handoff.json"
    rs_build_path = rs_root / "00_matched_data" / "build_report.json"
    rs_handoff = load_object(rs_handoff_path)
    rs_build = load_object(rs_build_path)
    if (
        rs_handoff.get("schema") != "post-qwen-rs-train-side-handoff-v1"
        or rs_handoff.get("passed") is not True
        or rs_handoff.get("heldout_loaded_during_training") is not False
        or rs_handoff.get("heldout_metrics_used_for_selection") is not False
        or rs_handoff.get("predeclared_chain_sha256")
        != sha256_file(chain_path)
        or rs_handoff.get("matched_data_build_sha256")
        != sha256_file(rs_build_path)
    ):
        raise ValueError("RS train-side handoff is invalid")
    expected_rs_schedule = {
        "learning_rate": rs_decl["learning_rate"],
        "epochs": rs_decl["epochs"],
        "max_steps": rs_decl["max_steps"],
        "batch_size": rs_decl["batch_size"],
        "grad_accum": rs_decl["grad_accum"],
        "seed": rs_decl["seed"],
        "lora_r": rs_decl["lora_r"],
        "lora_alpha": rs_decl["lora_alpha"],
        "lora_dropout": rs_decl["lora_dropout"],
        "load_4bit": rs_decl["load_4bit"],
        "gradient_checkpointing": rs_decl["gradient_checkpointing"],
        "bf16": rs_decl["bf16"],
        "fp16": rs_decl["fp16"],
    }
    rs_build_outputs = rs_build.get("outputs") or {}
    for name, checkpoint in (
        ("control", control_checkpoint),
        ("rs", rs_checkpoint),
    ):
        provenance = checkpoint["provenance"]
        output_key = "control" if name == "control" else "intervention"
        handoff_key = (
            "control_provenance_sha256"
            if name == "control"
            else "rs_provenance_sha256"
        )
        expected_train = rs_build_outputs.get(output_key) or {}
        if (
            provenance.get("heldout_loaded_during_training") is not False
            or (provenance.get("stage_contract") or {}).get("sha256")
            != sha256_file(chain_path)
            or checkpoint["run_provenance"]["sha256"]
            != rs_handoff.get(handoff_key)
            or provenance.get("train_file_sha256")
            != expected_train.get("sha256")
            or provenance.get("train_seal_sha256")
            != expected_train.get("seal_sha256")
            or provenance.get("eval_file_sha256") is not None
            or provenance.get("eval_seal_sha256") is not None
            or provenance.get("eval_strategy") != "no"
            or provenance.get("attn_implementation")
            != rs_decl["attn_implementation"]
            or (provenance.get("loss_contract") or {}).get(
                "sequence_distribution_nll"
            )
            is not False
            or any(
                (provenance.get("training_schedule") or {}).get(key)
                != value
                for key, value in expected_rs_schedule.items()
            )
            or Path(
                str((provenance.get("warmstart_checkpoint") or {}).get("path") or "")
            ).expanduser().resolve()
            != expected_qwen_path
            or (provenance.get("warmstart_checkpoint") or {}).get(
                "decoder_adapter_sha256"
            )
            != qwen_checkpoint["decoder_adapter_sha256"]
            or (provenance.get("warmstart_checkpoint") or {}).get(
                "source_overlay_sha256"
            )
            != qwen_checkpoint["source_overlay_sha256"]
            or (provenance.get("warmstart_checkpoint") or {}).get(
                "provenance_sha256"
            )
            != qwen_checkpoint["run_provenance"]["sha256"]
        ):
            raise ValueError(
                f"{name}: fitting touched heldout or used the wrong Qwen warmstart"
            )
    verpo_provenance = verpo_checkpoint["provenance"]
    verpo_run_contract_record = file_record(verpo_run_contract_path)
    run_generation = verpo_run_contract.get("generation") or {}
    run_verifier = verpo_run_contract.get("verifier") or {}
    run_reward = verpo_run_contract.get("reward") or {}
    run_judge = verpo_run_contract.get("judge") or {}
    run_optimizer = verpo_run_contract.get("optimizer") or {}
    run_runtime = verpo_run_contract.get("runtime") or {}
    run_task_sampling = verpo_run_contract.get("task_sampling") or {}
    declared_generation = {
        key: declared_verpo[key]
        for key in (
            "group_size",
            "rollout_batch_size",
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
        )
    }
    declared_verifier = {
        "workers": declared_verpo["reward_workers"],
        "timeout": declared_verpo["reward_timeout"],
        "stability_runs": declared_verpo["reward_stability_runs"],
    }
    declared_reward = {
        "verpo_alpha": declared_verpo["verpo_alpha"],
        "verpo_beta": declared_verpo["verpo_beta"],
        "kde_bandwidth": "population_std_over_2",
        "global_reward": "verifier_full_suite_binary",
        "local_reward": "density_calibrated_visible_per_test",
        "global_advantage": "global_reward_minus_group_mean",
        "local_advantage": "local_reward_minus_group_mean",
        "advantage_normalization_factor": 1,
        "unified_advantage": (
            "A_global + verpo_beta*A_local + judge_weight*A_teacher"
        ),
        "judge_weight": declared_verpo["judge_weight"],
        "teacher_signal": {
            "observed": "selected_compiling_failure_score_in_[0,1]",
            "unobserved": "missing_with_exact_zero_advantage",
            "advantage": "observed_score_minus_observed_subset_mean",
            "separately_centered_over_observed_subset": True,
            "full_pass_or_non_compiling_teacher_mask": False,
            "paper_extension": True,
        },
        "population_std_advantage_division": False,
    }
    declared_judge = {
        "mode": declared_verpo["judge_mode"],
        "api_style": declared_verpo["judge_api_style"],
        "model": declared_verpo["teacher"],
        "base_url": declared_verpo["judge_base_url"],
        "thinking_mode": declared_verpo["judge_thinking_mode"],
        "reasoning_mode": declared_verpo["judge_reasoning_mode"],
        "reasoning_effort": declared_verpo[
            "judge_reasoning_effort"
        ],
        "max_tokens": declared_verpo["judge_max_tokens"],
        "completion_retries": declared_verpo["judge_completion_retries"],
        "retry_max_tokens": declared_verpo["judge_retry_max_tokens"],
        "timeout_seconds": declared_verpo["judge_timeout_seconds"],
        "max_retries": declared_verpo["judge_max_retries"],
        "concurrency": declared_verpo["judge_concurrency"],
        "interval": declared_verpo["judge_interval"],
        "group_top_n": declared_verpo["judge_group_top_n"],
        "deadline_seconds": declared_verpo["judge_deadline_seconds"],
        "failure_policy": declared_verpo["judge_failure_policy"],
        "max_calls": declared_verpo["judge_max_calls"],
        "escalation_queue": declared_verpo["judge_escalation_queue"],
    }
    declared_optimizer = {
        "learning_rate": declared_verpo["learning_rate"],
        "weight_decay": declared_verpo["weight_decay"],
        "max_grad_norm": declared_verpo["max_grad_norm"],
        "ppo_clip": declared_verpo["ppo_clip"],
        "sft_replay_weight": declared_verpo["sft_replay_weight"],
        "on_policy_logprob_tolerance": declared_verpo[
            "on_policy_logprob_tolerance"
        ],
        "max_updates": declared_verpo["max_updates"],
        "checkpoint_interval": declared_verpo["checkpoint_interval"],
    }
    declared_runtime = {
        "seed": declared_verpo["seed"],
        "attn_implementation": declared_verpo["attn_implementation"],
        "load_4bit": declared_verpo["load_4bit"],
        "bf16": declared_verpo["bf16"],
        "fp16": declared_verpo["fp16"],
        "heldout_loaded_during_training": False,
    }
    if (
        Path(verpo_checkpoint["path"]).parent != declared_verpo_root
        or verpo_completed_path != declared_verpo_root / "completed.json"
        or verpo_provenance.get("stage") != "on-policy-direct-compact-verpo"
        or verpo_provenance.get("heldout_loaded_during_training") is not False
        or verpo_provenance.get("run_contract_sha256")
        != stable_sha256(verpo_run_contract)
        or verpo_run_contract.get("schema") != "direct-compact-verpo-run-v1"
        or (
            verpo_run_contract.get("predeclared_chain_contract") or {}
        ).get("sha256")
        != sha256_file(chain_path)
        or (verpo_run_contract.get("judge") or {}).get("model")
        != declared_verpo.get("teacher")
        or (verpo_run_contract.get("judge") or {}).get("base_url")
        != declared_verpo.get("judge_base_url")
        or (verpo_run_contract.get("judge") or {}).get("reasoning_mode")
        != "standard"
        or (verpo_run_contract.get("rollout_file") or {}).get("sha256")
        != (declared_verpo.get("rollout_dataset") or {}).get("sha256")
        or verpo_run_contract.get("rollout_seal_sha256")
        != (declared_verpo.get("rollout_seal") or {}).get("sha256")
        or (
            verpo_run_contract.get("verpo_feedback_public_manifest") or {}
        ).get("sha256")
        != (
            declared_verpo.get("feedback_public_manifest") or {}
        ).get("sha256")
        or run_task_sampling.get("policy")
        != declared_verpo.get("task_sampling_policy")
        or int(run_task_sampling.get("dataset_rows", -1))
        != int(declared_verpo.get("rollout_rows", -2))
        or int(run_task_sampling.get("planned_rollout_groups", -1))
        != int(declared_verpo.get("planned_rollout_groups", -2))
        or int(run_task_sampling.get("planned_unique_tasks", -1))
        != int(declared_verpo.get("planned_unique_tasks", -2))
        or run_task_sampling.get("planned_unique_fraction")
        != declared_verpo.get("planned_unique_fraction")
        or int(run_task_sampling.get("complete_dataset_cycles", -1))
        != int(declared_verpo.get("complete_dataset_cycles", -2))
        or int(run_task_sampling.get("partial_cycle_groups", -1))
        != int(declared_verpo.get("partial_cycle_groups", -2))
        or run_task_sampling.get("with_replacement_within_cycle") is not False
        or Path(
            str((verpo_run_contract.get("warmstart") or {}).get("path") or "")
        ).expanduser().resolve()
        != Path(rs_checkpoint["path"])
        or any(
            run_generation.get(key) != value
            for key, value in declared_generation.items()
        )
        or any(
            run_verifier.get(key) != value
            for key, value in declared_verifier.items()
        )
        or any(
            run_reward.get(key) != value
            for key, value in declared_reward.items()
        )
        or any(
            run_judge.get(key) != value
            for key, value in declared_judge.items()
        )
        or any(
            run_optimizer.get(key) != value
            for key, value in declared_optimizer.items()
        )
        or any(
            run_runtime.get(key) != value
            for key, value in declared_runtime.items()
        )
        or Path(
            str(
                (verpo_provenance.get("warmstart_checkpoint") or {}).get(
                    "path"
                )
                or ""
            )
        ).expanduser().resolve()
        != Path(rs_checkpoint["path"])
    ):
        raise ValueError(
            "VeRPO fitting touched heldout or used a non-predeclared warmstart"
        )

    assert common_generation is not None and common_score is not None
    expected_common_generation = {
        "decoder_model": expected_eval["decoder_model"],
        "decoder_revision": expected_eval["decoder_revision"],
        "attn_implementation": expected_eval["attn_implementation"],
        "num_rows": EXPECTED_HELDOUT_ROWS,
        "num_samples": expected_eval["num_samples"],
        "max_new_tokens": expected_eval["max_new_tokens"],
        "temperature": expected_eval["temperature"],
        "top_p": expected_eval["top_p"],
        "top_k": expected_eval["top_k"],
        "batch_size": expected_eval["batch_size"],
        "limit": expected_eval["limit"],
        "seed": expected_eval["seed"],
        "bf16": expected_eval["bf16"],
        "fp16": False,
    }
    if any(
        common_generation.get(key) != value
        for key, value in expected_common_generation.items()
    ):
        raise ValueError(
            "matched inference settings differ from the predeclared evaluation"
        )
    expected_common_score = {
        "k": expected_eval["k"],
        "timeout": expected_eval["timeout"],
        "stability_runs": expected_eval["stability_runs"],
    }
    if common_score != expected_common_score:
        raise ValueError(
            "matched scoring settings differ from the predeclared evaluation"
        )
    expected_contract_sha = (
        (payload.get("executable_train") or {}).get("contract") or {}
    ).get("sha256")
    if common_generation.get("contract_sha256") != expected_contract_sha:
        raise ValueError("evaluation compact contract differs from safe1578")

    result = {
        "schema": SCHEMA,
        "report_only": True,
        "used_for_stage_selection_or_launch": False,
        "all_training_complete_before_evaluation": True,
        "chain_contract": file_record(chain_path),
        "heldout": heldout_record,
        "heldout_seal": heldout_seal_record,
        "heldout_rows": EXPECTED_HELDOUT_ROWS,
        "heldout_task_ids_sha256": heldout_task_sha,
        "eval_views_report": file_record(view_report_path),
        "eval_public": view_records["public"],
        "eval_alignment": view_records["alignment"],
        "eval_tests": view_records["tests"],
        "common_generation": common_generation,
        "per_arm_direct_prompt_modes": dict(expected_prompt_modes),
        "common_scoring": common_score,
        "arms": arm_reports,
        "verpo_completed": file_record(verpo_completed_path),
        "verpo_run_contract": verpo_run_contract_record,
        "rs_train_side_handoff": file_record(rs_handoff_path),
        "rs_matched_data_build": file_record(rs_build_path),
        "invariants": {
            "exact_same_175_tasks": True,
            "exact_same_generation_settings_except_checkpoint_prompt_mode": True,
            "checkpoint_conditioned_prompt_modes_predeclared": True,
            "exact_same_scoring_settings": True,
            "measure_only_seal_verified": True,
            "prepared_views_deterministically_rederived_from_heldout": True,
            "no_stage_selected_from_heldout": True,
            "qwen_rs_control_verpo_checkpoint_provenance_verified": True,
            "generation_journals_hash_chained_and_terminal": True,
            "scoring_journals_hash_chained_and_terminal": True,
            "resampled_or_rerun_slots": 0,
        },
    }
    if output.exists():
        observed = load_object(output)
        if observed != result:
            raise ValueError(
                "existing evaluation suite differs from exact sealed inputs"
            )
        print(
            f"POST_QWEN_HELDOUT175_EVALUATION_REUSED output={output}",
            flush=True,
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    require_exact_or_write(output, result)
    print(
        "POST_QWEN_HELDOUT175_EVALUATION "
        + " ".join(
            f"{name}={report['pass_at_k']['count']}/"
            f"{EXPECTED_HELDOUT_ROWS}"
            for name, report in arm_reports.items()
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
