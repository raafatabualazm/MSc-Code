#!/usr/bin/env python3
"""Validate and report the two-seed opaque typed-contract-only control."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis_contract_only_control import contract_only_inference as inference
from analysis_contract_only_control import contract_only_view as view_builder
from analysis_contract_only_control import handoff_attestation as handoff_record
from analysis_contract_only_control.score_contract_only import PROVENANCE_SCHEMA
from analysis_contract_only_control.verify_smoke_replay import build_gate
from scripts.evaluation import score_direct_compact_passk as stock_scorer
from scripts.evaluation import t5gemma2_f2_passk_inference as base_inference
from scripts.evaluation import t5gemma2_measurement_audit_report as audit_report
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)


SCHEMA = "t5gemma2-contract-only-two-seed-report-v1"
SEEDS = (42, 43)
CURRENT_BASE_INFERENCE_SHA256 = (
    "30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d"
)
HISTORICAL_EVALUATOR_SHA256 = (
    "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
)
CURRENT_EVALUATOR_SHA256 = (
    "5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d"
)
EXPECTED_CURRENT_REPORT_SCHEMA = "t5gemma2-f2-intervention-multiseed-report-v1"
EXPECTED_RUNTIME_SCHEMA = "t5gemma2-measurement-runtime-compat-v1"
EXPECTED_PREREG_SCHEMA = "t5gemma2-contract-only-preregistration-v1"
EXPECTED_PREDECESSOR_REPORTER_SHA256 = (
    "89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "17645716115052bb48a906a4c7231c76ec28a9d7b66dc55bc53e669e990bee63"
)


def _read(path: str | Path, label: str) -> Any:
    resolved = Path(path).expanduser().resolve()
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {resolved}") from exc


def _parse_spec(value: str, label: str) -> tuple[int, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"{label} spec must be seed|predictions|score")
    seed = int(parts[0])
    return (
        seed,
        Path(parts[1]).expanduser().resolve(),
        Path(parts[2]).expanduser().resolve(),
    )


def _content_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256")
    }


def _validate_score_journal(
    *, arm: Mapping[str, Any], evaluation_path: Path, expected_k: int
) -> None:
    """Validate score chain/content without requiring one evaluator revision."""

    score_path = Path(arm["score_path"])
    predictions_path = Path(arm["predictions_path"])
    provenance_path = Path(str(predictions_path) + ".provenance.json")
    journal_path = Path(str(score_path) + ".evaluation.journal.jsonl")
    events = load_journal(journal_path)
    score = arm["score"]
    if not events:
        raise ValueError(f"empty score journal: {journal_path}")
    header = events[0]
    complete = events[-1]
    contract = header.get("contract")
    candidate_results = score.get("candidate_results") or []
    expected_evaluator = score.get("evaluator", {}).get("sha256")
    if (
        header.get("event") != "score_header"
        or not isinstance(contract, Mapping)
        or header.get("contract_sha256") != canonical_sha256(contract)
        or complete.get("event") != "score_complete"
        or contract.get("predictions_sha256") != sha256_file(predictions_path)
        or contract.get("prediction_provenance_sha256") != sha256_file(provenance_path)
        or contract.get("evaluation_sha256") != sha256_file(evaluation_path)
        or contract.get("evaluator_sha256") != expected_evaluator
        or int(contract.get("k", -1)) != expected_k
        or int(contract.get("timeout", -1)) != 30
        or int(contract.get("stability_runs", -1)) != 2
        or int(contract.get("slots", -1)) != len(candidate_results)
        or complete.get("candidate_results_canonical_sha256")
        != canonical_sha256(candidate_results)
        or int(complete.get("slots", -1)) != len(candidate_results)
        or _content_record(score.get("evaluation_journal") or {})
        != _content_record(journal_record(journal_path))
        or expected_evaluator
        not in {HISTORICAL_EVALUATOR_SHA256, CURRENT_EVALUATOR_SHA256}
    ):
        raise ValueError(f"score hash-chain contract failed: {score_path}")
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidate_results:
        by_task.setdefault(str(row.get("task_id") or ""), []).append(row)
    recomputed: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda row: int(row.get("sample_index", -1)))
        if [row.get("sample_index") for row in rows] != list(range(expected_k)):
            raise ValueError(f"{score_path}: candidate sample coverage differs")
        recomputed.append(
            {
                "task_id": task_id,
                "pass_at_1": bool(rows[0].get("passed")),
                "pass_at_k": any(bool(row.get("passed")) for row in rows),
                "compile_at_k": any(bool(row.get("compiled")) for row in rows),
                "passing_samples": sum(bool(row.get("passed")) for row in rows),
                "compiling_samples": sum(bool(row.get("compiled")) for row in rows),
            }
        )
    expected_task_results = sorted(
        score.get("task_results") or [], key=lambda row: str(row.get("task_id") or "")
    )
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in recomputed),
        "pass_at_k": sum(row["pass_at_k"] for row in recomputed),
        "compile_at_k": sum(row["compile_at_k"] for row in recomputed),
    }
    if recomputed != expected_task_results or any(
        score.get(metric, {}).get("count") != count
        for metric, count in counts.items()
    ):
        raise ValueError(f"score aggregates are inconsistent: {score_path}")
    task_count = len(recomputed)
    if int(score.get("tasks", -1)) != task_count or any(
        not math.isclose(
            float(score.get(metric, {}).get("rate", -1.0)),
            count / task_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for metric, count in counts.items()
    ):
        raise ValueError(f"score global rates are inconsistent: {score_path}")


def _model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "name",
            "revision",
            "config_sha256",
            "arm",
            "tokenizer_sha256",
            "warmstart_contract_sha256",
            "adapter",
        )
    }


def _metric(score: Mapping[str, Any], name: str) -> dict[str, Any]:
    return audit_report._metric(score, name)  # noqa: SLF001


def _distinct(score: Mapping[str, Any], k: int) -> dict[str, Any]:
    per_task: dict[str, list[str]] = {}
    for row in score.get("candidate_results") or []:
        per_task.setdefault(str(row.get("task_id") or ""), []).append(
            str(row.get("code_sha256") or "")
        )
    if not per_task or any(len(values) != k for values in per_task.values()):
        raise ValueError("candidate diversity input is incomplete")
    values = [len(set(per_task[task])) for task in sorted(per_task)]
    return {
        "mean_distinct_per_k": statistics.mean(values),
        "tasks_below_k": sum(value < k for value in values),
        "histogram": {
            str(key): value for key, value in sorted(Counter(values).items())
        },
    }


def _mcnemar_exact(pair: Mapping[str, Any]) -> dict[str, Any]:
    left = int(pair["left_only"])
    right = int(pair["right_only"])
    discordant = left + right
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left, right) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
    return {**dict(pair), "exact_two_sided_p": p_value}


def _pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    raw = audit_report._paired(left, right)  # noqa: SLF001
    result: dict[str, Any] = {
        "orientation": "left=typed_contract_only,right=comparator"
    }
    result.update(
        {
            metric: _mcnemar_exact(raw[metric])
            for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
        }
    )
    return result


def _arm_summary(arm: Mapping[str, Any], k: int) -> dict[str, Any]:
    return {
        "metrics": {
            name: _metric(arm["score"], name)
            for name in ("pass_at_1", "pass_at_k", "compile_at_k")
        },
        "candidate_level": audit_report._candidate_metrics(arm["score"]),  # noqa: SLF001
        "diversity": _distinct(arm["score"], k),
        "max_token_completions": int(
            arm["provenance"].get("max_token_completions", 0)
        ),
        "predictions_sha256": sha256_file(arm["predictions_path"]),
        "score_sha256": sha256_file(arm["score_path"]),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError("two-seed distribution is incomplete")
    return {
        "n": len(values),
        "values": list(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def _load_specs(
    values: Sequence[str],
    *,
    label: str,
    tasks: int,
    k: int,
    provenance_schema: str,
) -> dict[int, dict[str, Any]]:
    specs = [_parse_spec(value, label) for value in values]
    if tuple(sorted(seed for seed, _, _ in specs)) != SEEDS:
        raise ValueError(f"{label} arms must be exactly seeds 42 and 43")
    return {
        seed: audit_report._load_arm(  # noqa: SLF001
            label=f"{label}_seed{seed}",
            predictions_path=predictions,
            score_path=score,
            expected_tasks=tasks,
            expected_k=k,
            expected_provenance_schema=provenance_schema,
        )
        for seed, predictions, score in specs
    }


def _validate_preregistration(prereg: Mapping[str, Any]) -> None:
    settings = prereg.get("design") or {}
    hypothesis = prereg.get("hypotheses") or {}
    checkpoint = prereg.get("checkpoint") or {}
    registered_view = prereg.get("view") or {}
    sealed_inputs = prereg.get("sealed_inputs") or {}
    sealed_predecessor = prereg.get("sealed_predecessor") or {}
    if (
        prereg.get("schema") != EXPECTED_PREREG_SCHEMA
        or prereg.get("status") != "sealed_before_generation"
        or settings.get("seeds") != [42, 43]
        or settings.get("smoke_tasks") != 5
        or settings.get("tasks_per_full_seed") != 175
        or settings.get("k") != 10
        or settings.get("temperature") != 0.8
        or settings.get("top_p") != 0.95
        or settings.get("max_source_tokens") != 32768
        or settings.get("max_new_tokens") != 4096
        or settings.get("attention_implementation") != "sdpa"
        or settings.get("bf16") is not True
        or settings.get("timeout_seconds") != 30
        or settings.get("stability_runs") != 2
        or hypothesis.get("compile_at_10_strong_pattern_interval") != [160, 172]
        or hypothesis.get("pass_at_10_strong_pattern_interval") != [1, 3]
        or registered_view.get("expected_ordered_task_ids_sha256")
        != "9b93767fd4d0b4057bc752113faeb1efda9faa609e537e189350a6d874d6e38e"
        or registered_view.get("expected_ordered_source_sha256s_sha256")
        != "5da3f58c3d9d2c936fd5c02dbb54618a36aed493daa52315098b2e461f39708f"
        or registered_view.get("expected_row_transformations_sha256")
        != "b563744ca311992983d8c244a41c50fde38befbf6c09f0e8f8cd19fea30d719c"
        or prereg.get("oracle_caveat") != view_builder.ORACLE_CAVEAT
        or prereg.get("out_of_distribution_caveat") != view_builder.OOD_CAVEAT
        or checkpoint.get("name") != inference.EXPECTED_CHECKPOINT_NAME
        or checkpoint.get("file_sha256")
        != inference.EXPECTED_CHECKPOINT_FILE_SHA256
        or sealed_inputs.get("dataset_sha256")
        != base_inference.DATASET_SHA256
        or sealed_inputs.get("dataset_seal_sha256")
        != base_inference.DATASET_SEAL_SHA256
        or sealed_inputs.get("f2_sha256") != base_inference.F2_SHA256
        or sealed_inputs.get("f2_manifest_sha256")
        != base_inference.F2_MANIFEST_SHA256
        or sealed_predecessor.get("report_schema")
        != EXPECTED_CURRENT_REPORT_SCHEMA
        or sealed_predecessor.get("report_sha256")
        != EXPECTED_PREDECESSOR_REPORT_SHA256
        or sealed_predecessor.get("reporter_sha256")
        != EXPECTED_PREDECESSOR_REPORTER_SHA256
        or sealed_predecessor.get(
            "model_visible_source_bytes_identical_across_seeds"
        )
        is not True
        or sealed_predecessor.get("full_input_view_records_identical_across_seeds")
        is not False
        or sealed_predecessor.get("only_permitted_metadata_drift")
        != "row_transformations_sha256"
    ):
        raise ValueError("pre-result preregistration contract failed")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    full_evaluation = Path(args.evaluation).expanduser().resolve()
    baselines = _load_specs(
        args.baseline,
        label="baseline",
        tasks=args.expected_tasks,
        k=args.k,
        provenance_schema=audit_report.BASE_PROVENANCE_SCHEMA,
    )
    typed = _load_specs(
        args.typed,
        label="typed_f2",
        tasks=args.expected_tasks,
        k=args.k,
        provenance_schema=audit_report.ABLATION_PROVENANCE_SCHEMA,
    )
    control = _load_specs(
        args.control,
        label="typed_contract_only",
        tasks=args.expected_tasks,
        k=args.k,
        provenance_schema=PROVENANCE_SCHEMA,
    )
    for family in (baselines, typed, control):
        for arm in family.values():
            _validate_score_journal(
                arm=arm, evaluation_path=full_evaluation, expected_k=args.k
            )

    dataset_rows = view_builder._read_jsonl(args.evaluation, "held-out dataset")  # noqa: SLF001
    f2_rows = view_builder._read_jsonl(args.f2_jsonl, "held-out F2")  # noqa: SLF001
    rebuilt_sources, rebuilt_view = view_builder.build_input_view(
        dataset_rows=dataset_rows, f2_rows=f2_rows
    )
    rebuilt_ids = [str(row.get("task_id") or "") for row in dataset_rows]
    rebuilt_source_hashes = [view_builder._sha256_text(source) for source in rebuilt_sources]  # noqa: SLF001
    if (
        len(rebuilt_sources) != args.expected_tasks
        or any(
            view_builder._extract_binary_payload(source) != ""  # noqa: SLF001
            for source in rebuilt_sources
        )
    ):
        raise ValueError("rebuilt contract-only view is not exactly empty")

    reference_model = canonical_sha256(
        _model_identity(baselines[42]["provenance"]["model"])
    )
    expected_inference_sha = sha256_file(Path(inference.__file__).resolve())
    expected_builder_sha = sha256_file(Path(view_builder.__file__).resolve())
    score_contract_reference = (
        baselines[42]["score"]["evaluation"]["sha256"],
        baselines[42]["score"]["k"],
        baselines[42]["score"]["timeout"],
        baselines[42]["score"]["stability_runs"],
    )
    for seed in SEEDS:
        baseline = baselines[seed]
        typed_arm = typed[seed]
        control_arm = control[seed]
        generation_contract = control_arm["journal"][0].get("contract") or {}
        provenance = control_arm["provenance"]
        terminals = control_arm["journal"][1:-1]
        control_score_contract = (
            control_arm["score"]["evaluation"]["sha256"],
            control_arm["score"]["k"],
            control_arm["score"]["timeout"],
            control_arm["score"]["stability_runs"],
        )
        if (
            baseline["task_ids"] != rebuilt_ids
            or typed_arm["task_ids"] != rebuilt_ids
            or control_arm["task_ids"] != rebuilt_ids
            or canonical_sha256(_model_identity(baseline["provenance"]["model"]))
            != reference_model
            or canonical_sha256(_model_identity(typed_arm["provenance"]["model"]))
            != reference_model
            or canonical_sha256(_model_identity(provenance["model"]))
            != reference_model
            or provenance.get("sampling") != baseline["provenance"].get("sampling")
            or provenance.get("sampling") != typed_arm["provenance"].get("sampling")
            or control_arm["coordinates"] != baseline["coordinates"]
            or control_arm["coordinates"] != typed_arm["coordinates"]
            or control_score_contract != score_contract_reference
            or provenance.get("input_view") != view_builder.VIEW
            or provenance.get("heldout", {}).get("input_view") != rebuilt_view
            or provenance.get("heldout", {}).get("model_visible_fields")
            != [
                "gold_derived_opaque_types_and_arity",
                "task_invariant_empty_binary_payload",
            ]
            or provenance.get("f2_exposed_to_model") is not False
            or provenance.get("gold_derived_oracle_control") is not True
            or provenance.get("oracle_caveat") != view_builder.ORACLE_CAVEAT
            or provenance.get("out_of_distribution_caveat")
            != view_builder.OOD_CAVEAT
            or generation_contract.get("script_sha256") != expected_inference_sha
            or generation_contract.get("input_builder_script_sha256")
            != expected_builder_sha
            or generation_contract.get("base_inference_script_sha256")
            != CURRENT_BASE_INFERENCE_SHA256
            or generation_contract.get("source_truncation") is not False
            or generation_contract.get("runtime", {}).get("bf16") is not True
            or generation_contract.get("runtime", {}).get("attn_implementation")
            != "sdpa"
            or generation_contract.get("f2_exposed_to_model") is not False
            or generation_contract.get("no_training_or_checkpoint_write") is not True
            or [terminal.get("task_id") for terminal in terminals] != rebuilt_ids
            or [terminal.get("source_sha256") for terminal in terminals]
            != rebuilt_source_hashes
        ):
            raise ValueError(f"seed {seed}: pairing/model/privacy/source contract failed")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for relative, expected in inference.EXPECTED_CHECKPOINT_FILE_SHA256.items():
        if sha256_file(checkpoint / relative) != expected:
            raise ValueError(f"frozen checkpoint differs: {relative}")

    gold_path = Path(args.gold_score).expanduser().resolve()
    gold = _read(gold_path, "Rank-0 gold score")
    if (
        gold.get("schema") != audit_report.SCORE_SCHEMA
        or gold.get("tasks") != args.expected_tasks
        or gold.get("k") != 1
        or gold.get("pass_at_1", {}).get("count") != args.expected_tasks
        or gold.get("pass_at_k", {}).get("count") != args.expected_tasks
        or gold.get("compile_at_k", {}).get("count") != args.expected_tasks
    ):
        raise ValueError("Rank-0 gold round-trip failed")

    runtime_path = Path(args.runtime_compatibility).expanduser().resolve()
    runtime = _read(runtime_path, "runtime compatibility")
    if (
        runtime.get("schema") != EXPECTED_RUNTIME_SCHEMA
        or runtime.get("status") != "pass"
        or runtime.get("current_generation_replay", {}).get(
            "exact_prefix_reproduction"
        )
        is not True
        or runtime.get("current_generation_replay", {}).get(
            "model_identity_projection_identical"
        )
        is not True
        or runtime.get("current_scoring_replay", {}).get(
            "candidate_compile_pass_decisions_identical"
        )
        is not True
        or runtime.get("current_scoring_replay", {}).get("task_metrics_identical")
        is not True
    ):
        raise ValueError("historical/current runtime compatibility gate failed")

    current_path = Path(args.current_multiseed_report).expanduser().resolve()
    current = _read(current_path, "completed intervention multiseed report")
    typed_current = (current.get("interventions") or {}).get("typed_opaque_contract")
    if (
        current.get("schema") != EXPECTED_CURRENT_REPORT_SCHEMA
        or current.get("status") != "complete"
        or sha256_file(current_path) != EXPECTED_PREDECESSOR_REPORT_SHA256
        or current.get("script_sha256") != EXPECTED_PREDECESSOR_REPORTER_SHA256
        or current.get("design", {}).get("seeds") != [42, 43, 44, 45, 46]
        or current.get("design", {}).get("tasks_per_run") != args.expected_tasks
        or current.get("design", {}).get("k") != args.k
        or current.get("rank0_gold_roundtrip", {}).get("passed")
        != args.expected_tasks
        or current.get("model_visible_source_bytes_identical_across_seeds")
        is not True
        or current.get("full_input_view_records_identical_across_seeds")
        is not False
        or current.get("allowed_input_view_metadata_drift", {}).get("field")
        != "row_transformations_sha256"
        or current.get("allowed_input_view_metadata_drift", {}).get(
            "full_record_identity_not_claimed"
        )
        is not True
        or not isinstance(typed_current, Mapping)
    ):
        raise ValueError("predecessor multiseed report is not complete")
    for seed in SEEDS:
        current_seed = (typed_current.get("seeds") or {}).get(str(seed)) or {}
        if (
            current_seed.get("predictions_sha256")
            != sha256_file(typed[seed]["predictions_path"])
            or current_seed.get("score_sha256")
            != sha256_file(typed[seed]["score_path"])
        ):
            raise ValueError(f"typed+F2 seed {seed} differs from predecessor report")

    prereg_path = Path(args.preregistration).expanduser().resolve()
    prereg = _read(prereg_path, "preregistration")
    _validate_preregistration(prereg)
    sealed_inputs = prereg["sealed_inputs"]
    dataset_seal = Path(args.dataset_seal).expanduser().resolve()
    f2_manifest = Path(args.f2_manifest).expanduser().resolve()
    if (
        sha256_file(full_evaluation) != sealed_inputs["dataset_sha256"]
        or sha256_file(dataset_seal) != sealed_inputs["dataset_seal_sha256"]
        or sha256_file(Path(args.f2_jsonl).expanduser().resolve())
        != sealed_inputs["f2_sha256"]
        or sha256_file(f2_manifest) != sealed_inputs["f2_manifest_sha256"]
    ):
        raise ValueError("evaluation/F2 artifacts differ from preregistration")
    registered_view = prereg["view"]
    if (
        rebuilt_view.get("ordered_task_ids_sha256")
        != registered_view["expected_ordered_task_ids_sha256"]
        or rebuilt_view.get("ordered_source_sha256s_sha256")
        != registered_view["expected_ordered_source_sha256s_sha256"]
        or rebuilt_view.get("row_transformations_sha256")
        != registered_view["expected_row_transformations_sha256"]
        or rebuilt_view.get("summary", {}).get("arity_histogram")
        != registered_view["expected_arity_histogram"]
    ):
        raise ValueError("rebuilt input view differs from preregistered bytes")

    handoff_path = Path(args.handoff_attestation).expanduser().resolve()
    handoff = _read(handoff_path, "exact-EXITED handoff attestation")
    if (
        handoff.get("schema") != handoff_record.SCHEMA
        or handoff.get("status") != "pass"
        or handoff.get("script_sha256")
        != sha256_file(Path(handoff_record.__file__).resolve())
        or handoff.get("upstream_supervisor", {}).get("observed_state") != "EXITED"
        or handoff.get("predecessor_report", {}).get("sha256")
        != EXPECTED_PREDECESSOR_REPORT_SHA256
        or handoff.get("predecessor_report", {}).get("reporter_sha256")
        != EXPECTED_PREDECESSOR_REPORTER_SHA256
        or handoff.get("predecessor_report", {}).get(
            "recomputed_immediately_before_attestation"
        )
        is not True
        or handoff.get("predecessor_report", {}).get("stable_hash_gate_passed")
        is not True
        or handoff.get("resource_gates", {}).get("gpu_compute_processes_empty")
        is not True
        or handoff.get("downstream_start_authorized") is not True
    ):
        raise ValueError("exact-EXITED handoff attestation contract failed")

    smoke_args = argparse.Namespace(
        smoke_predictions=args.smoke_predictions,
        smoke_score=args.smoke_score,
        smoke_evaluation=args.smoke_evaluation,
        full_predictions=str(control[42]["predictions_path"]),
        full_score=str(control[42]["score_path"]),
        full_evaluation=args.evaluation,
        smoke_tasks=5,
        full_tasks=args.expected_tasks,
        k=args.k,
        timeout=30,
        stability_runs=2,
    )
    recomputed_smoke_gate = build_gate(smoke_args)
    smoke_gate_path = Path(args.smoke_gate).expanduser().resolve()
    if _read(smoke_gate_path, "smoke replay gate") != recomputed_smoke_gate:
        raise ValueError("published smoke replay gate differs from recomputation")

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        control_summary = _arm_summary(control[seed], args.k)
        control_summary["paired_vs_same_seed_baseline"] = _pair(
            control[seed]["score"], baselines[seed]["score"]
        )
        control_summary["paired_vs_same_seed_typed_plus_f2"] = _pair(
            control[seed]["score"], typed[seed]["score"]
        )
        per_seed[seed] = control_summary

    compile_counts = [
        per_seed[seed]["metrics"]["compile_at_k"]["count"] for seed in SEEDS
    ]
    pass_counts = [
        per_seed[seed]["metrics"]["pass_at_k"]["count"] for seed in SEEDS
    ]
    strong_pattern = all(value >= 160 for value in compile_counts) and all(
        value <= 3 for value in pass_counts
    )
    typed_correctness_advantage = all(
        typed[seed]["score"]["pass_at_k"]["count"]
        > control[seed]["score"]["pass_at_k"]["count"]
        for seed in SEEDS
    )
    typed_mcnemar_significant = all(
        per_seed[seed]["paired_vs_same_seed_typed_plus_f2"]["pass_at_k"][
            "exact_two_sided_p"
        ]
        <= 0.05
        for seed in SEEDS
    )
    if strong_pattern and typed_correctness_advantage and typed_mcnemar_significant:
        interpretation_class = "gated_high_compile_low_pass_dissociation"
        defensible_claim = (
            "Task-specific F2 content contributes correctness beyond an oracle "
            "type/arity contract under this frozen policy."
        )
    elif any(value < 160 for value in compile_counts):
        interpretation_class = "missing_input_ood_confounded"
        defensible_claim = (
            "The empty-input intervention also disrupted compilation, so missing-"
            "input/OOD effects confound channel-specific interpretation."
        )
    elif any(value > 3 for value in pass_counts):
        interpretation_class = "substantial_type_or_benchmark_prior_performance"
        defensible_claim = (
            "Contract-only correctness exceeded the preregistered floor in at "
            "least one seed; re-interpret the typed-contract result for substantial "
            "type or benchmark-prior performance before submission."
        )
    else:
        interpretation_class = "preregistered_dissociation_gate_not_met"
        defensible_claim = (
            "The preregistered strong-pattern gate was not met; report the paired "
            "effects without a channel-specific headline claim."
        )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "design": {
            "fresh_full_seeds": list(SEEDS),
            "fresh_full_seed_count": len(SEEDS),
            "tasks_per_full_seed": args.expected_tasks,
            "k": args.k,
            "smoke_tasks": 5,
            "smoke_candidate_slots": 50,
            "checkpoint": "original_enriched_sft_optstep348",
            "same_seed_sample_coordinates_paired": True,
            "input_view_identical_across_seeds": True,
            "tests_exposed_to_model": False,
            "f2_exposed_to_model": False,
            "gold_derived_oracle_control": True,
            "no_training_or_checkpoint_write": True,
        },
        "limitations": {
            "oracle_caveat": view_builder.ORACLE_CAVEAT,
            "out_of_distribution_caveat": view_builder.OOD_CAVEAT,
            "semantic_decoding_proven_by_this_control_alone": False,
            "n_two_uncertainty": (
                "Only two fresh full seeds are reported; sample SD is descriptive "
                "and no cross-seed equivalence claim is made."
            ),
        },
        "dependencies": {
            "preregistration": {
                "path": str(prereg_path),
                "sha256": sha256_file(prereg_path),
            },
            "predecessor_multiseed_report": {
                "path": str(current_path),
                "sha256": sha256_file(current_path),
            },
            "runtime_compatibility": {
                "path": str(runtime_path),
                "sha256": sha256_file(runtime_path),
            },
            "smoke_replay_gate": {
                "path": str(smoke_gate_path),
                "sha256": sha256_file(smoke_gate_path),
            },
            "exact_exited_handoff_attestation": {
                "path": str(handoff_path),
                "sha256": sha256_file(handoff_path),
            },
            "input_view_record_sha256": canonical_sha256(rebuilt_view),
            "sealed_inputs": {
                "dataset_sha256": sha256_file(full_evaluation),
                "dataset_seal_sha256": sha256_file(dataset_seal),
                "f2_sha256": sha256_file(Path(args.f2_jsonl).expanduser().resolve()),
                "f2_manifest_sha256": sha256_file(f2_manifest),
            },
            "inference_script_sha256": expected_inference_sha,
            "input_builder_script_sha256": expected_builder_sha,
            "scoring_admission_wrapper_sha256": sha256_file(
                Path(__file__).resolve().with_name("score_contract_only.py")
            ),
            "unchanged_stock_scorer_sha256": sha256_file(
                Path(stock_scorer.__file__).resolve()
            ),
            "unchanged_base_inference_sha256": sha256_file(
                Path(base_inference.__file__).resolve()
            ),
        },
        "rank0_gold_roundtrip": {
            "path": str(gold_path),
            "sha256": sha256_file(gold_path),
            "passed": args.expected_tasks,
            "tasks": args.expected_tasks,
        },
        "checkpoint_files": {
            relative: sha256_file(checkpoint / relative)
            for relative in inference.EXPECTED_CHECKPOINT_FILE_SHA256
        },
        "baseline": {
            str(seed): _arm_summary(baselines[seed], args.k) for seed in SEEDS
        },
        "typed_plus_f2": {
            str(seed): _arm_summary(typed[seed], args.k) for seed in SEEDS
        },
        "typed_contract_only": {
            "seeds": {str(seed): per_seed[seed] for seed in SEEDS},
            "count_distributions": {
                metric: _distribution(
                    [
                        per_seed[seed]["metrics"][metric]["count"]
                        for seed in SEEDS
                    ]
                )
                for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
            },
            "distinct_per_k_distribution": _distribution(
                [
                    per_seed[seed]["diversity"]["mean_distinct_per_k"]
                    for seed in SEEDS
                ]
            ),
        },
        "preregistered_interpretation_gate": {
            "compile_at_10_at_least_160_each_seed": all(
                value >= 160 for value in compile_counts
            ),
            "pass_at_10_at_most_3_each_seed": all(
                value <= 3 for value in pass_counts
            ),
            "strong_high_compile_low_pass_pattern": strong_pattern,
            "typed_plus_f2_correctness_advantage_each_seed": typed_correctness_advantage,
            "typed_plus_f2_pass_at_10_mcnemar_p_le_0_05_each_seed": (
                typed_mcnemar_significant
            ),
            "interpretation_class": interpretation_class,
            "defensible_claim": defensible_claim,
            "compile_equivalence_claimed": False,
            "semantic_decoding_claimed": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--typed", action="append", required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--dataset_seal", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--f2_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gold_score", required=True)
    parser.add_argument("--runtime_compatibility", required=True)
    parser.add_argument("--current_multiseed_report", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--handoff_attestation", required=True)
    parser.add_argument("--smoke_predictions", required=True)
    parser.add_argument("--smoke_score", required=True)
    parser.add_argument("--smoke_evaluation", required=True)
    parser.add_argument("--smoke_gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_tasks", type=int, default=175)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.expected_tasks <= 5 or args.k <= 0:
        parser.error("expected_tasks and k are invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
