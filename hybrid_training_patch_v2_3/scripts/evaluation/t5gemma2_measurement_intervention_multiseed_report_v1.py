#!/usr/bin/env python3
"""Seal the five-seed x86 input-intervention replication.

Seed 42 is reused from the completed measurement audit.  Seeds 43--46 are
fresh, paired generations for each of the three frozen input views.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation import t5gemma2_measurement_audit_report as base
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


SCHEMA = "t5gemma2-f2-intervention-multiseed-report-v1"
SEEDS = (42, 43, 44, 45, 46)
VIEWS = (
    "typed_opaque_contract",
    "constants_stripped",
    "semantic_body_swap",
)
WRAPPER_SHA256 = "27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60"
HISTORICAL_BASE_INFERENCE_SHA256 = (
    "564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9"
)
CURRENT_BASE_INFERENCE_SHA256 = (
    "30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d"
)
HISTORICAL_EVALUATOR_SHA256 = (
    "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
)
CURRENT_EVALUATOR_SHA256 = (
    "5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d"
)
INPUT_VIEW_METADATA_DIGEST_FIELD = "row_transformations_sha256"
INPUT_VIEW_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "view",
        "rows",
        "ordered_task_ids_sha256",
        "ordered_source_sha256s_sha256",
        INPUT_VIEW_METADATA_DIGEST_FIELD,
        "tests_exposed_to_model",
        "full_gold_targets_exposed_to_model",
        "summary",
    }
)


def _read(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - detail belongs in the error
        raise ValueError(f"could not read {label}: {path}") from exc


def _parse_baseline(value: str) -> tuple[int, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("baseline spec must be seed|predictions|score")
    return int(parts[0]), Path(parts[1]).expanduser().resolve(), Path(parts[2]).expanduser().resolve()


def _parse_arm(value: str) -> tuple[str, int, Path, Path]:
    parts = value.split("|", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError("arm spec must be view|seed|predictions|score")
    return (
        parts[0],
        int(parts[1]),
        Path(parts[2]).expanduser().resolve(),
        Path(parts[3]).expanduser().resolve(),
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError("five-seed distribution is incomplete")
    return {
        "values": list(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_sd": statistics.stdev(values),
    }


def _distinct(score: Mapping[str, Any]) -> dict[str, Any]:
    per_task: dict[str, list[str]] = {}
    for row in score.get("candidate_results") or []:
        per_task.setdefault(str(row["task_id"]), []).append(str(row["code_sha256"]))
    if len(per_task) == 0 or any(len(values) != 10 for values in per_task.values()):
        raise ValueError("distinct-candidate input is incomplete")
    values = [len(set(per_task[task_id])) for task_id in sorted(per_task)]
    return {
        "mean_distinct_per_10": statistics.mean(values),
        "tasks_below_10": sum(value < 10 for value in values),
        "histogram": {str(key): value for key, value in sorted(Counter(values).items())},
    }


def _mcnemar_exact(pair: Mapping[str, Any]) -> dict[str, Any]:
    left = int(pair["left_only"])
    right = int(pair["right_only"])
    discordant = left + right
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left, right) + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
    return {**dict(pair), "exact_two_sided_p": p_value}


def _arm_summary(arm: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        name: base._metric(arm["score"], name)  # noqa: SLF001
        for name in ("pass_at_1", "pass_at_k", "compile_at_k")
    }
    return {
        "metrics": metrics,
        "candidate_level": base._candidate_metrics(arm["score"]),  # noqa: SLF001
        "diversity": _distinct(arm["score"]),
        "max_token_completions": int(arm["provenance"].get("max_token_completions", 0)),
        "predictions_sha256": sha256_file(arm["predictions_path"]),
        "score_sha256": sha256_file(arm["score_path"]),
    }


def _model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore loader-only attestations added after seed 42."""

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


def _input_view_contract_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return every input-view field except the non-model-visible row metadata digest.

    The row-transformation record was extended after the historical seed-42
    run.  Its digest may therefore differ even when every rendered encoder
    source byte is unchanged.  No other field is excluded from equality.
    """

    if not isinstance(record, Mapping):
        raise ValueError("input-view record is not an object")
    missing = INPUT_VIEW_REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"input-view record is missing fields: {sorted(missing)}")
    for field in (
        "ordered_task_ids_sha256",
        "ordered_source_sha256s_sha256",
        INPUT_VIEW_METADATA_DIGEST_FIELD,
    ):
        value = record.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"input-view record has invalid {field}")
    return {
        key: value
        for key, value in record.items()
        if key != INPUT_VIEW_METADATA_DIGEST_FIELD
    }


def _require_matching_input_view_contract(
    view: str,
    expected_projection_sha256: str | None,
    record: Mapping[str, Any],
) -> str:
    projection_sha256 = canonical_sha256(_input_view_contract_projection(record))
    if (
        expected_projection_sha256 is not None
        and expected_projection_sha256 != projection_sha256
    ):
        raise ValueError(f"{view}: model-visible input-view contract differs across seeds")
    return projection_sha256


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    prior_path = Path(args.seed42_measurement_report).expanduser().resolve()
    prior = _read(prior_path, "seed-42 measurement report")
    if (
        prior.get("schema") != base.REPORT_SCHEMA
        or prior.get("status") != "complete"
        or prior.get("heldout_tasks") != args.expected_tasks
        or prior.get("k") != args.k
        or prior.get("interpretation_gate", {}).get("triggered") is not True
        or set(prior.get("input_ablations") or {}) != set(VIEWS)
    ):
        raise ValueError("seed-42 measurement report contract failed")

    compatibility_path = Path(args.runtime_compatibility).expanduser().resolve()
    compatibility = _read(compatibility_path, "runtime compatibility attestation")
    if (
        compatibility.get("schema") != "t5gemma2-measurement-runtime-compat-v1"
        or compatibility.get("status") != "pass"
        or compatibility.get("current_generation_replay", {}).get(
            "exact_prefix_reproduction"
        )
        is not True
        or compatibility.get("current_generation_replay", {}).get("rows", 0) < 5
        or compatibility.get("current_generation_replay", {}).get("candidates", 0)
        < 50
        or compatibility.get("current_generation_replay", {}).get(
            "model_identity_projection_identical"
        )
        is not True
        or compatibility.get("current_scoring_replay", {}).get(
            "candidate_compile_pass_decisions_identical"
        )
        is not True
        or compatibility.get("current_scoring_replay", {}).get(
            "task_metrics_identical"
        )
        is not True
        or compatibility.get("tests_exposed_to_model") is not False
        or compatibility.get("full_gold_targets_exposed_to_model") is not False
    ):
        raise ValueError("runtime compatibility attestation failed")

    baseline_specs = [_parse_baseline(value) for value in args.baseline]
    if sorted(seed for seed, _, _ in baseline_specs) != list(SEEDS):
        raise ValueError("baselines must be exactly seeds 42..46")
    baselines = {
        seed: base._load_arm(  # noqa: SLF001
            label=f"baseline_seed{seed}",
            predictions_path=predictions,
            score_path=score,
            expected_tasks=args.expected_tasks,
            expected_k=args.k,
            expected_provenance_schema=base.BASE_PROVENANCE_SCHEMA,
        )
        for seed, predictions, score in baseline_specs
    }

    arm_specs = [_parse_arm(value) for value in args.arm]
    if len(arm_specs) != len(VIEWS) * len(SEEDS):
        raise ValueError("exactly fifteen view/seed arms are required")
    if {(view, seed) for view, seed, _, _ in arm_specs} != {
        (view, seed) for view in VIEWS for seed in SEEDS
    }:
        raise ValueError("arms must be all three views at seeds 42..46")
    arms = {
        (view, seed): base._load_arm(  # noqa: SLF001
            label=f"{view}_seed{seed}",
            predictions_path=predictions,
            score_path=score,
            expected_tasks=args.expected_tasks,
            expected_k=args.k,
            expected_provenance_schema=base.ABLATION_PROVENANCE_SCHEMA,
        )
        for view, seed, predictions, score in arm_specs
    }

    reference = baselines[42]
    reference_model = canonical_sha256(
        _model_identity(reference["provenance"]["model"])
    )
    reference_tasks = reference["task_ids"]
    reference_score_contract = (
        reference["score"]["evaluation"]["sha256"],
        reference["score"]["k"],
        reference["score"]["timeout"],
        reference["score"]["stability_runs"],
    )
    for seed, baseline in baselines.items():
        generation_contract = baseline["journal"][0].get("contract") or {}
        score_contract = (
            baseline["score"]["evaluation"]["sha256"],
            baseline["score"]["k"],
            baseline["score"]["timeout"],
            baseline["score"]["stability_runs"],
        )
        if (
            baseline["task_ids"] != reference_tasks
            or canonical_sha256(
                _model_identity(baseline["provenance"]["model"])
            )
            != reference_model
            or score_contract != reference_score_contract
            or int(baseline["provenance"]["sampling"]["seed"]) != seed
            or generation_contract.get("script_sha256")
            != HISTORICAL_BASE_INFERENCE_SHA256
            or baseline["score"].get("evaluator", {}).get("sha256")
            != HISTORICAL_EVALUATOR_SHA256
        ):
            raise ValueError(f"baseline seed {seed} differs in task/model/scoring contract")

    view_contract_projections: dict[str, str] = {}
    view_full_record_hashes: dict[str, dict[str, str]] = {
        view: {} for view in VIEWS
    }
    view_row_metadata_hashes: dict[str, dict[str, str]] = {
        view: {} for view in VIEWS
    }
    view_source_hashes: dict[str, dict[str, str]] = {view: {} for view in VIEWS}
    for (view, seed), arm in arms.items():
        baseline = baselines[seed]
        score_contract = (
            arm["score"]["evaluation"]["sha256"],
            arm["score"]["k"],
            arm["score"]["timeout"],
            arm["score"]["stability_runs"],
        )
        provenance = arm["provenance"]
        generation_contract = arm["journal"][0].get("contract") or {}
        view_record = provenance.get("heldout", {}).get("input_view") or {}
        expected_base_inference = (
            HISTORICAL_BASE_INFERENCE_SHA256
            if seed == 42
            else CURRENT_BASE_INFERENCE_SHA256
        )
        expected_evaluator = (
            HISTORICAL_EVALUATOR_SHA256 if seed == 42 else CURRENT_EVALUATOR_SHA256
        )
        if (
            arm["task_ids"] != reference_tasks
            or canonical_sha256(_model_identity(provenance["model"]))
            != reference_model
            or score_contract != reference_score_contract
            or provenance.get("input_view") != view
            or view_record.get("view") != view
            or provenance.get("tests_exposed_to_model") is not False
            or provenance.get("full_gold_targets_exposed_to_model") is not False
            or generation_contract.get("source_truncation") is not False
            or int(provenance["sampling"]["seed"]) != seed
            or provenance["sampling"] != baseline["provenance"]["sampling"]
            or arm["coordinates"] != baseline["coordinates"]
            or generation_contract.get("script_sha256") != WRAPPER_SHA256
            or generation_contract.get("base_inference_script_sha256")
            != expected_base_inference
            or arm["score"].get("evaluator", {}).get("sha256")
            != expected_evaluator
        ):
            raise ValueError(f"{view} seed {seed}: pairing/privacy contract differs")
        projection_hash = _require_matching_input_view_contract(
            view,
            view_contract_projections.get(view),
            view_record,
        )
        view_contract_projections[view] = projection_hash
        view_full_record_hashes[view][str(seed)] = canonical_sha256(view_record)
        view_row_metadata_hashes[view][str(seed)] = view_record[
            INPUT_VIEW_METADATA_DIGEST_FIELD
        ]
        view_source_hashes[view][str(seed)] = view_record[
            "ordered_source_sha256s_sha256"
        ]

    # Bind reused seed-42 files to the already published audit rather than merely
    # accepting any schema-compatible artifacts.
    for view in VIEWS:
        prior_arm = prior["input_ablations"][view]
        current = arms[(view, 42)]
        if (
            prior_arm.get("predictions_sha256") != sha256_file(current["predictions_path"])
            or prior_arm.get("score_sha256") != sha256_file(current["score_path"])
        ):
            raise ValueError(f"{view}: reused seed-42 artifact differs from prior audit")

    constants_record = arms[("constants_stripped", 42)]["provenance"]["heldout"]["input_view"]
    constants_summary = constants_record.get("summary") or {}
    unchanged_ids = list(constants_summary.get("unchanged_task_ids") or [])
    if (
        len(unchanged_ids) != int(constants_summary.get("unchanged_no_literal_rows", -1))
        or constants_summary.get("unchanged_task_ids_sha256") != canonical_sha256(unchanged_ids)
    ):
        raise ValueError("constants-stripped unchanged-task partition is invalid")
    for seed in SEEDS:
        baseline_predictions = {
            row["id"]: row["predictions"] for row in baselines[seed]["predictions"]
        }
        constants_predictions = {
            row["id"]: row["predictions"]
            for row in arms[("constants_stripped", seed)]["predictions"]
        }
        if any(
            baseline_predictions[task_id] != constants_predictions[task_id]
            for task_id in unchanged_ids
        ):
            raise ValueError(
                f"constants-stripped seed {seed}: unchanged source changed predictions"
            )

    baseline_summaries = {seed: _arm_summary(arm) for seed, arm in baselines.items()}
    view_summaries: dict[str, Any] = {}
    for view in VIEWS:
        per_seed: dict[int, Any] = {}
        solve_frequency = {task_id: 0 for task_id in reference_tasks}
        for seed in SEEDS:
            arm = arms[(view, seed)]
            summary = _arm_summary(arm)
            paired = base._paired(arm["score"], baselines[seed]["score"])  # noqa: SLF001
            summary["paired_vs_same_seed_baseline"] = {
                metric: _mcnemar_exact(paired[metric])
                for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
            }
            per_seed[seed] = summary
            for row in arm["score"]["task_results"]:
                solve_frequency[row["task_id"]] += int(bool(row["pass_at_k"]))
        view_summaries[view] = {
            "seeds": {
                str(seed): per_seed[seed] for seed in SEEDS
            },
            "count_distributions": {
                metric: _distribution(
                    [per_seed[seed]["metrics"][metric]["count"] for seed in SEEDS]
                )
                for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
            },
            "distinct_per_10_distribution": _distribution(
                [per_seed[seed]["diversity"]["mean_distinct_per_10"] for seed in SEEDS]
            ),
            "unique_tasks_solved_across_five_seeds": sum(
                count > 0 for count in solve_frequency.values()
            ),
            "task_solve_frequency_histogram": {
                str(frequency): sum(count == frequency for count in solve_frequency.values())
                for frequency in range(6)
            },
            "input_view_contract_projection_sha256": view_contract_projections[view],
            "input_view_full_record_sha256_by_seed": view_full_record_hashes[view],
            "row_transformations_sha256_by_seed": view_row_metadata_hashes[view],
            "full_input_view_records_identical_across_seeds": len(
                set(view_full_record_hashes[view].values())
            )
            == 1,
            "row_transformation_metadata_identical_across_seeds": len(
                set(view_row_metadata_hashes[view].values())
            )
            == 1,
            "model_visible_source_bytes_identical_across_seeds": len(
                set(view_source_hashes[view].values())
            )
            == 1,
        }

    gold_path = Path(args.gold_score).expanduser().resolve()
    gold = _read(gold_path, "Rank-0 gold score")
    if (
        gold.get("schema") != base.SCORE_SCHEMA
        or gold.get("tasks") != args.expected_tasks
        or gold.get("k") != 1
        or gold.get("pass_at_1", {}).get("count") != args.expected_tasks
        or gold.get("pass_at_k", {}).get("count") != args.expected_tasks
        or gold.get("compile_at_k", {}).get("count") != args.expected_tasks
    ):
        raise ValueError("Rank-0 gold round-trip is incomplete")

    return {
        "schema": SCHEMA,
        "status": "complete",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dependencies": {
            "measurement_report_script_sha256": sha256_file(Path(base.__file__).resolve()),
            "seed42_measurement_report": {
                "path": str(prior_path),
                "sha256": sha256_file(prior_path),
            },
            "runtime_compatibility": {
                "path": str(compatibility_path),
                "sha256": sha256_file(compatibility_path),
                "generation_prefix_candidates_replayed": compatibility[
                    "current_generation_replay"
                ]["candidates"],
                "full_score_candidates_replayed": args.expected_tasks * args.k,
            },
        },
        "design": {
            "isa": "x86_64",
            "architecture": "native_t5gemma2_encoder_decoder",
            "checkpoint_frozen": True,
            "seeds": list(SEEDS),
            "fresh_runs": 12,
            "reused_seed42_runs": 3,
            "views": list(VIEWS),
            "tasks_per_run": args.expected_tasks,
            "k": args.k,
            "primary_endpoint": "pass_at_k",
            "tests_exposed_to_model": False,
            "no_frontier_api": True,
            "no_training_or_promotion": True,
            "historical_to_current_runtime_replay_gate_passed": True,
        },
        "rank0_gold_roundtrip": {
            "path": str(gold_path),
            "sha256": sha256_file(gold_path),
            "passed": args.expected_tasks,
            "tasks": args.expected_tasks,
        },
        "checkpoint_record_sha256": reference_model,
        "task_order_sha256": canonical_sha256(reference_tasks),
        "same_seed_sample_coordinates_paired": True,
        "input_view_contract_projections_identical_across_seeds": True,
        "model_visible_source_bytes_identical_across_seeds": all(
            len(set(view_source_hashes[view].values())) == 1 for view in VIEWS
        ),
        "full_input_view_records_identical_across_seeds": all(
            len(set(view_full_record_hashes[view].values())) == 1 for view in VIEWS
        ),
        "allowed_input_view_metadata_drift": {
            "field": INPUT_VIEW_METADATA_DIGEST_FIELD,
            "reason": (
                "row-transformation metadata schema evolved after historical seed 42; "
                "the complete remaining contract and ordered model-visible source-byte "
                "digests are identical"
            ),
            "full_record_identity_not_claimed": True,
        },
        "constants_unchanged_sources_prediction_identical_across_all_seeds": True,
        "baseline_seeds": {
            str(seed): baseline_summaries[seed] for seed in SEEDS
        },
        "interventions": view_summaries,
        "interpretation": {
            "seed42_is_reused_diagnostic_context": True,
            "seeds43_to46_are_fresh_replicates": True,
            "prior_noise_floor_gate_triggered": True,
            "report_seed_distribution_and_same_seed_pairing": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--seed42_measurement_report", required=True)
    parser.add_argument("--runtime_compatibility", required=True)
    parser.add_argument("--gold_score", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_tasks", type=int, default=175)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.expected_tasks <= 0 or args.k <= 0:
        parser.error("expected_tasks and k must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
