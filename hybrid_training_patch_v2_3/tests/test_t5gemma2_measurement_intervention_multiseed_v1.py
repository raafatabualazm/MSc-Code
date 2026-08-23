from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import (  # noqa: E402
    t5gemma2_measurement_intervention_multiseed_report_v1 as report,
)
from scripts.evaluation.durable_evaluation_journal import (  # noqa: E402
    append_event,
    canonical_sha256,
    journal_record,
    sha256_file,
)


def test_design_is_exactly_three_views_by_four_fresh_seeds() -> None:
    assert report.SEEDS == (42, 43, 44, 45, 46)
    assert report.VIEWS == (
        "typed_opaque_contract",
        "constants_stripped",
        "semantic_body_swap",
    )
    assert len(report.VIEWS) * (len(report.SEEDS) - 1) == 12


def test_distribution_and_exact_mcnemar_are_stable() -> None:
    distribution = report._distribution([7, 7, 7, 5, 3])
    assert distribution["mean"] == pytest.approx(5.8)
    assert distribution["sample_sd"] == pytest.approx(1.7888543819998317)
    paired = report._mcnemar_exact(
        {
            "left_only": 5,
            "right_only": 0,
            "equal": 170,
            "discordant": 5,
            "left_count": 12,
            "right_count": 7,
        }
    )
    assert paired["exact_two_sided_p"] == pytest.approx(0.0625)


def test_diversity_uses_code_hashes_at_exact_k10() -> None:
    candidates = []
    for task_id, unique in (("a", 10), ("b", 8)):
        for sample_index in range(10):
            candidates.append(
                {
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "code_sha256": f"{task_id}-{sample_index % unique}",
                }
            )
    result = report._distinct({"candidate_results": candidates})
    assert result["mean_distinct_per_10"] == 9
    assert result["tasks_below_10"] == 1
    assert result["histogram"] == {"8": 1, "10": 1}


def test_model_identity_ignores_only_loader_attestations() -> None:
    old = {
        "name": "model",
        "revision": "commit",
        "config_sha256": "a",
        "arm": "sft",
        "tokenizer_sha256": "b",
        "warmstart_contract_sha256": "c",
        "adapter": {"adapter_weights_sha256": "d"},
    }
    current = {
        **old,
        "training_stage_schema": "new-loader-attestation",
        "production_floor_eligible": True,
    }
    assert report._model_identity(old) == report._model_identity(current)
    current["adapter"] = {"adapter_weights_sha256": "changed"}
    assert report._model_identity(old) != report._model_identity(current)


def test_input_view_projection_allows_only_row_metadata_digest_drift() -> None:
    original = {
        "schema": "t5gemma2-f2-measurement-input-view-v1",
        "view": "typed_opaque_contract",
        "rows": 175,
        "ordered_task_ids_sha256": "a" * 64,
        "ordered_source_sha256s_sha256": "b" * 64,
        "row_transformations_sha256": "c" * 64,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "summary": {"intervention": "gold_derived_types_and_arity_only"},
    }
    metadata_only = {**original, "row_transformations_sha256": "d" * 64}
    assert report._input_view_contract_projection(
        original
    ) == report._input_view_contract_projection(metadata_only)
    expected = report._require_matching_input_view_contract(
        "typed_opaque_contract", None, original
    )
    assert (
        report._require_matching_input_view_contract(
            "typed_opaque_contract", expected, metadata_only
        )
        == expected
    )

    for field, changed in (
        ("ordered_source_sha256s_sha256", "e" * 64),
        ("ordered_task_ids_sha256", "f" * 64),
        ("view", "constants_stripped"),
        ("summary", {"intervention": "changed"}),
    ):
        candidate = {**metadata_only, field: changed}
        with pytest.raises(ValueError, match="contract differs across seeds"):
            report._require_matching_input_view_contract(
                "typed_opaque_contract", expected, candidate
            )


def test_input_view_projection_requires_source_and_metadata_digests() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        report._input_view_contract_projection(
            {
                "schema": "t5gemma2-f2-measurement-input-view-v1",
                "view": "typed_opaque_contract",
            }
        )


def test_build_report_happy_path_uses_journal_truncation_contract(
    tmp_path: Path,
) -> None:
    task_id = "task-0"
    expected_tasks = 1
    k = 10
    evaluation_sha256 = "e" * 64
    model = {
        "name": "test-model",
        "revision": "test-revision",
        "config_sha256": "a" * 64,
        "arm": "sft",
        "tokenizer_sha256": "b" * 64,
        "warmstart_contract_sha256": "c" * 64,
        "adapter": {"adapter_weights_sha256": "d" * 64},
    }

    def write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def sampling(seed: int) -> dict[str, object]:
        return {
            "num_samples": k,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "max_source_tokens": 32768,
            "max_new_tokens": 4096,
            "seed": seed,
            "seed_policy": "seed+task_index*100003+batch_start",
            "generation_batch_size": 10,
            "decoder_prefix_is_not_output": True,
            "sampled_eos_retained": True,
            "fabricated_eos": False,
        }

    def write_arm(
        *,
        label: str,
        seed: int,
        view: str | None,
    ) -> tuple[Path, Path]:
        arm_dir = tmp_path / label
        arm_dir.mkdir()
        predictions_path = arm_dir / "predictions.json"
        score_path = arm_dir / "score.json"
        if view in (None, "constants_stripped"):
            prefix = f"baseline-seed{seed}"
        else:
            prefix = f"{view}-seed{seed}"
        outputs = [f"{prefix}-candidate{index}" for index in range(k)]
        write_json(predictions_path, [{"id": task_id, "predictions": outputs}])

        journal_path = Path(str(predictions_path) + ".generation.journal.jsonl")
        if view is None:
            generation_contract = {
                "script_sha256": report.HISTORICAL_BASE_INFERENCE_SHA256,
                "source_truncation": False,
            }
        else:
            generation_contract = {
                "script_sha256": report.WRAPPER_SHA256,
                "base_inference_script_sha256": (
                    report.HISTORICAL_BASE_INFERENCE_SHA256
                    if seed == 42
                    else report.CURRENT_BASE_INFERENCE_SHA256
                ),
                "source_truncation": False,
            }
        append_event(
            journal_path,
            {"event": "header", "contract": generation_contract},
        )
        append_event(
            journal_path,
            {
                "event": "task_complete",
                "task_id": task_id,
                "candidates": [
                    {"sample_index": index, "seed": seed}
                    for index in range(k)
                ],
            },
        )
        append_event(journal_path, {"event": "complete"})

        provenance: dict[str, object] = {
            "schema": (
                report.base.BASE_PROVENANCE_SCHEMA
                if view is None
                else report.base.ABLATION_PROVENANCE_SCHEMA
            ),
            "num_rows": expected_tasks,
            "num_samples": k,
            "output_sha256": sha256_file(predictions_path),
            "generation_journal": journal_record(journal_path),
            "model": model,
            "sampling": sampling(seed),
            "no_frontier_api": True,
            "tests_exposed_to_model": False,
            "full_gold_targets_exposed_to_model": False,
        }
        if view is not None:
            if view == "constants_stripped":
                summary = {
                    "unchanged_no_literal_rows": 1,
                    "unchanged_task_ids": [task_id],
                    "unchanged_task_ids_sha256": canonical_sha256([task_id]),
                }
            else:
                summary = {}
            provenance.update(
                {
                    "input_view": view,
                    "heldout": {
                        "input_view": {
                            "schema": "t5gemma2-f2-measurement-input-view-v1",
                            "view": view,
                            "rows": expected_tasks,
                            "ordered_task_ids_sha256": canonical_sha256([task_id]),
                            "ordered_source_sha256s_sha256": "f" * 64,
                            "row_transformations_sha256": (
                                "1" * 64 if seed == 42 else "2" * 64
                            ),
                            "tests_exposed_to_model": False,
                            "full_gold_targets_exposed_to_model": False,
                            "summary": summary,
                        }
                    },
                }
            )
        # source_truncation intentionally exists only in the sealed journal
        # contract, matching the production generation artifacts.
        assert "source_truncation" not in provenance
        write_json(Path(str(predictions_path) + ".provenance.json"), provenance)

        evaluator_sha256 = (
            report.HISTORICAL_EVALUATOR_SHA256
            if view is None or seed == 42
            else report.CURRENT_EVALUATOR_SHA256
        )
        candidate_results = [
            {
                "task_id": task_id,
                "sample_index": index,
                "code_sha256": hashlib.sha256(
                    outputs[index].encode("utf-8")
                ).hexdigest(),
                "compiled": True,
                "passed": False,
                "diagnostic": "",
            }
            for index in range(k)
        ]
        write_json(
            score_path,
            {
                "schema": report.base.SCORE_SCHEMA,
                "tasks": expected_tasks,
                "k": k,
                "timeout": 30,
                "stability_runs": 2,
                "evaluation": {"sha256": evaluation_sha256},
                "evaluator": {"sha256": evaluator_sha256},
                "predictions": {"sha256": sha256_file(predictions_path)},
                "pass_at_1": {"count": 0, "rate": 0.0},
                "pass_at_k": {"count": 0, "rate": 0.0},
                "compile_at_k": {"count": 1, "rate": 1.0},
                "task_results": [
                    {
                        "task_id": task_id,
                        "pass_at_1": False,
                        "pass_at_k": False,
                        "compile_at_k": True,
                    }
                ],
                "candidate_results": candidate_results,
            },
        )
        return predictions_path, score_path

    baselines = {
        seed: write_arm(label=f"baseline-{seed}", seed=seed, view=None)
        for seed in report.SEEDS
    }
    arms = {
        (view, seed): write_arm(
            label=f"{view}-{seed}", seed=seed, view=view
        )
        for view in report.VIEWS
        for seed in report.SEEDS
    }

    prior_path = tmp_path / "seed42-measurement-report.json"
    write_json(
        prior_path,
        {
            "schema": report.base.REPORT_SCHEMA,
            "status": "complete",
            "heldout_tasks": expected_tasks,
            "k": k,
            "interpretation_gate": {"triggered": True},
            "input_ablations": {
                view: {
                    "predictions_sha256": sha256_file(arms[(view, 42)][0]),
                    "score_sha256": sha256_file(arms[(view, 42)][1]),
                }
                for view in report.VIEWS
            },
        },
    )
    compatibility_path = tmp_path / "runtime-compatibility.json"
    write_json(
        compatibility_path,
        {
            "schema": "t5gemma2-measurement-runtime-compat-v1",
            "status": "pass",
            "tests_exposed_to_model": False,
            "full_gold_targets_exposed_to_model": False,
            "current_generation_replay": {
                "exact_prefix_reproduction": True,
                "rows": 5,
                "candidates": 50,
                "model_identity_projection_identical": True,
            },
            "current_scoring_replay": {
                "candidate_compile_pass_decisions_identical": True,
                "task_metrics_identical": True,
            },
        },
    )
    gold_path = tmp_path / "gold-score.json"
    write_json(
        gold_path,
        {
            "schema": report.base.SCORE_SCHEMA,
            "tasks": expected_tasks,
            "k": 1,
            "pass_at_1": {"count": expected_tasks},
            "pass_at_k": {"count": expected_tasks},
            "compile_at_k": {"count": expected_tasks},
        },
    )

    args = argparse.Namespace(
        baseline=[
            f"{seed}|{baselines[seed][0]}|{baselines[seed][1]}"
            for seed in report.SEEDS
        ],
        arm=[
            f"{view}|{seed}|{arms[(view, seed)][0]}|{arms[(view, seed)][1]}"
            for view in report.VIEWS
            for seed in report.SEEDS
        ],
        seed42_measurement_report=str(prior_path),
        runtime_compatibility=str(compatibility_path),
        gold_score=str(gold_path),
        expected_tasks=expected_tasks,
        k=k,
    )
    result = report.build_report(args)

    assert result["status"] == "complete"
    assert result["baseline_seeds"]["42"]["metrics"]["compile_at_k"] == {
        "count": 1,
        "rate": 1.0,
    }
    assert all(
        result["interventions"][view]["distinct_per_10_distribution"]["mean"]
        == 10
        for view in report.VIEWS
    )
    assert result["input_view_contract_projections_identical_across_seeds"] is True
    assert result["model_visible_source_bytes_identical_across_seeds"] is True
    assert result["full_input_view_records_identical_across_seeds"] is False
    assert all(
        result["interventions"][view][
            "model_visible_source_bytes_identical_across_seeds"
        ]
        is True
        and result["interventions"][view][
            "full_input_view_records_identical_across_seeds"
        ]
        is False
        for view in report.VIEWS
    )


def test_launcher_is_fail_closed_and_view_major() -> None:
    reporter = (
        PATCH_ROOT
        / "scripts"
        / "evaluation"
        / "t5gemma2_measurement_intervention_multiseed_report_v1.py"
    )
    reporter_sha = hashlib.sha256(reporter.read_bytes()).hexdigest()
    compatibility = (
        PATCH_ROOT
        / "scripts"
        / "evaluation"
        / "verify_t5gemma2_measurement_runtime_compat_v1.py"
    )
    compatibility_sha = hashlib.sha256(compatibility.read_bytes()).hexdigest()
    launcher = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2_measurement_intervention_multiseed_v1.sh"
    ).read_text(encoding="utf-8")
    assert f'{reporter_sha} "${{REPORTER}}"' in launcher
    assert f'{compatibility_sha} "${{COMPAT_CHECKER}}"' in launcher
    assert "for view in typed_opaque_contract constants_stripped semantic_body_swap" in launcher
    assert "for seed in 43 44 45 46" in launcher
    assert "--max_source_tokens 32768 --max_new_tokens 4096" in launcher
    assert "checkpoint-optstep-000348" in launcher
    assert "interpretation_gate.triggered == true" in launcher
    assert "Rank-0 gold round-trip failed" in launcher
    assert "nvidia-smi --query-compute-apps=pid" in launcher
    assert "less than ${MIN_FREE_KIB} KiB is free" in launcher
    assert "intervention_multiseed_report.json" in launcher
    assert "--limit 5" in launcher
    assert "candidate_compile_pass_decisions_identical == true" in launcher
    assert '--runtime_compatibility "${RUNTIME_COMPAT}"' in launcher


def test_supervisor_never_autostarts() -> None:
    config = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2-measurement-intervention-multiseed-v1.conf"
    ).read_text(encoding="utf-8")
    assert "autostart=false" in config
    assert "autorestart=false" in config
    assert "command=/bin/bash /opt/supervisor-scripts/" in config
    assert "stopasgroup=true" in config
    assert "killasgroup=true" in config
