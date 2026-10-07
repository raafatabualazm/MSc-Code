from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import (  # noqa: E402
    t5gemma2_typed_seed_replication_inference_v1 as adapter,
)
from scripts.evaluation import (  # noqa: E402
    t5gemma2_typed_seed_replication_report_v1 as report,
)
from scripts.evaluation.durable_evaluation_journal import (  # noqa: E402
    canonical_sha256,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pass3_manifest_binds_checkpoint_lineage_and_privacy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    for relative, payload in {
        "adapter/adapter_model.safetensors": b"weights",
        "adapter/adapter_config.json": b"{}\n",
        "tokenizer/tokenizer.json": b"{}\n",
    }.items():
        destination = checkpoint / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    contract = {
        "schema": "test-pass3-schema-v1",
        "architecture": "native_encoder_decoder",
        "status": "training",
        "base_model": {
            "name": "google/t5gemma-2-4b-4b",
            "resolved_commit": "487d4acf21a4d70c70bf534265b5263c9424979e",
        },
        "privacy": {"tests_model_visible": False, "heldout_overlap": 0},
        "dataset": {
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "tests_model_visible": False,
            "heldout_overlap": 0,
        },
    }
    run_contract = checkpoint / "run_contract.json"
    run_contract.write_text(json.dumps(contract), encoding="utf-8")
    result = tmp_path / "result.json"
    audit = tmp_path / "audit.json"
    result.write_text("{}\n", encoding="utf-8")
    audit.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema": adapter.CHECKPOINT_MANIFEST_SCHEMA,
        "arm": "pass3",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_files": {
            relative: {
                "path": str((checkpoint / relative).resolve()),
                "sha256": _sha(checkpoint / relative),
            }
            for relative in sorted(adapter.REQUIRED_CHECKPOINT_FILES)
        },
        "training_result": {"path": str(result.resolve()), "sha256": _sha(result)},
        "training_audit": {"path": str(audit.resolve()), "sha256": _sha(audit)},
        "run_contract_schema": contract["schema"],
        "run_contract_canonical_sha256": canonical_sha256(contract),
        "lineage": {
            "parent_arm": "incumbent_update58",
            "parent_adapter_weights_sha256": adapter.UPDATE58_ADAPTER_SHA256,
        },
        "privacy": {
            "heldout_175_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "gold_implementation_model_visible": False,
            "semantic_parameter_names_model_visible": False,
            "prior_success_exclusion_applied": True,
            "known_contaminant_excluded": "sigless_6b1dd0c6b6fc",
        },
        "no_automatic_promotion": True,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = adapter.validate_pass3_manifest(
        manifest_path=manifest_path,
        expected_sha256=_sha(manifest_path),
        checkpoint=checkpoint,
    )
    assert observed["run_contract_schema"] == "test-pass3-schema-v1"

    corrupted = json.loads(json.dumps(manifest))
    corrupted["privacy"]["tests_model_visible"] = True
    manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="policy contract"):
        adapter.validate_pass3_manifest(
            manifest_path=manifest_path,
            expected_sha256=_sha(manifest_path),
            checkpoint=checkpoint,
        )


def test_candidate_summary_uses_extracted_code_hashes() -> None:
    candidates = []
    for task_index in range(report.EXPECTED_TASKS):
        for sample_index in range(report.EXPECTED_K):
            distinct_slot = 0 if task_index < 20 else sample_index
            candidates.append(
                {
                    "task_id": f"task-{task_index}",
                    "sample_index": sample_index,
                    "code_sha256": hashlib.sha256(
                        f"{task_index}:{distinct_slot}".encode()
                    ).hexdigest(),
                    "passed": task_index == 0 and sample_index == 0,
                }
            )
    summary = report._summarize_candidates(  # noqa: SLF001
        {"candidate_results": candidates}, k=10
    )
    assert summary["tasks_below_10_distinct"] == 20
    assert summary["distinct_extracted_code_per_10"] == pytest.approx(1570 / 175)
    assert summary["successes_per_solved_task"] == [1]
    assert summary["diversity_guardrail"]["passes"] is False


def test_paired_report_is_directional_and_does_not_promote() -> None:
    def arm(pass_ids: set[str], distinct: float) -> dict:
        task_ids = [f"t{i}" for i in range(4)]
        rows = [
            {
                "task_id": task_id,
                "pass_at_1": task_id in pass_ids,
                "pass_at_k": task_id in pass_ids,
                "compile_at_k": True,
            }
            for task_id in task_ids
        ]
        count = len(pass_ids)
        return {
            "task_ids": task_ids,
            "score": {"task_results": rows},
            "metrics": {
                "pass_at_1": {"count": count, "rate": count / 4},
                "pass_at_k": {"count": count, "rate": count / 4},
                "compile_at_k": {"count": 4, "rate": 1.0},
            },
            "distinct_extracted_code_per_10": distinct,
        }

    paired = report._paired(arm({"t0", "t1"}, 9.95), arm({"t1", "t2", "t3"}, 9.85))  # noqa: SLF001
    assert paired["pass_at_k"]["gains"] == 2
    assert paired["pass_at_k"]["losses"] == 1
    assert paired["distinct_extracted_code_per_10"][
        "difference_pass3_minus_incumbent"
    ] == pytest.approx(-0.10)


def test_launchers_pin_three_separate_resumable_arms_and_no_promotion() -> None:
    arm_launcher = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2_typed_seed_replication_arm_v1.sh"
    ).read_text(encoding="utf-8")
    report_launcher = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2_typed_seed_replication_report_v1.sh"
    ).read_text(encoding="utf-8")
    assert "for seed in 43 44 45 46" in arm_launcher
    assert "typed_sft|incumbent|pass3" in arm_launcher
    assert "t5gemma2_typed_seed_replication_sft_opt348_v1" in arm_launcher
    assert "t5gemma2_typed_seed_replication_update58_v1" in arm_launcher
    assert "t5gemma2_typed_seed_replication_pass3_v1" in arm_launcher
    assert "/usr/bin/flock" in arm_launcher
    assert "nvidia-smi --query-compute-apps=pid" in arm_launcher
    assert "--input_view typed_opaque_contract" in arm_launcher
    assert "--num_samples 10 --generation_batch_size 10" in arm_launcher
    assert "--temperature 0.8 --top_p 0.95" in arm_launcher
    assert "--max_source_tokens 32768 --max_new_tokens 4096" in arm_launcher
    assert "--stability_runs 2" in arm_launcher
    assert "pass3 checkpoint manifest and handoff SHA are required" in arm_launcher
    assert "update58_current_stack_seed42_k10" in report_launcher
    assert "typed_contract_seed42_k10" in report_launcher
    assert "promotion=not_performed" in report_launcher
    assert "OPTIONAL_PASS2_ROOT" in report_launcher


def test_report_policy_requires_four_confirmatory_seeds() -> None:
    assert report.EXPECTED_SEEDS == (42, 43, 44, 45, 46)
    assert report.CONFIRMATORY_SEEDS == (43, 44, 45, 46)
    args = report.parse_args(
        [
            "--arm",
            "typed_sft|42|p|s",
            "--evaluation-file",
            "eval",
            "--expected-wrapper-sha256",
            "a" * 64,
            "--expected-base-inference-sha256",
            "b" * 64,
            "--expected-evaluator-sha256",
            "c" * 64,
            "--expected-adapter-sha256",
            "d" * 64,
            "--output",
            "out",
        ]
    )
    assert args.expected_adapter_sha256 == "d" * 64
