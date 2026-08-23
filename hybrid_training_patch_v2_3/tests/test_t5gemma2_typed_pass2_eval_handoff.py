from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation import audit_t5gemma2_typed_pass2 as audit
from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts" / "evaluation" / "audit_t5gemma2_typed_pass2.py"
EVAL = ROOT / "deploy" / "vast" / "t5gemma2_typed_pass2_eval.sh"
HANDOFF = ROOT / "deploy" / "vast" / "t5gemma2_typed_pass2_eval_handoff.sh"
SUPERVISOR = (
    ROOT / "deploy" / "vast" / "t5gemma2-typed-pass2-eval-handoff.conf"
)


def _valid_contract() -> dict:
    schedule = [
        {"source_task_id": f"local-{index:03d}", "source_category": "local_student_new"}
        for index in range(audit.EXPECTED_LOCAL_ROWS)
    ] + [
        {"source_task_id": f"api-{index:03d}", "source_category": "external_teacher_new"}
        for index in range(audit.EXPECTED_API_ROWS)
    ]
    return {
        "schema": audit.PASS2_RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "runtime": {
            "trainer_sha256": audit.EXPECTED_PASS2_SCRIPT_SHA256,
            "trainer_profile": "typed_direct_only_rs_sft_pass2_local190_plus_dual_api",
        },
        "optimization": {
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "planned_updates": 54,
            "updates_per_epoch": 27,
            "learning_rate": 0.00002,
            "warmup_ratio": 0,
            "warmup_updates": 0,
            "seed": 42,
            "bf16": True,
            "gradient_checkpointing": True,
            "attn_implementation": "sdpa",
        },
        "dataset": {
            "schema": audit.PASS2_DATASET_SCHEMA,
            "rows": 209,
            "architecture": "native_encoder_decoder",
            "heldout_overlap": 0,
            "known_contaminant_excluded": audit.EXPECTED_CONTAMINANT,
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "repair_conditioned_prefixes_visible": False,
            "reasoning_model_visible": False,
            "all_targets_bound_to_generation_journals": True,
            "production_floor_eligible": True,
            "composition": {
                "verified_direct": 209,
                "local_student_new": 190,
                "external_teacher_new": 19,
                "prior_225_replay": 0,
                "gold_replay": 0,
                "repair_conditioned": 0,
                "reasoning_rows": 0,
                "gold_source_replay": 0,
                "independently_generated_exact_gold_matches": 13,
            },
            "local_harvest": {"rows": 190},
            "dual_api_harvest": {
                "schema": "t5gemma2-typed-dual-api-pass2-input-audit-v1",
                "status": "complete",
                "direct_code_only": True,
                "gold_source_replay": False,
                "heldout_175_model_visible": False,
                "heldout_175_used_for_generation_or_selection": False,
                "targets": {"rows": 19},
            },
            "prior_225_exclusion": {"rows": 225},
            "heldout_identity_audit": {"rows": 175},
            "full_acceptance_reverification": {
                "rows": 209,
                "passed": 209,
                "stability_runs": 2,
                "timeout_seconds": 30,
                "diagnostics_persisted": False,
                "tests_model_visible": False,
            },
            "schedule": schedule,
            "schedule_sha256": canonical_sha256(schedule),
        },
        "warmstart": {"checkpoint_name": "checkpoint-optstep-000058", "update": 58},
        "warmstart_contract_schema": audit.PASS1_RUN_SCHEMA,
        "lora": {
            "new_adapter_attached": False,
            "warmstart_weights_continued": True,
            "encoder_and_decoder_trainable": True,
            "vision_trainable": False,
        },
        "privacy": {
            "heldout_overlap": 0,
            "heldout_content_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "reasoning_persisted": False,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _training_args(tmp_path: Path, contract: dict) -> SimpleNamespace:
    stage = tmp_path / "stage"
    checkpoint = stage / "checkpoint-optstep-000054"
    paths = {
        "result": stage / "result.json",
        "root_contract": stage / "run_contract.json",
        "dataset_manifest": stage / "dataset_manifest.json",
        "latest_pointer": stage / "latest_checkpoint.json",
        "checkpoint_contract": checkpoint / "run_contract.json",
        "training_state": checkpoint / "training_state.pt",
        "adapter_weights": checkpoint / "adapter" / "adapter_model.safetensors",
        "adapter_config": checkpoint / "adapter" / "adapter_config.json",
        "tokenizer": checkpoint / "tokenizer" / "tokenizer.json",
    }
    contract_sha = canonical_sha256(contract)
    result = {
        "schema": audit.PASS2_RUN_SCHEMA,
        "status": "complete",
        "updates": 54,
        "planned_updates": 54,
        "rows": 209,
        "latest_checkpoint": "checkpoint-optstep-000054",
        "production_floor_eligible": True,
    }
    pointer = {
        "schema": audit.PASS2_CHECKPOINT_SCHEMA,
        "update": 54,
        "run_contract_sha256": contract_sha,
        "path": str(checkpoint.resolve()),
    }
    _write_json(paths["result"], result)
    _write_json(paths["root_contract"], contract)
    _write_json(paths["dataset_manifest"], contract["dataset"])
    _write_json(paths["latest_pointer"], pointer)
    _write_json(paths["checkpoint_contract"], contract)
    for key in ("training_state", "adapter_weights"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_bytes(key.encode())
    _write_json(paths["adapter_config"], {"r": 64})
    _write_json(paths["tokenizer"], {"model": "synthetic"})
    values: dict[str, object] = {"output": str(tmp_path / "audit.json")}
    for name, path in paths.items():
        values[name] = str(path)
        values[f"expected_{name}_sha256"] = sha256_file(path)
    return SimpleNamespace(**values)


def test_pass2_contract_accepts_exact_composition_and_rejects_api_drift() -> None:
    contract = _valid_contract()
    record = audit._validate_pass2_contract(contract)  # noqa: SLF001
    assert record == {
        "canonical_sha256": canonical_sha256(contract),
        "rows": 209,
        "local_rows": 190,
        "api_rows": 19,
        "planned_updates": 54,
        "warmstart_update": 58,
    }
    tampered = copy.deepcopy(contract)
    tampered["dataset"]["composition"]["external_teacher_new"] = 18
    with pytest.raises(ValueError, match="run/dataset/privacy/lineage"):
        audit._validate_pass2_contract(tampered)  # noqa: SLF001


def test_training_audit_gates_final_checkpoint_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _valid_contract()
    args = _training_args(tmp_path, contract)
    monkeypatch.setattr(
        audit,
        "_load_training_state",
        lambda _path: {
            "schema": audit.PASS2_CHECKPOINT_SCHEMA,
            "update": 54,
            "epoch": 2,
            "next_row": 0,
            "run_contract_sha256": canonical_sha256(contract),
            "optimizer": {},
            "scheduler": {},
            "rng": {},
        },
    )
    report = audit.audit_training(args)
    assert report["status"] == "pass"
    assert report["composition"] == {
        "rows": 209,
        "local_student_new": 190,
        "external_teacher_new": 19,
        "prior_225_replay": 0,
        "gold_replay": 0,
        "heldout_overlap": 0,
    }
    assert report["checkpoint"]["name"] == "checkpoint-optstep-000054"


def test_handoff_waits_for_exact_success_then_late_binds() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "t5gemma2-typed-dual-to-pass2-handoff" in text
    assert "RUNNING|STARTING" in text and "EXITED) break" in text
    assert 'STOPPED) blocked' in text and 'FATAL|BACKOFF|UNKNOWN|"") blocked' in text
    assert 'snapshot_one="$(sha256sum "${files[@]}")"' in text
    assert 'snapshot_two="$(sha256sum "${files[@]}")"' in text
    assert text.index("snapshot_two=") < text.index("export T5GEMMA_TYPED_PASS2_RESULT_SHA256")
    assert text.index("audit_t5gemma2_typed_pass2.py training") < text.index('exec "${EVAL_LAUNCHER}"')
    for fragment in (
        ".composition.rows == 209",
        ".composition.local_student_new == 190",
        ".composition.external_teacher_new == 19",
        ".composition.prior_225_replay == 0",
        ".composition.gold_replay == 0",
        ".composition.heldout_overlap == 0",
        'checkpoint-optstep-000054',
    ):
        assert fragment in text


def test_eval_is_typed_k10_seed42_4096_and_scores_full_plus_clean() -> None:
    text = EVAL.read_text(encoding="utf-8")
    for fragment in (
        "--input_view typed_opaque_contract",
        "--num_samples 10",
        "--generation_batch_size 10",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--temperature 0.8",
        "--top_p 0.95",
        "--seed 42",
        "--k 10 --workers 32 --timeout 30 --stability_runs 2",
        "typed_direct_pass2_seed42_k10_score_full175.json",
        "typed_direct_pass2_seed42_k10_score_clean174.json",
        "--exclude_task_id sigless_8bf7f40ca356",
    ):
        assert fragment in text
    assert "audit_t5gemma2_typed_pass2.py compare" not in text
    assert "previous_update58_audit" not in text


def test_comparison_guard_requires_same_base_inference_code() -> None:
    text = AUDITOR.read_text(encoding="utf-8")
    assert (
        'bg["header_contract"]["base_inference_script_sha256"]\n'
        '        == ag["header_contract"]["base_inference_script_sha256"]'
    ) in text


def test_supervisor_job_is_manual_and_fail_closed() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-pass2-eval-handoff]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
