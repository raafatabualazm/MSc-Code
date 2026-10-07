from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.evaluation.validate_t5gemma2_typed_c2_verpo_terminal import (
    GATE_UPDATES,
    classify_terminal,
)
from scripts.training import t5gemma2_typed_c2_verpo_pilot150 as pilot


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = (
    ROOT
    / "deploy/vast/t5gemma2_typed_c2_verpo_to_matched_eval_handoff_v1.sh"
)
CONF = (
    ROOT
    / "deploy/vast/t5gemma2-typed-c2-verpo-to-matched-eval-handoff-v1.conf"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contract() -> dict:
    return {
        "schema": "t5gemma2-typed-c2-verpo-pilot150-run-v1",
        "status": "training",
        "architecture": "native_encoder_decoder",
        "policy_architecture": "native_t5gemma2_encoder_decoder",
        "automatic_promotion_permitted": False,
        "production_floor_eligible": False,
        "private_holdback_exposed": False,
        "no_frontier_api": True,
        "llm_judge": False,
        "acceptance_tests_exposed": False,
        "optimization": {
            "max_updates": 150,
            "tasks_per_update": 1,
            "learning_rate": 1e-6,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "ppo_clip": 0.0,
            "sft_replay_weight": 0.0,
            "gold_or_sft_replay_gradient": False,
        },
        "sampling": {
            "group_size": 4,
            "repair_group_size": 4,
            "max_repair_parents": 2,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 8192,
            "max_source_tokens": 32768,
            "eos_token_ids": [1],
            "suppressed_token_ids": [0],
            "pad_before_eos_fail_closed": True,
        },
        "pilot": {
            "maximum_updates": 150,
            "gate_interval": 16,
            "mandatory_pause_after_first_gate": True,
            "later_gates_automatic": True,
        },
        "selection": {
            "tasks": 150,
            "trainer_opened_gold_or_target_source": False,
        },
        "seed": 42,
    }


def _metric(update: int, *, active: bool = True) -> dict:
    if active:
        global_advantages = [0.75, -0.25, -0.25, -0.25]
        local_advantages = [0.5, -0.5, 0.0, 0.0]
        unified_advantages = [1.25, -0.75, -0.25, -0.25]
    else:
        global_advantages = [0.0] * 4
        local_advantages = [0.0] * 4
        unified_advantages = [0.0] * 4
    trajectories = [
        {"actions": [7, 1], "action_tokens": 2, "sampled_pad_before_eos": 0}
        for _ in range(4)
    ]
    return {
        "schema": "t5gemma2-typed-c2-verpo-pilot150-rollout-v1",
        "update": update,
        "no_frontier_api": True,
        "trajectory_count": 4,
        "active_policy_trajectories": 4 if active else 0,
        "optimizer_step": active,
        "policy_loss": 0.1 if active else 0.0,
        "sft_replay_loss": 0.0,
        "grad_norm": 1.0 if active else 0.0,
        "max_on_policy_logprob_drift": 1e-5,
        "sampled_pad_before_eos": 0,
        "task_records": [
            {
                "base_reward": {
                    "global_advantages": global_advantages,
                    "local_advantages": local_advantages,
                    "compile_advantages": [0.0] * 4,
                    "unified_advantages": unified_advantages,
                },
                "trajectories": trajectories,
            }
        ],
    }


def _make_stage(stage: Path, *, stop_update: int | None = None) -> None:
    contract = _contract()
    contract_sha = canonical_sha256(contract)
    _write_json(stage / "run_contract.json", contract)
    final_update = stop_update or 150
    rows = [
        _metric(update, active=stop_update is None)
        for update in range(1, final_update + 1)
    ]
    for row in rows:
        row["run_contract_sha256"] = contract_sha
    (stage / "rollout_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for boundary in GATE_UPDATES:
        if boundary > final_update:
            break
        gate = pilot.evaluate_mechanics_gate(
            rows[boundary - 16 : boundary],
            run_contract_sha256=contract_sha,
            window_start=boundary - 15,
            window_end=boundary,
        )
        _write_json(stage / f"pilot_gate_update{boundary:06d}.json", gate)
    checkpoint = stage / f"checkpoint-optstep-{final_update:06d}"
    _write_json(checkpoint / "run_contract.json", contract)
    for relative in (
        "training_state.pt",
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "tokenizer/tokenizer.json",
    ):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    _write_json(
        stage / "latest_checkpoint.json",
        {
            "schema": "t5gemma2-typed-c2-verpo-pilot150-checkpoint-v1",
            "update": final_update,
            "path": str(checkpoint.resolve()),
            "run_contract_sha256": contract_sha,
        },
    )
    if stop_update is not None:
        result = {
            "schema": "t5gemma2-typed-c2-verpo-pilot150-run-v1",
            "status": "stopped_at_window_gate",
            "updates": stop_update,
            "latest_checkpoint": f"checkpoint-optstep-{stop_update:06d}",
            "gate_update": stop_update,
            "window_gate": "STOP",
            "automatic_promotion_performed": False,
            "production_floor_eligible": False,
            "no_frontier_api": True,
            "run_contract_sha256": contract_sha,
        }
    else:
        result = {
            "schema": "t5gemma2-typed-c2-verpo-pilot150-run-v1",
            "status": "complete",
            "updates": 150,
            "latest_checkpoint": "checkpoint-optstep-000150",
            "mechanics_gate": "GO",
            "window_gates_passed": list(GATE_UPDATES),
            "automatic_promotion_performed": False,
            "production_floor_eligible": False,
            "pilot_disposition": "discardable_not_for_automatic_promotion",
            "run_contract_sha256": contract_sha,
        }
    _write_json(stage / "result.json", result)


def test_exact_complete_update150_all_go_permits_evaluation(tmp_path: Path) -> None:
    stage = tmp_path / "pilot"
    _make_stage(stage)
    decision = classify_terminal(stage)
    assert decision["disposition"] == "EVALUATE"
    assert decision["terminal_update"] == 150
    assert decision["evaluation_permitted"] is True
    assert decision["automatic_promotion_performed"] is False


def test_stop_gate_is_clean_no_eval(tmp_path: Path) -> None:
    stage = tmp_path / "pilot"
    _make_stage(stage, stop_update=16)
    decision = classify_terminal(stage)
    assert decision["disposition"] == "STOP_NO_EVAL"
    assert decision["terminal_update"] == 16
    assert decision["evaluation_permitted"] is False


def test_tampered_gate_or_missing_result_blocks_evaluation(tmp_path: Path) -> None:
    stage = tmp_path / "pilot"
    _make_stage(stage)
    gate_path = stage / "pilot_gate_update000144.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["decision"] = "STOP"
    _write_json(gate_path, gate)
    assert classify_terminal(stage)["disposition"] == "BLOCKED_NO_EVAL"
    (stage / "result.json").unlink()
    assert classify_terminal(stage)["disposition"] == "BLOCKED_NO_EVAL"


def test_handoff_waits_for_exited_and_starts_only_exact_eval_program() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "t5gemma2-typed-c2-verpo-pilot150-v1" in text
    assert "t5gemma2-typed-c2-verpo-matched-eval8192-v1" in text
    assert "RUNNING|STARTING" in text and "EXITED) break" in text
    assert text.index('EXITED) break') < text.index('--stage "${STAGE}"')
    assert text.index('decision_two=') < text.index(
        '"${SUPERVISORCTL}" start "${EVAL_PROGRAM}"'
    )
    assert "STOP_NO_EVAL|BLOCKED_NO_EVAL" in text
    assert "automatic_promotion_performed == false" in text
    assert "private_holdback_read == false" in text
    assert "rm " not in text and "Remove-Item" not in text
    assert "holdback_alignment" not in text


def test_handoff_supervisor_is_manual_and_fail_closed() -> None:
    text = CONF.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-c2-verpo-to-matched-eval-handoff-v1]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
