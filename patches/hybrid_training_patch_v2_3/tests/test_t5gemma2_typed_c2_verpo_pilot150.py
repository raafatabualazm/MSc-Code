import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.training import t5gemma2_typed_c2_verpo_pilot150 as pilot


def _trajectory(*, eos: bool = True) -> dict:
    actions = [7, pilot.EXPECTED_EOS_TOKEN_ID if eos else 9]
    return {
        "actions": actions,
        "action_tokens": len(actions),
        "sampled_pad_before_eos": 0,
    }


def _metric(update: int, *, active: bool = True, eos: bool = True) -> dict:
    if active:
        global_advantage = [0.75, -0.25, -0.25, -0.25]
        local = [0.5, -0.5, 0.0, 0.0]
        compile_advantage = [0.0, 0.0, 0.0, 0.0]
        unified = [
            global_advantage[index] + local[index] for index in range(4)
        ]
        active_count = 4
        policy_loss = 0.1
        grad_norm = 1.0
        optimizer_step = True
    else:
        global_advantage = local = compile_advantage = unified = [0.0] * 4
        active_count = 0
        policy_loss = 0.0
        grad_norm = 0.0
        optimizer_step = False
    return {
        "schema": pilot.ROLLOUT_SCHEMA,
        "update": update,
        "run_contract_sha256": "a" * 64,
        "trajectory_count": 4,
        "active_policy_trajectories": active_count,
        "policy_loss": policy_loss,
        "sft_replay_loss": 0.0,
        "grad_norm": grad_norm,
        "optimizer_step": optimizer_step,
        "max_on_policy_logprob_drift": 1e-5 if active else 0.0,
        "sampled_pad_before_eos": 0,
        "task_records": [
            {
                "base_reward": {
                    "global_advantages": global_advantage,
                    "local_advantages": local,
                    "compile_advantages": compile_advantage,
                    "unified_advantages": unified,
                },
                "trajectories": [_trajectory(eos=eos) for _ in range(4)],
            }
        ],
    }


def _profile_args() -> argparse.Namespace:
    return argparse.Namespace(
        group_size=4,
        repair_group_size=4,
        max_repair_parents=2,
        tasks_per_update=1,
        max_updates=150,
        temperature=0.8,
        max_new_tokens=8192,
        max_source_tokens=32768,
        max_target_tokens=32768,
        verpo_alpha=2.0,
        local_weight=1.0,
        compile_weight=0.25,
        learning_rate=1e-6,
        weight_decay=0.0,
        max_grad_norm=1.0,
        ppo_clip=0.0,
        sft_replay_weight=0.0,
        on_policy_logprob_tolerance=2e-4,
        reward_workers=4,
        reward_timeout=30,
        reward_stability_runs=1,
        checkpoint_interval=1,
        keep_last_checkpoints=2,
        seed=42,
        attn_implementation="sdpa",
        bf16=True,
    )


def test_exact_profile_is_one_pass_pure_reward_and_8192() -> None:
    args = _profile_args()
    pilot._validate_exact_profile(args)  # noqa: SLF001
    args.sft_replay_weight = 0.02
    with pytest.raises(ValueError, match="profile differs"):
        pilot._validate_exact_profile(args)  # noqa: SLF001


def test_window_gate_go_and_stop() -> None:
    go_rows = [_metric(update) for update in range(17, 33)]
    gate = pilot.evaluate_mechanics_gate(
        go_rows,
        run_contract_sha256="a" * 64,
        window_start=17,
        window_end=32,
    )
    assert gate["decision"] == "GO"
    assert gate["criteria"]["base_unified_active_groups"]["observed"] == 16
    assert gate["criteria"]["base_local_noncollinear_groups"]["observed"] == 16
    assert gate["gate_sha256"] == canonical_sha256(
        {key: value for key, value in gate.items() if key != "gate_sha256"}
    )

    stop_rows = [_metric(update, active=False) for update in range(1, 17)]
    stopped = pilot.evaluate_mechanics_gate(
        stop_rows,
        run_contract_sha256="a" * 64,
        window_start=1,
        window_end=16,
    )
    assert stopped["decision"] == "STOP"
    assert stopped["criteria"]["zero_policy_updates"]["observed"] == 16


def test_window_gate_fails_closed_on_no_eos_or_degenerate_active_update() -> None:
    no_eos = [_metric(update) for update in range(1, 17)]
    no_eos[0] = _metric(1, eos=False)
    with pytest.raises(ValueError, match="EOS/action invariants"):
        pilot.evaluate_mechanics_gate(
            no_eos,
            run_contract_sha256="a" * 64,
            window_start=1,
            window_end=16,
        )
    zero_grad = [_metric(update) for update in range(1, 17)]
    zero_grad[0]["grad_norm"] = 0.0
    with pytest.raises(ValueError, match="active update is degenerate"):
        pilot.evaluate_mechanics_gate(
            zero_grad,
            run_contract_sha256="a" * 64,
            window_start=1,
            window_end=16,
        )
    zero_loss = [_metric(update) for update in range(1, 17)]
    zero_loss[0]["policy_loss"] = 0.0
    assert (
        pilot.evaluate_mechanics_gate(
            zero_loss,
            run_contract_sha256="a" * 64,
            window_start=1,
            window_end=16,
        )["decision"]
        == "GO"
    )


def test_repair_diagnostic_uses_candidate_only_neutral_main(monkeypatch) -> None:
    calls = []

    def visible(candidate, tests, task_id, **kwargs):
        del candidate, tests, task_id, kwargs
        return {
            "compiled": False,
            "full_pass": False,
            "test_passes": [False, False],
            "diagnostic": "visible oracle diagnostic",
        }

    def neutral(candidate, tests, task_id, **kwargs):
        calls.append((candidate, tests, task_id, kwargs))
        return False, False, "neutral syntax error", "unused"

    monkeypatch.setattr(pilot, "_BASE_SCORE_CANDIDATE", visible)
    monkeypatch.setattr(pilot.engine, "evaluate_dart_jit_tests_detail", neutral)
    detail = pilot.score_dart_candidate_neutral_repair(
        "int fn0() => ;",
        "void main(){ expect(fn0(), 7); }",
        "task",
        timeout=30,
        stability_runs=1,
    )
    assert calls[0][1] == "void main() {}"
    assert detail["diagnostic"] == "neutral syntax error"
    assert detail["repair_compiled"] is False
    assert detail["repair_diagnostic_source"] == "candidate_only_plus_neutral_main"


def test_update16_is_clean_pause_then_same_gate_allows_resume(
    tmp_path: Path, monkeypatch
) -> None:
    contract = {"schema": pilot.RUN_SCHEMA, "status": "training"}
    contract_sha = canonical_sha256(contract)
    (tmp_path / "run_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    metrics = [_metric(update) for update in range(1, 17)]
    for row in metrics:
        row["run_contract_sha256"] = contract_sha
    (tmp_path / "rollout_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
    )
    generated = SimpleNamespace(
        actions=(7, pilot.EXPECTED_EOS_TOKEN_ID), sampled_pad_before_eos=0
    )
    monkeypatch.setattr(pilot, "_BASE_GENERATE_GROUP", lambda **kwargs: [generated])
    monkeypatch.setattr(pilot, "_GATE_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pilot, "_BASE_GROUPS_STARTED", 16)
    monkeypatch.setattr(pilot, "_CONTINUE_AFTER_GATE16", False)
    with pytest.raises(pilot._MandatoryGatePause):  # noqa: SLF001
        pilot._guarded_generate_group(state_kind="base")  # noqa: SLF001
    phase = json.loads((tmp_path / "phase_status.json").read_text(encoding="utf-8"))
    gate = json.loads(
        (tmp_path / "pilot_gate_update000016.json").read_text(encoding="utf-8")
    )
    assert phase["status"] == "awaiting_explicit_resume_after_gate16"
    assert gate["decision"] == "GO"

    monkeypatch.setattr(pilot, "_BASE_GROUPS_STARTED", 16)
    monkeypatch.setattr(pilot, "_CONTINUE_AFTER_GATE16", True)
    assert pilot._guarded_generate_group(state_kind="base") == [generated]  # noqa: SLF001
    assert pilot._BASE_GROUPS_STARTED == 17  # noqa: SLF001


def test_target_free_task_view_loader_rejects_gold_field(tmp_path: Path) -> None:
    view = tmp_path / "view.jsonl"
    manifest_path = tmp_path / "manifest.json"
    source_hashes = []
    rows = []
    task_ids = []
    for position in range(pilot.EXPECTED_PROXY_TASKS):
        task_id = f"task-{position}"
        task_ids.append(task_id)
        source = f"typed source {position}"
        tests = "void main() { expect(1, 1); }"
        source_hash = pilot.engine.sha256_text(source)
        source_hashes.append(source_hash)
        rows.append(
            {
                "schema": pilot.TASK_VIEW_SCHEMA,
                "position": position,
                "task_id": task_id,
                "source": source,
                "source_sha256": source_hash,
                "typed_contract_sha256": "b" * 64,
                "feedback_tests": tests,
                "feedback_tests_sha256": pilot.engine.sha256_text(tests),
                "model_visible_fields": ["opaque_typed_contract", "F2.text"],
                "target_or_gold_present": False,
                "private_holdback_present": False,
            }
        )
    # The production loader pins this hash. Unit-test the structural rejection
    # before reaching that identity check by temporarily using the fixture hash.
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    view.write_text(payload, encoding="utf-8", newline="")
    body = {
        "schema": pilot.TASK_VIEW_MANIFEST_SCHEMA,
        "status": "complete",
        "rows": 150,
        "task_view": {
            "path": str(view.resolve()),
            "sha256": pilot._sha256_file(view),  # noqa: SLF001
            "rows": 150,
            "ordered_task_ids_sha256": pilot.EXPECTED_PROXY_TASK_IDS_SHA256,
            "ordered_source_sha256s_sha256": canonical_sha256(source_hashes),
        },
        "selection": {
            "ordered_task_ids_sha256": pilot.EXPECTED_PROXY_TASK_IDS_SHA256,
            "prior_candidates_actions_logprobs_rewards_reused": False,
            "proxy_summary_sha256": pilot.EXPECTED_PROXY_SUMMARY_SHA256,
            "proxy_journal_sha256": pilot.EXPECTED_PROXY_JOURNAL_SHA256,
            "proxy_chain_head_sha256": pilot.EXPECTED_PROXY_CHAIN_HEAD_SHA256,
            "proxy_contract_sha256": pilot.EXPECTED_PROXY_CONTRACT_SHA256,
        },
        "privacy": {
            "gold_or_target_in_task_view": False,
            "acceptance_tests_in_task_view": False,
            "private_holdback_in_task_view": False,
            "visible_train_feedback_tests_in_task_view": True,
        },
        "source_boundary": {"rows": 2386, "validated": True},
        "runtime": {
            "builder_sha256": pilot._sha256_file(  # noqa: SLF001
                Path(pilot.__file__).resolve().parents[1]
                / "preprocessing/build_t5gemma2_typed_c2_verpo_task_view.py"
            ),
            "typed_source_builder_sha256": pilot._sha256_file(  # noqa: SLF001
                Path(pilot.typed_sft.__file__).resolve()
            ),
            "pilot_profile_sha256": pilot._sha256_file(  # noqa: SLF001
                Path(pilot.__file__).resolve()
            ),
        },
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows[0]["dart_source"] = "gold"
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    view.write_text(payload, encoding="utf-8", newline="")
    manifest["task_view"]["sha256"] = pilot._sha256_file(view)  # noqa: SLF001
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(body)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row 0 differs"):
        pilot.load_task_view(view, manifest_path)
