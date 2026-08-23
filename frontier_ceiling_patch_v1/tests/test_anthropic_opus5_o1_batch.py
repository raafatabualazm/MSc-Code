from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import anthropic_opus5_o1_batch as o1
from frontier_core import PreflightError, atomic_write_json


def _summary(*, successes: int, task_prefix: str = "t") -> dict:
    tasks = [
        {"task_id": f"{task_prefix}{index:03d}", "passed": False}
        for index in range(175)
    ]
    return {
        "status": "primary_fixed_cap_complete",
        "metric": "primary_fixed_cap_8192",
        "capacity_adaptive": False,
        "length_is_failure": True,
        "model": "claude-sonnet-5",
        "effort": "high",
        "thinking": {"type": "adaptive"},
        "k": 2,
        "fixed_max_output_tokens": 8192,
        "tasks": 175,
        "logical_slots": 350,
        "bad_gate_threshold_successes": 9,
        "bad_gate_triggered": successes <= 9,
        "config_sha256": "a" * 64,
        "pass_at_2_fixed_8192": {
            "successes": successes,
            "total": 175,
            "rate": successes / 175,
        },
        "task_results": tasks,
    }


def _write_pair(root: Path, *, opus: int, codex: int) -> None:
    for arm, successes in (("opus", opus), ("codex", codex)):
        path = root / arm / "primary_8192_summary.json"
        path.parent.mkdir(parents=True)
        atomic_write_json(path, _summary(successes=successes))


def test_bad_gate_accepts_if_either_complete_sonnet_arm_is_at_most_nine(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, opus=9, codex=14)
    gate = o1.load_sonnet_bad_gate(tmp_path)
    assert gate == o1.load_sonnet_bad_gate(tmp_path)
    assert gate["bad_gate_satisfied"] is True
    assert gate["minimum_successes"] == 9
    assert gate["definition"] == "min(opus_successes,codex_successes)<=9"
    assert gate["budget"]["opus_pair_base_cap_usd"] == 50.176
    assert gate["budget"]["combined_base_cap_usd"] == pytest.approx(90.3168)
    assert gate["budget"]["base_cap_headroom_usd"] == pytest.approx(8.6832)


def test_bad_gate_rejects_if_both_sonnet_arms_exceed_nine(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, opus=10, codex=11)
    with pytest.raises(PreflightError, match="gate is closed"):
        o1.load_sonnet_bad_gate(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "claude-opus-5"),
        ("k", 1),
        ("fixed_max_output_tokens", 16384),
        ("length_is_failure", False),
        ("logical_slots", 349),
    ],
)
def test_bad_gate_rejects_noncanonical_sonnet_summary(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _write_pair(tmp_path, opus=8, codex=12)
    path = tmp_path / "opus" / "primary_8192_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    atomic_write_json(path, data)
    with pytest.raises(PreflightError, match="summary field"):
        o1.load_sonnet_bad_gate(tmp_path)


def test_bad_gate_rejects_different_task_order(tmp_path: Path) -> None:
    _write_pair(tmp_path, opus=8, codex=12)
    path = tmp_path / "codex" / "primary_8192_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["task_results"][0], data["task_results"][1] = (
        data["task_results"][1],
        data["task_results"][0],
    )
    atomic_write_json(path, data)
    with pytest.raises(PreflightError, match="same sealed task order"):
        o1.load_sonnet_bad_gate(tmp_path)


def test_opus_native_request_is_exact_k1_8k_adaptive_high() -> None:
    prior = o1._GATE_ATTESTATION
    o1._GATE_ATTESTATION = {
        "schema": "test",
        "budget": {"within_overall_base_cap": True},
    }
    try:
        with o1.configured_engine():
            request = o1.batch._request_for_spec(
                SimpleNamespace(model="claude-opus-5"),
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "x"},
                    ]
                },
                {
                    "custom_id": "a00_s00_t000_c08192",
                    "cap": 8192,
                },
            )
            assert request == {
                "custom_id": "a00_s00_t000_c08192",
                "params": {
                    "model": "claude-opus-5",
                    "max_tokens": 8192,
                    "system": "s",
                    "messages": [{"role": "user", "content": "x"}],
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                },
            }
    finally:
        o1._GATE_ATTESTATION = prior


def test_strict_two_arm_worst_batch_cap_is_exact() -> None:
    specs = [{"cap": 8192} for _ in range(175)]
    with o1.configured_engine():
        per_arm = o1.batch.worst_batch_cost(specs)
    assert per_arm == pytest.approx(25.088)
    assert per_arm * 2 == pytest.approx(50.176)
    assert 40.1408 + per_arm * 2 <= 99.0


def test_opus_metric_rewrite_preserves_refusal_transparency(tmp_path: Path) -> None:
    marker = {
        "status": "invalid_refusal_dominated",
        "valid_as_capability_ceiling": False,
        "ceiling_claim_allowed": False,
    }
    stop_report = {
        "native_stop_reason_counts": {"refusal": 158, "end_turn": 1},
        "native_stop_category_counts": {"refusal:unspecified": 158},
    }
    atomic_write_json(
        tmp_path / "primary_8192_summary.json",
        {
            "pass_at_2_fixed_8192": {"successes": 0, "total": 175},
            "compile_at_2_fixed_8192": {"successes": 1, "total": 175},
            "bad_gate_threshold_successes": 9,
            "bad_gate_triggered": True,
            "capability_metric_assessment": marker,
            "anthropic_native_stop_report": stop_report,
        },
    )
    atomic_write_json(
        tmp_path / "progress.json",
        {
            "primary_fixed_cap_8192": {
                "pass_at_2_fixed_8192": {"successes": 0, "total": 175},
                "bad_gate_triggered": True,
                "capability_metric_assessment": marker,
                "anthropic_native_stop_report": stop_report,
            },
            "capability_metric_assessment": marker,
            "anthropic_native_stop_report": stop_report,
        },
    )
    prior = o1._GATE_ATTESTATION
    o1._GATE_ATTESTATION = {
        "schema": "test",
        "budget": {"within_overall_base_cap": True},
    }
    try:
        result = o1._rewrite_metric_artifacts(tmp_path)
    finally:
        o1._GATE_ATTESTATION = prior

    assert result is not None
    primary = json.loads(
        (tmp_path / "primary_8192_summary.json").read_text(encoding="utf-8")
    )
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert primary["capability_metric_assessment"] == marker
    assert primary["anthropic_native_stop_report"] == stop_report
    assert progress["primary_fixed_cap_8192"]["capability_metric_assessment"] == marker
    assert progress["capability_metric_assessment"] == marker


def test_specialized_parser_rejects_auto_and_seals_opus_contract() -> None:
    prior = o1._GATE_ATTESTATION
    o1._GATE_ATTESTATION = {
        "schema": "test",
        "budget": {"within_overall_base_cap": True},
    }
    argv = [
        "--action",
        "preflight",
        "--model",
        "claude-opus-5",
        "--input-mode",
        "prematerialized_f2",
        "--k",
        "1",
        "--max-output-tokens",
        "8192",
        "--expected-task-count",
        "175",
        "--screen-cost-cap-usd",
        "25.088",
        "--prompt-jsonl",
        "prompt.jsonl",
        "--prompt-manifest",
        "prompt.manifest.json",
        "--eval-jsonl",
        "eval.jsonl",
        "--eval-seal",
        "eval.seal.json",
        "--pair-manifest",
        "pair.json",
        "--pair-arm-key",
        "opus_real_fn0_cfg",
        "--expected-prompt-jsonl-sha256",
        "a" * 64,
        "--expected-prompt-manifest-sha256",
        "b" * 64,
        "--expected-eval-jsonl-sha256",
        "c" * 64,
        "--expected-eval-seal-sha256",
        "d" * 64,
        "--expected-pair-manifest-sha256",
        "e" * 64,
        "--evaluator-module",
        "evaluator.py",
        "--expected-evaluator-sha256",
        "f" * 64,
        "--dart",
        "dart",
        "--expected-dart-sha256",
        "0" * 64,
    ]
    try:
        with o1.configured_engine():
            args = o1.batch.parse_args(argv)
            assert args.model == "claude-opus-5"
            assert args.k == 1
            assert args.max_output_tokens == 8192
            with pytest.raises(PreflightError, match="auto is disabled"):
                o1.batch.parse_args(["--action", "auto", *argv[2:]])
    finally:
        o1._GATE_ATTESTATION = prior


def test_prior_paid_submission_blocks_any_second_create(tmp_path: Path) -> None:
    config_sha = "c" * 64
    row = {
        "schema": o1.SCHEMA,
        "config_sha256": config_sha,
        "event_type": "batch_submitted",
        "batch_id": "msgbatch_test",
    }
    (tmp_path / "batch_events.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with o1.configured_engine():
        with pytest.raises(
            o1.batch.audited.RunFailure,
            match="exactly one paid Batch creation",
        ):
            o1._assert_no_prior_paid_create(tmp_path, config_sha)


def test_paid_create_requires_both_exact_full_arm_preflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = o1._GATE_ATTESTATION
    gate = {
        "schema": "test-gate",
        "budget": {"within_overall_base_cap": True},
    }
    o1._GATE_ATTESTATION = gate
    monkeypatch.setattr(o1, "OPUS_RUN_ROOT", tmp_path)
    try:
        for arm, expected in o1.SEALED_ARM_IDENTITIES.items():
            arm_dir = tmp_path / arm
            arm_dir.mkdir()
            sealed = {
                "pair_manifest_sha256": o1.PAIR_MANIFEST_SHA256,
                **expected,
            }
            provenance = {
                "status": "preflight_only_complete",
                "tasks_selected": 175,
                "config": {
                    "model_requested": "claude-opus-5",
                    "k": 1,
                    "max_output_tokens": 8192,
                    "expected_evaluator_sha256": o1.EVALUATOR_SHA256,
                    "pair_arm_key": expected["pair_arm_key"],
                    "sealed_inputs": sealed,
                    "anthropic_opus5_o1_gate": {
                        "sonnet_bad_gate_attestation": gate,
                    },
                },
            }
            atomic_write_json(arm_dir / "provenance.json", provenance)
        o1._assert_both_exact_arms_preflighted(tmp_path / "opus")
        (tmp_path / "codex" / "provenance.json").unlink()
        with pytest.raises(
            o1.batch.audited.RunFailure,
            match="both exact O1 arms",
        ):
            o1._assert_both_exact_arms_preflighted(tmp_path / "opus")
    finally:
        o1._GATE_ATTESTATION = prior


def test_single_cap_ladder_turns_length_into_failure_without_retry() -> None:
    with o1.configured_engine():
        assert o1.batch.CAP_LADDER == (8192,)
        plans = [{"task_id": "t0"}]
        specs = o1.batch.pending_request_specs(plans, [], [])
        assert len(specs) == 1
        assert specs[0]["cap"] == 8192
