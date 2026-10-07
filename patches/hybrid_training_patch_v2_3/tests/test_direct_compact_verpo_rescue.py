from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation.direct_compact_qwen_inference import (
    load_rescue_conditioning_plan,
)
from scripts.training import direct_compact_verpo_rescue as rescue


VALID_F2 = (
    "F2\n"
    "C0\n"
    "\n"
    "Ax86_64\n"
    "Ea\n"
    "D\n"
    "S\n"
    "B\n"
    "a{jne}|tb\n"
    "b{ret}|\n"
    "X\n"
)
GUIDE = "Decode the exact lossless-semantic-f2 representation."
SHA = "a" * 64


def _provenance(*, output_sha256: str = SHA) -> dict[str, Any]:
    return {
        "dataset_sha256": "6" * 64,
        "alignment_sha256": "7" * 64,
        "selected_role": "fit",
        "decoder_model": "student",
        "decoder_revision": "rev",
        "model_config_sha256": None,
        "decoder_adapter": None,
        "decoder_adapter_sha256": None,
        "source_overlay_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "codebook_sha256": "3" * 64,
        "codec_sha256": "4" * 64,
        "tokenizer_json_sha256": "5" * 64,
        "direct_prompt_mode": "code_only_v1",
        "attn_implementation": "flash_attention_2",
        "max_new_tokens": 64,
        "num_samples": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "batch_size": 1,
        "limit": 0,
        "seed": 7,
        "bf16": True,
        "fp16": False,
        "precision": "bf16",
        "sampling_seed_policy": (
            "sha256(base_seed,source_plan_sha256,"
            "base_candidate_rank,task_id)"
        ),
        "started_without_terminal_policy": (
            "retry_identical_seeded_batch_with_hash_chained_receipt"
        ),
        "resampled_slots": 0,
        "output_sha256": output_sha256,
    }


def _detail(passes: list[bool], *, diagnostic: str = "") -> dict[str, Any]:
    compiled = True
    return {
        "compiled": compiled,
        "full_pass": compiled and all(passes),
        "test_passes": passes,
        "diagnostic": diagnostic,
    }


def _base_score(
    candidate: str,
    tests: str,
    identity: str,
    **_kwargs: Any,
) -> dict[str, Any]:
    del candidate, tests, identity
    return _detail([False, False], diagnostic="expected 2, got 0")


def _make_plan(
    *,
    select_k: int = 1,
    repairs: int = 1,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    values = candidates or [
        "int f(int x) { return 0; }",
        "int f(int x) { if (x == 1) return 2; return 0; }",
        "int f(int x) { for (;;) { break; } return x; }",
    ]
    system_sha = rescue.sha256_text(GUIDE)
    return rescue.build_pilot_plan(
        [{"task_id": "t0", "predictions": values}],
        [{"task_id": "t0", "feedback_tests": "VISIBLE"}],
        [
            {
                "task_id": "t0",
                "representation_schema": "lossless-semantic-f2",
                "text": VALID_F2,
                "text_sha256": rescue.sha256_text(VALID_F2),
                "system_prompt_sha256": system_sha,
            }
        ],
        inference_provenance=_provenance(),
        source_format_guide=GUIDE,
        source_format_guide_sha256=system_sha,
        input_records={"fixture": {"sha256": SHA, "size_bytes": 1}},
        select_k=select_k,
        repairs_per_candidate=repairs,
        workers=2,
        score_fn=_base_score,
    )


def _accepted_result(group: Mapping[str, Any]) -> dict[str, Any]:
    diagnoses = []
    for candidate in group["candidates"]:
        base_index = int(candidate["base_candidate_index"])
        diagnoses.append(
            {
                "group_index": base_index,
                "accepted": True,
                "rejection_reasons": [],
                "fault_class": "wrong_branch",
                "edit_location": {"operation": "unknown"},
                "evidence": [],
                "explanation": "The visible branch disagrees with execution.",
                "repair_steps": ["Correct the branch and preserve all else."],
            }
        )
    return {
        "schema": "verpo-judge-diagnose-result-v1",
        "guidance_mode": rescue.JUDGE_CALL_MODE,
        "diagnoses": diagnoses,
    }


class _FakeJudge:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def diagnose_group(self, payload: Mapping[str, Any], **kwargs: Any) -> Any:
        del kwargs
        self.calls.append(payload)
        return _accepted_result(
            {
                "candidates": [
                    {"base_candidate_index": row["group_index"]}
                    for row in payload["candidates"]
                ]
            }
        )

    def telemetry(self) -> dict[str, Any]:
        return {"receipt_count": 0, "calls": len(self.calls)}

    def receipt_attestation_since(self, _cursor: int) -> None:
        return None


def _diagnoses(plan: Mapping[str, Any]) -> dict[str, Any]:
    judge = _FakeJudge()
    result = rescue.diagnose_pilot_plan(plan, judge=judge)
    assert len(judge.calls) == len(plan["groups"])
    return result


def _repair_run(
    plan: Mapping[str, Any],
    conditioning: Mapping[str, Any],
    code: str,
    *,
    output_sha256: str,
) -> dict[str, Any]:
    provenance = _provenance(output_sha256=output_sha256)
    provenance["rescue_conditioning"] = {
        "schema": conditioning["schema"],
        "sha256": rescue.canonical_sha256(conditioning),
        "arm": conditioning["arm"],
        "base_candidate_rank": conditioning["base_candidate_rank"],
        "repairs_per_candidate": conditioning["repairs_per_candidate"],
        "source_plan_sha256": conditioning["source_plan_sha256"],
    }
    provenance["prediction_token_ids_persisted"] = True
    row = conditioning["rows"][0]
    return {
        "outputs": [
            {
                "task_id": "t0",
                "predictions": [code],
                "prediction_token_ids": [[7, 8]],
                "conditioning_sha256": row["conditioning_sha256"],
            }
        ],
        "provenance": provenance,
        "output_file_sha256": output_sha256,
    }


def test_diversity_and_power_are_predeclared_not_rollout_index_based() -> None:
    values = [
        "int f() { return 0; }",
        "int f() { return 1; }",
        "int f() { while (true) { break; } return 9; }",
    ]
    first, report = rescue.max_min_diverse_indices(values, 2)
    permuted = [values[2], values[0], values[1]]
    second, _ = rescue.max_min_diverse_indices(permuted, 2)

    assert {values[index] for index in first} == {
        permuted[index] for index in second
    }
    assert report["rollout_index_used_as_diversity_measure"] is False
    power = rescue.mcnemar_power_plan(10)
    assert power["adequately_powered_under_assumption"] is False
    assert "Underpowered" in power["warning"]


def test_plan_selects_only_all_zero_group_and_contains_no_private_tests() -> None:
    plan = _make_plan(select_k=2, repairs=3)

    assert len(plan["groups"]) == 1
    assert len(plan["groups"][0]["candidates"]) == 2
    assert plan["budget"]["judge_calls_planned"] == 1
    assert plan["budget"]["repair_slots_planned_total"] == 24
    assert all(
        "reward_holdback_tests" not in group
        and "acceptance_tests" not in group
        for group in plan["groups"]
    )


def test_one_judge_call_derives_both_feedback_arms_and_resume_reuses() -> None:
    plan = _make_plan()
    judge = _FakeJudge()
    starts: list[Mapping[str, Any]] = []
    terminals: list[Mapping[str, Any]] = []
    first = rescue.diagnose_pilot_plan(
        plan,
        judge=judge,
        before_call=starts.append,
        after_call=terminals.append,
    )
    assert len(judge.calls) == len(starts) == len(terminals) == 1
    assert rescue._has_forbidden_api_key(judge.calls[0]) is None

    modes = first["rows"][0]["modes"]
    with_steps = modes["diagnosis_and_steps"]["diagnoses"][0]
    diagnosis_only = modes["diagnosis_only"]["diagnoses"][0]
    assert diagnosis_only["repair_steps"] == []
    assert {
        key: value
        for key, value in with_steps.items()
        if key != "repair_steps"
    } == {
        key: value
        for key, value in diagnosis_only.items()
        if key != "repair_steps"
    }

    replay = _FakeJudge()
    resumed = rescue.diagnose_pilot_plan(
        plan,
        judge=replay,
        completed_results={
            terminals[0]["call_key"]: terminals[0]["result"]
        },
    )
    assert replay.calls == []
    assert resumed["rows"] == first["rows"]


def test_materialization_has_all_four_fixed_slot_arms_and_loads(
    tmp_path: Path,
) -> None:
    plan = _make_plan(select_k=2, repairs=2)
    materialized = rescue.materialize_conditioning_plans(
        plan, _diagnoses(plan)
    )
    manifest = rescue.write_materialized_plans(
        materialized, tmp_path / "plans"
    )

    assert len(manifest["plans"]) == 8
    assert set(manifest["arms"]) == set(rescue.ARM_ORDER)
    for record in manifest["plans"]:
        loaded = load_rescue_conditioning_plan(
            record["artifact"]["path"]
        )
        assert len(loaded["rows"]) == 1
        assert loaded["repairs_per_candidate"] == 2


def test_score_uses_visible_selection_then_holdback_gates_exports() -> None:
    plan = _make_plan()
    materialized = rescue.materialize_conditioning_plans(
        plan, _diagnoses(plan)
    )
    code_by_arm = {
        "plain_resample": "FULL_BOTH",
        "compiler_only": "PARTIAL_BOTH",
        "diagnosis_only": "VISIBLE_ONLY",
    }
    runs: dict[str, Mapping[str, Any]] = {}
    for index, (arm, code) in enumerate(code_by_arm.items(), start=1):
        key = rescue.plan_key(arm, 0)
        runs[key] = _repair_run(
            plan,
            materialized["plans"][key],
            code,
            output_sha256=f"{index}" * 64,
        )

    private_calls: list[str] = []

    def scorer(
        code: str,
        tests: str,
        identity: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if tests == "VISIBLE":
            outcomes = {
                "FULL_BOTH": [True, True],
                "PARTIAL_BOTH": [True, False],
                "VISIBLE_ONLY": [True, False],
            }.get(code, [False, False])
        else:
            private_calls.append(code)
            outcomes = {
                "FULL_BOTH": [True, True],
                "PARTIAL_BOTH": [True, False],
                "VISIBLE_ONLY": [False, False],
            }.get(code, [False, False])
        return _detail(outcomes, diagnostic=identity)

    result = rescue.score_rescue_outputs(
        plan,
        materialized,
        runs,
        [{"task_id": "t0", "reward_holdback_tests": "PRIVATE"}],
        private_holdback_commitment={
            "sha256": "f" * 64,
            "size_bytes": 9,
        },
        workers=4,
        score_fn=scorer,
    )

    assert [row["code"] for row in result["rs_sft_targets"]] == [
        "FULL_BOTH"
    ]
    assert [row["chosen"] for row in result["preference_pairs"]] == [
        "PARTIAL_BOTH"
    ]
    assert "VISIBLE_ONLY" not in {
        row["chosen"] for row in result["preference_pairs"]
    }
    assert "diagnosis_and_steps" not in private_calls
    assert result["report"]["metrics_by_arm"]["diagnosis_and_steps"][
        "missing_output_slots"
    ] == 1
    assert result["report"]["metrics_by_arm"]["diagnosis_only"][
        "visible_change_x_private_change_2x2"
    ]["visible_only_overfit"] == 1
    contrast = result["report"]["primary_paired_contrast"]
    assert contrast["predeclared"] is True
    assert contrast["pairs"] == 1
    assert contrast["table"] == {
        "neither_rescued": 0,
        "control_only_rescued": 1,
        "treatment_only_rescued": 0,
        "both_rescued": 0,
    }
    assert contrast["exact_two_sided_binomial_p_value"] == 1.0
    assert all(
        row["eligible_for_on_policy_verpo_update"] is False
        for row in result["preference_pairs"]
    )


def test_feedback_report_requires_exact_report_and_output_records(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.jsonl"
    f2 = tmp_path / "f2.jsonl"
    manifest = tmp_path / "f2.manifest.json"
    rollout.write_text("{}\n", encoding="utf-8")
    f2.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "feedback.json"
    report = {
        "schema": rescue.FEEDBACK_VIEW_REPORT_SCHEMA,
        "status": "complete",
        "outputs": {
            "rollout": rescue.file_record(rollout),
            "f2": rescue.file_record(f2),
            "f2_manifest": rescue.file_record(manifest),
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = rescue.sha256_file(report_path)

    validated = rescue.validate_feedback_view_report(
        report_path,
        report_sha,
        expected_outputs={
            "rollout": rollout,
            "f2": f2,
            "f2_manifest": manifest,
        },
    )
    assert validated["report_record"]["sha256"] == report_sha
    rollout.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(rescue.RescueError, match="differs"):
        rescue.validate_feedback_view_report(
            report_path,
            report_sha,
            expected_outputs={"rollout": rollout},
        )


def test_diagnosis_journal_reuses_terminal_and_rejects_orphan(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "diagnosis.jsonl"
    call_key = "b" * 64
    contract = {
        "source_plan_sha256": "c" * 64,
        "ordered_call_keys": [call_key],
        "judge": {"model": "fake"},
        "grounding_validator_schema": "grounding",
        "result_schema": rescue.DIAGNOSIS_ARTIFACT_SCHEMA,
    }
    rescue._append_diagnosis_journal_event(
        journal,
        event="journal_header",
        payload={
            "contract": contract,
            "contract_sha256": rescue.canonical_sha256(contract),
        },
    )
    rescue._append_diagnosis_journal_event(
        journal,
        event="diagnosis_started",
        payload={"call_key": call_key},
    )
    with pytest.raises(rescue.RescueError, match="automatic retry is forbidden"):
        rescue.inspect_diagnosis_journal(
            journal, expected_contract=contract
        )

    terminal_journal = tmp_path / "terminal.jsonl"
    rescue._append_diagnosis_journal_event(
        terminal_journal,
        event="journal_header",
        payload={
            "contract": contract,
            "contract_sha256": rescue.canonical_sha256(contract),
        },
    )
    rescue._append_diagnosis_journal_event(
        terminal_journal,
        event="diagnosis_started",
        payload={"call_key": call_key},
    )
    result = {"diagnoses": [], "guidance_mode": rescue.JUDGE_CALL_MODE}
    rescue._append_diagnosis_journal_event(
        terminal_journal,
        event="diagnosis_terminal",
        payload={
            "call_key": call_key,
            "result": result,
            "result_sha256": rescue.canonical_sha256(result),
            "receipt_attestation": None,
        },
    )
    state = rescue.inspect_diagnosis_journal(
        terminal_journal, expected_contract=contract
    )
    assert state["completed_results"] == {call_key: result}


def test_rejected_receipt_resume_keeps_receipt_and_accepted_id_counts_separate(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "rejected-receipt.jsonl"
    call_key = "d" * 64
    contract = {
        "source_plan_sha256": "e" * 64,
        "ordered_call_keys": [call_key],
        "judge": {"model": "fake"},
        "grounding_validator_schema": "grounding",
        "result_schema": rescue.DIAGNOSIS_ARTIFACT_SCHEMA,
    }
    rescue._append_diagnosis_journal_event(
        journal,
        event="journal_header",
        payload={
            "contract": contract,
            "contract_sha256": rescue.canonical_sha256(contract),
        },
    )
    rescue._append_diagnosis_journal_event(
        journal,
        event="diagnosis_started",
        payload={"call_key": call_key},
    )
    receipt_base = {
        "schema": "verpo-judge-response-receipt-v1",
        "receipt_index": 1,
        "previous_receipt_sha256": (
            rescue.DIAGNOSIS_JOURNAL_GENESIS_SHA256
        ),
        "response": {"id": "rejected-response"},
        "validation": {"accepted": False},
    }
    receipt = {
        **receipt_base,
        "receipt_sha256": rescue.canonical_sha256(receipt_base),
    }
    result = {"diagnoses": [], "guidance_mode": rescue.JUDGE_CALL_MODE}
    attestation = {
        "receipt_count_before_step": 0,
        "receipt_count_this_step": 1,
        "cumulative_receipt_count": 1,
        "previous_receipt_chain_sha256": (
            rescue.DIAGNOSIS_JOURNAL_GENESIS_SHA256
        ),
        "cumulative_receipt_chain_sha256": receipt["receipt_sha256"],
        "receipts": [receipt],
    }
    rescue._append_diagnosis_journal_event(
        journal,
        event="diagnosis_terminal",
        payload={
            "call_key": call_key,
            "result": result,
            "result_sha256": rescue.canonical_sha256(result),
            "receipt_attestation": attestation,
        },
    )
    state = rescue.inspect_diagnosis_journal(
        journal, expected_contract=contract
    )
    assert state["receipt_count"] == 1
    assert state["response_id_sha256s"] == []

    judge = rescue.VerpoJudge(
        model="fake",
        base_url="https://fake.invalid",
        api_style="openai_compatible_chat",
        max_tokens=128,
        retry_max_tokens=128,
        max_retries=0,
        completion_retries=0,
        max_calls=0,
        receipt_chain_seed=state["receipt_chain_sha256"],
        receipt_index_offset=state["receipt_count"],
        prior_response_id_sha256s=state["response_id_sha256s"],
    )
    assert judge.telemetry()["receipt_count"] == 1
    assert judge.telemetry()["unique_response_ids"] == 0


def test_diagnose_command_is_exactly_resumable_without_second_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan()
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "diagnoses.json"
    journal_path = tmp_path / "diagnoses.journal.jsonl"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    class FakeCommandJudge:
        provider_calls = 0

        def __init__(self, **kwargs: Any) -> None:
            self.model = kwargs.get("model") or "fake-model"
            self.base_url = kwargs.get("base_url") or "https://fake.invalid"
            self.api_style = (
                kwargs.get("api_style") or "openai_compatible_chat"
            )
            self.concurrency = kwargs.get("concurrency") or 1
            self.max_tokens = kwargs.get("max_tokens") or 128
            self.timeout_seconds = kwargs.get("timeout_seconds") or 10.0
            self.max_retries = kwargs.get("max_retries", 0)
            self.completion_retries = kwargs.get("completion_retries", 0)
            self.retry_max_tokens = kwargs.get("retry_max_tokens") or 128
            self.thinking_mode = (
                kwargs.get("thinking_mode") or "provider_default"
            )
            self.reasoning_effort = (
                kwargs.get("reasoning_effort") or "high"
            )
            self.reasoning_mode = kwargs.get("reasoning_mode") or "standard"
            self.chat_json_schema = bool(
                kwargs.get("chat_json_schema", False)
            )
            self.max_calls = kwargs.get("max_calls")
            self.fail_closed = True
            self.receipt_count = int(kwargs.get("receipt_index_offset", 0))
            self.receipt_head = kwargs.get(
                "receipt_chain_seed",
                rescue.DIAGNOSIS_JOURNAL_GENESIS_SHA256,
            )

        def telemetry(self) -> dict[str, Any]:
            return {
                "model": self.model,
                "base_url": self.base_url,
                "api_style": self.api_style,
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
                "completion_retries_allowed": self.completion_retries,
                "thinking_mode": self.thinking_mode,
                "reasoning_effort": self.reasoning_effort,
                "reasoning_mode": self.reasoning_mode,
                "chat_json_schema": self.chat_json_schema,
                "fail_closed": True,
                "receipt_count": self.receipt_count,
                "receipt_chain_sha256": self.receipt_head,
            }

        def diagnose_group(
            self, payload: Mapping[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            type(self).provider_calls += 1
            return _accepted_result(
                {
                    "candidates": [
                        {"base_candidate_index": row["group_index"]}
                        for row in payload["candidates"]
                    ]
                }
            )

        def receipt_attestation_since(self, _cursor: int) -> None:
            return None

    monkeypatch.setattr(rescue, "VerpoJudge", FakeCommandJudge)
    args = argparse.Namespace(
        plan=str(plan_path),
        output=str(output_path),
        receipt_journal=str(journal_path),
        model="fake-model",
        base_url="https://fake.invalid",
        api_style="openai_compatible_chat",
        max_tokens=128,
        timeout_seconds=10.0,
        max_retries=0,
        thinking_mode="provider_default",
        reasoning_effort="high",
        max_calls=0,
    )
    rescue._diagnose_command(args)
    first = output_path.read_bytes()
    rescue._diagnose_command(args)

    assert output_path.read_bytes() == first
    assert FakeCommandJudge.provider_calls == 1
    events = rescue._load_diagnosis_journal_events(journal_path)
    assert [event["event"] for event in events] == [
        "journal_header",
        "diagnosis_started",
        "diagnosis_terminal",
        "diagnosis_complete",
    ]


def test_bundle_missing_outputs_fails_by_default_and_can_be_sealed(
    tmp_path: Path,
) -> None:
    plan = _make_plan()
    materialized = rescue.materialize_conditioning_plans(
        plan, _diagnoses(plan)
    )
    written_dir = tmp_path / "plans"
    rescue.write_materialized_plans(materialized, written_dir)
    loaded = rescue._load_materialized_dir(written_dir)
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    with pytest.raises(rescue.RescueError, match="lack inference outputs"):
        rescue.build_inference_bundle(
            plan, loaded, outputs, allow_missing=False
        )
    bundle = rescue.build_inference_bundle(
        plan, loaded, outputs, allow_missing=True
    )
    assert bundle["status"] == "complete_with_missing_runs"
    assert len(bundle["missing_generatable_runs"]) == 4
    assert bundle["missing_runs_count_as_itt_failures"] is True


def test_repair_checkpoint_mismatch_fails_closed() -> None:
    plan = _make_plan()
    materialized = rescue.materialize_conditioning_plans(
        plan, _diagnoses(plan)
    )
    key = rescue.plan_key("plain_resample", 0)
    run = _repair_run(
        plan,
        materialized["plans"][key],
        "FULL_BOTH",
        output_sha256="9" * 64,
    )
    run["provenance"]["source_overlay_sha256"] = "0" * 64
    with pytest.raises(rescue.RescueError, match="different student"):
        rescue.score_rescue_outputs(
            plan,
            materialized,
            {key: run},
            [{"task_id": "t0", "reward_holdback_tests": "PRIVATE"}],
            private_holdback_commitment={
                "sha256": "f" * 64,
                "size_bytes": 1,
            },
            score_fn=_base_score,
        )


def test_cross_arm_generation_binding_includes_precision_and_rng() -> None:
    plan = _make_plan()
    materialized = rescue.materialize_conditioning_plans(
        plan, _diagnoses(plan)
    )
    first_key = rescue.plan_key("plain_resample", 0)
    second_key = rescue.plan_key("compiler_only", 0)
    first = _repair_run(
        plan,
        materialized["plans"][first_key],
        "FIRST",
        output_sha256="8" * 64,
    )
    second = _repair_run(
        plan,
        materialized["plans"][second_key],
        "SECOND",
        output_sha256="9" * 64,
    )
    second["provenance"]["bf16"] = False
    second["provenance"]["precision"] = "fp32"
    with pytest.raises(rescue.RescueError, match="generation settings"):
        rescue.score_rescue_outputs(
            plan,
            materialized,
            {first_key: first, second_key: second},
            [{"task_id": "t0", "reward_holdback_tests": "PRIVATE"}],
            private_holdback_commitment={
                "sha256": "f" * 64,
                "size_bytes": 1,
            },
            score_fn=_base_score,
        )


def test_diagnose_rejects_any_sdk_or_completion_retry_budget(
    tmp_path: Path,
) -> None:
    plan = _make_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    args = argparse.Namespace(
        plan=str(plan_path),
        output=str(tmp_path / "diagnoses.json"),
        receipt_journal=str(tmp_path / "journal.jsonl"),
        model="fake",
        base_url="https://fake.invalid",
        api_style="openai_compatible_chat",
        max_tokens=128,
        timeout_seconds=10.0,
        max_retries=1,
        thinking_mode="provider_default",
        reasoning_effort="high",
        max_calls=0,
    )
    with pytest.raises(rescue.RescueError, match="max_retries=0"):
        rescue._diagnose_command(args)


def test_cli_exposes_plan_diagnose_materialize_bundle_and_score() -> None:
    for command in ("plan", "diagnose", "materialize", "bundle", "score"):
        with pytest.raises(SystemExit) as exc:
            rescue.parse_args([command, "--help"])
        assert exc.value.code == 0
