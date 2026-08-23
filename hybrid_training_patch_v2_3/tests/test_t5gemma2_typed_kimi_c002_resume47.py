from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    sha256_file,
)
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_api_rescue_c002_resume47 as adapter
from scripts.training import t5gemma2_typed_kimi_c002_resume47 as controller


def test_resume_budget_and_schedule_contract() -> None:
    assert adapter.TAIL_TASKS == 47
    assert adapter.MAX_INPUT_TOKENS == 30_720
    assert adapter.EXPECTED_MAX_TAIL_PROMPT_BYTE_UPPER_BOUND == 29_901
    assert controller.INITIAL_WORST_USD == controller.Decimal("7.219200")
    assert controller.RETRY_WORST_USD_PER_TASK == controller.Decimal("0.215040")
    assert controller.MAX_RETRY_TASKS == 4
    assert controller.EXPECTED_CUMULATIVE_PREFIX_SPEND == controller.Decimal(
        "4.325709"
    )


def test_retry_is_all_or_none_and_never_exceeds_four_tasks() -> None:
    abundant = controller.Decimal("8")
    assert controller._retry_should_skip(retry_tasks=0, remaining=abundant) is False
    assert controller._retry_should_skip(retry_tasks=4, remaining=abundant) is False
    assert controller._retry_should_skip(retry_tasks=5, remaining=abundant) is True
    assert controller._retry_should_skip(
        retry_tasks=4, remaining=controller.Decimal("0.86")
    ) is True
    assert (
        controller.STATED_BALANCE
        - controller.EXPECTED_CUMULATIVE_PREFIX_SPEND
        - controller.INITIAL_WORST_USD
        == controller.Decimal("0.895091")
    )


def test_initial_selection_reconstructs_50_then_slices_exact_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = list(adapter.PREFIX_TASK_IDS) + [f"tail-{index}" for index in range(47)]
    # The production digest is pinned; use a fixture digest only inside this test.
    monkeypatch.setattr(adapter, "TAIL_SCHEDULE_SHA256", canonical_sha256(ids[3:]))
    rows = [(i, SimpleNamespace(task_id=task_id), {}) for i, task_id in enumerate(ids)]
    monkeypatch.setattr(
        adapter.c002,
        "phase_selection",
        lambda **_kwargs: (rows, {"mode": cascade.PHASE_KIMI_INITIAL}),
    )
    observed: list[str] = []

    def validate(selected):
        observed.extend(row[1].task_id for row in selected)
        return [], []

    monkeypatch.setattr(adapter, "_validate_reconstructed_source", validate)
    selected, record = adapter.phase_selection(
        args=SimpleNamespace(phase=cascade.PHASE_KIMI_INITIAL, max_tasks=47),
        all_visible_zero=rows,
        prior_records=[],
    )
    assert observed == ids
    assert [row[1].task_id for row in selected] == ids[3:]
    assert not set(adapter.PREFIX_TASK_IDS) & {row[1].task_id for row in selected}
    assert record["paid_prefix_recalled"] is False


def test_prompt_preflight_checks_every_tail_slot_before_live_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_bound = adapter.EXPECTED_MAX_TAIL_PROMPT_BYTE_UPPER_BOUND
    overhead = len(base.SYSTEM_PROMPT.encode("utf-8")) + 1024
    prompts = ["x"] * 46 + ["x" * (target_bound - overhead)]
    slots = [
        SimpleNamespace(task_id=f"task-{index}", prompt=prompt)
        for index, prompt in enumerate(prompts)
    ]
    monkeypatch.setattr(adapter, "TAIL_SCHEDULE_SHA256", canonical_sha256([s.task_id for s in slots]))
    monkeypatch.setattr(adapter, "_ORIGINAL_BUILD_SLOTS", lambda *_args, **_kwargs: slots)
    monkeypatch.setattr(base, "_slot_binding", lambda slot: {"task_id": slot.task_id})
    monkeypatch.setattr(adapter, "_PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(adapter, "_REQUESTED_PHASE", cascade.PHASE_KIMI_INITIAL)
    assert adapter._preflight_build_slots([], samples_per_parent=1) == slots
    record = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert record["slots_checked"] == 47
    assert record["max_prompt_byte_upper_bound"] == 29_901
    assert record["all_selected_slots_checked_before_first_live_call"] is True
    assert record["provider_credentials_required"] is False


def test_prompt_preflight_rejects_one_oversized_slot_without_partial_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overhead = len(base.SYSTEM_PROMPT.encode("utf-8")) + 1024
    slot = SimpleNamespace(
        task_id="oversized", prompt="x" * (adapter.MAX_INPUT_TOKENS - overhead + 1)
    )
    monkeypatch.setattr(adapter, "_ORIGINAL_BUILD_SLOTS", lambda *_a, **_k: [slot])
    monkeypatch.setattr(adapter, "_PREFLIGHT_PATH", tmp_path / "must-not-exist.json")
    monkeypatch.setattr(adapter, "_REQUESTED_PHASE", cascade.PHASE_KIMI_RETRY)
    with pytest.raises(ValueError, match="exceeds"):
        adapter._preflight_build_slots([], samples_per_parent=1)
    assert not (tmp_path / "must-not-exist.json").exists()


def test_source_loader_accepts_only_exact_three_paid_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "typed_api_rescue.journal.jsonl"
    contract = {
        "schema": cascade.RUN_SCHEMA,
        "phase": cascade.PHASE_KIMI_INITIAL,
        "cohort_index": 2,
        "inputs": {"sealed": True},
        "selection": {
            "scheduled_tasks": 50,
            "scheduled_slots": 50,
            "task_ids_sha256": adapter.ORIGINAL_SCHEDULE_SHA256,
            "slot_bindings_sha256": "a" * 64,
        },
        "budget": {
            "max_input_tokens_per_call": 16_384,
            "max_output_tokens_per_call": 4_096,
        },
    }
    append_event(
        journal,
        {
            "event": "header",
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    charges = [(2000, 1600, 30_000_000), (2500, 1600, 30_000_000), (2787, 1672, 34_941_000)]
    for index, (task_id, charge) in enumerate(zip(adapter.PREFIX_TASK_IDS, charges, strict=True)):
        request = f"request-{index}"
        binding = {"slot_position": index, "task_id": task_id, "request_sha256": request}
        append_event(journal, {"event": "call_intent", **binding})
        code = f"void fn0() {{ /* {index} */ }}"
        append_event(
            journal,
            {
                "event": "call_result",
                **binding,
                "status": "response",
                "parse_accepted": True,
                "code": code,
                "code_sha256": base.sha256_text(code),
                "usage": {
                    "charged_input_tokens": charge[0],
                    "charged_output_tokens": charge[1],
                    "charged_usd_nanos": charge[2],
                },
            },
        )
    plan = {
        "schema": cascade.PLAN_SCHEMA,
        "status": "complete",
        "phase": cascade.PHASE_KIMI_INITIAL,
        "cohort_index": 2,
        "provider_credentials_read": False,
        "frontier_api_calls": False,
        "inputs_sha256": canonical_sha256(contract["inputs"]),
        "selection": {
            "scheduled_tasks": 50,
            "scheduled_calls": 50,
            "task_ids_sha256": adapter.ORIGINAL_SCHEDULE_SHA256,
        },
        "budget": contract["budget"],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    head = Path(journal_record(journal)["chain_head_path"])
    monkeypatch.setattr(adapter, "SOURCE_PLAN_SHA256", sha256_file(plan_path))
    monkeypatch.setattr(adapter, "SOURCE_JOURNAL_SHA256", sha256_file(journal))
    monkeypatch.setattr(adapter, "SOURCE_CHAIN_HEAD_SHA256", sha256_file(head))
    evidence = adapter.load_source_evidence(
        plan_path=plan_path, journal_path=journal, chain_head_path=head
    )
    assert len(evidence.events) == 7
    assert evidence.journal_record["event_count"] == 7


def test_launchers_are_gpu_isolated_secret_free_and_nonrestarting() -> None:
    root = Path(__file__).resolve().parents[1]
    phase = (root / "deploy/vast/t5gemma2_typed_api_rescue_c002_resume47.sh").read_text(encoding="utf-8")
    top = (root / "deploy/vast/t5gemma2_typed_kimi_c002_resume47.sh").read_text(encoding="utf-8")
    conf = (root / "deploy/vast/t5gemma2-typed-kimi-c002-resume47.conf").read_text(encoding="utf-8")
    assert "MAX_TASKS=47" in phase
    assert "--max_input_tokens_per_call 30720" in phase
    assert "T5GEMMA_TYPED_C002_PROMPT_PREFLIGHT_SHA256" in phase
    assert "live prompt preflight differs" in phase
    assert "OPENROUTER_API_KEY=" not in top
    assert "f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3" in top
    assert 'CUDA_VISIBLE_DEVICES=""' in top
    assert "exec nice -n 10" in top
    assert "[t]5gemma2_typed_kimi_continuation_c002.py" in top
    assert "autostart=false" in conf
    assert "autorestart=false" in conf
    assert "exitcodes=0,78" in conf
    assert "t5gemma2_typed_kimi_c002_resume47.log" in conf
