from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frontier_passk_anthropic_batch as batch


def _plans(count: int = 3) -> list[dict]:
    return [
        {
            "task_id": f"task_{index}",
            "messages": [
                {"role": "system", "content": "sealed system"},
                {"role": "user", "content": f"sealed prompt {index}"},
            ],
        }
        for index in range(count)
    ]


def test_initial_pending_order_is_sample_major_then_task_order() -> None:
    specs = batch.pending_request_specs(_plans(), [], [])
    assert [(row["sample_index"], row["task_index"], row["cap"]) for row in specs] == [
        (0, 0, 8192),
        (0, 1, 8192),
        (0, 2, 8192),
        (1, 0, 8192),
        (1, 1, 8192),
        (1, 2, 8192),
    ]
    assert len({row["custom_id"] for row in specs}) == len(specs)


def test_length_advances_same_logical_slot_through_cap_ladder() -> None:
    plans = _plans(1)
    attempts = [
        {
            "task_id": "task_0",
            "sample_index": 0,
            "requested_max_tokens": 8192,
            "cap_attempt_index": 0,
            "result_type": "succeeded",
            "finish_reason": "length",
        }
    ]
    specs = batch.pending_request_specs(plans, attempts, [])
    sample_zero = next(row for row in specs if row["sample_index"] == 0)
    assert sample_zero["cap"] == 16384
    assert sample_zero["cap_attempt_index"] == 0


def test_transport_error_retries_same_cap_with_bounded_attempt_index() -> None:
    plans = _plans(1)
    attempts = [
        {
            "task_id": "task_0",
            "sample_index": 0,
            "requested_max_tokens": 8192,
            "cap_attempt_index": 0,
            "result_type": "expired",
            "finish_reason": None,
        }
    ]
    specs = batch.pending_request_specs(plans, attempts, [])
    sample_zero = next(row for row in specs if row["sample_index"] == 0)
    assert sample_zero["cap"] == 8192
    assert sample_zero["cap_attempt_index"] == 1


def test_terminal_slot_is_removed_from_pending() -> None:
    plans = _plans(1)
    terminals = [{"task_id": "task_0", "sample_index": 0}]
    specs = batch.pending_request_specs(plans, [], terminals)
    assert [(row["task_id"], row["sample_index"]) for row in specs] == [("task_0", 1)]


def test_request_uses_native_adaptive_high_and_no_sampling() -> None:
    args = SimpleNamespace(model=batch.MODEL_ID)
    spec = {
        "custom_id": "a00_s00_t000_c08192",
        "cap": 8192,
    }
    request = batch._request_for_spec(args, _plans(1)[0], spec)
    params = request["params"]
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    assert params["max_tokens"] == 8192
    assert "temperature" not in params
    assert "top_p" not in params
    assert "top_k" not in params


def test_two_arm_initial_strict_worst_cost_is_below_41_dollars() -> None:
    one_arm = batch.pending_request_specs(_plans(175), [], [])
    assert len(one_arm) == 350
    one_arm_cost = batch.worst_batch_cost(one_arm)
    assert one_arm_cost == pytest.approx(20.0704)
    assert one_arm_cost * 2 == pytest.approx(40.1408)
    assert one_arm_cost * 2 < 41.0
    assert one_arm_cost <= batch.DEFAULT_ARM_COST_CAP_USD


def test_actual_cost_uses_batch_rates_and_all_successful_attempts() -> None:
    rows = [
        {
            "result_type": "succeeded",
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000},
        },
        {
            "result_type": "expired",
            "usage": None,
        },
    ]
    cost = batch.actual_batch_cost(rows)
    assert cost["estimated_input_usd"] == pytest.approx(1.0)
    assert cost["estimated_output_usd"] == pytest.approx(0.5)
    assert cost["estimated_total_usd"] == pytest.approx(1.5)


def test_active_submission_requires_exactly_zero_or_one() -> None:
    submitted = {
        "event_type": "batch_submitted",
        "batch_id": "msgbatch_1",
    }
    assert batch._active_submission([submitted]) == submitted
    assert (
        batch._active_submission(
            [
                submitted,
                {"event_type": "batch_harvested", "batch_id": "msgbatch_1"},
            ]
        )
        is None
    )
    with pytest.raises(Exception, match="more than one"):
        batch._active_submission(
            [submitted, {"event_type": "batch_submitted", "batch_id": "msgbatch_2"}]
        )


def test_provider_token_audit_counts_each_unique_exact_prompt(tmp_path) -> None:
    calls: list[dict] = []

    class Counter:
        def count_tokens(self, **kwargs):
            calls.append(kwargs)
            return {"input_tokens": 1234 + len(calls)}

    class Models:
        def list(self, **kwargs):
            assert kwargs == {"limit": 100}
            return {"data": [{"id": batch.MODEL_ID}]}

    client = SimpleNamespace(messages=Counter(), models=Models())
    args = SimpleNamespace(model=batch.MODEL_ID, workers=2)
    plans = _plans(3)
    for index, plan in enumerate(plans):
        plan["prompt_sha256"] = f"{index:064x}"
    audit = batch._count_input_tokens(
        args,
        out=tmp_path,
        plans=plans,
        config_sha="a" * 64,
        client=client,
        api_key="secret",
    )

    assert len(calls) == 3
    assert audit["unique_prompts_counted"] == 3
    assert audit["logical_requests_covered"] == 6
    assert audit["all_counts_within_cap"] is True
    assert audit["model_catalog_attestation"]["target_model_present"] is True
    for call in calls:
        assert call["model"] == batch.MODEL_ID
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"] == {"effort": "high"}
        assert "temperature" not in call
        assert "top_p" not in call


def test_token_count_retries_only_declared_transient_errors(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    class Counter:
        def count_tokens(self, **kwargs):
            del kwargs
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("transient")
            return {"input_tokens": 12}

    monkeypatch.setattr(batch, "_retryable_token_count_error", lambda exc: True)
    monkeypatch.setattr(batch.time, "sleep", sleeps.append)
    result = batch._count_tokens_with_retries(
        SimpleNamespace(messages=Counter()),
        {"model": batch.MODEL_ID, "messages": []},
    )

    assert result == {"input_tokens": 12}
    assert calls == 3
    assert sleeps == [0.5, 1.0]


def test_token_count_does_not_retry_non_transient_error(monkeypatch) -> None:
    calls = 0

    class Counter:
        def count_tokens(self, **kwargs):
            del kwargs
            nonlocal calls
            calls += 1
            raise ValueError("bad request")

    monkeypatch.setattr(batch, "_retryable_token_count_error", lambda exc: False)
    with pytest.raises(ValueError, match="bad request"):
        batch._count_tokens_with_retries(
            SimpleNamespace(messages=Counter()),
            {"model": batch.MODEL_ID, "messages": []},
        )
    assert calls == 1


def test_final_cap_length_is_forced_capacity_failure() -> None:
    terminal = SimpleNamespace(
        candidate_valid=True,
        terminal_reason="candidate_valid",
        response_id="msg_1",
        finish_reason="length",
        code_sha256="b" * 64,
    )
    evaluation = batch._failed_evaluation()
    payload = batch._outcome_payload(
        config_sha="a" * 64,
        task_id="task_0",
        task_index=0,
        sample_index=0,
        batch_id="msgbatch_1",
        custom_id="a00_s00_t000_c65536",
        cap=65536,
        terminal=terminal,
        evaluator_record={"sha256": "c" * 64, "entrypoint": "evaluate"},
        evaluation=evaluation,
        evaluation_performed=False,
        metric_role="capacity_adaptive",
        capacity_exhausted=True,
    )
    assert payload["candidate_valid"] is False
    assert payload["passed"] is False
    assert payload["terminal_reason"] == "capacity_exhausted_at_65536"


def test_outcome_payload_preserves_native_refusal_diagnostics() -> None:
    terminal = SimpleNamespace(
        candidate_valid=False,
        terminal_reason="finish_reason is 'content_filter', not 'stop'",
        response_id="msg_refusal",
        finish_reason="content_filter",
        code_sha256=None,
    )
    payload = batch._outcome_payload(
        config_sha="a" * 64,
        task_id="task_0",
        task_index=0,
        sample_index=0,
        batch_id="msgbatch_1",
        custom_id="a00_s00_t000_c08192",
        cap=8192,
        terminal=terminal,
        evaluator_record={"sha256": "c" * 64, "entrypoint": "evaluate"},
        evaluation=batch._failed_evaluation(),
        evaluation_performed=False,
        metric_role="primary_fixed_cap_8192",
        capacity_exhausted=False,
        native_stop_reason="refusal",
        native_stop_details_category="cyber",
        native_stop_category="refusal:cyber",
    )

    # Compatibility semantics remain unchanged.
    assert payload["finish_reason"] == "content_filter"
    assert payload["native_stop_reason"] == "refusal"
    assert payload["native_stop_details_category"] == "cyber"
    assert payload["native_stop_category"] == "refusal:cyber"


def test_historical_batch_rows_recover_native_stop_from_attempt_response() -> None:
    normalized = batch.sync.normalize_anthropic_response(
        {
            "id": "msg_refusal",
            "model": batch.MODEL_ID,
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {"category": "cyber"},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    attempts = [
        {
            "batch_id": "msgbatch_1",
            "custom_id": "slot_1",
            "normalized_response": normalized,
        }
    ]
    historical_terminal = {
        "task_id": "task_0",
        "sample_index": 0,
        "batch_id": "msgbatch_1",
        "custom_id": "slot_1",
        "finish_reason": "content_filter",
    }

    enriched = batch._rows_with_native_stop_metadata([historical_terminal], attempts)
    assert enriched[0]["finish_reason"] == "content_filter"
    assert enriched[0]["native_stop_reason"] == "refusal"
    assert enriched[0]["native_stop_category"] == "refusal:cyber"

    report = batch.sync.anthropic_metric_transparency(
        enriched,
        task_ids=["task_0"],
        k=1,
        complete=True,
    )
    assert (
        report["capability_metric_assessment"]["status"] == "invalid_refusal_dominated"
    )
    assert report["capability_metric_assessment"]["ceiling_claim_allowed"] is False
