from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import frontier_core as core
import frontier_passk as runner


MODEL = "deepseek-v4-pro"
POLICY_SHA = "a" * 64
CONFIG_SHA = "b" * 64
PROMPT_SHA = "c" * 64
CAP = 131_072


def _response(
    *,
    response_id: str = "resp-1",
    finish_reason: str = "stop",
    content: str = "dynamic fn0() => 7;",
    refusal: object = None,
    model: str = MODEL,
) -> dict[str, object]:
    return {
        "id": response_id,
        "model": model,
        "created": 123,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": content,
                    "reasoning_content": "reason",
                    "refusal": refusal,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def _classified(response: dict[str, object]) -> core.TerminalProviderResponse:
    return core.classify_terminal_provider_response(
        response,
        expected_model=MODEL,
        max_prompt_tokens=12_000,
        requested_max_tokens=CAP,
    )


def _terminal_row(
    *,
    task_id: str = "t0",
    sample_index: int = 0,
    attempt_index: int = 0,
    response: dict[str, object] | None = None,
) -> dict[str, object]:
    response = response or _response()
    value = _classified(response)
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "api_attempt",
        "attempt_id": f"{task_id}.s{sample_index}.a{attempt_index}",
        "config_sha256": CONFIG_SHA,
        "slot_policy_sha256": POLICY_SHA,
        "task_id": task_id,
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "prompt_sha256": PROMPT_SHA,
        "requested_model": MODEL,
        "requested_max_tokens": CAP,
        "provider": "deepseek",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "response_received": True,
        "slot_terminal": True,
        "candidate_valid": value.candidate_valid,
        "terminal_reason": value.terminal_reason,
        "transport_retry": False,
        "transport_error": None,
        "fatal_response_contract": False,
        "response_id": value.response_id,
        "resolved_model": value.response_model,
        "response_created": value.response_created,
        "finish_reason": value.finish_reason,
        "budget_charge_tokens": value.usage["total_tokens"],
        "usage": value.usage,
        "content": value.content,
        "reasoning_content": value.reasoning_content,
        "code": value.code,
        "code_sha256": value.code_sha256,
        "response": response,
    }


def _transport_row(
    *,
    task_id: str = "t0",
    sample_index: int = 0,
    attempt_index: int = 0,
) -> dict[str, object]:
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "api_attempt",
        "attempt_id": f"{task_id}.s{sample_index}.a{attempt_index}",
        "config_sha256": CONFIG_SHA,
        "slot_policy_sha256": POLICY_SHA,
        "task_id": task_id,
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "prompt_sha256": PROMPT_SHA,
        "requested_model": MODEL,
        "requested_max_tokens": CAP,
        "provider": "deepseek",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "response_received": False,
        "slot_terminal": False,
        "candidate_valid": None,
        "terminal_reason": None,
        "transport_retry": True,
        "retryable_transport": True,
        "transport_error": "api_exception:APIConnectionError:lost",
        "fatal_response_contract": False,
        "budget_charge_tokens": CAP + 12_000,
        "usage": None,
        "response": None,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load(path: Path, *, k: int = 2):
    return runner.load_resume_attempts(
        path,
        config_sha=CONFIG_SHA,
        prompt_map={"t0": {"prompt_sha256": PROMPT_SHA}},
        budget=core.TokenBudget(0),
        requested_model=MODEL,
        k=k,
        max_prompt_tokens=12_000,
        requested_max_tokens=CAP,
        max_transport_attempts_per_slot=6,
        slot_policy_sha256=POLICY_SHA,
    )


def test_length_response_with_safe_fn0_is_terminal_and_evaluable() -> None:
    value = _classified(_response(finish_reason="length"))
    assert value.finish_reason == "length"
    assert value.candidate_valid is True
    assert value.terminal_reason == "candidate_valid"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_response(content=""), "response_content_is_empty"),
        (_response(refusal="cannot comply"), "response_contains_refusal"),
        (_response(content="void main() {}"), "unsafe_or_invalid_candidate"),
        (_response(content="dynamic helper() => 7;"), "unsafe_or_invalid_candidate"),
    ],
)
def test_returned_model_invalids_are_terminal_candidates(
    response: dict[str, object],
    reason: str,
) -> None:
    value = _classified(response)
    assert value.candidate_valid is False
    assert reason in value.terminal_reason


def test_zero_completion_empty_response_is_terminal_not_fatal() -> None:
    response = _response(content="")
    response["usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 0,
        "total_tokens": 100,
    }
    value = _classified(response)
    assert value.candidate_valid is False
    assert value.terminal_reason == "response_content_is_empty"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"id": ""}),
        lambda value: value.update({"model": "wrong-model"}),
        lambda value: value.update({"usage": None}),
        lambda value: value.update(
            {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 20,
                    "total_tokens": 20,
                }
            }
        ),
        lambda value: value.update(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 999,
                }
            }
        ),
        lambda value: value.update(
            {
                "usage": {
                    "prompt_tokens": 12_001,
                    "completion_tokens": 20,
                    "total_tokens": 12_021,
                }
            }
        ),
    ],
)
def test_identity_and_usage_contract_breaches_are_fatal(mutate) -> None:
    response = _response()
    mutate(response)
    with pytest.raises(core.ResponseContractError):
        _classified(response)


def test_resume_accepts_transport_then_one_terminal_invalid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempts.jsonl"
    invalid = _terminal_row(
        attempt_index=1,
        response=_response(content="void main() {}"),
    )
    _write_rows(path, [_transport_row(), invalid])
    terminal, next_attempt = _load(path)
    assert terminal[("t0", 0)]["candidate_valid"] is False
    assert next_attempt[("t0", 0)] == 2


def test_resume_rejects_post_terminal_attempt(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [_terminal_row(), _transport_row(attempt_index=1)])
    with pytest.raises(runner.RunFailure, match="post-terminal"):
        _load(path)


def test_resume_rejects_terminal_field_tampering(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    row = _terminal_row()
    row["candidate_valid"] = False
    _write_rows(path, [row])
    with pytest.raises(runner.RunFailure, match="tampered"):
        _load(path)


def test_resume_rejects_duplicate_response_ids_across_slots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempts.jsonl"
    rows = [
        _terminal_row(sample_index=0),
        _terminal_row(sample_index=1),
    ]
    _write_rows(path, rows)
    with pytest.raises(runner.RunFailure, match="duplicate terminal response id"):
        _load(path)


def test_resume_rejects_legacy_schema_and_out_of_range_slot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempts.jsonl"
    legacy = _terminal_row()
    legacy["schema"] = "audited-frontier-passk-v1"
    _write_rows(path, [legacy])
    with pytest.raises(runner.RunFailure, match="incompatible"):
        _load(path)
    _write_rows(path, [_terminal_row(sample_index=2)])
    with pytest.raises(runner.RunFailure, match="invalid task/sample"):
        _load(path, k=2)


def test_resume_rejects_cap_policy_and_index_gaps(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    wrong_cap = _terminal_row()
    wrong_cap["requested_max_tokens"] = 65_536
    _write_rows(path, [wrong_cap])
    with pytest.raises(runner.RunFailure, match="cap mismatch"):
        _load(path)
    gap = _terminal_row(attempt_index=1)
    _write_rows(path, [gap])
    with pytest.raises(runner.RunFailure, match="contiguous"):
        _load(path)


def test_retryability_is_restricted_to_transport_and_selected_statuses() -> None:
    APIConnectionError = type("APIConnectionError", (Exception,), {})
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    assert runner.is_retryable_api_exception(APIConnectionError("lost"))
    assert runner.is_retryable_api_exception(APITimeoutError("slow"))
    assert not runner.is_retryable_api_exception(TypeError("bug"))
    assert not runner.is_retryable_api_exception(
        SimpleNamespace(status_code=400)  # type: ignore[arg-type]
    )
    assert runner.is_retryable_api_exception(
        SimpleNamespace(status_code=429)  # type: ignore[arg-type]
    )
    assert runner.is_retryable_api_exception(
        SimpleNamespace(status_code=503)  # type: ignore[arg-type]
    )


def test_make_request_uses_exact_fixed_cap() -> None:
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _response()

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    args = SimpleNamespace(
        model=MODEL,
        temperature=0.8,
        top_p=0.95,
        timeout_seconds=7200,
        extra_body={},
    )
    runner.make_request(
        client,
        args,
        [{"role": "user", "content": "x"}],
        requested_max_tokens=CAP,
    )
    assert captured["max_tokens"] == CAP
    assert captured["timeout"] == 7200


def test_end_to_end_exact_slots_retry_only_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    APIConnectionError = type("APIConnectionError", (Exception,), {})
    scripted: list[object] = [
        APIConnectionError("lost"),
        _response(response_id="r0", finish_reason="length"),
        _response(response_id="r1", refusal="no"),
        _response(response_id="r2", content="void main() {}"),
    ]
    requests: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            value = scripted.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI),
    )
    monkeypatch.setattr(
        runner,
        "api_credentials",
        lambda _args: ("secret", "https://example.invalid"),
    )

    def evaluate(code, _tests, _evaluation_id, **_kwargs):
        return True, "fn0" in code, "ok", code

    evaluator_record = {
        "sha256": "e" * 64,
        "entrypoint": "evaluate_dart_jit_tests_detail",
    }
    evaluator_module = SimpleNamespace(
        evaluate_dart_jit_tests_detail=evaluate,
    )
    monkeypatch.setattr(
        runner,
        "import_evaluator",
        lambda *_args, **_kwargs: (evaluator_module, evaluator_record),
    )
    monkeypatch.setattr(runner, "retry_delay", lambda *_args: 0.0)

    args = SimpleNamespace(
        resume=True,
        expected_evaluator_sha256="e" * 64,
        expected_dart_sha256="f" * 64,
        evaluator_module=tmp_path / "evaluator.py",
        dart=tmp_path / "dart",
        provider="deepseek",
        model=MODEL,
        k=3,
        workers=1,
        max_prompt_tokens=12_000,
        max_output_tokens=CAP,
        max_attempts_per_sample=3,
        temperature=0.8,
        top_p=0.95,
        timeout_seconds=7200,
        extra_body={},
        budget=0,
        eval_stability_runs=2,
        eval_timeout_seconds=30,
        retry_max_seconds=0.0,
        retry_base_seconds=0.0,
        dataset_label="fixture",
        arm="compact",
        input_mode="decoded_compact",
        pair_arm_key=None,
    )
    policy = runner.fixed_slot_policy(args)
    policy_sha = core.stable_sha256(policy)
    provenance = {
        "config": {
            "slot_policy": policy,
            "slot_policy_sha256": policy_sha,
        },
        "evaluator": evaluator_record,
        "dataset": {"sha256": "1" * 64},
        "task_set_sha256": core.stable_sha256(["t0"]),
        "artifacts": {},
    }
    messages = [{"role": "user", "content": "prompt"}]
    plan = {
        "task_id": "t0",
        "messages": messages,
        "prompt_sha256": core.stable_sha256(messages),
        "row": {"acceptance_tests": "void main() {}"},
    }
    prompt_map = {
        "t0": {
            "prompt_sha256": plan["prompt_sha256"],
            "token_count": {"estimated_prompt_tokens": 100},
        }
    }
    (tmp_path / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
    summary = runner.run_api_and_evaluation(
        args,
        out=tmp_path,
        plans=[plan],
        prompt_map=prompt_map,
        config_sha=CONFIG_SHA,
        provenance=provenance,
    )
    assert len(requests) == 4
    assert all(request["max_tokens"] == CAP for request in requests)
    assert summary["terminal_responses"] == 3
    assert summary["evaluable_candidates"] == 1
    assert summary["invalid_candidates"] == 2
    assert summary["transport_retries"] == 1
    assert summary["length_slots"] == 1
    assert summary["pass_at_k"]["successes"] == 1
    outcomes = [
        json.loads(line)
        for line in (tmp_path / "outcomes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(outcomes) == 3
    assert sum(outcome["evaluation_performed"] for outcome in outcomes) == 1
