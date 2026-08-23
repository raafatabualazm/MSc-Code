from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from frontier_ceiling_patch_v1 import seal_stopped_32k_pilot as seal

PAIR_SHA = "a" * 64
EVALUATOR_SHA = "b" * 64
PAIR_ARM_KEY = "opus_real_fn0_cfg"


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _response(response_id: str, finish: str, completion_tokens: int) -> dict:
    return {
        "id": response_id,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": "```dart\nint fn0() => 1;\n```"},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
        },
    }


def _outcome(
    *,
    config_sha: str,
    attempt_id: str,
    code_sha: str,
    sample: int,
    compiled: bool,
    passed: bool,
) -> dict:
    return {
        "config_sha256": config_sha,
        "task_id": "task-1",
        "sample_index": sample,
        "attempt_id": attempt_id,
        "code_sha256": code_sha,
        "evaluator_sha256": EVALUATOR_SHA,
        "evaluator_entrypoint": "evaluate_dart_jit_tests_detail",
        "completion_attestation_id": seal.REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced": True,
        "completion_attestation_satisfied_all_runs": passed,
        "compiled": compiled,
        "passed": passed,
        "stability_runs": [
            {
                "compiled": compiled,
                "passed": passed,
                "evaluated_source_sha256": "c" * 64,
                "completion_attestation_id": seal.REQUIRED_ATTESTATION_ID,
                "completion_attestation_required": True,
                "completion_attestation_satisfied": passed,
            }
        ],
    }


def _fixture(run_dir: Path) -> str:
    run_dir.mkdir()
    config = {
        "provider": "deepseek",
        "model_requested": "deepseek-v4-pro",
        "arm": "compact",
        "input_mode": "prematerialized_f2",
        "pair_arm_key": PAIR_ARM_KEY,
        "k": 2,
        "max_output_tokens": 32768,
        "expected_task_count": 1,
        "expected_evaluator_sha256": EVALUATOR_SHA,
        "sealed_inputs": {
            "pair_manifest_sha256": PAIR_SHA,
            "pair_arm_key": PAIR_ARM_KEY,
        },
    }
    config_sha = _sha(config)
    _write_json(
        run_dir / "provenance.json",
        {
            "status": "running",
            "config": config,
            "config_sha256": config_sha,
            "evaluator": {"sha256": EVALUATOR_SHA},
            "artifacts": {"pair_manifest": {"sha256": PAIR_SHA}},
            "source_pair_manifest_claims": {
                "sha256": PAIR_SHA,
                "pair_arm_key": PAIR_ARM_KEY,
            },
        },
    )
    _write_jsonl(run_dir / "tasks.jsonl", [{"task_id": "task-1"}])
    messages = [{"role": "user", "content": "decompile"}]
    prompt_sha = _sha(messages)
    _write_jsonl(
        run_dir / "prompts.jsonl",
        [
            {
                "task_id": "task-1",
                "arm": "compact",
                "input_mode": "prematerialized_f2",
                "messages": messages,
                "prompt_sha256": prompt_sha,
            }
        ],
    )
    code0 = "int fn0() => 1;"
    code1 = "int fn0() => 2;"
    attempts = [
        {
            "attempt_id": "a0-rejected",
            "config_sha256": config_sha,
            "task_id": "task-1",
            "sample_index": 0,
            "attempt_index": 0,
            "prompt_sha256": prompt_sha,
            "requested_model": "deepseek-v4-pro",
            "provider": "deepseek",
            "valid": False,
            "invalid_reason": "finish_reason is 'length', not 'stop'",
            "usage": _response("r0", "length", 32768)["usage"],
            "response": _response("r0", "length", 32768),
        },
        {
            "attempt_id": "a0-selected",
            "config_sha256": config_sha,
            "task_id": "task-1",
            "sample_index": 0,
            "attempt_index": 1,
            "prompt_sha256": prompt_sha,
            "requested_model": "deepseek-v4-pro",
            "provider": "deepseek",
            "valid": True,
            "invalid_reason": None,
            "code": code0,
            "code_sha256": hashlib.sha256(code0.encode()).hexdigest(),
            "response_id": "r1",
            "resolved_model": "deepseek-v4-pro",
            "finish_reason": "stop",
            "usage": _response("r1", "stop", 500)["usage"],
            "response": _response("r1", "stop", 500),
        },
        {
            "attempt_id": "a1-selected",
            "config_sha256": config_sha,
            "task_id": "task-1",
            "sample_index": 1,
            "attempt_index": 0,
            "prompt_sha256": prompt_sha,
            "requested_model": "deepseek-v4-pro",
            "provider": "deepseek",
            "valid": True,
            "invalid_reason": None,
            "code": code1,
            "code_sha256": hashlib.sha256(code1.encode()).hexdigest(),
            "response_id": "r2",
            "resolved_model": "deepseek-v4-pro",
            "finish_reason": "stop",
            "usage": _response("r2", "stop", 400)["usage"],
            "response": _response("r2", "stop", 400),
        },
    ]
    _write_jsonl(run_dir / "attempts.jsonl", attempts)
    _write_jsonl(
        run_dir / "outcomes.jsonl",
        [
            _outcome(
                config_sha=config_sha,
                attempt_id="a0-selected",
                code_sha=attempts[1]["code_sha256"],
                sample=0,
                compiled=True,
                passed=True,
            ),
            _outcome(
                config_sha=config_sha,
                attempt_id="a1-selected",
                code_sha=attempts[2]["code_sha256"],
                sample=1,
                compiled=False,
                passed=False,
            ),
        ],
    )
    (run_dir / "runner.log").write_text("stopped\n", encoding="utf-8")
    return config_sha


def _inactive(_unit: str) -> dict[str, str]:
    return {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
    }


def _audit_kwargs(config_sha: str) -> dict:
    return {
        "service_units": ["pilot.service"],
        "expected_config_sha256": config_sha,
        "expected_pair_manifest_sha256": PAIR_SHA,
        "expected_pair_arm_key": PAIR_ARM_KEY,
        "expected_evaluator_sha256": EVALUATOR_SHA,
        "expected_k": 2,
        "expected_task_count": 1,
        "unit_query": _inactive,
    }


def test_builds_stable_read_only_invalid_seal_with_exact_partial_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    config_sha = _fixture(run_dir)
    before = seal.snapshot_files(run_dir)

    audit = seal.build_audit(run_dir, **_audit_kwargs(config_sha))

    assert seal.snapshot_files(run_dir) == before
    assert audit["status"] == "invalid_for_definitive_ceiling"
    assert audit["invalid_for_definitive_ceiling_reasons"] == [
        "output_cap_censoring",
        "model_output_rejection_sampling",
    ]
    diagnostics = audit["diagnostics"]
    assert diagnostics["output_cap_censoring"][
        "censored_terminal_provider_responses"
    ] == 1
    assert diagnostics["rejection_sampling"][
        "slots_with_multiple_provider_responses"
    ] == 1
    assert diagnostics["rejection_sampling"][
        "selected_valid_slots_after_rejected_provider_response"
    ] == 1
    partial = diagnostics["selected_candidate_partial_metrics"]
    assert partial["candidate_pass"]["rate_fraction"] == "1/2"
    assert partial["complete_selected_task_pass_at_k"]["rate_fraction"] == "1/1"
    assert diagnostics["fixed_slot_reconstruction"][
        "definitive_fixed_slot_metric_reconstructable"
    ] is False
    assert {row["relative_path"] for row in audit["source_run"]["files"]} == {
        "attempts.jsonl",
        "outcomes.jsonl",
        "prompts.jsonl",
        "provenance.json",
        "runner.log",
        "tasks.jsonl",
    }


def test_exclusive_writer_creates_only_sibling_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    config_sha = _fixture(run_dir)
    audit = seal.build_audit(run_dir, **_audit_kwargs(config_sha))
    output = seal.default_output_path(run_dir)
    written = seal.write_sibling_audit_exclusive(run_dir, output, audit)
    assert written == output
    assert output.parent == run_dir.parent
    assert not output.is_relative_to(run_dir)
    with pytest.raises(seal.SealError, match="refusing to overwrite"):
        seal.write_sibling_audit_exclusive(run_dir, output, audit)


def test_active_service_fails_before_an_audit_can_be_written(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot"
    config_sha = _fixture(run_dir)

    def active(_unit: str) -> dict[str, str]:
        return {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
        }

    with pytest.raises(seal.SealError, match="not inactive"):
        kwargs = _audit_kwargs(config_sha)
        kwargs["unit_query"] = active
        seal.build_audit(
            run_dir,
            **kwargs,
        )
    assert not seal.default_output_path(run_dir).exists()


def test_malformed_attempt_journal_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot"
    config_sha = _fixture(run_dir)
    (run_dir / "attempts.jsonl").write_text("{broken\n", encoding="utf-8")
    with pytest.raises(seal.SealError, match="cannot parse attempt journal"):
        seal.build_audit(
            run_dir,
            **_audit_kwargs(config_sha),
        )
