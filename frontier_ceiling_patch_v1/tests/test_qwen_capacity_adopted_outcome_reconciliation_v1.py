from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import qwen37_capacity_adopted_outcome_reconciliation_v1 as reconciliation
import qwen37_capacity_adopted_outcome_monitor_v1 as monitor
import qwen37_primary_effective_status_v8 as status_v8


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    workspace = tmp_path / "workspace"
    run_root = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
    )
    capacity_root = (
        run_root / "qwen37_capacity_v6_0608_codex_mc12k_tb8k"
    )
    capacity_root.mkdir(parents=True)
    diagnostic_root = (
        run_root
        / "qwen37_clean_v4_0608_codex_k2_mc12k_tol10_tb8k"
    )
    diagnostic_root.mkdir(parents=True)
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 100,
        "total_tokens": 200,
        "reasoning_tokens": 50,
        "answer_tokens": 50,
    }
    target: dict[str, object] = {
        "schema": reconciliation.CAPACITY_SCHEMA,
        "record_type": "capacity_target",
        "selection_id": "s" * 64,
        "arm": "codex",
        "task_id": "sigless_test",
        "global_sample_index": 5,
        "prompt_sha256": "p" * 64,
        "selection_reads_outcomes": False,
    }
    target["selection_record_sha256"] = reconciliation.canonical_sha(target)
    _write_jsonl(capacity_root / "targets.jsonl", [target])
    capacity_config = {
        "schema": reconciliation.CAPACITY_SCHEMA,
        "arm": "codex",
        "partition": "0608",
        "out": str(capacity_root.resolve()),
        "contract_sha256": (
            reconciliation.EXPECTED_CAPACITY_CONTRACT_SHA256
        ),
        "targets_sha256": reconciliation.sha256_file(
            capacity_root / "targets.jsonl"
        ),
        "runtime_identity": {
            "capacity_runner_sha256": (
                reconciliation.EXPECTED_CAPACITY_ENTRY_SHA256
            )
        },
    }
    (capacity_root / "provenance.json").write_text(
        json.dumps(
            {
                "config": capacity_config,
                "config_sha256": reconciliation.canonical_sha(
                    capacity_config
                ),
            }
        ),
        encoding="utf-8",
    )
    slot_policy = {"k": 2}
    diagnostic_config = {
        "slot_policy": slot_policy,
        "slot_policy_sha256": reconciliation.canonical_sha(slot_policy),
        "api_base_url_sha256": "e" * 64,
    }
    diagnostic_config_sha = reconciliation.canonical_sha(
        diagnostic_config
    )
    (diagnostic_root / "provenance.json").write_text(
        json.dumps(
            {
                "config": diagnostic_config,
                "config_sha256": diagnostic_config_sha,
            }
        ),
        encoding="utf-8",
    )
    terminal: dict[str, object] = {
        "schema": reconciliation.EXPECTED_DIAGNOSTIC_SCHEMA,
        "task_id": "sigless_test",
        "sample_index": 0,
        "attempt_id": "attempt-1",
        "response_id": "response-1",
        "requested_model": "qwen3.7-max-2026-06-08",
        "resolved_model": "qwen3.7-max-2026-06-08",
        "finish_reason": "stop",
        "candidate_valid": False,
        "terminal_reason": "unsafe_or_invalid_candidate:test",
        "code_sha256": "c" * 64,
        "prompt_sha256": "p" * 64,
        "config_sha256": diagnostic_config_sha,
        "response_received": True,
        "slot_terminal": True,
        "usage": usage,
    }
    outcome: dict[str, object] = {
        "schema": reconciliation.EXPECTED_DIAGNOSTIC_SCHEMA,
        "record_type": "candidate_outcome",
        "task_id": "sigless_test",
        "sample_index": 0,
        "attempt_id": "attempt-1",
        "response_id": "response-1",
        "finish_reason": "stop",
        "candidate_valid": False,
        "terminal_reason": "unsafe_or_invalid_candidate:test",
        "code_sha256": "c" * 64,
        "config_sha256": diagnostic_config_sha,
        "evaluator_entrypoint": "evaluate_dart_jit_tests_detail",
        "evaluator_sha256": reconciliation.EXPECTED_EVALUATOR_SHA256,
        "evaluation_performed": False,
        "compiled": False,
        "passed": False,
        "completion_attestation_id": (
            reconciliation.runner.REQUIRED_ATTESTATION_ID
        ),
        "completion_attestation_enforced": False,
        "completion_attestation_satisfied_all_runs": False,
        "stability_runs": [],
        "evaluated_at": "2026-07-26T00:00:00Z",
    }
    _write_jsonl(diagnostic_root / "attempts.jsonl", [terminal])
    _write_jsonl(diagnostic_root / "outcomes.jsonl", [outcome])
    effective: dict[str, object] = {
        "schema": reconciliation.CAPACITY_SCHEMA,
        "record_type": "effective_capacity_terminal",
        "selection_id": "s" * 64,
        "overlay_contract_sha256": (
            reconciliation.EXPECTED_CAPACITY_CONTRACT_SHA256
        ),
        "arm": "codex",
        "task_id": "sigless_test",
        "global_sample_index": 5,
        "origin": "adopted_clean_diagnostic",
        "effective_source_directory": str(diagnostic_root.resolve()),
        "effective_source_config_sha256": diagnostic_config_sha,
        "effective_source_slot_policy_sha256": diagnostic_config[
            "slot_policy_sha256"
        ],
        "effective_endpoint_sha256": "e" * 64,
        "effective_attempt_id": "attempt-1",
        "response_id": "response-1",
        "finish_reason": "stop",
        "candidate_valid": False,
        "terminal_reason": "unsafe_or_invalid_candidate:test",
        "code_sha256": "c" * 64,
        "compiled": False,
        "passed": False,
        "usage": usage,
        "canonical_terminal_row_sha256": (
            reconciliation.canonical_sha(terminal)
        ),
        "canonical_outcome_row_sha256": (
            reconciliation.canonical_sha(outcome)
        ),
    }
    _write_jsonl(capacity_root / "effective_terminals.jsonl", [effective])
    _write_jsonl(capacity_root / "outcomes.jsonl", [])
    return workspace, capacity_root, target, effective


def test_reconciliation_is_local_exact_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, capacity_root, _, _ = _fixture(tmp_path)

    def prepare(
        workspace_arg: Path,
        capacity_arg: Path,
        *,
        arm: str,
        partition: str,
    ) -> tuple[Path, dict[str, object]]:
        del workspace_arg, arm, partition
        overlay = capacity_arg / reconciliation.OVERLAY_DIRECTORY
        overlay.mkdir(parents=True, exist_ok=True)
        return overlay, {"config_sha256": "f" * 64}

    monkeypatch.setattr(reconciliation, "_prepare_overlay", prepare)
    first = reconciliation.reconcile_one(
        workspace, capacity_root, arm="codex", partition="0608"
    )
    second = reconciliation.reconcile_one(
        workspace, capacity_root, arm="codex", partition="0608"
    )
    assert first["newly_appended"] == 1
    assert first["newly_appended_compiled"] == 0
    assert first["newly_appended_passed"] == 0
    assert second["newly_appended"] == 0
    rows = reconciliation.load_effective_outcomes(capacity_root)
    assert set(rows) == {"s" * 64}
    assert rows["s" * 64]["provider_calls"] == 0
    assert rows["s" * 64]["stability_runs"] == []


def test_reconciliation_rejects_source_outcome_rebinding(
    tmp_path: Path,
) -> None:
    workspace, capacity_root, target, effective = _fixture(tmp_path)
    diagnostic = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
        / "qwen37_clean_v4_0608_codex_k2_mc12k_tol10_tb8k"
        / "outcomes.jsonl"
    )
    row = json.loads(diagnostic.read_text(encoding="utf-8"))
    row["evaluated_at"] = "2026-07-26T00:00:01Z"
    _write_jsonl(diagnostic, [row])
    with pytest.raises(
        reconciliation.AuditError,
        match="legacy effective outcome binding",
    ):
        reconciliation.transform_diagnostic_outcome(
            workspace=workspace,
            capacity_root=capacity_root,
            target=target,
            effective=effective,
            arm="codex",
            partition="0608",
        )


def test_v8_reporter_consumes_exact_reconciled_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, capacity_root, _, effective = _fixture(tmp_path)

    def prepare(
        workspace_arg: Path,
        capacity_arg: Path,
        *,
        arm: str,
        partition: str,
    ) -> tuple[Path, dict[str, object]]:
        del workspace_arg, arm, partition
        overlay = capacity_arg / reconciliation.OVERLAY_DIRECTORY
        overlay.mkdir(parents=True, exist_ok=True)
        return overlay, {"config_sha256": "f" * 64}

    monkeypatch.setattr(reconciliation, "_prepare_overlay", prepare)
    reconciliation.reconcile_one(
        workspace, capacity_root, arm="codex", partition="0608"
    )
    feed: dict[str, object] = {
        "schema": reconciliation.CAPACITY_SCHEMA,
        "record_type": "capacity_effective_terminal_feed",
        "selection_id": "s" * 64,
        "overlay_contract_sha256": (
            reconciliation.EXPECTED_CAPACITY_CONTRACT_SHA256
        ),
        "arm": "codex",
        "task_id": "sigless_test",
        "global_sample_index": 5,
        "response_id": "response-1",
        "finish_reason": "stop",
        "candidate_valid": False,
        "code_sha256": "c" * 64,
        "validated_usage": effective["usage"],
        "effective_terminal_canonical_row_sha256": effective[
            "canonical_terminal_row_sha256"
        ],
    }
    immutable = dict(feed)
    feed["terminal_feed_payload_sha256"] = (
        reconciliation.canonical_sha(immutable)
    )
    _write_jsonl(
        capacity_root / "effective_terminal_feed.jsonl", [feed]
    )
    monkeypatch.setattr(
        status_v8.parent,
        "CAPACITY_OUTPUTS",
        (("0608", "codex", capacity_root.name),),
    )
    monkeypatch.setattr(
        status_v8.capacity_length,
        "validate_feed_terminal",
        lambda *args, **kwargs: None,
    )
    terminals, outcomes, _, _, progress = status_v8.capacity_outputs(
        workspace,
        expected_contract_sha256=(
            reconciliation.EXPECTED_CAPACITY_CONTRACT_SHA256
        ),
    )
    assert len(terminals) == len(outcomes) == 1
    assert progress[0]["base_outcomes"] == 0
    assert progress[0]["reconciled_adopted_outcomes"] == 1


def test_v8_rejects_reconciliation_entry_hash_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = tmp_path / "frontier_ceiling_patch_v1"
    patch.mkdir()

    def fake_sha(path: Path) -> str:
        if path.name.endswith("_reconciliation_v1.py"):
            return "0" * 64
        return status_v8.EXPECTED_RECONCILIATION_CONTRACT_SHA256

    monkeypatch.setattr(status_v8.runner, "sha256_file", fake_sha)
    with pytest.raises(
        status_v8.AuditError,
        match="reconciliation dependency hash mismatch",
    ):
        status_v8.validate_reconciliation_dependencies(patch)


def test_monitor_rejects_extension_hash_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    patch = workspace / "frontier_ceiling_patch_v1"
    patch.mkdir(parents=True)
    monitor_entry_sha = monitor.runner.sha256_file(
        Path(monitor.__file__).resolve()
    )
    extension = {
        "schema": "qwen37-capacity-adopted-outcome-extension-v1",
        "monitor": {
            "entry_sha256": monitor_entry_sha
        },
        "reconciliation": {
            "entry_sha256": "0" * 64,
            "contract_sha256": (
                reconciliation.EXPECTED_CONTRACT_SHA256
            ),
        },
        "effective_status": {
            "entry_sha256": monitor.EXPECTED_STATUS_ENTRY_SHA256
        },
    }
    (patch / monitor.EXTENSION_NAME).write_text(
        json.dumps(extension), encoding="utf-8"
    )

    def fake_sha(path: Path) -> str:
        if path.name == Path(monitor.__file__).name:
            return monitor_entry_sha
        if path.name == reconciliation.CONTRACT_NAME:
            return reconciliation.EXPECTED_CONTRACT_SHA256
        if path.name == "qwen37_primary_effective_status_v8.py":
            return monitor.EXPECTED_STATUS_ENTRY_SHA256
        return monitor.EXPECTED_RECONCILIATION_ENTRY_SHA256

    monkeypatch.setattr(monitor.runner, "sha256_file", fake_sha)
    with pytest.raises(
        reconciliation.AuditError,
        match="dependency/extension mismatch",
    ):
        monitor.validate_dependencies(workspace)
