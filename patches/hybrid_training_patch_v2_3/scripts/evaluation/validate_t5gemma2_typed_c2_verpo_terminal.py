#!/usr/bin/env python3
"""Classify the sealed terminal state of the typed Arm-C2 VeRPO pilot.

This validator is intentionally read-only.  It never opens the private holdback,
promotes a checkpoint, deletes an artifact, or starts another process.  The
Supervisor handoff may start the matched evaluation only for ``EVALUATE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.training import t5gemma2_typed_c2_verpo_pilot150 as pilot


RUN_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-checkpoint-v1"
ROLLOUT_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-rollout-v1"
GATE_SCHEMA = "t5gemma2-typed-c2-verpo-window-gate-v1"
GATE_UPDATES = tuple(range(16, 145, 16))
FINAL_UPDATE = 150


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "t5gemma2-typed-c2-verpo-eval-handoff-decision-v1",
        "status": "complete",
        "disposition": "BLOCKED_NO_EVAL",
        "reason": reason,
        "evaluation_permitted": False,
        "automatic_promotion_performed": False,
        "private_holdback_read": False,
    }


def _validate_contract(contract: Mapping[str, Any]) -> None:
    optimization = contract.get("optimization") or {}
    sampling = contract.get("sampling") or {}
    pilot_contract = contract.get("pilot") or {}
    selection = contract.get("selection") or {}
    if not (
        contract.get("schema") == RUN_SCHEMA
        and contract.get("status") == "training"
        and contract.get("architecture") == "native_encoder_decoder"
        and contract.get("policy_architecture")
        == "native_t5gemma2_encoder_decoder"
        and contract.get("automatic_promotion_permitted") is False
        and contract.get("production_floor_eligible") is False
        and contract.get("private_holdback_exposed") is False
        and contract.get("no_frontier_api") is True
        and contract.get("llm_judge") is False
        and contract.get("acceptance_tests_exposed") is False
        and optimization.get("max_updates") == FINAL_UPDATE
        and optimization.get("tasks_per_update") == 1
        and optimization.get("learning_rate") == 1e-6
        and optimization.get("weight_decay") == 0.0
        and optimization.get("max_grad_norm") == 1.0
        and optimization.get("ppo_clip") == 0.0
        and optimization.get("sft_replay_weight") == 0.0
        and optimization.get("gold_or_sft_replay_gradient") is False
        and sampling.get("group_size") == 4
        and sampling.get("repair_group_size") == 4
        and sampling.get("max_repair_parents") == 2
        and sampling.get("temperature") == 0.8
        and sampling.get("top_p") == 1.0
        and sampling.get("top_k") == 0
        and sampling.get("max_new_tokens") == 8192
        and sampling.get("max_source_tokens") == 32768
        and sampling.get("eos_token_ids") == [1]
        and sampling.get("suppressed_token_ids") == [0]
        and sampling.get("pad_before_eos_fail_closed") is True
        and pilot_contract.get("maximum_updates") == FINAL_UPDATE
        and pilot_contract.get("gate_interval") == 16
        and pilot_contract.get("mandatory_pause_after_first_gate") is True
        and pilot_contract.get("later_gates_automatic") is True
        and selection.get("tasks") == 150
        and selection.get("trainer_opened_gold_or_target_source") is False
        and contract.get("seed") == 42
    ):
        raise ValueError("root run contract differs from the sealed pilot profile")


def _read_metrics(path: Path, contract_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank rollout metric at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"rollout metric {line_number} is not an object")
            if not (
                value.get("schema") == ROLLOUT_SCHEMA
                and value.get("update") == line_number
                and value.get("run_contract_sha256") == contract_sha256
                and value.get("no_frontier_api") is True
            ):
                raise ValueError(f"rollout metric {line_number} binding differs")
            rows.append(value)
    return rows


def _validate_pointer(stage: Path, contract_sha256: str, update: int) -> list[Path]:
    pointer_path = stage / "latest_checkpoint.json"
    pointer = _read_object(pointer_path, "latest checkpoint pointer")
    checkpoint = Path(str(pointer.get("path") or "")).resolve()
    expected_checkpoint = (stage / f"checkpoint-optstep-{update:06d}").resolve()
    if not (
        pointer.get("schema") == CHECKPOINT_SCHEMA
        and pointer.get("update") == update
        and pointer.get("run_contract_sha256") == contract_sha256
        and checkpoint == expected_checkpoint
    ):
        raise ValueError("latest checkpoint pointer differs")
    required = [
        checkpoint / "run_contract.json",
        checkpoint / "training_state.pt",
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "tokenizer" / "tokenizer.json",
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise ValueError("terminal checkpoint is incomplete")
    checkpoint_contract = _read_object(required[0], "checkpoint run contract")
    if canonical_sha256(checkpoint_contract) != contract_sha256:
        raise ValueError("checkpoint run contract differs")
    return [pointer_path, *required]


def _validate_gate(
    stage: Path,
    rows: list[dict[str, Any]],
    contract_sha256: str,
    boundary: int,
) -> tuple[dict[str, Any], Path]:
    path = stage / f"pilot_gate_update{boundary:06d}.json"
    recorded = _read_object(path, f"gate {boundary}")
    recomputed = pilot.evaluate_mechanics_gate(
        rows[boundary - 16 : boundary],
        run_contract_sha256=contract_sha256,
        window_start=boundary - 15,
        window_end=boundary,
    )
    if recorded != recomputed:
        raise ValueError(f"gate {boundary} differs from metric recomputation")
    if not (
        recorded.get("schema") == GATE_SCHEMA
        and recorded.get("status") == "pass"
        and recorded.get("gate_update") == boundary
        and recorded.get("window_start_update") == boundary - 15
        and recorded.get("window_end_update") == boundary
        and (recorded.get("criteria") or {}).get("integrity", {}).get("pass")
        is True
        and recorded.get("automatic_promotion_performed") is False
        and recorded.get("private_holdback_read") is False
    ):
        raise ValueError(f"gate {boundary} metadata differs")
    return recorded, path


def classify_terminal(stage: Path) -> dict[str, Any]:
    """Return the fail-closed handoff decision for an already EXITED pilot."""

    stage = stage.resolve()
    result_path = stage / "result.json"
    if not result_path.is_file():
        return _blocked("training supervisor exited without result.json")
    try:
        result = _read_object(result_path, "training result")
        contract_path = stage / "run_contract.json"
        metrics_path = stage / "rollout_metrics.jsonl"
        contract = _read_object(contract_path, "root run contract")
        _validate_contract(contract)
        contract_sha256 = canonical_sha256(contract)
        if result.get("run_contract_sha256") != contract_sha256:
            raise ValueError("training result is not bound to the root contract")
        if result.get("schema") != RUN_SCHEMA:
            raise ValueError("training result schema differs")
        rows = _read_metrics(metrics_path, contract_sha256)
        common_paths = [result_path, contract_path, metrics_path]

        if result.get("status") == "stopped_at_window_gate":
            update = result.get("updates")
            if not (
                type(update) is int
                and update in GATE_UPDATES
                and len(rows) == update
                and result.get("latest_checkpoint")
                == f"checkpoint-optstep-{update:06d}"
                and result.get("gate_update") == update
                and result.get("window_gate") == "STOP"
                and result.get("automatic_promotion_performed") is False
                and result.get("production_floor_eligible") is False
                and result.get("no_frontier_api") is True
            ):
                raise ValueError("STOP result differs")
            evidence_paths = common_paths + _validate_pointer(
                stage, contract_sha256, update
            )
            for boundary in GATE_UPDATES:
                if boundary > update:
                    break
                gate, gate_path = _validate_gate(
                    stage, rows, contract_sha256, boundary
                )
                evidence_paths.append(gate_path)
                expected = "STOP" if boundary == update else "GO"
                if gate.get("decision") != expected:
                    raise ValueError(
                        f"gate {boundary} decision is not expected {expected}"
                    )
            return {
                "schema": "t5gemma2-typed-c2-verpo-eval-handoff-decision-v1",
                "status": "complete",
                "disposition": "STOP_NO_EVAL",
                "reason": f"sealed mechanics gate STOP at update {update}",
                "terminal_update": update,
                "evaluation_permitted": False,
                "automatic_promotion_performed": False,
                "private_holdback_read": False,
                "evidence_bundle_sha256": canonical_sha256(
                    {str(path): _sha256_file(path) for path in evidence_paths}
                ),
            }

        if result.get("status") != "complete":
            return _blocked(f"non-evaluable terminal result status={result.get('status')!r}")
        if not (
            result.get("updates") == FINAL_UPDATE
            and len(rows) == FINAL_UPDATE
            and result.get("latest_checkpoint") == "checkpoint-optstep-000150"
            and result.get("mechanics_gate") == "GO"
            and result.get("window_gates_passed") == list(GATE_UPDATES)
            and result.get("automatic_promotion_performed") is False
            and result.get("production_floor_eligible") is False
            and result.get("pilot_disposition")
            == "discardable_not_for_automatic_promotion"
        ):
            raise ValueError("complete result differs")
        evidence_paths = common_paths + _validate_pointer(
            stage, contract_sha256, FINAL_UPDATE
        )
        for boundary in GATE_UPDATES:
            gate, gate_path = _validate_gate(
                stage, rows, contract_sha256, boundary
            )
            evidence_paths.append(gate_path)
            if gate.get("decision") != "GO":
                raise ValueError(f"gate {boundary} is not GO")
        return {
            "schema": "t5gemma2-typed-c2-verpo-eval-handoff-decision-v1",
            "status": "complete",
            "disposition": "EVALUATE",
            "reason": "exact complete update-150 state with all sealed gates GO",
            "terminal_update": FINAL_UPDATE,
            "evaluation_permitted": True,
            "automatic_promotion_performed": False,
            "private_holdback_read": False,
            "evidence_bundle_sha256": canonical_sha256(
                {str(path): _sha256_file(path) for path in evidence_paths}
            ),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return _blocked(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            classify_terminal(Path(args.stage)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
