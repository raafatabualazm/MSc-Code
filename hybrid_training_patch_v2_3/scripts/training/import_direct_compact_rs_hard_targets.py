#!/usr/bin/env python3
"""Import one sealed provider's code-only RS-SFT hard targets.

The importer is intentionally provider-neutral, but never provider-anonymous.
Every source row must carry the exact provider identity sealed by its harvest,
the sealed fit/executable-view bindings, and the verifier receipt that marked
the code as passing.  This stage verifies those bindings, independently
re-runs the production acceptance harness, canonicalizes exact duplicates, and
emits the generic manifest accepted by ``build_direct_compact_rs_sft.py``.

It never opens held-out bytes.  Historical heldout175 is represented only by
the commitment already validated by the executable-view report.

For the native Qwen3.7 auxiliary collector, ``--source_targets`` is its
``verified_repairs.jsonl``, ``--source_seal`` is ``run_contract.json``, and
``--source_report`` is ``build_report.json``.  Each must be accompanied by its
out-of-band expected SHA-256.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from models.direct_compact_causal import (
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.preprocessing.build_multifunction_executable_view import (
    validate_executable_view,
)
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256 as journal_canonical_sha256,
    journal_record,
    load_journal,
)
from scripts.training.collect_qwen37_auxiliary_repairs import (
    JOURNAL_SCHEMA as QWEN37_JOURNAL_SCHEMA,
    LEDGER_SCHEMA as QWEN37_LEDGER_SCHEMA,
    OUTPUT_SCHEMA as QWEN37_OUTPUT_SCHEMA,
    PINNED_MODELS as QWEN37_PINNED_MODELS,
    REPORT_SCHEMA as QWEN37_REPORT_SCHEMA,
    RUN_CONTRACT_SCHEMA as QWEN37_RUN_CONTRACT_SCHEMA,
    validate_journal_state as validate_qwen37_journal_state,
)


SOURCE_ROW_SCHEMA = "direct-compact-recertified-code-target-v1"
SOURCE_SEAL_SCHEMA = "direct-compact-recertified-code-target-seal-v1"
IMPORTED_ROW_SCHEMA = "direct-compact-rs-hard-target-v1"
IMPORTED_SEAL_SCHEMA = "direct-compact-rs-hard-target-seal-v1"
IMPORT_MANIFEST_SCHEMA = "direct-compact-rs-hard-target-import-v1"
SOURCE_REPORT_SCHEMA = "direct-compact-recertified-code-target-report-v1"
PROVIDER_KEY_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PROHIBITED_REASONING_FIELDS = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "raw_reasoning_content",
        "chain_of_thought",
        "cot",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def read_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object: {resolved}")
    return value


def read_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"{label} has a blank row at {resolved}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{label} row is not an object at "
                    f"{resolved}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {resolved}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows),
    )


def load_evaluator(path: Path, expected_sha256: str) -> Any:
    if sha256_file(path) != expected_sha256:
        raise ValueError("production evaluator SHA-256 mismatch")
    name = "direct_compact_hard_target_evaluator_" + expected_sha256
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import production evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha256_file(path) != expected_sha256:
        raise ValueError("production evaluator changed during import")
    for attribute in (
        "evaluate_dart_jit_tests_detail",
        "COMPLETION_ATTESTATION_ID",
    ):
        if not hasattr(module, attribute):
            raise ValueError(
                f"production evaluator lacks required {attribute}"
            )
    return module


def _require_sha(value: Any, label: str) -> str:
    digest = str(value or "")
    if SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} is not a SHA-256 digest")
    return digest


def validate_provider(value: Any, provider_key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source provider identity is missing")
    provider = dict(value)
    returned = provider.get("returned_models")
    if (
        provider.get("key") != provider_key
        or not str(provider.get("organization") or "").strip()
        or not str(provider.get("api") or "").strip()
        or not str(provider.get("requested_model") or "").strip()
        or provider.get("returned_model_must_equal_requested") is not True
        or not isinstance(returned, list)
        or not returned
        or any(not isinstance(item, str) or not item for item in returned)
        or set(returned) != {str(provider["requested_model"])}
    ):
        raise ValueError(
            "provider identity is incomplete or permits model relabeling"
        )
    return provider


def contains_prohibited_reasoning(value: Any) -> bool:
    """Reject reasoning payloads at any depth while allowing hash receipts."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in PROHIBITED_REASONING_FIELDS:
                return True
            if contains_prohibited_reasoning(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(contains_prohibited_reasoning(item) for item in value)
    return False


def observe_dart_version(evaluator: Any) -> str:
    dart_binary = str(getattr(evaluator, "DART_BIN", "dart") or "dart")
    try:
        result = subprocess.run(
            [dart_binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot observe the active Dart runtime") from exc
    observed = str(result.stderr or result.stdout or "").strip()
    if result.returncode != 0 or not observed:
        raise ValueError("active Dart runtime version probe failed")
    return observed


def _validated_file_record(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} file record is missing")
    try:
        path = Path(str(value.get("path") or "")).expanduser().resolve()
        size = int(value.get("size_bytes", -1))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} file record is malformed") from exc
    if (
        not path.is_file()
        or (expected_path is not None and path != expected_path.resolve())
        or value.get("sha256") != sha256_file(path)
        or size != path.stat().st_size
    ):
        raise ValueError(f"{label} file record differs from local bytes")
    return path, dict(value)


def _record_matches(
    actual: Any,
    expected: Any,
    label: str,
) -> None:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError(f"{label} records are missing")
    for field in ("sha256", "size_bytes"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{label} record {field} differs")


def validate_source_seal(
    *,
    seal: Mapping[str, Any],
    source_targets: Path,
    source_report: Path,
    provider_key: str,
    provider: Mapping[str, Any],
    base_train: Path,
    base_seal: Path,
    contract: Path,
    executable_report: Path,
    executable_view: Mapping[str, Any],
    evaluator_sha256: str,
    completion_attestation_id: str,
    dart_version: str,
    stability_runs: int,
    expected_parent_fit_rows: int,
    expected_executable_rows: int,
) -> None:
    fit = seal.get("fit_universe")
    verifier = seal.get("verifier")
    if (
        seal.get("schema") != SOURCE_SEAL_SCHEMA
        or seal.get("status") != "complete"
        or seal.get("provider_key") != provider_key
        or dict(seal.get("provider") or {}) != dict(provider)
        or seal.get("code_only") is not True
        or seal.get("reasoning_is_not_training_target") is not True
        or int(seal.get("rows", -1)) <= 0
        or seal.get("output_sha256") != sha256_file(source_targets)
        or seal.get("source_report_sha256") != sha256_file(source_report)
        or not isinstance(fit, Mapping)
        or not isinstance(verifier, Mapping)
    ):
        raise ValueError("source hard-target seal contract failed")
    if (
        fit.get("base_train_sha256") != sha256_file(base_train)
        or fit.get("base_train_seal_sha256") != sha256_file(base_seal)
        or fit.get("contract_sha256") != sha256_file(contract)
        or fit.get("executable_view_report_sha256")
        != sha256_file(executable_report)
        or int(fit.get("parent_fit_rows", -1))
        != expected_parent_fit_rows
        or int(fit.get("executable_rows", -1))
        != expected_executable_rows
        or int(fit.get("heldout_rows", -1)) != 175
        or fit.get("heldout_task_ids_sha256")
        != executable_view["heldout_task_ids_sha256"]
        or int(fit.get("heldout_intersection_count", -1)) != 0
        or fit.get("heldout_bytes_opened_during_harvest") is not False
    ):
        raise ValueError("source fit2776/heldout175 binding failed")
    if (
        verifier.get("evaluator_sha256") != evaluator_sha256
        or verifier.get("completion_attestation_id")
        != completion_attestation_id
        or verifier.get("dart_version") != dart_version
        or int(verifier.get("stability_runs", -1)) != stability_runs
        or verifier.get("all_candidates_compiled") is not True
        or verifier.get("all_candidates_passed") is not True
        or verifier.get("acceptance_holdback_exposed_to_provider") is not False
        or verifier.get("heldout_tests_exposed_to_provider") is not False
    ):
        raise ValueError("source verifier contract failed")
    if (
        _require_sha(
            verifier.get("verifier_implementation_sha256"),
            "source verifier implementation",
        )
        != evaluator_sha256
    ):
        raise ValueError("source verifier implementation is not production")
    _require_sha(
        seal.get("source_journal_chain_head_sha256"),
        "source journal chain head",
    )


def validate_qwen37_source(
    *,
    provider_key: str,
    source_targets: Path,
    source_contract_path: Path,
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    executable_report: Mapping[str, Any],
    executable_view: Mapping[str, Any],
    base: Mapping[str, Mapping[str, Any]],
    fit_binding: Mapping[str, str],
    expected_evaluator_sha256: str,
    completion_attestation_id: str,
    dart_version: str,
    stability_runs: int,
) -> dict[str, Any]:
    """Validate and normalize the native Qwen3.7 auxiliary collector."""

    model = str(report.get("model") or "")
    endpoint = str(report.get("endpoint") or "")
    if provider_key != "qwen37" and not provider_key.startswith("qwen37_"):
        raise ValueError(
            "native Qwen3.7 artifacts require a qwen37 provider key"
        )
    inputs = contract.get("inputs")
    transport = contract.get("transport")
    contamination = contract.get("contamination_contract")
    compatibility = contract.get("training_compatibility")
    verifier = contract.get("verifier")
    report_contamination = report.get("contamination_controls")
    report_compatibility = report.get("compatibility")
    parent = executable_report.get("parent")
    if (
        report.get("schema") != QWEN37_REPORT_SCHEMA
        or model not in QWEN37_PINNED_MODELS
        or not endpoint
        or contract.get("schema") != QWEN37_RUN_CONTRACT_SCHEMA
        or contract.get("exact_pinned_model") != model
        or contract.get("returned_model_must_equal_requested") is not True
        or contract.get("endpoint") != endpoint
        or contract.get("mode")
        != "auxiliary_verified_rs_sft_hard_targets_only"
        or report.get("run_contract_sha256")
        != journal_canonical_sha256(contract)
        or not isinstance(inputs, Mapping)
        or not isinstance(transport, Mapping)
        or not isinstance(contamination, Mapping)
        or not isinstance(compatibility, Mapping)
        or not isinstance(verifier, Mapping)
        or not isinstance(report_contamination, Mapping)
        or not isinstance(report_compatibility, Mapping)
        or not isinstance(parent, Mapping)
    ):
        raise ValueError("native Qwen3.7 source contract failed")

    repairs_path, repairs_record = _validated_file_record(
        report.get("verified_repairs_artifact"),
        "Qwen3.7 verified repairs",
        expected_path=source_targets,
    )
    if repairs_path != source_targets.resolve():
        raise AssertionError("Qwen repair path validation drifted")
    _record_matches(inputs.get("fit"), parent.get("train"), "Qwen fit2776")
    _record_matches(
        inputs.get("fit_seal"),
        parent.get("train_seal"),
        "Qwen fit2776 seal",
    )
    fit_seal_path, _fit_seal_record = _validated_file_record(
        inputs.get("fit_seal"), "Qwen fit2776 seal"
    )
    fit_seal = read_json(fit_seal_path, "Qwen fit2776 seal")
    frozen_contract = inputs.get("frozen_contract")
    if not isinstance(frozen_contract, Mapping):
        raise ValueError("Qwen run contract lacks frozen split contract")
    if (
        int(fit_seal.get("rows", -1)) != 2776
        or fit_seal.get("selected_role") != "fit"
        or fit_seal.get("training_allowed") is not True
        or fit_seal.get("heldout_measure_only") is not False
        or fit_seal.get("output_sha256")
        != (inputs.get("fit") or {}).get("sha256")
        or fit_seal.get("contract_sha256")
        != frozen_contract.get("sha256")
        or not isinstance(fit_seal.get("heldout_commitment"), Mapping)
        or inputs.get("heldout_artifact_opened") is not False
    ):
        raise ValueError("Qwen fit2776 source seal contract failed")

    if (
        transport.get("api") != "synchronous_chat_completions"
        or int(transport.get("n", -1)) != 1
        or int(transport.get("workers", -1)) != 1
        or transport.get("one_terminal_logical_draw_per_task") is not True
        or contamination.get("fit_rows") != 2776
        or contamination.get("heldout_artifact_opened") is not False
        or contamination.get("tests_in_provider_messages") is not False
        or contamination.get("gold_in_provider_messages") is not False
        or contamination.get("raw_diagnostic_in_provider_messages") is not False
        or contamination.get("compressed_enriched_assembly_in_provider_messages")
        is not True
        or contamination.get("compressed_cfg_in_provider_messages") is not True
        or compatibility.get(
            "auxiliary_verified_rs_sft_hard_targets_only"
        )
        is not True
        or compatibility.get("qwen38_sequence_kl") is not False
        or compatibility.get("qwen38_cot") is not False
        or compatibility.get("qwen38_union") is not False
        or report_contamination.get("fit_rows") != 2776
        or report_contamination.get("heldout_artifact_opened") is not False
        or report_contamination.get("provider_received_tests") is not False
        or report_contamination.get("provider_received_gold") is not False
        or report_contamination.get(
            "provider_received_raw_compiler_diagnostics"
        )
        is not False
        or report_contamination.get(
            "provider_received_compressed_enriched_assembly"
        )
        is not True
        or report_contamination.get("provider_received_compressed_cfg")
        is not True
        or report_compatibility.get(
            "auxiliary_verified_rs_sft_hard_target_import_allowed"
        )
        is not True
        or report_compatibility.get("qwen38_sequence_kl_import_allowed")
        is not False
        or report_compatibility.get("qwen38_cot_import_allowed") is not False
        or report_compatibility.get("qwen38_union_import_allowed") is not False
    ):
        raise ValueError("Qwen source contamination/compatibility gate failed")
    if (
        verifier.get("implementation_sha256")
        != expected_evaluator_sha256
        or int(verifier.get("stability_runs", -1)) != stability_runs
        or verifier.get("completion_attestation")
        != completion_attestation_id
    ):
        raise ValueError("Qwen source verifier differs from production")

    report_journal = report.get("journal")
    if not isinstance(report_journal, Mapping):
        raise ValueError("Qwen report lacks a journal receipt")
    journal_path = Path(
        str(report_journal.get("path") or "")
    ).expanduser().resolve()
    observed_journal = journal_record(journal_path)
    if dict(report_journal) != observed_journal:
        raise ValueError("Qwen journal receipt differs from its hash chain")
    events = load_journal(journal_path)
    state = validate_qwen37_journal_state(
        events,
        contract_sha256=journal_canonical_sha256(contract),
        budget_cap=int((contract.get("budget") or {}).get("cap_tokens", -1)),
    )
    if state.complete is None or state.orphan_task_id is not None:
        raise ValueError("Qwen journal is not terminal")
    header_contract = (events[0].get("run_contract") or {}) if events else {}
    if (
        events[0].get("schema") != QWEN37_JOURNAL_SCHEMA
        or header_contract.get("sha256") != sha256_file(source_contract_path)
        or Path(str(header_contract.get("path") or "")).expanduser().resolve()
        != source_contract_path.resolve()
    ):
        raise ValueError("Qwen journal does not bind the source run contract")

    ledger_path, _ledger_record = _validated_file_record(
        report.get("token_ledger"), "Qwen token ledger"
    )
    ledger = read_json(ledger_path, "Qwen token ledger")
    statuses = Counter(
        str(row.get("status") or "") for row in state.terminals.values()
    )
    if (
        ledger.get("schema") != QWEN37_LEDGER_SCHEMA
        or ledger.get("model") != model
        or ledger.get("endpoint") != endpoint
        or ledger.get("journal") != observed_journal
        or int(ledger.get("logical_draws", -1)) != len(state.terminals)
        or int(ledger.get("budget_debit_tokens", -1))
        != state.budget_debits
        or int(ledger.get("budget_cap_tokens", -1))
        != int((contract.get("budget") or {}).get("cap_tokens", -2))
        or int(ledger.get("provider_reported_actual_tokens", -1))
        != state.actual_usage_tokens
        or int(
            ledger.get(
                "unknown_usage_slots_charged_at_full_reservation", -1
            )
        )
        != state.unknown_usage_slots
        or int(ledger.get("remaining_tokens", -1))
        != int((contract.get("budget") or {}).get("cap_tokens", -2))
        - state.budget_debits
        or int(report.get("logical_draws", -1)) != len(state.terminals)
        or report.get("terminal_statuses") != dict(statuses)
    ):
        raise ValueError("Qwen report/ledger differs from terminal journal")

    provider = validate_provider(
        {
            "key": provider_key,
            "organization": "Alibaba Cloud Model Studio",
            "api": transport["api"],
            "requested_model": model,
            "returned_model_must_equal_requested": True,
            "returned_models": [model],
        },
        provider_key,
    )
    source_rows = read_jsonl(source_targets, "Qwen verified repairs")
    normalized_rows: list[dict[str, Any]] = []
    excluded_task_ids: list[str] = []
    seen_tasks: set[str] = set()
    expected_source_attestation = {
        "verifier_sha256": expected_evaluator_sha256,
        "completion_attestation": completion_attestation_id,
        "stability_runs": stability_runs,
        "passed": True,
    }
    expected_verification = {
        "evaluator_sha256": expected_evaluator_sha256,
        "completion_attestation_id": completion_attestation_id,
        "dart_version": dart_version,
        "stability_runs": stability_runs,
        "compiled": True,
        "passed": True,
        "acceptance_holdback_exposed_to_provider": False,
        "heldout_tests_exposed_to_provider": False,
    }
    for index, row in enumerate(source_rows):
        task_id = str(row.get("task_id") or "")
        code = str(row.get("target") or "")
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        source = row.get("source")
        attestation = row.get("attestation")
        terminal = state.terminals.get(task_id)
        started = state.starts.get(task_id)
        failed_candidate = (
            started.get("candidate") if isinstance(started, Mapping) else None
        )
        if (
            row.get("schema") != QWEN37_OUTPUT_SCHEMA
            or not task_id
            or task_id in seen_tasks
            or not code.strip()
            or row.get("target_sha256") != code_sha
            or row.get("target_mode") != "final_dart_code_only"
            or row.get("reasoning_in_target") is not False
            or row.get("training_use")
            != "auxiliary_verified_rs_sft_hard_target_only"
            or not isinstance(source, Mapping)
            or not isinstance(attestation, Mapping)
            or source.get("model") != model
            or source.get("endpoint") != endpoint
            or any(
                attestation.get(field) != expected
                for field, expected in expected_source_attestation.items()
            )
            or not isinstance(terminal, Mapping)
            or not isinstance(started, Mapping)
            or not isinstance(failed_candidate, Mapping)
            or started.get("requested_model") != model
            or started.get("endpoint") != endpoint
            or failed_candidate.get("task_id") != task_id
            or source.get("failed_candidate_code_sha256")
            != failed_candidate.get("code_sha256")
            or source.get("priority")
            != failed_candidate.get("priority_name")
            or terminal.get("status") != "verified_pass"
            or terminal.get("requested_model") != model
            or terminal.get("returned_model") != model
            or terminal.get("returned_model_matches_requested") is not True
            or terminal.get("endpoint") != endpoint
            or terminal.get("code") != code
            or terminal.get("code_sha256") != code_sha
            or source.get("provider_request_id")
            != terminal.get("provider_request_id")
            or source.get("system_fingerprint")
            != terminal.get("system_fingerprint")
            or source.get("provider_response_sha256")
            != terminal.get("provider_response_sha256")
            or source.get("raw_content_sha256")
            != terminal.get("raw_content_sha256")
            or source.get("raw_reasoning_sha256")
            != terminal.get("raw_reasoning_sha256")
            or contains_prohibited_reasoning(row)
        ):
            raise ValueError(f"Qwen verified repair row {index} is inconsistent")
        seen_tasks.add(task_id)
        terminal_verification = terminal.get("verification") or {}
        for field, expected in (
            ("verifier_sha256", expected_evaluator_sha256),
            ("completion_attestation", completion_attestation_id),
            ("stability_runs", stability_runs),
            ("compiled", True),
            ("passed", True),
            ("harness_completion_attested", True),
        ):
            if terminal_verification.get(field) != expected:
                raise ValueError(
                    f"Qwen terminal {task_id} verifier field {field} differs"
                )
        if task_id not in base:
            if task_id not in set(executable_view.get("excluded_task_ids") or []):
                raise ValueError(
                    f"Qwen target {task_id} is outside the sealed fit universe"
                )
            excluded_task_ids.append(task_id)
            continue
        provenance = {
            "native_schema": QWEN37_OUTPUT_SCHEMA,
            "source_row": index,
            "source": dict(source),
            "attestation": dict(attestation),
            "journal_terminal_event_sha256": terminal[
                "journal_event_sha256"
            ],
        }
        normalized_rows.append(
            {
                "schema": SOURCE_ROW_SCHEMA,
                "provider_key": provider_key,
                "provider": provider,
                "task_id": task_id,
                "code": code,
                "code_sha256": code_sha,
                "fit_bindings": dict(fit_binding),
                "verification": expected_verification,
                "source_provenance": provenance,
                "source_provenance_sha256": stable_sha256(provenance),
            }
        )
    if (
        int(report.get("verified_repairs", -1)) != len(source_rows)
        or statuses.get("verified_pass", 0) != len(source_rows)
        or not normalized_rows
    ):
        raise ValueError("Qwen materialized repair accounting differs")

    fit_universe = {
        **dict(fit_binding),
        "parent_fit_rows": 2776,
        "executable_rows": int(executable_view["rows"]),
        "heldout_rows": 175,
        "heldout_task_ids_sha256": executable_view[
            "heldout_task_ids_sha256"
        ],
        "heldout_intersection_count": 0,
        "heldout_bytes_opened_during_harvest": False,
    }
    return {
        "provider": provider,
        "rows": normalized_rows,
        "source_rows": len(source_rows),
        "excluded_task_ids": sorted(excluded_task_ids),
        "fit_universe": fit_universe,
        "source_journal_chain_head_sha256": observed_journal[
            "head_event_sha256"
        ],
        "source_verifier_implementation_sha256": expected_evaluator_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base_train", required=True, type=Path)
    parser.add_argument("--base_train_seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--executable_view_report", required=True, type=Path)
    parser.add_argument(
        "--expected_executable_view_report_sha256", required=True
    )
    parser.add_argument("--provider_key", required=True)
    parser.add_argument("--source_targets", required=True, type=Path)
    parser.add_argument(
        "--source_seal",
        required=True,
        type=Path,
        help=(
            "Generic source seal, or native Qwen3.7 run_contract.json."
        ),
    )
    parser.add_argument(
        "--expected_source_seal_sha256",
        required=True,
        help="Out-of-band SHA-256 for --source_seal.",
    )
    parser.add_argument(
        "--source_report",
        required=True,
        type=Path,
        help="Generic source report, or native Qwen3.7 build_report.json.",
    )
    parser.add_argument("--expected_source_report_sha256", required=True)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--expected_evaluator_sha256", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_parent_fit_rows", type=int, default=2776)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=2)
    return parser.parse_args()


def import_targets(args: argparse.Namespace) -> dict[str, Any]:
    provider_key = str(args.provider_key).strip().lower()
    if PROVIDER_KEY_RE.fullmatch(provider_key) is None:
        raise ValueError("provider_key must be a stable lowercase identifier")
    if (
        args.expected_parent_fit_rows != 2776
        or args.workers <= 0
        or args.timeout <= 0
        or args.stability_runs < 2
    ):
        raise ValueError(
            "production import requires fit2776, positive runtime limits, "
            "and at least two stability runs"
        )

    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "base_train": args.base_train,
            "base_train_seal": args.base_train_seal,
            "contract": args.contract,
            "executable_view_report": args.executable_view_report,
            "source_targets": args.source_targets,
            "source_seal": args.source_seal,
            "source_report": args.source_report,
            "evaluator": args.evaluator,
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        sha256_file(paths["executable_view_report"])
        != args.expected_executable_view_report_sha256
        or sha256_file(paths["source_seal"])
        != args.expected_source_seal_sha256
        or sha256_file(paths["source_report"])
        != args.expected_source_report_sha256
        or sha256_file(paths["evaluator"])
        != args.expected_evaluator_sha256
    ):
        raise ValueError("an expected import input SHA-256 does not match")

    contract = DirectCompactContract.load(paths["contract"])
    base_seal_value = validate_join_seal(
        paths["base_train"],
        paths["base_train_seal"],
        paths["contract"],
        expected_role="fit",
    )
    executable_report_value = read_json(
        paths["executable_view_report"], "executable-view report"
    )
    executable_outputs = executable_report_value.get("outputs") or {}
    executable_view = validate_executable_view(
        dataset=paths["base_train"],
        seal=paths["base_train_seal"],
        f2=(executable_outputs.get("f2") or {}).get("path", ""),
        f2_manifest=(executable_outputs.get("f2_manifest") or {}).get(
            "path", ""
        ),
        build_report=paths["executable_view_report"],
        expected_build_report_sha256=(
            args.expected_executable_view_report_sha256
        ),
        contract=paths["contract"],
        verify_heldout=False,
        expected_parent_rows=args.expected_parent_fit_rows,
    )
    base_rows = read_jsonl(paths["base_train"], "executable fit dataset")
    expected_executable_rows = int(
        executable_view.get("rows", len(base_rows))
    )
    if (
        len(base_rows) != int(base_seal_value["rows"])
        or len(base_rows) != expected_executable_rows
        or int(executable_view.get("parent_rows", -1))
        != args.expected_parent_fit_rows
        or int(executable_view.get("heldout_rows", -1)) != 175
        or executable_view.get(
            "heldout_bytes_opened_during_validation", False
        )
    ):
        raise ValueError("executable fit2776 view contract failed")
    base: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in base:
            raise ValueError("executable fit dataset has duplicate task IDs")
        contract.validate_row(row, task_id)
        base[task_id] = row

    evaluator = load_evaluator(
        paths["evaluator"], args.expected_evaluator_sha256
    )
    completion_attestation_id = str(
        evaluator.COMPLETION_ATTESTATION_ID
    )
    dart_version = str(base_seal_value.get("dart_version") or "")
    observed_dart_version = observe_dart_version(evaluator)
    if (
        base_seal_value.get("evaluator_sha256")
        != args.expected_evaluator_sha256
        or base_seal_value.get("completion_attestation_id")
        != completion_attestation_id
        or int(base_seal_value.get("stability_runs", -1))
        != args.stability_runs
        or not dart_version
        or observed_dart_version != dart_version
    ):
        raise ValueError(
            "requested recertification harness differs from the sealed "
            "production fit verifier"
        )
    fit_binding = {
        "base_train_sha256": sha256_file(paths["base_train"]),
        "base_train_seal_sha256": sha256_file(paths["base_train_seal"]),
        "contract_sha256": sha256_file(paths["contract"]),
        "executable_view_report_sha256": sha256_file(
            paths["executable_view_report"]
        ),
    }
    source_report_value = read_json(
        paths["source_report"], "source target report"
    )
    source_seal_value = read_json(
        paths["source_seal"], "source target seal/contract"
    )
    native_qwen37 = (
        source_report_value.get("schema") == QWEN37_REPORT_SCHEMA
    )
    if native_qwen37:
        native = validate_qwen37_source(
            provider_key=provider_key,
            source_targets=paths["source_targets"],
            source_contract_path=paths["source_seal"],
            report=source_report_value,
            contract=source_seal_value,
            executable_report=executable_report_value,
            executable_view=executable_view,
            base=base,
            fit_binding=fit_binding,
            expected_evaluator_sha256=args.expected_evaluator_sha256,
            completion_attestation_id=completion_attestation_id,
            dart_version=dart_version,
            stability_runs=args.stability_runs,
        )
        provider = native["provider"]
        source_rows = native["rows"]
        source_rows_total = int(native["source_rows"])
        source_excluded_task_ids = list(native["excluded_task_ids"])
        source_seal = {
            "fit_universe": native["fit_universe"],
            "source_journal_chain_head_sha256": native[
                "source_journal_chain_head_sha256"
            ],
            "verifier": {
                "verifier_implementation_sha256": native[
                    "source_verifier_implementation_sha256"
                ]
            },
        }
    else:
        source_seal = source_seal_value
        provider = validate_provider(source_seal.get("provider"), provider_key)
        if (
            source_report_value.get("schema") != SOURCE_REPORT_SCHEMA
            or source_report_value.get("status") != "complete"
            or source_report_value.get("provider_key") != provider_key
            or dict(source_report_value.get("provider") or {}) != provider
            or source_report_value.get("source_targets_sha256")
            != sha256_file(paths["source_targets"])
            or source_report_value.get("verifier_implementation_sha256")
            != args.expected_evaluator_sha256
            or source_report_value.get("code_only") is not True
            or source_report_value.get("reasoning_is_not_training_target")
            is not True
        ):
            raise ValueError("generic source report contract failed")
        validate_source_seal(
            seal=source_seal,
            source_targets=paths["source_targets"],
            source_report=paths["source_report"],
            provider_key=provider_key,
            provider=provider,
            base_train=paths["base_train"],
            base_seal=paths["base_train_seal"],
            contract=paths["contract"],
            executable_report=paths["executable_view_report"],
            executable_view=executable_view,
            evaluator_sha256=args.expected_evaluator_sha256,
            completion_attestation_id=completion_attestation_id,
            dart_version=dart_version,
            stability_runs=args.stability_runs,
            expected_parent_fit_rows=args.expected_parent_fit_rows,
            expected_executable_rows=expected_executable_rows,
        )
        source_rows = read_jsonl(
            paths["source_targets"], "source hard targets"
        )
        if len(source_rows) != int(source_seal["rows"]):
            raise ValueError("source target rows differ from source seal")
        source_rows_total = len(source_rows)
        source_excluded_task_ids = []

    ordered_candidate_keys: list[dict[str, str]] = []
    source_task_ids: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    expected_verifier = {
        "evaluator_sha256": args.expected_evaluator_sha256,
        "completion_attestation_id": completion_attestation_id,
        "dart_version": dart_version,
        "stability_runs": args.stability_runs,
        "compiled": True,
        "passed": True,
        "acceptance_holdback_exposed_to_provider": False,
        "heldout_tests_exposed_to_provider": False,
    }
    for index, row in enumerate(source_rows):
        task_id = str(row.get("task_id") or "")
        code = str(row.get("code") or "")
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        provenance = row.get("source_provenance")
        if (
            row.get("schema") != SOURCE_ROW_SCHEMA
            or row.get("provider_key") != provider_key
            or dict(row.get("provider") or {}) != provider
            or not task_id
            or task_id not in base
            or not code.strip()
            or row.get("code_sha256") != code_sha
            or dict(row.get("fit_bindings") or {}) != fit_binding
            or dict(row.get("verification") or {}) != expected_verifier
            or not isinstance(provenance, Mapping)
            or row.get("source_provenance_sha256")
            != stable_sha256(provenance)
            or contains_prohibited_reasoning(row)
        ):
            raise ValueError(
                f"source target row {index} violates code-only fit contract"
            )
        source_task_ids.append(task_id)
        ordered_candidate_keys.append(
            {"task_id": task_id, "code_sha256": code_sha}
        )
        grouped.setdefault((task_id, code_sha), []).append(
            {
                "source_row": index,
                "source_row_sha256": stable_sha256(row),
                "source_provenance": dict(provenance),
                "source_provenance_sha256": stable_sha256(provenance),
                "code": code,
            }
        )
    if not native_qwen37 and (
        source_seal.get("task_set_sha256")
        != stable_sha256(sorted(set(source_task_ids)))
        or source_seal.get("ordered_candidate_keys_sha256")
        != stable_sha256(ordered_candidate_keys)
    ):
        raise ValueError("source task/candidate digest differs from seal")

    candidates: list[dict[str, Any]] = []
    for (task_id, code_sha), contributors in sorted(grouped.items()):
        contributors.sort(
            key=lambda value: (
                value["source_provenance_sha256"],
                value["source_row_sha256"],
                value["source_row"],
            )
        )
        candidates.append(
            {
                "task_id": task_id,
                "code": contributors[0]["code"],
                "code_sha256": code_sha,
                "contributors": contributors,
            }
        )

    def recertify(candidate: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(candidate["task_id"])
        code = str(candidate["code"])
        tests = str(
            base[task_id].get("acceptance_tests")
            or base[task_id].get("tests")
            or ""
        )
        # A fence is the evaluator's lossless full-program input mode.  Without
        # it, a leading enum/class/extension could be mistaken for prose.
        raw = f"```dart\n{code.rstrip()}\n```"
        compiled, passed, diagnostic, _source = (
            evaluator.evaluate_dart_jit_tests_detail(
                raw,
                tests,
                f"{provider_key}_{task_id}_{candidate['code_sha256'][:12]}",
                timeout=args.timeout,
                stability_runs=args.stability_runs,
            )
        )
        return {
            **candidate,
            "compiled": bool(compiled),
            "passed": bool(passed),
            "diagnostic_sha256": hashlib.sha256(
                str(diagnostic or "").encode()
            ).hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        recertified = list(pool.map(recertify, candidates))
    failed = [
        {
            "task_id": row["task_id"],
            "code_sha256": row["code_sha256"],
            "compiled": row["compiled"],
            "passed": row["passed"],
            "diagnostic_sha256": row["diagnostic_sha256"],
        }
        for row in recertified
        if not row["compiled"] or not row["passed"]
    ]
    if failed:
        raise ValueError(
            f"{len(failed)} sealed source targets failed independent "
            f"recertification: {failed[:10]!r}"
        )

    imported_rows: list[dict[str, Any]] = []
    for row in recertified:
        contributors = [
            {
                "source_row": value["source_row"],
                "source_row_sha256": value["source_row_sha256"],
                "source_provenance": value["source_provenance"],
                "source_provenance_sha256": value[
                    "source_provenance_sha256"
                ],
            }
            for value in row["contributors"]
        ]
        imported_rows.append(
            {
                "schema": IMPORTED_ROW_SCHEMA,
                "ok": True,
                "task_id": row["task_id"],
                "code": row["code"],
                "code_sha256": row["code_sha256"],
                "provider_key": provider_key,
                "provider": provider,
                "provider_provenance": {
                    "source_targets_sha256": sha256_file(
                        paths["source_targets"]
                    ),
                    "source_seal_sha256": sha256_file(
                        paths["source_seal"]
                    ),
                    "source_report_sha256": sha256_file(
                        paths["source_report"]
                    ),
                    "source_journal_chain_head_sha256": source_seal[
                        "source_journal_chain_head_sha256"
                    ],
                    "contributors": contributors,
                },
                "independent_recertification": {
                    **expected_verifier,
                    "diagnostic_sha256": row["diagnostic_sha256"],
                },
            }
        )
    imported_rows.sort(
        key=lambda row: (row["task_id"], row["code_sha256"])
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "repairs": output_dir / "verified_repairs.jsonl",
        "seal": output_dir / "verified_repairs.seal.json",
        "manifest": output_dir / "import_manifest.json",
    }
    write_jsonl(output_paths["repairs"], imported_rows)
    imported_ids = [str(row["task_id"]) for row in imported_rows]
    imported_keys = [
        {
            "task_id": str(row["task_id"]),
            "code_sha256": str(row["code_sha256"]),
        }
        for row in imported_rows
    ]
    seal_value = {
        "schema": IMPORTED_SEAL_SCHEMA,
        "selected_role": "fit",
        "training_allowed": True,
        "provider_key": provider_key,
        "provider": provider,
        "code_only": True,
        "rows": len(imported_rows),
        "unique_tasks": len(set(imported_ids)),
        "output_sha256": sha256_file(output_paths["repairs"]),
        "ordered_candidate_keys_sha256": stable_sha256(imported_keys),
        "task_set_sha256": stable_sha256(sorted(set(imported_ids))),
        "fit_universe": dict(source_seal["fit_universe"]),
        "verifier": {
            **expected_verifier,
            "all_candidates_compiled": True,
            "all_candidates_passed": True,
            "source_verifier_implementation_sha256": source_seal[
                "verifier"
            ]["verifier_implementation_sha256"],
        },
        "source_seal_sha256": sha256_file(paths["source_seal"]),
        "importer_sha256": sha256_file(Path(__file__).resolve()),
        "heldout_bytes_opened_during_import": False,
    }
    write_json(output_paths["seal"], seal_value)
    manifest = {
        "schema": IMPORT_MANIFEST_SCHEMA,
        "status": "complete",
        "provider_key": provider_key,
        "provider": provider,
        "requested_model": provider["requested_model"],
        "api": provider["api"],
        "code_only": True,
        "reasoning_is_not_training_target": True,
        "inputs": {
            "base_train": file_record(paths["base_train"]),
            "base_train_seal": file_record(paths["base_train_seal"]),
            "contract": file_record(paths["contract"]),
            "executable_view_report": file_record(
                paths["executable_view_report"]
            ),
            "source_targets": file_record(paths["source_targets"]),
            "source_seal": file_record(paths["source_seal"]),
            "source_report": file_record(paths["source_report"]),
            "evaluator": file_record(paths["evaluator"]),
        },
        "outputs": {
            "verified_repairs": file_record(output_paths["repairs"]),
            "verified_repairs_seal": file_record(output_paths["seal"]),
        },
        "counts": {
            "source_rows": source_rows_total,
            "source_rows_execution_eligible": len(source_rows),
            "source_rows_excluded_by_executable_view": len(
                source_excluded_task_ids
            ),
            "source_excluded_task_ids": source_excluded_task_ids,
            "source_unique_task_code_pairs": len(candidates),
            "exact_source_duplicates_collapsed": (
                len(source_rows) - len(candidates)
            ),
            "output_rows": len(imported_rows),
            "unique_tasks": len(set(imported_ids)),
            "independent_recertification_failures": 0,
        },
        "fit_universe": dict(source_seal["fit_universe"]),
        "verifier": seal_value["verifier"],
        "source_journal_chain_head_sha256": source_seal[
            "source_journal_chain_head_sha256"
        ],
        "source_schema": (
            QWEN37_REPORT_SCHEMA if native_qwen37 else SOURCE_REPORT_SCHEMA
        ),
        "invariants": {
            "provider_identity_preserved": True,
            "provider_relabeling_permitted": False,
            "deterministic_exact_code_dedupe": True,
            "all_outputs_independently_recertified": True,
            "reasoning_excluded_from_training_targets": True,
            "fit2776_membership_bound": True,
            "heldout175_intersection_zero": True,
            "heldout_bytes_opened_during_import": False,
            "source_nonexecutable_targets_never_imported": True,
        },
    }
    write_json(output_paths["manifest"], manifest)
    print(
        "DIRECT_COMPACT_HARD_TARGET_IMPORT_COMPLETE "
        f"provider={provider_key} rows={len(imported_rows)} "
        f"tasks={len(set(imported_ids))} "
        f"sha256={sha256_file(output_paths['repairs'])}",
        flush=True,
    )
    return manifest


def main() -> int:
    import_targets(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
