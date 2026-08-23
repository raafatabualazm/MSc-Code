#!/usr/bin/env python3
"""Model-scoped circuit breaker for exact Qwen free-quota exhaustion.

The guard watches only an explicit clean-v4/v5 run/service allowlist.  On the
first newly observed response-less HTTP 403 free-quota boundary, it stops the
allowlisted units for that exact requested model and writes an atomic receipt.
It never stops a different Qwen model, and no DeepSeek unit can be constructed
from this file.

Generic HTTP 429 rate limiting is deliberately ignored.  Error classification
is delegated to ``authorize_qwen_free_quota_resume``, which recognizes only the
two exact HTTP 403 provider payloads observed in these runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import authorize_qwen_free_quota_resume as quota_error


SCHEMA = "qwen37-model-free-quota-guard-v1"
EVENT_SCHEMA = "qwen37-model-free-quota-stop-receipt-v1"
ALLOWED_MODELS = frozenset(
    {
        "qwen3.7-max-2026-05-17",
        "qwen3.7-max-preview",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
    }
)


@dataclass(frozen=True)
class Target:
    model: str
    run_id: str
    unit: str


TARGETS = (
    Target(
        "qwen3.7-max-2026-05-17",
        "qwen37_clean_v4_0517_opus_k3_mc12k_tol10_tb8k",
        "frontier-qwen37-clean-v4-0517-opus-k3-mc12k-tol10-tb8k.service",
    ),
    Target(
        "qwen3.7-max-2026-05-17",
        "qwen37_clean_v4_0517_codex_k3_mc12k_tol10_tb8k",
        "frontier-qwen37-clean-v4-0517-codex-k3-mc12k-tol10-tb8k.service",
    ),
    Target(
        "qwen3.7-max-2026-05-17",
        "qwen37_clean_v5_supplement_0517_opus_k2_mc12k_tol10_tb8k",
        (
            "frontier-qwen37-clean-v5-supplement-0517-opus-k2-"
            "mc12k-tol10-tb8k.service"
        ),
    ),
    Target(
        "qwen3.7-max-2026-05-17",
        "qwen37_clean_v5_supplement_0517_codex_k2_mc12k_tol10_tb8k",
        (
            "frontier-qwen37-clean-v5-supplement-0517-codex-k2-"
            "mc12k-tol10-tb8k.service"
        ),
    ),
    Target(
        "qwen3.7-max-preview",
        "qwen37_clean_v4_preview_opus_k2_mc12k_tol10_tb8k",
        "frontier-qwen37-clean-v4-preview-opus-k2-mc12k-tol10-tb8k.service",
    ),
    Target(
        "qwen3.7-max-preview",
        "qwen37_clean_v4_preview_codex_k2_mc12k_tol10_tb8k",
        "frontier-qwen37-clean-v4-preview-codex-k2-mc12k-tol10-tb8k.service",
    ),
    Target(
        "qwen3.7-max-preview",
        "qwen37_clean_v5_supplement_preview_opus_k3_mc12k_tol10_tb8k",
        (
            "frontier-qwen37-clean-v5-supplement-preview-opus-k3-"
            "mc12k-tol10-tb8k.service"
        ),
    ),
    Target(
        "qwen3.7-max-preview",
        "qwen37_clean_v5_supplement_preview_codex_k3_mc12k_tol10_tb8k",
        (
            "frontier-qwen37-clean-v5-supplement-preview-codex-k3-"
            "mc12k-tol10-tb8k.service"
        ),
    ),
    # The viable unpaired diagnostic shards.  Their moderation-rejected Opus
    # siblings are intentionally absent.
    Target(
        "qwen3.7-max-2026-05-20",
        "qwen37_clean_v4_0520_codex_k3_mc12k_tol10_tb8k",
        "frontier-qwen37-clean-v4-0520-codex-k3-mc12k-tol10-tb8k.service",
    ),
    Target(
        "qwen3.7-max-2026-06-08",
        "qwen37_clean_v4_0608_codex_k2_mc12k_tol10_tb8k",
        "frontier-qwen37-clean-v4-0608-codex-k2-mc12k-tol10-tb8k.service",
    ),
)


class GuardError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sealed_mapping() -> list[dict[str, str]]:
    return [
        {"model": target.model, "run_id": target.run_id, "unit": target.unit}
        for target in TARGETS
    ]


MAPPING_SHA256 = stable_sha256(sealed_mapping())


def targets_for_model(model: str) -> tuple[Target, ...]:
    if model not in ALLOWED_MODELS:
        raise GuardError(f"model is not allowlisted: {model}")
    selected = tuple(target for target in TARGETS if target.model == model)
    if not selected:
        raise GuardError(f"model has no sealed targets: {model}")
    if any("deepseek" in target.unit.lower() for target in selected):
        raise GuardError("sealed target mapping contains a forbidden DeepSeek unit")
    return selected


def load_processed_evidence(receipt_dir: Path) -> set[str]:
    processed: set[str] = set()
    if not receipt_dir.exists():
        return processed
    for path in sorted(receipt_dir.glob("event_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GuardError(f"cannot parse existing receipt {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema") != EVENT_SCHEMA:
            raise GuardError(f"foreign receipt in guard directory: {path}")
        if value.get("mapping_sha256") != MAPPING_SHA256:
            raise GuardError(f"receipt mapping hash mismatch: {path}")
        evidence = value.get("evidence")
        if not isinstance(evidence, list):
            raise GuardError(f"receipt evidence is malformed: {path}")
        for item in evidence:
            if not isinstance(item, dict):
                raise GuardError(f"receipt evidence item is malformed: {path}")
            digest = item.get("row_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise GuardError(f"receipt evidence digest is malformed: {path}")
            processed.add(digest)
    return processed


def scan_model(
    run_root: Path,
    model: str,
    processed: set[str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for target in targets_for_model(model):
        path = run_root / target.run_id / "attempts.jsonl"
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                digest = sha256_bytes(raw)
                if digest in processed:
                    continue
                try:
                    row = json.loads(raw)
                except Exception as exc:
                    raise GuardError(
                        f"cannot parse {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise GuardError(f"{path}:{line_number} is not an object")
                variant = quota_error.qwen_free_quota_variant(row)
                if variant is None:
                    continue
                if row.get("requested_model") != model:
                    raise GuardError(
                        f"{path}:{line_number} exact boundary has a foreign model"
                    )
                evidence.append(
                    {
                        "run_id": target.run_id,
                        "unit": target.unit,
                        "journal": str(path),
                        "line_number": line_number,
                        "row_sha256": digest,
                        "task_id": row.get("task_id"),
                        "sample_index": row.get("sample_index"),
                        "attempt_index": row.get("attempt_index"),
                        "attempt_id": row.get("attempt_id"),
                        "provider_error_variant": variant,
                        "response_received": False,
                        "http_status": 403,
                    }
                )
    return evidence


def unit_state(unit: str) -> dict[str, str | None]:
    completed = subprocess.run(
        (
            "systemctl",
            "show",
            unit,
            "--property=ActiveState",
            "--property=SubState",
            "--property=Result",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return {
        "active_state": fields.get("ActiveState"),
        "sub_state": fields.get("SubState"),
        "result": fields.get("Result"),
    }


def stop_model_units(model: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets_for_model(model):
        before = unit_state(target.unit)
        completed = subprocess.run(
            ("systemctl", "stop", target.unit),
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        after = unit_state(target.unit)
        results.append(
            {
                "unit": target.unit,
                "run_id": target.run_id,
                "before": before,
                "stop_returncode": completed.returncode,
                "stderr_sha256": sha256_bytes(
                    completed.stderr.encode("utf-8", errors="replace")
                ),
                "after": after,
            }
        )
    return results


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def trip_guard(
    *,
    model: str,
    evidence: list[dict[str, Any]],
    receipt_dir: Path,
) -> dict[str, Any]:
    if not evidence:
        raise GuardError("cannot trip guard without exact evidence")
    if any(item.get("http_status") != 403 for item in evidence):
        raise GuardError("guard evidence contains a non-403 event")
    tripped_at = utc_now()
    receipt: dict[str, Any] = {
        "schema": EVENT_SCHEMA,
        "status": "stopping",
        "tripped_at": tripped_at,
        "model": model,
        "scope": "exact_requested_model_only",
        "mapping_sha256": MAPPING_SHA256,
        "evidence": evidence,
        "ignored_error_policy": {
            "http_429_rate_limits_never_trip": True,
            "generic_insufficient_quota_never_trips": True,
        },
        "deepseek_units_targeted": False,
    }
    receipt_dir.mkdir(parents=True, exist_ok=True)
    event_digest = stable_sha256(
        {
            "model": model,
            "row_sha256": sorted(item["row_sha256"] for item in evidence),
        }
    )
    receipt_path = receipt_dir / f"event_{event_digest}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stop_results"] = stop_model_units(model)
    receipt["status"] = "stopped_model_units"
    receipt["finished_at"] = utc_now()
    atomic_write_json(receipt_path, receipt)
    return receipt


def poll(
    *,
    run_root: Path,
    receipt_dir: Path,
    model: str,
    poll_seconds: float,
    once: bool,
) -> int:
    targets_for_model(model)
    if poll_seconds <= 0:
        raise GuardError("poll interval must be positive")
    while True:
        processed = load_processed_evidence(receipt_dir)
        evidence = scan_model(run_root, model, processed)
        if evidence:
            receipt = trip_guard(
                model=model,
                evidence=evidence,
                receipt_dir=receipt_dir,
            )
            print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
            return 0
        if once:
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "status": "armed_no_boundary",
                        "model": model,
                        "mapping_sha256": MAPPING_SHA256,
                        "targets": len(targets_for_model(model)),
                    },
                    sort_keys=True,
                )
            )
            return 0
        time.sleep(poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/workspace/artifacts/frontier_ceiling_two_enrichments/runs"
        ),
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path(
            "/workspace/artifacts/frontier_ceiling_two_enrichments/"
            "qwen37_quota_guard_v1"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return poll(
            run_root=args.run_root.expanduser().resolve(),
            receipt_dir=args.receipt_dir.expanduser().resolve(),
            model=args.model,
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
