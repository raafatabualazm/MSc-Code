#!/usr/bin/env python3
"""Submit only the two unbilled Sonnet primary slots that hit credit capacity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import frontier_passk_anthropic_batch as batch
from frontier_core import JsonlJournal, load_json, sha256_file, stable_sha256, utc_now

EXPECTED_RUNNER_SHA256 = (
    "86aa896c2f7c97d90006e5c69ff77ed61432df9d33780d2e950343112833c5d8"
)
EXPECTED_OUT = Path(
    "/workspace/artifacts/frontier_ceiling_two_enrichments/runs/"
    "anthropic_sonnet5_batch_screen_k2_warm_v1/opus"
)
EXPECTED_INITIAL_BATCH_ID = "msgbatch_016QhZR7xz4bGPbNkqir5Pju"
EXPECTED_FAILED_CUSTOM_IDS = {
    "a00_s00_t075_c08192",
    "a00_s01_t065_c08192",
}
EXPECTED_RECOVERED_ERROR_TYPE = "provider_credit_balance_exhausted_retryable"


def _cli_value(argv: Sequence[str], flag: str) -> str:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    raise RuntimeError(f"{flag} is required")


def _select_primary_retries(
    original: Any,
    plans: Sequence[Mapping[str, Any]],
    slot_attempts: Sequence[Mapping[str, Any]],
    terminals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specs = original(plans, slot_attempts, terminals)
    if not slot_attempts and not terminals:
        # Preserve the original preflight cost computation in batch.main().
        return specs
    failed_rows = {
        str(row.get("custom_id") or ""): row
        for row in slot_attempts
        if row.get("batch_id") == EXPECTED_INITIAL_BATCH_ID
        and row.get("result_type") == "errored"
    }
    if set(failed_rows) != EXPECTED_FAILED_CUSTOM_IDS:
        raise RuntimeError("sealed credit-error attempt set changed")
    if any(
        row.get("error_type") != EXPECTED_RECOVERED_ERROR_TYPE
        for row in failed_rows.values()
    ):
        raise RuntimeError("credit errors were not recovery-classified")
    failed_slots = {
        (str(row["task_id"]), int(row["sample_index"]))
        for row in failed_rows.values()
    }
    selected = [
        dict(spec)
        for spec in specs
        if (str(spec["task_id"]), int(spec["sample_index"])) in failed_slots
    ]
    if len(selected) != 2:
        raise RuntimeError(f"expected two primary retries, found {len(selected)}")
    for spec in selected:
        if (
            int(spec["cap"]) != batch.CAP_LADDER[0]
            or int(spec["cap_attempt_index"]) != 1
        ):
            raise RuntimeError("selected slot is not a same-cap first retry")
    return selected


def _authorize(argv: Sequence[str]) -> None:
    if _cli_value(argv, "--action") != "submit":
        raise RuntimeError("primary credit recovery permits --action submit only")
    out = Path(_cli_value(argv, "--out")).resolve()
    if out != EXPECTED_OUT:
        raise RuntimeError(f"unexpected retry output directory: {out}")
    if sha256_file(Path(batch.__file__).resolve()) != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("sealed Anthropic batch runner hash changed")

    provenance = load_json(out / "provenance.json", "Sonnet provenance")
    config_sha = str(provenance.get("config_sha256") or "")
    events = batch._batch_events(out, config_sha)
    if batch._active_submission(events) is not None:
        raise RuntimeError("a Sonnet batch is already active")
    harvested = {
        str(row.get("batch_id") or "")
        for row in events
        if row.get("event_type") == "batch_harvested"
    }
    if EXPECTED_INITIAL_BATCH_ID not in harvested:
        raise RuntimeError("initial Sonnet batch has not been fully harvested")

    attempts = batch._slot_attempts(out, config_sha)
    failed_rows = [
        row
        for row in attempts
        if row.get("batch_id") == EXPECTED_INITIAL_BATCH_ID
        and row.get("result_type") == "errored"
    ]
    if {str(row.get("custom_id") or "") for row in failed_rows} != (
        EXPECTED_FAILED_CUSTOM_IDS
    ):
        raise RuntimeError("native failed-slot set differs from authorization")
    authorization = {
        "schema": batch.SCHEMA,
        "event_type": "primary_credit_retries_authorized",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "source_batch_id": EXPECTED_INITIAL_BATCH_ID,
        "retry_script_sha256": sha256_file(Path(__file__).resolve()),
        "sealed_batch_runner_sha256": EXPECTED_RUNNER_SHA256,
        "source_failed_custom_ids": sorted(EXPECTED_FAILED_CUSTOM_IDS),
        "source_failed_rows_sha256": stable_sha256(failed_rows),
        "authorized_request_count": 2,
        "same_logical_slots": True,
        "same_cap": batch.CAP_LADDER[0],
        "successful_slots_reissued": False,
        "length_slots_submitted": False,
    }
    prior = [
        row
        for row in events
        if row.get("event_type") == "primary_credit_retries_authorized"
    ]
    if prior:
        comparable = dict(authorization)
        comparable.pop("recorded_at")
        prior_comparable = dict(prior[-1])
        prior_comparable.pop("recorded_at", None)
        if prior_comparable != comparable:
            raise RuntimeError("foreign primary-retry authorization")
    else:
        JsonlJournal(out / "batch_events.jsonl").append(authorization)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    _authorize(raw)
    original = batch.pending_request_specs
    batch.pending_request_specs = (
        lambda plans, slot_attempts, terminals: _select_primary_retries(
            original, plans, slot_attempts, terminals
        )
    )
    return batch.main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
