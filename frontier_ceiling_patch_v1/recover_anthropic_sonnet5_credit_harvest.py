#!/usr/bin/env python3
"""Resume the sealed Sonnet harvest past two provider credit-cap errors.

The native Batch result remains immutable.  This companion only reclassifies
the two exact, unbilled provider-capacity errors as retryable transport errors
so the other 348 successful responses can be harvested.  Its authorization is
written into the original batch event journal before the recovery runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import frontier_passk_anthropic_batch as batch
from frontier_core import JsonlJournal, load_json, load_jsonl, sha256_file, utc_now


EXPECTED_RUNNER_SHA256 = (
    "86aa896c2f7c97d90006e5c69ff77ed61432df9d33780d2e950343112833c5d8"
)
EXPECTED_OUT = Path(
    "/workspace/artifacts/frontier_ceiling_two_enrichments/runs/"
    "anthropic_sonnet5_batch_screen_k2_warm_v1/opus"
)
EXPECTED_BATCH_ID = "msgbatch_016QhZR7xz4bGPbNkqir5Pju"
EXPECTED_ERROR_CUSTOM_IDS = {
    "a00_s00_t075_c08192",
    "a00_s01_t065_c08192",
}
EXPECTED_NATIVE_ERROR_TYPE = "invalid_request_error"
EXPECTED_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)
RECOVERED_ERROR_TYPE = "provider_credit_balance_exhausted_retryable"


def _cli_value(argv: Sequence[str], flag: str) -> str:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    raise RuntimeError(f"{flag} is required")


def _native_error(payload: Any) -> tuple[str, str]:
    value = payload if isinstance(payload, Mapping) else {}
    nested = value.get("error") if isinstance(value.get("error"), Mapping) else value
    return str(nested.get("type") or ""), str(nested.get("message") or "")


def recover_result_payload(
    raw_result: Mapping[str, Any],
    original: Any,
) -> tuple[str, Any]:
    result_type, payload = original(raw_result)
    custom_id = str(raw_result.get("custom_id") or "")
    error_type, message = _native_error(payload)
    if custom_id in EXPECTED_ERROR_CUSTOM_IDS:
        if (
            result_type != "errored"
            or error_type != EXPECTED_NATIVE_ERROR_TYPE
            or message != EXPECTED_MESSAGE
        ):
            raise RuntimeError(
                f"sealed credit-error result changed for {custom_id}"
            )
        return (
            "errored",
            {
                "type": RECOVERED_ERROR_TYPE,
                "message": message,
                "native_error_type": error_type,
            },
        )
    return result_type, payload


def _authorize(argv: Sequence[str]) -> None:
    if _cli_value(argv, "--action") != "harvest":
        raise RuntimeError("credit recovery permits --action harvest only")
    out = Path(_cli_value(argv, "--out")).resolve()
    if out != EXPECTED_OUT:
        raise RuntimeError(f"unexpected recovery output directory: {out}")
    runner_path = Path(batch.__file__).resolve()
    if sha256_file(runner_path) != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("sealed Anthropic batch runner hash changed")

    provenance = load_json(out / "provenance.json", "Sonnet provenance")
    config_sha = str(provenance.get("config_sha256") or "")
    if len(config_sha) != 64:
        raise RuntimeError("missing sealed Sonnet config hash")
    events = batch._batch_events(out, config_sha)
    active = batch._active_submission(events)
    if active is None or str(active.get("batch_id")) != EXPECTED_BATCH_ID:
        raise RuntimeError("expected Sonnet batch is not active for harvest")

    results_path = out / f"batch_results_{EXPECTED_BATCH_ID}.jsonl"
    results = load_jsonl(results_path, "native Sonnet batch results")
    observed_errors: dict[str, tuple[str, str, str]] = {}
    for row in results:
        result_type, payload = batch._result_payload(row)
        if result_type == "succeeded":
            continue
        error_type, message = _native_error(payload)
        observed_errors[str(row.get("custom_id") or "")] = (
            result_type,
            error_type,
            message,
        )
    expected_error = (
        "errored",
        EXPECTED_NATIVE_ERROR_TYPE,
        EXPECTED_MESSAGE,
    )
    if set(observed_errors) != EXPECTED_ERROR_CUSTOM_IDS or any(
        value != expected_error for value in observed_errors.values()
    ):
        raise RuntimeError("native Batch error set differs from sealed recovery set")

    authorization = {
        "schema": batch.SCHEMA,
        "event_type": "credit_error_harvest_recovery_authorized",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": EXPECTED_BATCH_ID,
        "recovery_script_sha256": sha256_file(Path(__file__).resolve()),
        "sealed_batch_runner_sha256": EXPECTED_RUNNER_SHA256,
        "native_results_sha256": sha256_file(results_path),
        "native_error_custom_ids": sorted(EXPECTED_ERROR_CUSTOM_IDS),
        "native_error_type": EXPECTED_NATIVE_ERROR_TYPE,
        "recovered_error_type": RECOVERED_ERROR_TYPE,
        "native_results_modified": False,
        "successful_slots_reissued": False,
    }
    prior = [
        row
        for row in events
        if row.get("event_type")
        == "credit_error_harvest_recovery_authorized"
    ]
    if prior:
        comparable = dict(authorization)
        comparable.pop("recorded_at")
        prior_comparable = dict(prior[-1])
        prior_comparable.pop("recorded_at", None)
        if prior_comparable != comparable:
            raise RuntimeError("foreign credit-harvest recovery authorization")
    else:
        JsonlJournal(out / "batch_events.jsonl").append(authorization)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    _authorize(raw)
    original = batch._result_payload
    batch._result_payload = lambda row: recover_result_payload(row, original)
    return batch.main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
