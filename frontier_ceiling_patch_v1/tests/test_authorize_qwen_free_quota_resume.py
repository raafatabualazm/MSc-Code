from __future__ import annotations

import copy
import sys
from pathlib import Path

PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import authorize_qwen_free_quota_resume as tool
import frontier_passk as runner


MODEL = "qwen3.7-max-2026-05-17"
FREE_MESSAGE = (
    "The free quota has been exhausted. To continue accessing the model on a "
    "paid basis, please complete your payment information （or disable the "
    '"use free tier only" mode in the management console if already completed).'
)


def boundary_row(variant: str = "AllocationQuota.FreeTierOnly") -> dict[str, object]:
    if variant == "AllocationQuota.FreeTierOnly":
        payload = (
            "{'error': {'message': " + repr(FREE_MESSAGE)
            + ", 'type': 'AllocationQuota.FreeTierOnly', 'param': None, "
            "'code': 'AllocationQuota.FreeTierOnly'}, "
            "'id': 'chatcmpl-123', 'request_id': '123'}"
        )
    elif variant == "insufficient_quota":
        payload = (
            "{'error': {'message': "
            + repr(
                FREE_MESSAGE.replace("（", "(")
                .replace("）", ")")
                .removesuffix(".")
            )
            + ", 'id': '123', 'type': 'insufficient_quota', "
            "'code': 'insufficient_quota'}}"
        )
    else:
        raise AssertionError(variant)
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "api_attempt",
        "provider": "qwen",
        "requested_model": MODEL,
        "response_received": False,
        "slot_terminal": False,
        "candidate_valid": None,
        "terminal_reason": None,
        "transport_retry": True,
        "retryable_transport": False,
        "fatal_response_contract": False,
        "usage": None,
        "response": None,
        "transport_error": tool.ERROR_PREFIX + payload,
    }


def test_accepts_both_exact_observed_http_403_variants() -> None:
    allocation = boundary_row()
    insufficient = boundary_row("insufficient_quota")
    assert tool.qwen_free_quota_variant(allocation) == (
        "AllocationQuota.FreeTierOnly"
    )
    assert tool.qwen_free_quota_variant(insufficient) == "insufficient_quota"


def test_rejects_429_insufficient_quota_rate_limit() -> None:
    row = boundary_row("insufficient_quota")
    row["transport_error"] = str(row["transport_error"]).replace(
        "Error code: 403", "Error code: 429"
    )
    assert not tool.is_exact_qwen_free_quota_boundary(row)


def test_rejects_generic_insufficient_quota_message() -> None:
    row = boundary_row("insufficient_quota")
    row["transport_error"] = str(row["transport_error"]).replace(
        tool.FREE_QUOTA_SENTENCE,
        "You exceeded your current quota.",
    )
    assert not tool.is_exact_qwen_free_quota_boundary(row)


def test_rejects_a_returned_provider_response() -> None:
    row = boundary_row()
    row["response_received"] = True
    assert not tool.is_exact_qwen_free_quota_boundary(row)


def test_rejects_foreign_model_and_provider() -> None:
    foreign_model = boundary_row()
    foreign_model["requested_model"] = "deepseek-v4-pro"
    assert not tool.is_exact_qwen_free_quota_boundary(foreign_model)
    foreign_provider = boundary_row()
    foreign_provider["provider"] = "deepseek"
    assert not tool.is_exact_qwen_free_quota_boundary(foreign_provider)


def test_rejects_payload_shape_drift() -> None:
    row = boundary_row()
    row["transport_error"] = str(row["transport_error"]).replace(
        "'request_id': '123'",
        "'request_id': '123', 'extra': 'unexpected'",
    )
    assert not tool.is_exact_qwen_free_quota_boundary(row)


def test_rejects_an_already_authorized_row() -> None:
    row = copy.deepcopy(boundary_row())
    row["resume_override"] = {"schema": tool.OVERRIDE_SCHEMA}
    assert not tool.is_exact_qwen_free_quota_boundary(row)


def test_exact_run_allowlist_excludes_moderation_rejected_opus_shards() -> None:
    assert (
        "qwen37_clean_v4_0520_opus_k3_mc12k_tol10_tb8k"
        not in tool.ALLOWED_RUN_MODELS
    )
    assert (
        "qwen37_clean_v4_0608_opus_k2_mc12k_tol10_tb8k"
        not in tool.ALLOWED_RUN_MODELS
    )
    assert (
        tool.ALLOWED_RUN_MODELS[
            "qwen37_clean_v4_0520_codex_k3_mc12k_tol10_tb8k"
        ]
        == "qwen3.7-max-2026-05-20"
    )
