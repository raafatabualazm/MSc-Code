from __future__ import annotations

import copy
import sys
from pathlib import Path

PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import authorize_data_inspection_retry as tool
import frontier_passk as runner


def boundary_row() -> dict[str, object]:
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "api_attempt",
        "provider": "qwen",
        "response_received": False,
        "slot_terminal": False,
        "candidate_valid": None,
        "terminal_reason": None,
        "transport_retry": True,
        "retryable_transport": False,
        "fatal_response_contract": False,
        "usage": None,
        "response": None,
        "transport_error": (
            "api_exception:BadRequestError:Error code: 400 - "
            "{'error': {'code': 'data_inspection_failed'}}"
        ),
    }


def test_exact_data_inspection_boundary_accepts_only_exact_shape() -> None:
    assert tool.is_exact_data_inspection_boundary(boundary_row())


def test_data_inspection_boundary_rejects_a_returned_response() -> None:
    row = copy.deepcopy(boundary_row())
    row["response_received"] = True
    assert not tool.is_exact_data_inspection_boundary(row)


def test_data_inspection_boundary_rejects_other_400_code() -> None:
    row = copy.deepcopy(boundary_row())
    row["transport_error"] = (
        "api_exception:BadRequestError:Error code: 400 - "
        "{'error': {'code': 'invalid_parameter'}}"
    )
    assert not tool.is_exact_data_inspection_boundary(row)


def test_data_inspection_boundary_rejects_existing_override() -> None:
    row = copy.deepcopy(boundary_row())
    row["resume_override"] = {"schema": "already-authorized"}
    assert not tool.is_exact_data_inspection_boundary(row)
