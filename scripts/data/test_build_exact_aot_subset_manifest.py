from __future__ import annotations

import copy

import pytest

from scripts.data.build_exact_aot_subset_manifest import (
    AOT_ROW_SCHEMA,
    SubsetManifestError,
    build_projection,
)


def _aot(task_id: str, row: int) -> dict:
    return {
        "schema": AOT_ROW_SCHEMA,
        "task_id": task_id,
        "split": "train",
        "split_row": row,
        "aot_path": f"aot/train/{row}.aot",
        "aot_sha256": f"{row + 1:064x}",
        "aot_size_bytes": 10,
        "analysis_program_sha256": "a" * 64,
        "function_source_sha256": "b" * 64,
        "producer": {"script_sha256": "c" * 64},
    }


def test_projection_preserves_rows_and_separates_model_membership() -> None:
    full = [_aot("unused", 0), _aot("dev-a", 1), _aot("train-a", 2)]
    original = copy.deepcopy(full)
    selected, membership = build_projection(
        full_rows=full,
        train_rows=[{"task_id": "train-a"}],
        dev_rows=[{"task_id": "dev-a"}],
        expected_train_rows=1,
        expected_dev_rows=1,
    )
    assert full == original
    assert selected == [original[2], original[1]]
    assert membership[0]["model_role"] == "train"
    assert membership[1]["model_role"] == "dev"
    # The held-out row truthfully retains its upstream pool split.
    assert selected[1]["split"] == "train"
    assert membership[1]["source_split"] == "train"


def test_projection_rejects_overlap() -> None:
    with pytest.raises(SubsetManifestError, match="train_dev_task_overlap"):
        build_projection(
            full_rows=[_aot("same", 0)],
            train_rows=[{"task_id": "same"}],
            dev_rows=[{"task_id": "same"}],
            expected_train_rows=1,
            expected_dev_rows=1,
        )


def test_projection_rejects_missing_task() -> None:
    with pytest.raises(
        SubsetManifestError, match="tasks_missing_from_full_manifest"
    ):
        build_projection(
            full_rows=[_aot("train-a", 0)],
            train_rows=[{"task_id": "train-a"}],
            dev_rows=[{"task_id": "dev-a"}],
            expected_train_rows=1,
            expected_dev_rows=1,
        )


def test_projection_rejects_duplicate_source_positions() -> None:
    first = _aot("train-a", 0)
    second = _aot("dev-a", 0)
    with pytest.raises(
        SubsetManifestError,
        match="duplicate_full_manifest_split_position",
    ):
        build_projection(
            full_rows=[first, second],
            train_rows=[{"task_id": "train-a"}],
            dev_rows=[{"task_id": "dev-a"}],
            expected_train_rows=1,
            expected_dev_rows=1,
        )
