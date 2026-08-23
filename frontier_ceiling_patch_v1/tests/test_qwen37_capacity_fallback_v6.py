from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import qwen37_capacity_fallback_v6 as fallback


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sealed_test_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    eval_path = tmp_path / "eval.jsonl"
    tasks: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for index in range(fallback.EXPECTED_TASKS):
        task_id = f"sigless_{index:012x}"
        tests = f"\nFuture<void> main() async {{ /* {index} */ }}\n"
        tests_sha = fallback.runner.sha256_text(tests)
        tasks.append(
            {
                "task_id": task_id,
                "tests_sha256": tests_sha,
                "acceptance_tests_sha256": tests_sha,
                "tests_equal_acceptance_tests": True,
            }
        )
        eval_rows.append(
            {
                "task_id": task_id,
                "tests": tests,
                "acceptance_tests": tests,
            }
        )
    _write_jsonl(source / "tasks.jsonl", tasks)
    _write_jsonl(eval_path, eval_rows)
    (source / "provenance.json").write_text(
        json.dumps(
            {
                "config": {
                    "sealed_inputs": {
                        "eval_jsonl": str(eval_path),
                        "eval_jsonl_sha256": fallback.sha256_file(eval_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return source, eval_path


def test_partition_map_is_disjoint_and_covers_all_primary_indices() -> None:
    indices = [
        value
        for shard in fallback.SOURCE_SHARDS
        for value in shard.global_indices
    ]
    assert sorted(indices) == list(range(10))
    assert len(indices) == len(set(indices))
    assert fallback.PARTITIONS["0520"]["indices"] == (0, 1, 2, 3, 4)
    assert fallback.PARTITIONS["0608"]["indices"] == (5, 6, 7, 8, 9)
    assert fallback.PARTITIONS["0520"]["diagnostic_reuse_indices"] == (
        0,
        1,
        2,
    )
    assert fallback.PARTITIONS["0608"]["diagnostic_reuse_indices"] == (5, 6)


def test_alias_order_is_pair_deterministic_and_exact() -> None:
    assert fallback.PARTITIONS["0520"]["aliases"] == (
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-preview",
        "qwen3.7-max",
    )
    assert fallback.PARTITIONS["0608"]["aliases"] == (
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-preview",
        "qwen3.7-max",
    )


@pytest.mark.parametrize(
    "text",
    [
        (
            "api_exception:PermissionDeniedError:Error code: 403 - "
            "{'error': {'message': 'The free quota has been exhausted.', "
            "'code': 'AllocationQuota.FreeTierOnly'}}"
        ),
        (
            "api_exception:PermissionDeniedError:Error code: 403 - "
            "{'error': {'message': 'The free quota has been exhausted.', "
            "'code': 'insufficient_quota'}}"
        ),
    ],
)
def test_exact_quota_boundary_accepts_only_sealed_codes(text: str) -> None:
    assert fallback.exact_quota_403(text)


@pytest.mark.parametrize(
    "text",
    [
        "Error code: 429 - rate limit",
        "Error code: 403 - forbidden",
        "Error code: 400 - data_inspection_failed",
        "AllocationQuota.FreeTierOnly",
    ],
)
def test_non_quota_errors_do_not_select_capacity_targets(text: str) -> None:
    assert not fallback.exact_quota_403(text)


def test_data_inspection_is_exact_and_separate_from_quota() -> None:
    text = "BadRequestError: Error code: 400 - data_inspection_failed"
    assert fallback.data_inspection_failed(text)
    assert not fallback.exact_quota_403(text)


def test_selection_id_binds_arm_slot_and_source_config() -> None:
    common = {
        "shard_key": "base_0517_k3",
        "task_id": "sigless_a",
        "global_index": 0,
        "source_config_sha256": "a" * 64,
    }
    opus = fallback.selection_id(arm="opus", **common)
    codex = fallback.selection_id(arm="codex", **common)
    assert opus != codex
    assert len(opus) == 64
    changed = fallback.selection_id(
        arm="opus", **{**common, "global_index": 1}
    )
    assert changed != opus


def test_diagnostic_local_mapping_is_one_to_one() -> None:
    assert [
        fallback._diagnostic_local_index("0520", value)
        for value in (0, 1, 2)
    ] == [0, 1, 2]
    assert [
        fallback._diagnostic_local_index("0608", value)
        for value in (5, 6)
    ] == [0, 1]


def test_load_tests_uses_hash_sealed_eval_jsonl(tmp_path: Path) -> None:
    source, _ = _sealed_test_source(tmp_path)
    task_id = "sigless_000000000011"
    tests = fallback._load_tests(source, task_id)
    assert tests == "\nFuture<void> main() async { /* 17 */ }\n"
    assert "acceptance_tests" not in json.loads(
        (source / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )


def test_load_tests_rejects_eval_file_sha_change(tmp_path: Path) -> None:
    source, eval_path = _sealed_test_source(tmp_path)
    with eval_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(fallback.AuditError, match="eval JSONL SHA mismatch"):
        fallback._load_tests(source, "sigless_000000000000")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tests_equal_acceptance_tests", False),
        ("tests_sha256", "0" * 64),
        ("acceptance_tests_sha256", "1" * 64),
    ],
)
def test_load_tests_rejects_task_hash_binding_change(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source, _ = _sealed_test_source(tmp_path)
    rows = [
        json.loads(line)
        for line in (source / "tasks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[0][field] = value
    _write_jsonl(source / "tasks.jsonl", rows)
    with pytest.raises(fallback.AuditError, match="test-hash binding mismatch"):
        fallback._load_tests(source, "sigless_000000000000")


def test_load_tests_rejects_eval_test_inequality(tmp_path: Path) -> None:
    source, eval_path = _sealed_test_source(tmp_path)
    rows = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["acceptance_tests"] = "\nvoid main() {}\n"
    _write_jsonl(eval_path, rows)
    provenance = json.loads(
        (source / "provenance.json").read_text(encoding="utf-8")
    )
    provenance["config"]["sealed_inputs"][
        "eval_jsonl_sha256"
    ] = fallback.sha256_file(eval_path)
    (source / "provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    with pytest.raises(
        fallback.AuditError,
        match="tests/acceptance_tests differ",
    ):
        fallback._load_tests(source, "sigless_000000000000")


@pytest.mark.parametrize(
    "changed_flag",
    [
        "allow_preview",
        "include_undated_alias",
        "include_source_alias",
    ],
)
def test_capacity_epoch_rejects_alias_flag_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_flag: str,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    env_file = tmp_path / "Qwen.env"
    env_file.write_text("sealed placeholder\n", encoding="utf-8")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1/")
    flags = {
        "allow_preview": False,
        "include_undated_alias": False,
        "include_source_alias": False,
        "only_undated_alias": False,
    }
    fallback.register_capacity_epoch(
        out,
        capacity_epoch="epoch-1",
        credential_source="secondary_qwen_env",
        credential_env_file=env_file,
        **flags,
    )
    flags[changed_flag] = True
    with pytest.raises(
        fallback.AuditError,
        match="rebound to different endpoint semantics",
    ):
        fallback.register_capacity_epoch(
            out,
            capacity_epoch="epoch-1",
            credential_source="secondary_qwen_env",
            credential_env_file=env_file,
            **flags,
        )


def test_current_generic_route_is_undated_only() -> None:
    aliases = fallback._enabled_route_aliases(
        partition="0520",
        allow_preview=False,
        include_undated_alias=True,
        include_source_alias=False,
        only_undated_alias=True,
    )
    assert aliases == ["qwen3.7-max"]


def test_current_preview_then_0517_routes_are_deterministic() -> None:
    preview = fallback._enabled_route_aliases(
        partition="0520",
        allow_preview=True,
        include_undated_alias=False,
        include_source_alias=False,
        only_undated_alias=False,
    )
    assert preview == [
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-preview",
    ]
    source = fallback._enabled_route_aliases(
        partition="0520",
        allow_preview=True,
        include_undated_alias=False,
        include_source_alias=True,
        only_undated_alias=False,
    )
    assert source == [
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-preview",
        "qwen3.7-max-2026-05-17",
    ]


def test_fallback_route_enables_all_five_aliases() -> None:
    aliases = fallback._enabled_route_aliases(
        partition="0608",
        allow_preview=True,
        include_undated_alias=True,
        include_source_alias=True,
        only_undated_alias=False,
    )
    assert aliases == [
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-preview",
        "qwen3.7-max",
        "qwen3.7-max-2026-05-17",
    ]


def test_all_dated_aliases_exactly_unavailable_stops_epoch() -> None:
    aliases = [
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
    ]
    assert fallback._all_enabled_aliases_exactly_unavailable(
        aliases,
        set(aliases),
    )


def test_diagnostic_wait_with_usable_alias_does_not_stop_epoch() -> None:
    aliases = [
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
    ]
    assert not fallback._all_enabled_aliases_exactly_unavailable(
        aliases,
        {"qwen3.7-max-2026-05-20"},
    )


def test_new_capacity_epochs_do_not_modify_completed_effective_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    effective_path = out / "effective_terminals.jsonl"
    immutable = (
        '{"selection_id":"sealed-slot","response_received":true,'
        '"slot_terminal":true}\n'
    ).encode()
    effective_path.write_bytes(immutable)
    env_file = tmp_path / "Qwen.env"
    env_file.write_text("sealed placeholder\n", encoding="utf-8")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1/")
    stages = [
        ("dated", False, False, False, False),
        ("preview", True, False, False, False),
        ("source", True, False, True, False),
        ("generic", False, True, False, True),
        ("fallback", True, True, True, False),
    ]
    for name, preview, undated, source, undated_only in stages:
        fallback.register_capacity_epoch(
            out,
            capacity_epoch=f"epoch-{name}",
            credential_source="secondary_qwen_env",
            credential_env_file=env_file,
            allow_preview=preview,
            include_undated_alias=undated,
            include_source_alias=source,
            only_undated_alias=undated_only,
        )
        assert effective_path.read_bytes() == immutable
