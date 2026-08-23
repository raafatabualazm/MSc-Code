from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import t5gemma2_enriched_sft as base
from scripts.training import t5gemma2_typed_contract_sft as typed


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _f2(task_id: str, text: str) -> dict[str, object]:
    return {
        "schema": base.F2_ROW_SCHEMA,
        "representation_schema": base.REPRESENTATION_SCHEMA,
        "task_id": task_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "verified": dict(base._REQUIRED_F2_ATTESTATIONS),
    }


def test_contract_handles_real_parser_edges_without_semantic_names() -> None:
    cases = {
        "String fn0(start, end) => '$start:$end';": (
            "String fn0(dynamic p0, dynamic p1)",
            ["dynamic", "dynamic"],
        ),
        "int fn0(List<int> rolls, [bool doubleNext = false]) => 0;": (
            "int fn0(List<int> p0)",
            ["List<int>"],
        ),
        "Set<int> fn0(List<int> samples, [int index = 0]) => {};": (
            "Set<int> fn0(List<int> p0)",
            ["List<int>"],
        ),
    }
    for source, (expected, parameter_types) in cases.items():
        signature, record = typed.opaque_contract_signature(source)
        assert signature == expected
        assert record["parameter_types"] == parameter_types
        assert record["semantic_parameter_names_exposed"] is False
    with pytest.raises(ValueError, match="non-semantic scalar"):
        typed.opaque_contract_signature(
            "String fn0([String topic = 'semantic hint']) => topic;"
        )


def test_typed_join_pins_holdout_and_never_serializes_it(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    f2_path = tmp_path / "train_f2.jsonl"
    heldout_path = tmp_path / "dev.jsonl"
    train = [
        {
            "task_id": "train-0",
            "dart_source": "int fn0(int count) => count + 1;",
            "acceptance_tests": "TRAIN_TEST_0",
        },
        {
            "task_id": "train-1",
            "dart_source": "bool fn0(String value) => value.isEmpty;",
            "acceptance_tests": "TRAIN_TEST_1",
        },
    ]
    f2 = [_f2("train-0", "F2-A"), _f2("train-1", "F2-B")]
    heldout = [
        {
            "task_id": "dev-0",
            "dart_source": "Set<int> fn0(List<String> secret) => <int>{};",
            "acceptance_tests": "NEVER_VISIBLE",
        }
    ]
    _write_jsonl(train_path, train)
    _write_jsonl(f2_path, f2)
    _write_jsonl(heldout_path, heldout)
    pairs, manifest = typed.load_typed_text_pairs(
        train_path,
        f2_path,
        expected_dataset_sha256=base.sha256_file(train_path),
        expected_f2_sha256=base.sha256_file(f2_path),
        expected_rows=2,
        heldout_path=heldout_path,
        expected_heldout_sha256=base.sha256_file(heldout_path),
        expected_heldout_rows=1,
    )
    assert "int fn0(int p0)" in pairs[0].source
    assert "count" not in pairs[0].source
    assert "bool fn0(String p0)" in pairs[1].source
    assert manifest["schema"] == typed.RUN_SCHEMA
    assert manifest["model_visible_fields"] == ["opaque_typed_contract", "F2.text"]
    assert manifest["heldout"]["model_visible"] is False
    assert "NEVER_VISIBLE" not in json.dumps(manifest)

    contaminated = [dict(train[0])]
    _write_jsonl(heldout_path, contaminated)
    with pytest.raises(ValueError, match="task-ID overlap"):
        typed.load_typed_text_pairs(
            train_path,
            f2_path,
            expected_dataset_sha256=base.sha256_file(train_path),
            expected_f2_sha256=base.sha256_file(f2_path),
            expected_rows=2,
            heldout_path=heldout_path,
            expected_heldout_sha256=base.sha256_file(heldout_path),
            expected_heldout_rows=1,
        )


def test_semantic_duplicate_must_be_explicitly_excluded(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    f2_path = tmp_path / "train_f2.jsonl"
    heldout_path = tmp_path / "dev.jsonl"
    train = [
        {
            "task_id": "train-duplicate",
            "dart_source": "void fn0() => print('a');",
            "acceptance_tests": "SAME_ACCEPTANCE_TEST",
        },
        {
            "task_id": "train-clean",
            "dart_source": "int fn0() => 1;",
            "acceptance_tests": "TRAIN_ONLY_TEST",
        },
    ]
    heldout = [
        {
            "task_id": "dev-alias",
            "dart_source": "void fn0() { print('a'); }",
            "acceptance_tests": "SAME_ACCEPTANCE_TEST",
        }
    ]
    _write_jsonl(train_path, train)
    _write_jsonl(f2_path, [_f2("train-duplicate", "F2-D"), _f2("train-clean", "F2-C")])
    _write_jsonl(heldout_path, heldout)
    kwargs = {
        "expected_dataset_sha256": base.sha256_file(train_path),
        "expected_f2_sha256": base.sha256_file(f2_path),
        "expected_rows": 2,
        "heldout_path": heldout_path,
        "expected_heldout_sha256": base.sha256_file(heldout_path),
        "expected_heldout_rows": 1,
    }
    with pytest.raises(ValueError, match="acceptance-test overlap"):
        typed.load_typed_text_pairs(train_path, f2_path, **kwargs)
    pairs, manifest = typed.load_typed_text_pairs(
        train_path,
        f2_path,
        exclude_train_task_ids=["train-duplicate"],
        **kwargs,
    )
    assert [pair.task_id for pair in pairs] == ["train-clean"]
    assert manifest["input_rows"] == 2
    assert manifest["rows"] == 1
    assert manifest["training_exclusions"]["task_ids"] == ["train-duplicate"]


def test_all_synced_2776_gold_sources_have_a_contract_when_present() -> None:
    dataset = (
        PATCH_ROOT.parent
        / "transfer_staging_t5gemma2"
        / "workspace"
        / "multifunction_v1"
        / "expanded2776"
        / "build"
        / "train_multifunction_binary_expanded_2776.jsonl"
    )
    if not dataset.is_file():
        pytest.skip("synced 2,776-row training dataset is absent")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    contracts = [
        typed.opaque_contract_signature(row["dart_source"])[0] for row in rows
    ]
    assert len(contracts) == 2776
    assert all(" fn0(" in contract for contract in contracts)


def test_launcher_is_fresh_base_two_epoch_and_distinct() -> None:
    launcher = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2_typed_contract_sft.sh"
    ).read_text(encoding="utf-8")
    config = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2-typed-contract-sft-2epoch.conf"
    ).read_text(encoding="utf-8")
    assert "t5gemma2_typed_contract_sft.py" in launcher
    assert "--expected_rows 2776" in launcher
    assert "--exclude_train_task_id sigless_6b1dd0c6b6fc" in launcher
    assert "--expected_heldout_rows 175" in launcher
    assert "--epochs 2" in launcher
    assert "--lora_rank 64" in launcher
    assert "--lora_alpha 128" in launcher
    assert "T5GEMMA_TYPED_SFT_FRESH_BASE" in launcher
    assert "--resume_checkpoint" in launcher
    assert "enriched_sft_2epoch_v1" not in launcher
    assert "t5gemma2_4b4b_typed_contract_sft_2epoch_v1" in config
    assert "[program:t5gemma-typed-contract-sft-2epoch]" in config
