import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file
from scripts.training import t5gemma2_enriched_sft as sft


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_heldout_loader_exposes_only_f2_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dev.jsonl"
    f2 = tmp_path / "dev_f2.jsonl"
    seal = tmp_path / "dev.seal.json"
    manifest = tmp_path / "dev_f2.manifest.json"
    task_ids = ["measure-0", "measure-1"]
    dataset_rows = [
        {
            "task_id": task_id,
            "dart_source": f"int fn0() => {index};",
            "acceptance_tests": "SECRET TEST",
        }
        for index, task_id in enumerate(task_ids)
    ]
    f2_rows = []
    for index, task_id in enumerate(task_ids):
        text = f"CONST {index}\nCFG B0 -> RET"
        f2_rows.append(
            {
                "schema": sft.F2_ROW_SCHEMA,
                "representation_schema": sft.REPRESENTATION_SCHEMA,
                "task_id": task_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "verified": dict(sft._REQUIRED_F2_ATTESTATIONS),
            }
        )
    _write_jsonl(dataset, dataset_rows)
    _write_jsonl(f2, f2_rows)
    task_hash = canonical_sha256(task_ids)
    _write_json(
        manifest,
        {
            "schema": "verified-api-readable-compact-v2",
            "rows": 2,
            "dataset": {"sha256": sha256_file(dataset)},
            "output": {"sha256": sha256_file(f2)},
            "task_set_sha256": task_hash,
            "invariants": {"train_dev_representation_contract_identical": True},
        },
    )
    _write_json(
        seal,
        {
            "schema": "compact-public-private-join-seal-v1",
            "heldout_measure_only": True,
            "training_allowed": False,
            "selected_role": "measure",
            "rows": 2,
            "output_sha256": sha256_file(dataset),
            "f2_output": {"sha256": sha256_file(f2)},
            "f2_manifest": {"sha256": sha256_file(manifest)},
            "task_set_sha256": task_hash,
            "completion_attestation_id": "per-run-256-bit-marker-exactly-once-v1",
        },
    )
    monkeypatch.setattr(inference, "HELDOUT_ROWS", 2)
    monkeypatch.setattr(inference, "DATASET_SHA256", sha256_file(dataset))
    monkeypatch.setattr(inference, "DATASET_SEAL_SHA256", sha256_file(seal))
    monkeypatch.setattr(inference, "F2_SHA256", sha256_file(f2))
    monkeypatch.setattr(inference, "F2_MANIFEST_SHA256", sha256_file(manifest))
    monkeypatch.setattr(inference, "TASK_SET_SHA256", task_hash)

    rows, record = inference.load_heldout_rows(
        dataset=dataset,
        dataset_seal=seal,
        f2_jsonl=f2,
        f2_manifest=manifest,
    )

    assert [row.task_id for row in rows] == task_ids
    assert "CONST 0" in rows[0].source
    assert "SECRET TEST" not in rows[0].source
    assert "int fn0" not in rows[0].source
    assert record["model_visible_fields"] == ["F2.text"]
    assert record["tests_serialized_to_model"] is False
    assert record["gold_targets_serialized_to_model"] is False


def _candidate(sample_index: int, text: str) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "seed": sample_index + 42,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "action_tokens": 3,
        "eos_observed": True,
        "max_token_completion": False,
    }


def test_generation_journal_requires_exact_order_and_candidate_hashes() -> None:
    rows = [
        inference.EvaluationRow("t0", "source", hashlib.sha256(b"source").hexdigest())
    ]
    contract = {"schema": inference.INFERENCE_SCHEMA, "arm": "base"}
    terminal = {
        "event": "task_terminal",
        "schema": inference.JOURNAL_SCHEMA,
        "task_index": 0,
        "task_id": "t0",
        "source_sha256": rows[0].source_sha256,
        "candidates": [_candidate(0, "int fn0() => 1;")],
    }
    predictions = [{"id": "t0", "predictions": ["int fn0() => 1;"]}]
    events = [
        {
            "event": "header",
            "schema": inference.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
        terminal,
        {
            "event": "complete",
            "schema": inference.JOURNAL_SCHEMA,
            "rows": 1,
            "predictions_canonical_sha256": canonical_sha256(predictions),
        },
    ]

    terminals, complete = inference._journal_state(
        events, contract=contract, rows=rows, num_samples=1
    )
    assert complete is True
    assert terminals[0]["task_id"] == "t0"

    corrupted = json.loads(json.dumps(events))
    corrupted[1]["candidates"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="candidate"):
        inference._journal_state(corrupted, contract=contract, rows=rows, num_samples=1)


def test_sampling_seed_is_paired_and_coordinate_stable() -> None:
    assert inference.sample_seed(42, 0, 0) == 42
    assert inference.sample_seed(42, 1, 0) == 100_045
    assert inference.sample_seed(42, 1, 3) == 100_048
    with pytest.raises(ValueError):
        inference.sample_seed(-1, 0, 0)
