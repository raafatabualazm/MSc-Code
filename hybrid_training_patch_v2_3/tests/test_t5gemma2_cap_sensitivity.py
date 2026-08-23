import hashlib
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import t5gemma2_cap_sensitivity as sensitivity
from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import canonical_sha256


def _candidate(
    sample_index: int,
    text: str,
    *,
    capped: bool = False,
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "seed": 42,
        "batch_position": sample_index,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "action_tokens": 4096 if capped else 3,
        "eos_observed": not capped,
        "max_token_completion": capped,
    }


def _source_fixture() -> tuple[
    list[inference.EvaluationRow],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    rows = [
        inference.EvaluationRow(
            f"task-{index}",
            f"source-{index}",
            hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        )
        for index in range(2)
    ]
    model = {"name": inference.MODEL_NAME, "arm": "sft", "adapter": {"x": 1}}
    heldout = {"selected_rows": 2, "model_visible_fields": ["F2.text"]}
    contract: dict[str, object] = {
        "schema": inference.INFERENCE_SCHEMA,
        "script_sha256": "a" * 64,
        "arm": "sft",
        "model": model,
        "heldout": heldout,
        "sampling": {
            "num_samples": 10,
            "generation_batch_size": 10,
            "max_source_tokens": 32768,
            "max_new_tokens": 4096,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "seed": 42,
            "seed_policy": "seed+task_index*100003+batch_start",
            "decoder_prefix_is_not_output": True,
            "sampled_eos_retained": True,
            "fabricated_eos": False,
        },
        "runtime": {
            "torch": "test",
            "cuda": "test",
            "bf16": True,
            "attn_implementation": "sdpa",
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "targets_exposed_to_model": False,
        "source_truncation": False,
    }
    terminals: list[dict[str, object]] = []
    capped_slots = {(0, 2), (0, 8), (1, 4)}
    for task_index, row in enumerate(rows):
        terminals.append(
            {
                "event": "task_terminal",
                "schema": inference.JOURNAL_SCHEMA,
                "task_index": task_index,
                "task_id": row.task_id,
                "source_sha256": row.source_sha256,
                "candidates": [
                    _candidate(
                        sample_index,
                        f"old-{task_index}-{sample_index}",
                        capped=(task_index, sample_index) in capped_slots,
                    )
                    for sample_index in range(10)
                ],
            }
        )
    predictions = [
        {
            "id": terminal["task_id"],
            "predictions": [
                candidate["text"] for candidate in terminal["candidates"]
            ],
        }
        for terminal in terminals
    ]
    events: list[dict[str, object]] = [
        {
            "event": "header",
            "schema": inference.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
        *terminals,
        {
            "event": "complete",
            "schema": inference.JOURNAL_SCHEMA,
            "rows": 2,
            "predictions_canonical_sha256": canonical_sha256(predictions),
        },
    ]
    return rows, model, heldout, events


def test_plan_selects_only_capped_slots_and_preserves_original_batches() -> None:
    rows, model, heldout, events = _source_fixture()

    contract, terminals, batches, selected = sensitivity.build_replay_plan(
        source_events=events,
        rows=rows,
        expected_arm="sft",
        expected_model=model,
        expected_heldout=heldout,
        target_max_new_tokens=8192,
        expected_capped=3,
    )

    assert contract["sampling"]["max_new_tokens"] == 4096
    assert len(terminals) == 2
    assert [(row["task_index"], row["sample_index"]) for row in selected] == [
        (0, 2),
        (0, 8),
        (1, 4),
    ]
    assert batches == [
        sensitivity.ReplayBatch(0, "task-0", 0, 10, (2, 8)),
        sensitivity.ReplayBatch(1, "task-1", 0, 10, (4,)),
    ]

    with pytest.raises(ValueError, match="capped-slot count"):
        sensitivity.build_replay_plan(
            source_events=events,
            rows=rows,
            expected_arm="sft",
            expected_model=model,
            expected_heldout=heldout,
            target_max_new_tokens=8192,
            expected_capped=23,
        )


def _extended(
    source_candidate: dict[str, object],
    sample_index: int,
) -> dict[str, object]:
    text = str(source_candidate["text"]) + "-extended"
    return {
        "sample_index": sample_index,
        "seed": 42,
        "batch_position": sample_index,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "action_tokens": 5000,
        "eos_observed": True,
        "max_token_completion": False,
        "prefix_action_tokens": 4096,
        "prefix_text_sha256": source_candidate["text_sha256"],
        "source_text_sha256": source_candidate["text_sha256"],
        "source_prefix_verified": True,
    }


def test_sensitivity_journal_merges_only_selected_slots_and_checks_prefix() -> None:
    rows, model, heldout, source_events = _source_fixture()
    _, source_terminals, batches, _ = sensitivity.build_replay_plan(
        source_events=source_events,
        rows=rows,
        expected_arm="sft",
        expected_model=model,
        expected_heldout=heldout,
        target_max_new_tokens=8192,
        expected_capped=3,
    )
    contract = {"schema": sensitivity.RUN_SCHEMA, "selected_slots": 3}
    batch_terminals: list[dict[str, object]] = []
    for schedule_index, batch in enumerate(batches):
        source_terminal = source_terminals[batch.task_index]
        batch_terminals.append(
            {
                "event": "batch_terminal",
                "schema": sensitivity.JOURNAL_SCHEMA,
                "schedule_index": schedule_index,
                "task_index": batch.task_index,
                "task_id": batch.task_id,
                "batch_start": batch.batch_start,
                "batch_count": batch.batch_count,
                "source_sha256": source_terminal["source_sha256"],
                "candidates": [
                    _extended(
                        source_terminal["candidates"][sample_index],
                        sample_index,
                    )
                    for sample_index in batch.selected_sample_indices
                ],
            }
        )
    merged = sensitivity.merge_predictions(
        source_terminals=source_terminals,
        sensitivity_terminals=batch_terminals,
    )
    events: list[dict[str, object]] = [
        {
            "event": "header",
            "schema": sensitivity.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
        *batch_terminals,
        {
            "event": "complete",
            "schema": sensitivity.JOURNAL_SCHEMA,
            "replayed_batches": 2,
            "selected_slots": 3,
            "predictions_canonical_sha256": canonical_sha256(merged),
        },
    ]

    terminals, complete, merged_hash = sensitivity.sensitivity_journal_state(
        events,
        contract=contract,
        batches=batches,
        source_terminals=source_terminals,
        source_cap=4096,
    )

    assert complete is True
    assert merged_hash == canonical_sha256(merged)
    assert merged[0]["predictions"][2] == "old-0-2-extended"
    assert merged[0]["predictions"][8] == "old-0-8-extended"
    assert merged[0]["predictions"][3] == "old-0-3"
    assert merged[1]["predictions"][4] == "old-1-4-extended"
    assert sum(len(terminal["candidates"]) for terminal in terminals) == 3

    corrupted = [dict(event) for event in events]
    corrupted[1] = dict(corrupted[1])
    corrupted[1]["candidates"] = [
        dict(candidate) for candidate in corrupted[1]["candidates"]
    ]
    corrupted[1]["candidates"][0]["prefix_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate"):
        sensitivity.sensitivity_journal_state(
            corrupted,
            contract=contract,
            batches=batches,
            source_terminals=source_terminals,
            source_cap=4096,
        )
