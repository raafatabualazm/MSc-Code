from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import score_direct_compact_passk as scoring
from scripts.evaluation import t5gemma2_measurement_audit_inputs as inputs
from scripts.evaluation import t5gemma2_measurement_audit_report as report


def _f2(prefix: str, structure: str) -> str:
    return inputs.render_f2(prefix=prefix, structure=structure)


def _rows(count: int = 5):
    dataset = []
    f2 = []
    for index in range(count):
        task_id = f"task-{index}"
        text = _f2(
            f'// strings ["literal-{index}"]\n'
            f'// externals[X=index]:T|["runtime-{index}"]\n',
            "Ax86_64\nE一\nD\n一mov:A,A\nS\nB2\n"
            + "一" * (index + 1)
            + "\nX\n",
        )
        dataset.append(
            {
                "task_id": task_id,
                "dart_source": f"List<int> fn0(String semantic{index}, int count{index}) => <int>[];",
                "acceptance_tests": "SECRET",
            }
        )
        f2.append(
            {
                "task_id": task_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return dataset, f2


def test_f2_partition_uses_utf8_byte_length_and_round_trips() -> None:
    prefix = '// strings ["é"]\n'
    structure = "Ax86_64\nE一\nD\nS\nB2\n一\nX\n"
    text = _f2(prefix, structure)
    parsed = inputs.parse_f2(text)
    assert parsed.prefix == prefix
    assert parsed.prefix_bytes == len(prefix.encode("utf-8"))
    assert parsed.structure == structure


def test_semantic_body_swap_is_a_balanced_length_sorted_derangement() -> None:
    dataset, f2 = _rows(5)
    sources, record = inputs.build_input_view(
        dataset_rows=dataset,
        f2_rows=f2,
        view="semantic_body_swap",
    )
    originals = [inputs.parse_f2(row["text"]) for row in f2]
    transformed = [
        inputs.parse_f2(
            source.split("<enriched_binary>\n", 1)[1].split(
                "\n</enriched_binary>", 1
            )[0]
        )
        for source in sources
    ]
    assert [row.prefix for row in transformed] == [row.prefix for row in originals]
    assert sorted(row.structure_sha256 for row in transformed) == sorted(
        row.structure_sha256 for row in originals
    )
    assert all(
        left.structure_sha256 != right.structure_sha256
        for left, right in zip(originals, transformed, strict=True)
    )
    assert record["summary"]["exact_derangement"] is True
    assert record["summary"]["median_absolute_structure_byte_delta"] >= 0
    assert record["summary"]["maximum_absolute_structure_byte_delta"] >= 0


def test_constants_stripped_preserves_externals_and_structure_exactly() -> None:
    dataset, f2 = _rows(3)
    sources, record = inputs.build_input_view(
        dataset_rows=dataset,
        f2_rows=f2,
        view="constants_stripped",
    )
    for source, original_row in zip(sources, f2, strict=True):
        transformed_text = source.split("<enriched_binary>\n", 1)[1].split(
            "\n</enriched_binary>", 1
        )[0]
        transformed = inputs.parse_f2(transformed_text)
        original = inputs.parse_f2(original_row["text"])
        assert transformed.structure == original.structure
        assert "// strings" not in transformed.prefix
        assert "// externals" in transformed.prefix
    assert record["summary"]["changed_rows"] == 3
    assert record["summary"]["external_call_identities_preserved"] is True


def test_typed_opaque_contract_has_types_arity_and_only_p_names() -> None:
    signature, record = inputs.opaque_contract_signature(
        "Map<String, int> fn0(List<String> semanticNames, int factor) => {};"
    )
    assert signature == "Map<String, int> fn0(List<String> p0, int p1)"
    assert record["parameter_types"] == ["List<String>", "int"]
    assert record["arity"] == 2
    assert "semanticNames" not in signature
    assert "factor" not in signature
    optional_signature, optional_record = inputs.opaque_contract_signature(
        "int fn0([int semantic = 1]) => 1;"
    )
    assert optional_signature == "int fn0()"
    assert optional_record["omitted_optional_parameter_count"] == 1
    dataset, f2 = _rows(3)
    _, view_record = inputs.build_input_view(
        dataset_rows=dataset,
        f2_rows=f2,
        view="typed_opaque_contract",
    )
    assert view_record["summary"]["arity_histogram"] == {"2": 3}
    assert view_record["summary"]["return_type_histogram"] == {
        "List<int>": 3
    }
    eval_sources, _ = inputs.build_input_view(
        dataset_rows=dataset[:1],
        f2_rows=f2[:1],
        view="typed_opaque_contract",
    )
    historical_instruction = (
        "Use this exact opaque top-level Dart interface (types and arity only; "
        "parameter names are neutral): List<int> fn0(String p0, int p1).\n"
    )
    assert eval_sources[0].count(historical_instruction) == 1
    assert "Implement a top-level fn0 callable through" not in eval_sources[0]


def test_measurement_provenance_is_admitted_only_with_full_privacy_contract() -> None:
    provenance = {
        "schema": "t5gemma2-f2-measurement-ablation-provenance-v1",
        "input_view": "constants_stripped",
        "heldout": {
            "input_view": {
                "schema": "t5gemma2-f2-measurement-input-view-v1",
                "view": "constants_stripped",
            }
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "output_sha256": "a" * 64,
        "num_samples": 10,
    }
    scoring.validate_prediction_provenance(
        provenance, prediction_sha256="a" * 64, k=10
    )
    corrupted = json.loads(json.dumps(provenance))
    corrupted["tests_exposed_to_model"] = True
    with pytest.raises(ValueError, match="contract failed"):
        scoring.validate_prediction_provenance(
            corrupted, prediction_sha256="a" * 64, k=10
        )


def test_full_launcher_pins_all_replicates_and_views() -> None:
    launcher = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2_measurement_audit.sh"
    ).read_text(encoding="utf-8")
    assert "for seed in 43 44 45 46" in launcher
    assert "semantic_body_swap constants_stripped typed_opaque_contract" in launcher
    assert "two_epoch_k10_predictions.json" in launcher
    assert "--max_source_tokens 32768" in launcher
    assert "--max_new_tokens 4096" in launcher
    assert "t5gemma-measurement-baseline-reseeds" in launcher
    assert "baseline_state" in launcher
    assert "gold_roundtrip/gold_k1_score.json" in launcher


def test_synced_real_seed42_artifact_is_report_compatible_when_present() -> None:
    workspace = PATCH_ROOT.parent
    predictions = (
        workspace
        / "artifacts"
        / "t5gemma2_sft_epoch_ablation_passk_v1"
        / "two_epoch_k10_predictions.json"
    )
    score = predictions.with_name("two_epoch_k10_score.json")
    if not predictions.is_file() or not score.is_file():
        pytest.skip("synced seed-42 artifact is not part of this checkout")
    arm = report._load_arm(
        label="real_seed42",
        predictions_path=predictions,
        score_path=score,
        expected_tasks=175,
        expected_k=10,
        expected_provenance_schema=report.BASE_PROVENANCE_SCHEMA,
    )
    assert len(arm["task_ids"]) == 175
    assert report._candidate_metrics(arm["score"])["candidates"] == 1750
