from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "hybrid_training_patch_v2_3"
for root in (WORKSPACE, PROJECT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analysis_contract_only_control import contract_only_inference as inference
from analysis_contract_only_control import contract_only_report as report
from analysis_contract_only_control import contract_only_view as view
from analysis_contract_only_control import score_contract_only as score_wrapper
from scripts.evaluation import t5gemma2_measurement_audit_inputs as audit_inputs


def _rows(count: int = 3):
    dataset = []
    f2_rows = []
    for index in range(count):
        task_id = f"task-{index}"
        f2_text = audit_inputs.render_f2(
            prefix=(
                f'// strings ["F2_CANARY_{index}"]\n'
                f'// externals[X=canary{index}]:T|["identity"]\n'
            ),
            structure=f"Ax86_64\nEentry\nD\nCANARY_STRUCTURE_{index}\nS\nB2\nX\n",
        )
        dataset.append(
            {
                "task_id": task_id,
                "dart_source": (
                    f"List<int> fn0(String semanticName{index}, int count{index}) "
                    "=> <int>[];"
                ),
                "acceptance_tests": f"TEST_CANARY_{index}",
            }
        )
        f2_rows.append(
            {
                "task_id": task_id,
                "text": f2_text,
                "text_sha256": hashlib.sha256(f2_text.encode()).hexdigest(),
            }
        )
    return dataset, f2_rows


def test_view_is_exact_historical_typed_prompt_with_zero_binary_bytes() -> None:
    dataset, f2_rows = _rows()
    sources, record = view.build_input_view(
        dataset_rows=dataset,
        f2_rows=f2_rows,
    )
    expected_instruction = (
        "Use this exact opaque top-level Dart interface (types and arity only; "
        "parameter names are neutral): List<int> fn0(String p0, int p1).\n"
    )
    for index, source in enumerate(sources):
        assert source.count(expected_instruction) == 1
        assert view._extract_binary_payload(source) == ""
        assert f"F2_CANARY_{index}" not in source
        assert f"CANARY_STRUCTURE_{index}" not in source
        assert f"canary{index}" not in source
        assert f"TEST_CANARY_{index}" not in source
        assert f"semanticName{index}" not in source
        assert f"count{index}" not in source
        assert source.endswith("<enriched_binary>\n\n</enriched_binary>\n")
    summary = record["summary"]
    assert record["view"] == view.VIEW
    assert record["f2_exposed_to_model"] is False
    assert summary["binary_placeholder"] == {
        "text": "",
        "utf8_hex": "",
        "utf8_bytes": 0,
        "sha256": view.EMPTY_SHA256,
        "task_invariant": True,
        "placement": "exactly_between_enriched_binary_open_and_close_tags",
    }
    assert summary["gold_derived_oracle_control"] is True
    assert summary["oracle_caveat"] == view.ORACLE_CAVEAT
    assert summary["out_of_distribution_caveat"] == view.OOD_CAVEAT


def test_view_fails_closed_on_f2_digest_or_identity_change() -> None:
    dataset, f2_rows = _rows()
    f2_rows[0]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sealed F2 text/digest"):
        view.build_input_view(dataset_rows=dataset, f2_rows=f2_rows)
    dataset, f2_rows = _rows()
    f2_rows[1]["task_id"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        view.build_input_view(dataset_rows=dataset, f2_rows=f2_rows)


def test_smoke_dataset_is_exact_first_five_and_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "full.jsonl"
    lines = [
        json.dumps({"task_id": f"task-{index}", "acceptance_tests": "x"})
        .encode()
        + b"\r\n"
        for index in range(6)
    ]
    source.write_bytes(b"".join(lines))
    target = tmp_path / "first5.jsonl"
    record = view.materialize_smoke_dataset(
        source_dataset=source, output=target, rows=5
    )
    assert target.read_bytes() == b"".join(lines[:5])
    assert record["ordered_task_ids"] == [f"task-{index}" for index in range(5)]
    assert view.materialize_smoke_dataset(
        source_dataset=source, output=target, rows=5
    ) == record
    target.write_bytes(b"different\n")
    with pytest.raises(ValueError, match="differs from exact payload"):
        view.materialize_smoke_dataset(source_dataset=source, output=target, rows=5)


def _valid_provenance() -> dict:
    dataset, f2_rows = _rows(1)
    _sources, view_record = view.build_input_view(
        dataset_rows=dataset, f2_rows=f2_rows
    )
    return {
        "schema": score_wrapper.PROVENANCE_SCHEMA,
        "input_view": view.VIEW,
        "output_sha256": "a" * 64,
        "num_samples": 10,
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "gold_interface_types_and_arity_exposed_to_model": True,
        "f2_exposed_to_model": False,
        "recovered_constants_exposed_to_model": False,
        "f2_structure_exposed_to_model": False,
        "external_call_identities_exposed_to_model": False,
        "gold_derived_oracle_control": True,
        "deployable_type_recovery_frontend_evaluated": False,
        "oracle_caveat": view.ORACLE_CAVEAT,
        "out_of_distribution_caveat": view.OOD_CAVEAT,
        "heldout": {
            "input_view": view_record,
            "model_visible_fields": [
                "gold_derived_opaque_types_and_arity",
                "task_invariant_empty_binary_payload",
            ],
            "binary_payload": {
                "text": "",
                "utf8_hex": "",
                "utf8_bytes": 0,
                "sha256": view.EMPTY_SHA256,
                "task_invariant": True,
            },
            "f2_serialized_to_model": False,
            "recovered_constants_serialized_to_model": False,
            "f2_structure_serialized_to_model": False,
            "external_call_identities_serialized_to_model": False,
            "gold_derived_oracle_control": True,
            "deployable_type_recovery_frontend_evaluated": False,
            "oracle_caveat": view.ORACLE_CAVEAT,
            "out_of_distribution_caveat": view.OOD_CAVEAT,
        },
    }


@pytest.mark.parametrize(
    "path,value",
    [
        (("f2_exposed_to_model",), True),
        (("oracle_caveat",), "missing"),
        (("heldout", "f2_serialized_to_model"), True),
        (("heldout", "input_view", "summary", "f2_utf8_bytes_serialized_to_model"), 1),
        (("heldout", "input_view", "summary", "binary_placeholder", "text"), "x"),
    ],
)
def test_scorer_admission_rejects_every_privacy_or_oracle_drift(
    path: tuple[str, ...], value: object
) -> None:
    provenance = _valid_provenance()
    score_wrapper.validate_contract_only_provenance(
        provenance, prediction_sha256="a" * 64, k=10
    )
    cursor = provenance
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError, match="provenance contract failed"):
        score_wrapper.validate_contract_only_provenance(
            provenance, prediction_sha256="a" * 64, k=10
        )


def test_checkpoint_gate_rejects_non_optstep348_before_file_reads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen optstep348"):
        inference._require_original_enriched_checkpoint(
            tmp_path / "typed-checkpoint", {}, {}
        )


def test_preregistered_thresholds_and_caveats_are_machine_checked() -> None:
    prereg = json.loads(
        (WORKSPACE / "analysis_contract_only_control" / "preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    report._validate_preregistration(prereg)
    assert prereg["hypotheses"]["compile_at_10_strong_pattern_interval"] == [160, 172]
    assert prereg["hypotheses"]["pass_at_10_strong_pattern_interval"] == [1, 3]
    assert "does not prove semantic decoding" in prereg["out_of_distribution_caveat"]


def test_actual_175_view_matches_preregistered_hashes_when_staged_data_exists() -> None:
    data = (
        WORKSPACE
        / "transfer_staging_t5gemma2_eval"
        / "workspace"
        / "multifunction_v1"
        / "build"
    )
    dataset_path = data / "dev_multifunction_binary.jsonl"
    f2_path = data / "dev_multifunction_binary_f2.jsonl"
    if not dataset_path.is_file() or not f2_path.is_file():
        pytest.skip("sealed full175 staging data not available")
    dataset = view._read_jsonl(dataset_path, "dataset")
    f2_rows = view._read_jsonl(f2_path, "F2")
    sources, record = view.build_input_view(dataset_rows=dataset, f2_rows=f2_rows)
    prereg = json.loads(
        (WORKSPACE / "analysis_contract_only_control" / "preregistration.json").read_text()
    )["view"]
    assert len(sources) == 175
    assert record["ordered_task_ids_sha256"] == prereg["expected_ordered_task_ids_sha256"]
    assert (
        record["ordered_source_sha256s_sha256"]
        == prereg["expected_ordered_source_sha256s_sha256"]
    )
    assert (
        record["row_transformations_sha256"]
        == prereg["expected_row_transformations_sha256"]
    )
    assert record["summary"]["arity_histogram"] == prereg["expected_arity_histogram"]
    assert all(view._extract_binary_payload(source) == "" for source in sources)


def test_runner_is_review_gated_post_current_and_uses_same_settings() -> None:
    launcher = (
        WORKSPACE
        / "analysis_contract_only_control"
        / "run_contract_only_control.sh"
    ).read_text(encoding="utf-8")
    assert "t5gemma2_4b4b_enriched_sft_2epoch_v1" in launcher
    assert "t5gemma2_4b4b_typed_contract_sft" not in launcher
    assert "T5GEMMA_CONTRACT_ONLY_REVIEWED_BUNDLE_SHA256" in launcher
    assert ".t5gemma2_typed_seed_replication_gpu.lock" in launcher
    assert "flock -w" in launcher
    assert "MIN_FREE_KIB" in launcher and "5242880" in launcher
    assert "MIN_GPU_FREE_MIB" in launcher and "5120" in launcher
    assert "--seed 42 --limit 5" in launcher
    assert 'evaluation_first5.jsonl"' in launcher
    assert "--seed 42 --output" in launcher
    assert "--seed 43 --output" in launcher
    assert "--temperature 0.8 --top_p 0.95" in launcher
    assert "--max_source_tokens 32768 --max_new_tokens 4096" in launcher
    assert "--k 10 --workers 32 --timeout 30 --stability_runs 2" in launcher
    assert "intervention_multiseed_report.json" in launcher
    assert "17645716115052bb48a906a4c7231c76ec28a9d7b66dc55bc53e669e990bee63" in launcher
    assert "89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a" in launcher
    assert "handoff_attestation.py" in launcher
    assert "--mode verify" in launcher
    assert '--handoff_attestation "${HANDOFF_ATTESTATION}"' in launcher
    assert "Rank-0 gold round-trip failed" in launcher
    assert "no_training_or_checkpoint_write" in launcher
    assert "ssh " not in launcher and "nohup" not in launcher


def test_supervisor_never_autostarts() -> None:
    control_config = (
        WORKSPACE
        / "analysis_contract_only_control"
        / "t5gemma2-contract-only-control-v1.conf"
    ).read_text(encoding="utf-8")
    handoff_config = (
        WORKSPACE
        / "analysis_contract_only_control"
        / "t5gemma2-contract-only-handoff-v1.conf"
    ).read_text(encoding="utf-8")
    for config in (control_config, handoff_config):
        assert "autostart=false" in config
        assert "autorestart=false" in config
        assert "stopasgroup=true" in config
        assert "killasgroup=true" in config
        assert "T5GEMMA_CONTRACT_ONLY_REVIEWED_BUNDLE_SHA256" in config
    assert "run_contract_only_control.sh" in control_config
    assert "handoff_after_current.sh" in handoff_config


def test_handoff_requires_exact_exited_and_recomputes_predecessor_report() -> None:
    handoff = (
        WORKSPACE
        / "analysis_contract_only_control"
        / "handoff_after_current.sh"
    ).read_text(encoding="utf-8")
    assert "RUNNING|STARTING|STOPPING" in handoff
    assert "STOPPED|FATAL|BACKOFF|UNKNOWN" in handoff
    assert "exact EXITED is required" in handoff
    assert '"${PYTHON_BIN}" "${REPORTER}"' in handoff
    assert "89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a" in handoff
    assert "17645716115052bb48a906a4c7231c76ec28a9d7b66dc55bc53e669e990bee63" in handoff
    assert "predecessor multiseed report did not reproduce" in handoff
    assert "report_sha_one" in handoff and "report_sha_two" in handoff
    assert "MIN_FREE_KIB" in handoff and "MIN_GPU_FREE_MIB" in handoff
    assert 'start "${DOWNSTREAM_PROGRAM}"' in handoff
    assert "Idempotent duplicate guard" in handoff
    assert "--mode create" in handoff
    assert '--handoff_attestation "${HANDOFF_ATTESTATION}"' in handoff
    assert "could not resume incomplete downstream control" in handoff
