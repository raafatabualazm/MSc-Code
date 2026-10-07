from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V3,
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.training import build_verpo_rescue_transfer as builder


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(builder.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def make_contract(path: Path) -> DirectCompactContract:
    contract = DirectCompactContract(
        schema="direct-compact-causal-v1",
        codec_sha256="a" * 64,
        codebook_sha256="b" * 64,
        tokenizer_json_sha256="c" * 64,
        tokenizer_fingerprint_sha256="d" * 64,
        model_config_sha256="e" * 64,
        decoder_model="test/decoder",
        decoder_revision="immutable-test-revision",
        target_function="fn0",
        target_language="Dart",
        dfg_extractor_sha256="f" * 64,
        lossless_domain="scrubbed_canonical_graph",
        base_vocab_size=4,
        source_token_ids=(4,),
        source_token_expansions=((4, (2,)),),
    )
    path.write_text(
        json.dumps(contract.as_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def base_row(
    contract: DirectCompactContract,
    task_id: str,
    *,
    gold: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "lang": "Dart",
        "function": "fn0",
        "compact_input_ids": [4, 2],
        "compact_codec_sha256": contract.codec_sha256,
        "compact_codebook_sha256": contract.codebook_sha256,
        "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
        "dart_source": gold or f"int fn0(int x) => x + {len(task_id)};",
        "feedback_tests": f"assert(fn0(0) == {len(task_id)});",
    }


def write_fit_fixture(
    root: Path,
    rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, DirectCompactContract]:
    contract_path = root / "contract.json"
    contract = make_contract(contract_path)
    rollout = root / "rollout.jsonl"
    write_jsonl(
        rollout,
        rows
        or [
            base_row(contract, "fit-a"),
            base_row(contract, "fit-b"),
            base_row(contract, "fit-c"),
        ],
    )
    seal = {
        "schema": "compact-public-private-join-seal-v1",
        "selected_role": "fit",
        "training_allowed": True,
        "heldout_measure_only": False,
        "output_sha256": sha256_file(rollout),
        "contract_sha256": sha256_file(contract_path),
        "rows": len(read_jsonl(rollout)),
    }
    seal_path = root / "rollout.seal.json"
    seal_path.write_text(
        json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rollout, seal_path, contract_path, contract


def full_repair(
    task_id: str,
    code: str,
    *,
    arm: str = "diagnosis_only",
    base_candidate_rank: int = 0,
    repair_rank: int = 0,
) -> dict[str, object]:
    return {
        "schema": builder.FULL_REPAIR_SCHEMA,
        "task_id": task_id,
        "code": code,
        "code_sha256": builder.sha256_text(code),
        "target_mode": "final_dart_code_only",
        "reasoning_in_target": False,
        "student_checkpoint_sha256": "1" * 64,
        "source_plan_sha256": "2" * 64,
        "visible_full_pass": True,
        "development_reward_holdback_full_pass": True,
        "development_reward_holdback_tests_sha256": builder.sha256_text(
            f"private-{task_id}"
        ),
        "development_holdback_consumed_for_transfer_selection": True,
        "final_175_holdout_touched": False,
        "provider_saw_development_holdback": False,
        "contributors": [
            {
                "arm": arm,
                "base_candidate_rank": base_candidate_rank,
                "repair_rank": repair_rank,
            }
        ],
    }


def partial_preference(
    task_id: str,
    chosen: str,
    rejected: str,
    *,
    holdback_delta: int = 1,
) -> dict[str, object]:
    return {
        "schema": builder.PARTIAL_PREFERENCE_SCHEMA,
        "task_id": task_id,
        "chosen": chosen,
        "chosen_sha256": builder.sha256_text(chosen),
        "rejected": rejected,
        "rejected_sha256": builder.sha256_text(rejected),
        "chosen_visible_passed_tests": 1,
        "rejected_visible_passed_tests": 0,
        "chosen_holdback_delta_passed_tests": holdback_delta,
        "off_policy": True,
        "different_conditioning_prefixes": True,
        "eligible_for_on_policy_verpo_update": False,
        "kept_separate_from_rs_sft_targets": True,
        "source_plan_sha256": "2" * 64,
    }


def write_score_report(
    path: Path,
    repairs: Path,
    preferences: Path,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": builder.SCORE_REPORT_SCHEMA,
        "status": "complete",
        "source_plan_sha256": "2" * 64,
        "student_checkpoint_sha256": "1" * 64,
        "exports": {
            "rs_sft_rows": len(read_jsonl(repairs)),
            "partial_preference_rows": len(read_jsonl(preferences)),
            "rs_sft_requires_full_visible_and_holdback": True,
            "preference_pairs_are_separate_off_policy": True,
        },
        "privacy": {
            "holdback_test_source_persisted": False,
            "holdback_diagnostic_persisted": False,
            "holdback_exposed_to_provider": False,
            "reference_dart_exposed_to_provider": False,
            "final_175_holdout_touched": False,
            "development_holdback_is_now_consumed_for_transfer_selection": True,
        },
        "export_artifacts": {
            "rs_sft_targets": {
                "path": str(repairs.resolve()),
                "sha256": sha256_file(repairs),
                "size_bytes": repairs.stat().st_size,
            },
            "preference_pairs": {
                "path": str(preferences.resolve()),
                "sha256": sha256_file(preferences),
                "size_bytes": preferences.stat().st_size,
            },
        },
    }
    if mutate is not None:
        mutate(body)
    body[builder.SCORE_REPORT_HASH_FIELD] = builder.canonical_sha256(body)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    return body


def build_transfer(
    rollout: Path,
    seal: Path,
    contract_path: Path,
    repairs: Path,
    output: Path,
    *,
    preferences: Path | None = None,
    min_unique_repairs: int = 400,
    allow_low_coverage_smoke: bool = True,
) -> dict[str, object]:
    if preferences is None:
        preferences = output.parent / f"{output.name}.preferences.jsonl"
        write_jsonl(preferences, [])
    score_report = output.parent / f"{output.name}.score.json"
    write_score_report(score_report, repairs, preferences)
    return builder.build_rescue_transfer(
        rollout,
        seal,
        contract_path,
        repairs,
        score_report,
        preferences,
        output,
        min_unique_repairs=min_unique_repairs,
        allow_low_coverage_smoke=allow_low_coverage_smoke,
    )


def test_builds_exact_matched_arms_and_separate_partial_preferences(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    repair_a = full_repair(
        "fit-a", "int fn0(int x) => x + 101;", repair_rank=1
    )
    alternative_a = full_repair(
        "fit-a",
        "int fn0(int x) => 101 + x;",
        arm="compiler_only",
    )
    repair_b = full_repair("fit-b", "int fn0(int x) => x + 202;")
    repairs = tmp_path / "repairs.jsonl"
    # One byte-identical duplicate is removed; one distinct alternative is
    # resolved by a content-derived stable key, never input order.
    write_jsonl(
        repairs,
        [repair_b, repair_a, copy.deepcopy(repair_a), alternative_a],
    )
    partial_a = partial_preference(
        "fit-a",
        "int fn0(int x) => x + 7;",
        "int fn0(int x) => 0;",
    )
    partial_b_no_private_gain = partial_preference(
        "fit-b",
        "int fn0(int x) => x + 8;",
        "int fn0(int x) => 0;",
        holdback_delta=0,
    )
    preferences = tmp_path / "preferences.jsonl"
    write_jsonl(
        preferences,
        [
            partial_b_no_private_gain,
            partial_a,
            copy.deepcopy(partial_a),
        ],
    )

    output = tmp_path / "output"
    report = build_transfer(
        rollout,
        seal,
        contract_path,
        repairs,
        output,
        preferences=preferences,
    )

    intervention = read_jsonl(output / builder.INTERVENTION_FILENAME)
    control = read_jsonl(output / builder.CONTROL_FILENAME)
    assert len(intervention) == len(control) == 4
    assert [row["task_id"] for row in intervention] == [
        "fit-a",
        "fit-a",
        "fit-b",
        "fit-b",
    ]
    assert [row["task_id"] for row in control] == [
        row["task_id"] for row in intervention
    ]
    gold_by_task = {
        row["task_id"]: row["dart_source"] for row in read_jsonl(rollout)
    }
    for offset in (0, 2):
        task_id = intervention[offset]["task_id"]
        assert intervention[offset]["supervised_target"] != gold_by_task[task_id]
        assert intervention[offset + 1]["supervised_target"] == gold_by_task[task_id]
        assert control[offset]["supervised_target"] == gold_by_task[task_id]
        assert control[offset + 1]["supervised_target"] == gold_by_task[task_id]
    for row in intervention + control:
        assert "supervised_target" in row
        assert "dart_source" not in row
        assert "feedback_tests" not in row
        assert "acceptance_tests" not in row
        assert "reasoning" not in row
        assert "judge" not in row

    validate_join_seal(
        output / builder.INTERVENTION_FILENAME,
        output / builder.INTERVENTION_SEAL_FILENAME,
        contract_path,
        expected_role="fit",
    )
    validate_join_seal(
        output / builder.CONTROL_FILENAME,
        output / builder.CONTROL_SEAL_FILENAME,
        contract_path,
        expected_role="fit",
    )
    intervention_seal = json.loads(
        (output / builder.INTERVENTION_SEAL_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    control_seal = json.loads(
        (output / builder.CONTROL_SEAL_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert intervention_seal["training_allowed"] is False
    assert control_seal["training_allowed"] is False
    preference_rows = read_jsonl(output / builder.PREFERENCE_FILENAME)
    assert len(preference_rows) == 1
    assert preference_rows[0]["task_id"] == "fit-a"
    assert preference_rows[0]["chosen"] == partial_a["chosen"]
    assert "supervised_target" not in preference_rows[0]
    assert preference_rows[0]["off_policy"] is True
    preference_seal = json.loads(
        (output / builder.PREFERENCE_SEAL_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert preference_seal["output_sha256"] == sha256_file(
        output / builder.PREFERENCE_FILENAME
    )
    assert preference_seal["eligible_for_sft"] is False

    assert report["counts"] == {
        "selected_full_repair_tasks": 2,
        "intervention_rows": 4,
        "intervention_repair_rows": 2,
        "intervention_gold_replay_rows": 2,
        "control_gold_rows": 4,
        "partial_preference_rows": 1,
    }
    assert report["full_repairs"]["exact_duplicates_removed"] == 1
    assert (
        report["partial_preferences"]["excluded_without_private_improvement"]
        == 1
    )
    assert report["partial_preferences"]["exact_duplicates_removed"] == 1
    assert report["coverage_gate"] == {
        "minimum_unique_repairs": 400,
        "observed_unique_repairs": 2,
        "production_coverage_met": False,
        "allow_low_coverage_smoke": True,
        "low_coverage_smoke_bypass_used": True,
        "training_use_permitted": False,
    }


def test_full_selection_and_outputs_are_independent_of_input_order(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    candidates = [
        full_repair("fit-a", "int fn0(int x) => x + 11;"),
        full_repair(
            "fit-a",
            "int fn0(int x) => 11 + x;",
            arm="plain_resample",
            repair_rank=2,
        ),
        full_repair("fit-b", "int fn0(int x) => x + 22;"),
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, candidates)
    write_jsonl(second, list(reversed(candidates)))
    build_transfer(
        rollout, seal, contract_path, first, tmp_path / "out-first"
    )
    build_transfer(
        rollout, seal, contract_path, second, tmp_path / "out-second"
    )
    for filename in (
        builder.INTERVENTION_FILENAME,
        builder.CONTROL_FILENAME,
        builder.SCHEDULE_FILENAME,
    ):
        assert (tmp_path / "out-first" / filename).read_bytes() == (
            tmp_path / "out-second" / filename
        ).read_bytes()


def test_gold_identical_repairs_do_not_count_and_non_gold_is_preferred(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    gold = {
        row["task_id"]: row["dart_source"] for row in read_jsonl(rollout)
    }
    genuine = "int fn0(int x) => 500 + x;"
    repairs = tmp_path / "repairs.jsonl"
    write_jsonl(
        repairs,
        [
            full_repair("fit-a", gold["fit-a"]),
            full_repair(
                "fit-a",
                genuine,
                arm="compiler_only",
                repair_rank=1,
            ),
            full_repair("fit-b", gold["fit-b"]),
        ],
    )
    output = tmp_path / "gold-filtered"
    report = build_transfer(
        rollout,
        seal,
        contract_path,
        repairs,
        output,
        min_unique_repairs=1,
        allow_low_coverage_smoke=False,
    )

    intervention = read_jsonl(output / builder.INTERVENTION_FILENAME)
    assert [row["task_id"] for row in intervention] == ["fit-a", "fit-a"]
    assert intervention[0]["supervised_target"] == genuine
    assert intervention[0]["supervised_target"] != gold["fit-a"]
    assert report["coverage_gate"]["observed_unique_repairs"] == 1
    assert report["full_repairs"][
        "gold_identical_input_rows_excluded"
    ] == 2
    assert report["full_repairs"][
        "gold_identical_unique_task_code_excluded"
    ] == 2
    assert report["full_repairs"][
        "tasks_with_only_gold_identical_repairs"
    ] == 1
    assert report["invariants"][
        "all_sft_repairs_byte_differ_from_original_gold"
    ] is True


def test_all_gold_identical_repairs_fail_before_transfer_output(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    gold = {
        row["task_id"]: row["dart_source"] for row in read_jsonl(rollout)
    }
    repairs = tmp_path / "only-gold.jsonl"
    write_jsonl(
        repairs,
        [
            full_repair("fit-a", gold["fit-a"]),
            full_repair("fit-b", gold["fit-b"]),
        ],
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        builder.RescueTransferError,
        match="zero genuine non-gold",
    ):
        build_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("code_sha256", "0" * 64, "code SHA-256 mismatch"),
        ("visible_full_pass", False, "full visible/private"),
        (
            "development_reward_holdback_full_pass",
            False,
            "full visible/private",
        ),
        ("final_175_holdout_touched", True, "full visible/private"),
        ("reasoning_in_target", True, "full visible/private"),
    ],
)
def test_full_repair_attestations_and_hashes_fail_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    row = full_repair("fit-a", "int fn0(int x) => x;")
    row[field] = value
    repairs = tmp_path / "repairs.jsonl"
    write_jsonl(repairs, [row])
    with pytest.raises(builder.RescueTransferError, match=message):
        build_transfer(
            rollout, seal, contract_path, repairs, tmp_path / "output"
        )


def test_unknown_tasks_extra_reasoning_and_conflicting_duplicates_fail_closed(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    unknown = full_repair("measure-175", "int fn0(int x) => x;")
    repairs = tmp_path / "unknown.jsonl"
    write_jsonl(repairs, [unknown])
    with pytest.raises(builder.RescueTransferError, match="outside the sealed fit"):
        build_transfer(
            rollout, seal, contract_path, repairs, tmp_path / "unknown-out"
        )

    extra = full_repair("fit-a", "int fn0(int x) => x;")
    extra["reasoning"] = "private chain of thought"
    write_jsonl(tmp_path / "extra.jsonl", [extra])
    with pytest.raises(builder.RescueTransferError, match="field set differs"):
        build_transfer(
            rollout,
            seal,
            contract_path,
            tmp_path / "extra.jsonl",
            tmp_path / "extra-out",
        )

    first = full_repair("fit-a", "int fn0(int x) => x;")
    conflicting = copy.deepcopy(first)
    conflicting["contributors"] = [
        {
            "arm": "compiler_only",
            "base_candidate_rank": 0,
            "repair_rank": 0,
        }
    ]
    write_jsonl(tmp_path / "conflict.jsonl", [first, conflicting])
    with pytest.raises(
        builder.RescueTransferError, match="conflicting provenance"
    ):
        build_transfer(
            rollout,
            seal,
            contract_path,
            tmp_path / "conflict.jsonl",
            tmp_path / "conflict-out",
        )


def test_partial_schema_hashes_and_off_policy_flags_fail_closed(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    repairs = tmp_path / "repairs.jsonl"
    write_jsonl(
        repairs,
        [full_repair("fit-a", "int fn0(int x) => x + 1;")],
    )
    bad = partial_preference(
        "fit-b",
        "int fn0(int x) => x + 2;",
        "int fn0(int x) => 0;",
    )
    bad["eligible_for_on_policy_verpo_update"] = True
    preferences = tmp_path / "preferences.jsonl"
    write_jsonl(preferences, [bad])
    with pytest.raises(builder.RescueTransferError, match="off-policy separation"):
        build_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            tmp_path / "output",
            preferences=preferences,
        )

    bad = partial_preference(
        "fit-b",
        "int fn0(int x) => x + 2;",
        "int fn0(int x) => 0;",
    )
    bad["chosen_sha256"] = "0" * 64
    write_jsonl(tmp_path / "bad-hash.jsonl", [bad])
    with pytest.raises(builder.RescueTransferError, match="chosen SHA-256 mismatch"):
        build_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            tmp_path / "hash-output",
            preferences=tmp_path / "bad-hash.jsonl",
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        {"hidden_tests": "secret"},
        {"chain_of_thought": "private reasoning"},
        {"private_metadata": {"oracle": "secret"}},
    ],
)
def test_sealed_rollout_oracle_and_reasoning_fields_fail_closed(
    tmp_path: Path, forbidden: dict[str, object]
) -> None:
    contract_path = tmp_path / "contract.json"
    contract = make_contract(contract_path)
    row = base_row(contract, "fit-a")
    row.update(forbidden)
    rollout, seal, contract_path, _contract = write_fit_fixture(
        tmp_path, [row]
    )
    repairs = tmp_path / "repairs.jsonl"
    write_jsonl(
        repairs,
        [full_repair("fit-a", "int fn0(int x) => x + 1;")],
    )
    with pytest.raises(
        builder.RescueTransferError, match="forbidden oracle/reasoning"
    ):
        build_transfer(
            rollout, seal, contract_path, repairs, tmp_path / "output"
        )


def test_zero_full_targets_and_tampered_join_fail_closed(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(builder.RescueTransferError, match="zero rows"):
        build_transfer(
            rollout, seal, contract_path, empty, tmp_path / "empty-output"
        )

    repairs = tmp_path / "repairs.jsonl"
    write_jsonl(
        repairs,
        [full_repair("fit-a", "int fn0(int x) => x + 1;")],
    )
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(builder.canonical_json(read_jsonl(rollout)[0]) + "\n")
    with pytest.raises(ValueError, match="sealed dataset SHA-256 mismatch"):
        build_transfer(
            rollout, seal, contract_path, repairs, tmp_path / "tampered-output"
        )


def test_score_report_self_digest_and_exact_export_binding_fail_closed(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    repairs = tmp_path / "repairs.jsonl"
    preferences = tmp_path / "preferences.jsonl"
    write_jsonl(
        repairs,
        [full_repair("fit-a", "int fn0(int x) => x + 1;")],
    )
    write_jsonl(preferences, [])

    tampered_report = tmp_path / "tampered-score.json"
    report = write_score_report(tampered_report, repairs, preferences)
    report["status"] = "incomplete"
    tampered_report.write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(
        builder.RescueTransferError, match="self-digest mismatch"
    ):
        builder.build_rescue_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            tampered_report,
            preferences,
            tmp_path / "tampered-output",
            allow_low_coverage_smoke=True,
        )

    wrong_export_report = tmp_path / "wrong-export-score.json"

    def change_export(body: dict[str, object]) -> None:
        exports = body["export_artifacts"]
        assert isinstance(exports, dict)
        target = exports["rs_sft_targets"]
        assert isinstance(target, dict)
        target["sha256"] = "0" * 64

    write_score_report(
        wrong_export_report,
        repairs,
        preferences,
        mutate=change_export,
    )
    with pytest.raises(
        builder.RescueTransferError, match="exact scorer export"
    ):
        builder.build_rescue_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            wrong_export_report,
            preferences,
            tmp_path / "wrong-export-output",
            allow_low_coverage_smoke=True,
        )


def test_score_report_source_plan_and_checkpoint_must_match_rows(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    repairs = tmp_path / "repairs.jsonl"
    preferences = tmp_path / "preferences.jsonl"
    write_jsonl(
        repairs,
        [full_repair("fit-a", "int fn0(int x) => x + 1;")],
    )
    write_jsonl(preferences, [])
    score_report = tmp_path / "score.json"
    write_score_report(
        score_report,
        repairs,
        preferences,
        mutate=lambda body: body.update(
            {
                "source_plan_sha256": "3" * 64,
                "student_checkpoint_sha256": "4" * 64,
            }
        ),
    )
    with pytest.raises(
        builder.RescueTransferError,
        match="source-plan/checkpoint bindings differ",
    ):
        builder.build_rescue_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            score_report,
            preferences,
            tmp_path / "output",
            allow_low_coverage_smoke=True,
        )


def test_production_coverage_gate_and_smoke_only_join_seals(
    tmp_path: Path,
) -> None:
    rollout, seal, contract_path, _contract = write_fit_fixture(tmp_path)
    repairs = tmp_path / "repairs.jsonl"
    preferences = tmp_path / "preferences.jsonl"
    write_jsonl(
        repairs,
        [full_repair("fit-a", "int fn0(int x) => x + 1;")],
    )
    write_jsonl(preferences, [])
    score_report = tmp_path / "score.json"
    write_score_report(score_report, repairs, preferences)

    with pytest.raises(
        builder.RescueTransferError, match="production minimum is 400"
    ):
        builder.build_rescue_transfer(
            rollout,
            seal,
            contract_path,
            repairs,
            score_report,
            preferences,
            tmp_path / "blocked",
        )

    smoke_report = build_transfer(
        rollout,
        seal,
        contract_path,
        repairs,
        tmp_path / "smoke",
    )
    smoke_seal = json.loads(
        (
            tmp_path
            / "smoke"
            / builder.INTERVENTION_SEAL_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert smoke_report["coverage_gate"]["training_use_permitted"] is False
    assert smoke_seal["training_allowed"] is False

    production_report = build_transfer(
        rollout,
        seal,
        contract_path,
        repairs,
        tmp_path / "production",
        min_unique_repairs=1,
        allow_low_coverage_smoke=False,
    )
    production_seal = json.loads(
        (
            tmp_path
            / "production"
            / builder.INTERVENTION_SEAL_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert production_report["coverage_gate"]["training_use_permitted"] is True
    assert production_seal["training_allowed"] is True


def test_v3_fails_explicitly_before_deriving_unsafe_pool_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    repairs = tmp_path / "repairs.jsonl"
    repairs.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        builder.DirectCompactContract,
        "load",
        lambda _path: SimpleNamespace(schema=CONTRACT_SCHEMA_V3),
    )
    with pytest.raises(
        builder.RescueTransferError,
        match="cannot safely derive selected-task pool-use aggregates",
    ):
        build_transfer(
            rollout, seal, contract, repairs, tmp_path / "output"
        )
