from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import t5gemma2_semantic_rescue_ablation as rescue


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _detail(passed: int, total: int = 2) -> dict[str, Any]:
    passes = [index < passed for index in range(total)]
    diagnostic = "" if passed == total else f"passed {passed}/{total}"
    return {
        "compiled": True,
        "full_pass": passed == total,
        "test_passes": passes,
        "passed_tests": passed,
        "total_tests": total,
        "diagnostic": diagnostic,
        "diagnostic_sha256": rescue.sha256_text(diagnostic),
    }


def _group() -> dict[str, Any]:
    source = "F2\nC0\n"
    parents = []
    for index, code in enumerate(("PARENT_ZERO", "PARENT_ONE")):
        parents.append(
            {
                "base_candidate_index": index,
                "candidate": code,
                "candidate_sha256": rescue.sha256_text(code),
                "visible_detail": _detail(0),
            }
        )
    return {
        "task_id": "task-0",
        "encoder_source": source,
        "encoder_source_sha256": rescue.sha256_text(source),
        "visible_tests": "VISIBLE_TESTS",
        "split_binding_sha256": SHA_B,
        "parents": parents,
    }


def _diagnoses(*, reject_second: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(2):
        accepted = not (reject_second and index == 1)
        rows.append(
            {
                "group_index": index,
                "accepted": accepted,
                "rejection_reasons": [] if accepted else ["ungrounded_evidence"],
                "fault_class": "wrong_constant" if accepted else "unknown",
                "edit_location": {"operation": "return"},
                "evidence": [],
                "explanation": "The visible result uses the wrong constant.",
                "repair_steps": ["Replace only the return constant."],
            }
        )
    return {
        "schema": rescue.DIAGNOSIS_SCHEMA,
        "status": "complete",
        "source_plan_sha256": SHA_A,
        "rows": [{"task_id": "task-0", "result": {"diagnoses": rows}}],
    }


def test_local_harvest_projection_is_an_explicit_privacy_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {"arm": "sft", "adapter_weights_sha256": SHA_A}
    task_ids_sha = rescue.canonical_sha256(["task-0"])
    contract = {
        "schema": rescue.LOCAL_RUN_SCHEMA,
        "checkpoint": checkpoint,
        "sampling": {
            "base_samples": rescue.BASE_DRAWS,
            "max_source_tokens": 32768,
            "max_new_tokens": 4096,
        },
        "schedule": {"task_ids_sha256": task_ids_sha},
    }
    actual_journal = {
        "sha256": SHA_A,
        "chain_head_sha256": SHA_B,
        "event_count": 3,
        "head_event_sha256": SHA_C,
    }
    report = {
        "schema": "t5gemma2-local-rs-sft-pilot-report-v1",
        "status": "complete",
        "checkpoint": checkpoint,
        "journal": dict(actual_journal),
    }
    candidates = []
    safe_generation_keys = {
        "action_tokens",
        "batch_position",
        "encoder_tokens",
        "eos_observed",
        "group_sample_index",
        "max_token_completion",
        "seed",
        "text_sha256",
    }
    for index in range(rescue.BASE_DRAWS):
        code = f"candidate-{index}"
        candidates.append(
            {
                "origin": "base",
                "sample_index": index,
                "code": code,
                "code_sha256": rescue.sha256_text(code),
                "generation": {
                    "seed": index,
                    "action_tokens": 3,
                    "eos_observed": True,
                    "max_token_completion": False,
                    "batch_position": index,
                    "encoder_tokens": 11,
                    "group_sample_index": index,
                    "text_sha256": rescue.sha256_text(code),
                    # A projection advertised as whitelist-only must not copy
                    # future/debug receipt fields merely because their key is
                    # not on today's private-key denylist.
                    "provider_debug_trace": "SECRET_DEBUG_BYTES",
                },
                "private_gate_results": {"passed": True},
                "selected_target": "SECRET_TARGET",
                "gold_code": "SECRET_GOLD",
            }
        )
    events = [
        {
            "event": "header",
            "schema": rescue.LOCAL_JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": rescue.canonical_sha256(contract),
        },
        {
            "event": "task_terminal",
            "schema": rescue.LOCAL_JOURNAL_SCHEMA,
            "task_position": 0,
            "task_id": "task-0",
            "source_sha256": SHA_A,
            "split_binding_sha256": SHA_B,
            "journal_event_sha256": SHA_C,
            "base_candidates": candidates,
            "private_gate_results": {"passed": True},
            "reward_holdback_tests": "SECRET_HOLDBACK",
        },
        {
            "event": "complete",
            "schema": rescue.LOCAL_JOURNAL_SCHEMA,
            "tasks": 1,
            "terminal_task_ids_sha256": task_ids_sha,
        },
    ]
    monkeypatch.setattr(rescue, "_read_json", lambda *_args: report)
    monkeypatch.setattr(rescue, "load_journal", lambda _path: events)
    monkeypatch.setattr(rescue, "journal_record", lambda _path: actual_journal)
    monkeypatch.setattr(
        rescue,
        "_file_record",
        lambda _path, **_kwargs: {"name": "report.json", "sha256": SHA_A},
    )

    _checkpoint, rows, _source = rescue._validate_local_source(
        Path("harvest.jsonl"), Path("report.json")
    )

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "task_id",
        "source_sha256",
        "split_binding_sha256",
        "source_terminal_event_sha256",
        "base_candidates",
    }
    for candidate in row["base_candidates"]:
        assert set(candidate) == {
            "sample_index",
            "code",
            "code_sha256",
            "generation",
        }
        assert set(candidate["generation"]) == safe_generation_keys
    serialized = json.dumps(rows, sort_keys=True)
    assert "SECRET" not in serialized


def test_materialized_schedule_is_three_arms_four_slots_and_arm_independent_seed() -> None:
    plan = {"groups": [_group()]}
    slots = rescue._materialize_slots(
        plan, SHA_A, _diagnoses(), seed=9102026
    )

    assert len(slots) == 12
    assert Counter(slot["arm"] for slot in slots) == {
        "plain_resample": 4,
        "compiler_only": 4,
        "semantic_judge": 4,
    }
    coordinates: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for slot in slots:
        coordinates[(slot["parent_rank"], slot["repair_rank"])][
            slot["arm"]
        ] = slot["seed"]
    assert len(coordinates) == 4
    assert all(
        set(by_arm) == set(rescue.ARM_ORDER)
        and len(set(by_arm.values())) == 1
        for by_arm in coordinates.values()
    )
    assert len({slot["slot_id"] for slot in slots}) == 12


def test_rejected_diagnosis_reserves_two_slots_and_makes_no_generation_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = {
        "schema": rescue.PLAN_SCHEMA,
        "status": "complete",
        "checkpoint": {"arm": "sft", "adapter_weights_sha256": SHA_A},
        "groups": [_group()],
    }
    diagnoses = _diagnoses(reject_second=True)
    generated_seeds: list[int] = []
    appended: list[dict[str, Any]] = []
    completed = False

    def fake_read(_path: str | Path, label: str) -> dict[str, Any]:
        return plan if label == "semantic rescue plan" else diagnoses

    def fake_append(_path: Path, event: Mapping[str, Any]) -> None:
        nonlocal completed
        row = dict(event)
        row["journal_event_sha256"] = rescue.canonical_sha256(row)
        appended.append(row)
        completed = row.get("event") == "complete" or completed

    def fake_state(
        _path: Path, *, contract: Mapping[str, Any], slots: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        del contract, slots
        terminals = [row for row in appended if row.get("event") == "slot_terminal"]
        return {
            "events": list(appended),
            "terminals": terminals,
            "complete": completed,
        }

    def fake_generator(**_kwargs: Any):
        def one(_source: str, seed: int) -> dict[str, Any]:
            generated_seeds.append(seed)
            return {
                "text": "GENERATED",
                "seed": seed,
                "action_tokens": 1,
                "eos_observed": True,
                "max_token_completion": False,
            }

        return one

    monkeypatch.setattr(rescue.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rescue, "_read_json", fake_read)
    monkeypatch.setattr(
        rescue,
        "_require_seal",
        lambda _value, **kwargs: SHA_A
        if kwargs["label"] == "rescue plan"
        else SHA_B,
    )
    monkeypatch.setattr(
        rescue,
        "_checkpoint_record",
        lambda *_args: ({"checkpoint": True}, plan["checkpoint"]),
    )
    monkeypatch.setattr(
        rescue,
        "_runtime_provenance",
        lambda: {"code_bundle_sha256": SHA_C},
    )
    monkeypatch.setattr(
        rescue,
        "_generation_journal_state",
        fake_state,
    )
    monkeypatch.setattr(
        rescue,
        "load_policy",
        lambda **_kwargs: (object(), object(), plan["checkpoint"]),
    )
    monkeypatch.setattr(
        rescue,
        "_preflight_slot_sources",
        lambda *_args, **_kwargs: {
            "slot_sources_checked": 12,
            "unique_sources_checked": 5,
            "max_observed_source_tokens": 8,
            "source_token_counts_sha256": SHA_A,
            "truncation_used": False,
        },
    )
    monkeypatch.setattr(rescue, "append_event", fake_append)
    monkeypatch.setattr(rescue, "_cached_runtime_generator", fake_generator)
    monkeypatch.setattr(
        rescue,
        "_build_generation_artifact",
        lambda **_kwargs: {"schema": rescue.GENERATION_SCHEMA, "status": "complete"},
    )
    monkeypatch.setattr(rescue, "require_exact_or_write", lambda *_args: None)

    args = argparse.Namespace(
        plan="plan.json",
        diagnoses="diagnoses.json",
        sft_checkpoint=str(tmp_path / "checkpoint"),
        output=str(tmp_path / "generation.json"),
        journal=str(tmp_path / "generation.journal.jsonl"),
        seed=77,
        max_source_tokens=32768,
        max_new_tokens=4096,
        temperature=0.8,
        top_p=0.95,
        attn_implementation="sdpa",
        bf16=True,
    )
    rescue.generate(args)

    terminals = [row for row in appended if row.get("event") == "slot_terminal"]
    assert len(terminals) == 12
    assert len(generated_seeds) == 10
    rejected = [row for row in terminals if row["status"] == "diagnosis_rejected"]
    assert len(rejected) == 2
    assert all(row["candidate"] is None for row in rejected)
    expected = rescue._materialize_slots(
        plan, SHA_A, diagnoses, seed=args.seed
    )
    assert generated_seeds == [slot["seed"] for slot in expected if slot["generate"]]


def test_score_replays_visible_artifact_before_holdback_and_uses_common_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    group = _group()
    plan_body = {
        "schema": rescue.PLAN_SCHEMA,
        "status": "complete",
        "checkpoint": {"arm": "sft", "adapter_weights_sha256": SHA_A},
        "groups": [group],
    }
    plan = rescue._seal(plan_body, rescue.PLAN_HASH)
    plan_sha = plan[rescue.PLAN_HASH]

    visible_passes: dict[str, int] = {}
    private_passes = {
        "PARENT_ZERO": 0,
        "PARENT_ONE": 1,
        "plain_resample_WINNER": 2,
        "compiler_only_WINNER": 1,
        "semantic_judge_WINNER": 2,
    }
    records: list[dict[str, Any]] = []
    for arm in rescue.ARM_ORDER:
        for index in range(rescue.SLOTS_PER_ARM):
            code = f"{arm}_{'WINNER' if index == 0 else f'loser_{index}'}"
            visible_passes[code] = 2 if index == 0 else 0
            records.append(
                {
                    "slot_id": f"task-0:{arm}:p{index // 2}:r{index % 2}",
                    "task_id": "task-0",
                    "arm": arm,
                    "parent_rank": index // 2,
                    "repair_rank": index % 2,
                    "status": "generated",
                    "candidate": {"text": code},
                }
            )
    generation_body = {
        "schema": rescue.GENERATION_SCHEMA,
        "status": "complete",
        "source_plan_sha256": plan_sha,
        "records": records,
    }
    generation = rescue._seal(generation_body, rescue.GENERATION_HASH)
    plan_path = tmp_path / "plan.json"
    generation_path = tmp_path / "generation.json"
    visible_path = tmp_path / "visible.json"
    score_path = tmp_path / "score.json"
    holdback_path = tmp_path / "holdback.private.jsonl"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    holdback_path.write_text(
        json.dumps(
            {
                "task_id": "task-0",
                "feedback_tests": "VISIBLE_TESTS",
                "reward_holdback_tests": "PRIVATE_TESTS",
                "visible_count": 2,
                "holdback_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected_holdback_sha = rescue.sha256_file(holdback_path)

    monkeypatch.setattr(
        rescue,
        "_runtime_provenance",
        lambda: {
            "code": {"native_scorer": {"sha256": SHA_C}},
            "code_bundle_sha256": SHA_A,
        },
    )

    def score_visible(
        jobs: list[tuple[str, str, str]], **_kwargs: Any
    ) -> list[dict[str, Any]]:
        return [_detail(visible_passes[code]) for code, _tests, _identity in jobs]

    monkeypatch.setattr(rescue, "_score_many", score_visible)
    monkeypatch.setattr(
        rescue,
        "_visible_detail",
        lambda candidate, _tests, _identity, **_kwargs: _detail(
            private_passes[candidate]
        ),
    )
    monkeypatch.setattr(rescue, "_split_binding", lambda *_args: SHA_B)
    monkeypatch.setattr(
        rescue, "extract_expect_spans", lambda _tests: ["case-0", "case-1"]
    )

    order: list[str] = []
    original_replay = rescue._visible_artifact_records
    original_sha256_file = rescue.sha256_file

    def replay_then_record(*args: Any, **kwargs: Any):
        result = original_replay(*args, **kwargs)
        order.append("visible_replayed")
        return result

    def guarded_sha(path: str | Path) -> str:
        if Path(path).resolve() == holdback_path.resolve():
            assert visible_path.is_file()
            order.append("holdback_hashed")
        return original_sha256_file(path)

    monkeypatch.setattr(rescue, "_visible_artifact_records", replay_then_record)
    monkeypatch.setattr(rescue, "sha256_file", guarded_sha)

    result = rescue.score(
        argparse.Namespace(
            plan=str(plan_path),
            generation=str(generation_path),
            private_holdback=str(holdback_path),
            expected_private_holdback_sha256=expected_holdback_sha,
            visible_output=str(visible_path),
            output=str(score_path),
            reward_timeout=30,
            stability_runs=2,
            workers=1,
        )
    )

    assert order[:2] == ["visible_replayed", "holdback_hashed"]
    by_arm = {row["arm"]: row for row in result["task_arm_results"]}
    assert {
        row["selected"]["common_baseline_holdback"]["passed_tests"]
        for row in by_arm.values()
    } == {1}
    assert by_arm["plain_resample"]["genuine_rescue"] is True
    assert by_arm["compiler_only"]["genuine_rescue"] is False
    assert by_arm["semantic_judge"]["genuine_rescue"] is True
    targets = result["exports"]["rs_sft_targets"]
    assert {row["code"] for row in targets} == {
        "plain_resample_WINNER",
        "semantic_judge_WINNER",
    }
    assert all(row["common_baseline_holdback_improved"] for row in targets)
    assert result["selection_policy"]["common_baseline_shared_by_all_arms"] is True
    assert result["selection_policy"]["all_visible_scores_complete_before_holdback_open"] is True


def test_cli_exposes_anthropic_and_visible_public_phase_arguments() -> None:
    diagnose = rescue.parse_args(
        [
            "diagnose",
            "--plan",
            "plan.json",
            "--output",
            "diagnoses.json",
            "--journal",
            "diagnoses.journal.jsonl",
            "--api-style",
            "anthropic_messages",
            "--thinking-mode",
            "adaptive",
            "--model",
            "claude-sonnet-5",
        ]
    )
    assert diagnose.api_style == "anthropic_messages"
    assert diagnose.thinking_mode == "adaptive"
    assert diagnose.model == "claude-sonnet-5"
    assert not hasattr(diagnose, "private_holdback")

    plan = rescue.parse_args(
        [
            "plan",
            "--projection",
            "projection.json",
            "--rollout-file",
            "visible-rollout.jsonl",
            "--f2-jsonl",
            "f2.jsonl",
            "--f2-manifest",
            "f2-manifest.json",
            "--public-manifest",
            "feedback.public.json",
            "--output",
            "plan.json",
        ]
    )
    assert plan.public_manifest == "feedback.public.json"
    assert plan.rollout_file == "visible-rollout.jsonl"
    assert not hasattr(plan, "private_holdback")

    score = rescue.parse_args(
        [
            "score",
            "--plan",
            "plan.json",
            "--generation",
            "generation.json",
            "--private-holdback",
            "holdback.private.jsonl",
            "--expected-private-holdback-sha256",
            SHA_A,
            "--visible-output",
            "visible-selection.json",
            "--output",
            "score.json",
        ]
    )
    assert score.visible_output == "visible-selection.json"
    assert score.private_holdback == "holdback.private.jsonl"
