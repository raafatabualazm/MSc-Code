#!/usr/bin/env python3
"""Single-seed HOLD gate for typed fold+gold Arm C.

Promotion eligibility remains anchored to the 18/175 update58 incumbent and
the pre-registered 9.90 diversity bar.  This audit additionally validates and
pairs Arm C against the completed Arm-B seed-42 artifact, which is the direct
practical-recipe comparison.  One seed may veto but can never promote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation import audit_t5gemma2_typed_pass3_promotion as shared
from scripts.evaluation.durable_evaluation_journal import (
    require_exact_or_write,
    sha256_file,
)


REPORT_SCHEMA = "t5gemma2-typed-fold-gold-replay-single-seed-promotion-gate-v2"
ARM_B_DECISION_SCHEMA = "t5gemma2-typed-fold-single-seed-promotion-gate-v1"


def _metric_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "pass_at_1",
            "pass_at_10",
            "compile_at_10",
            "distinct_mean",
            "distinct_counts_sha256",
            "tasks_below_10_distinct",
        )
    }


def _paired(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        current["task_order"] != baseline["task_order"]
        or current["sample_coordinates_sha256"] != baseline["sample_coordinates_sha256"]
    ):
        raise ValueError("Arm C/Arm B exact seed-42 pairing differs")
    result: dict[str, dict[str, Any]] = {}
    for key in ("passk", "pass1", "compk"):
        gains = sum(
            current["task_metrics"][task_id][key]
            and not baseline["task_metrics"][task_id][key]
            for task_id in current["task_order"]
        )
        losses = sum(
            baseline["task_metrics"][task_id][key]
            and not current["task_metrics"][task_id][key]
            for task_id in current["task_order"]
        )
        result[key] = {
            "gains": gains,
            "losses": losses,
            "discordant": gains + losses,
            "exact_two_sided_p": shared._exact_p(gains, losses),  # noqa: SLF001
        }
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    delegated = argparse.Namespace(
        pass3_score=args.arm_c_score,
        expected_pass3_score_sha256=args.expected_arm_c_score_sha256,
        pass3_predictions=args.arm_c_predictions,
        expected_pass3_predictions_sha256=args.expected_arm_c_predictions_sha256,
        pass3_provenance=args.arm_c_provenance,
        expected_pass3_provenance_sha256=args.expected_arm_c_provenance_sha256,
        update58_score=args.update58_score,
        expected_update58_score_sha256=args.expected_update58_score_sha256,
        update58_predictions=args.update58_predictions,
        expected_update58_predictions_sha256=args.expected_update58_predictions_sha256,
        update58_provenance=args.update58_provenance,
        expected_update58_provenance_sha256=args.expected_update58_provenance_sha256,
        collapse_checker=args.collapse_checker,
        expected_collapse_checker_sha256=args.expected_collapse_checker_sha256,
        output="unused",
    )
    report = shared.audit(delegated)

    arm_c_score_path = shared._pin(  # noqa: SLF001
        args.arm_c_score, args.expected_arm_c_score_sha256, "Arm C score"
    )
    arm_b_score_path = shared._pin(  # noqa: SLF001
        args.arm_b_score, args.expected_arm_b_score_sha256, "Arm B score"
    )
    arm_b_predictions_path = shared._pin(  # noqa: SLF001
        args.arm_b_predictions,
        args.expected_arm_b_predictions_sha256,
        "Arm B predictions",
    )
    arm_b_provenance_path = shared._pin(  # noqa: SLF001
        args.arm_b_provenance,
        args.expected_arm_b_provenance_sha256,
        "Arm B provenance",
    )
    arm_b_decision_path = shared._pin(  # noqa: SLF001
        args.arm_b_decision,
        args.expected_arm_b_decision_sha256,
        "Arm B decision",
    )
    arm_c = shared._validate_score(
        arm_c_score_path, label="Arm C score"
    )  # noqa: SLF001
    arm_b = shared._validate_score(
        arm_b_score_path, label="Arm B score"
    )  # noqa: SLF001
    arm_b_generation = shared._validate_generation(  # noqa: SLF001
        predictions_path=arm_b_predictions_path,
        provenance_path=arm_b_provenance_path,
        score=arm_b,
        label="Arm B",
    )
    arm_b_decision = shared._read_object(  # noqa: SLF001
        arm_b_decision_path, "Arm B decision"
    )
    decision_inputs = arm_b_decision.get("inputs")
    fold_input = (
        decision_inputs.get("fold") if isinstance(decision_inputs, Mapping) else None
    )
    fold_generation = (
        fold_input.get("generation") if isinstance(fold_input, Mapping) else None
    )
    if (
        arm_b_decision.get("schema") != ARM_B_DECISION_SCHEMA
        or arm_b_decision.get("status") != "pass"
        or arm_b_decision.get("automatic_promotion_performed") is not False
        or arm_b_decision.get("verpo_launched") is not False
        or arm_b_decision.get("decision", {}).get("promotion_status")
        != "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
        or not isinstance(fold_input, Mapping)
        or fold_input.get("sha256") != args.expected_arm_b_score_sha256
        or not isinstance(fold_generation, Mapping)
        or fold_generation.get("predictions", {}).get("sha256")
        != args.expected_arm_b_predictions_sha256
        or fold_generation.get("provenance", {}).get("sha256")
        != args.expected_arm_b_provenance_sha256
    ):
        raise ValueError(
            "Arm B decision does not seal the supplied comparison artifacts"
        )

    arm_c_vs_arm_b = _paired(arm_c, arm_b)
    report["schema"] = REPORT_SCHEMA
    report["arm"] = "typed_fold_production_eligible_gold_replay_arm_c2_v2"
    report["shared_gate_logic"] = {
        "path": str(Path(shared.__file__).resolve()),
        "sha256": sha256_file(Path(shared.__file__).resolve()),
        "semantics": "same_update58_floor_diversity_and_full175_k10_seed42_gate",
    }
    report["inputs"]["arm_c"] = report["inputs"].pop("pass3")
    report["inputs"]["arm_b"] = {
        "path": str(arm_b_score_path),
        "sha256": sha256_file(arm_b_score_path),
        "task_order_sha256": arm_b["task_order_sha256"],
        "sample_coordinates_sha256": arm_b["sample_coordinates_sha256"],
        "generation": arm_b_generation,
        "decision": {
            "path": str(arm_b_decision_path),
            "sha256": sha256_file(arm_b_decision_path),
        },
    }
    report["metrics"]["arm_c"] = report["metrics"].pop("pass3")
    report["metrics"]["arm_b"] = _metric_view(arm_b)
    report["metrics"]["arm_c_vs_arm_b_paired"] = arm_c_vs_arm_b
    report["metrics"]["paired_vs_update58"] = report["metrics"].pop("paired")
    report["decision"]["promotion_status"] = "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
    report["decision"]["promoted_checkpoint"] = None
    report["decision"]["verpo_status"] = "HOLD"
    report["decision"]["arm_b_comparison_is_single_seed_diagnostic_only"] = True
    report["decision"][
        "arm_c_estimand"
    ] = (
        "practical_B_plus_1to1_production_eligible_replay_recipe_"
        "not_pure_gold_content_causality"
    )
    report["decision"]["matched_arm_b_same_seeds_required_for_promotion"] = True
    report["replication_status"]["validated_arm_c_seeds"] = report[
        "replication_status"
    ].pop("validated_pass3_seeds")
    report["replication_status"]["additional_matched_arm_c_seeds_required"] = report[
        "replication_status"
    ].pop("additional_matched_pass3_seeds_required")
    report["replication_status"]["arm_b_same_seed_comparators_required"] = True
    report["automatic_promotion_performed"] = False
    report["verpo_launched"] = False
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for prefix in ("arm_c", "arm_b", "update58"):
        parser.add_argument(f"--{prefix.replace('_', '-')}-score", required=True)
        parser.add_argument(
            f"--expected-{prefix.replace('_', '-')}-score-sha256", required=True
        )
        parser.add_argument(f"--{prefix.replace('_', '-')}-predictions", required=True)
        parser.add_argument(
            f"--expected-{prefix.replace('_', '-')}-predictions-sha256", required=True
        )
        parser.add_argument(f"--{prefix.replace('_', '-')}-provenance", required=True)
        parser.add_argument(
            f"--expected-{prefix.replace('_', '-')}-provenance-sha256", required=True
        )
    parser.add_argument("--arm-b-decision", required=True)
    parser.add_argument("--expected-arm-b-decision-sha256", required=True)
    parser.add_argument("--collapse-checker", required=True)
    parser.add_argument("--expected-collapse-checker-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(args)
        require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_ARM_C_PROMOTION_GATE_BLOCKED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_ARM_C_PROMOTION_GATE_COMPLETE "
        + json.dumps(
            {
                "promotion_status": report["decision"]["promotion_status"],
                "seed42_diagnostic_eligible": report["decision"][
                    "seed42_diagnostic_eligible"
                ],
                "verpo_status": report["decision"]["verpo_status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
