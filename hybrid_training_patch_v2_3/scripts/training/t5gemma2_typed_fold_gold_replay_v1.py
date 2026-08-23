#!/usr/bin/env python3
"""Arm C2: the sealed 458-row typed fold plus 1:1 typed gold replay.

Arm C is deliberately isolated from fold-only Arm B.  It reconstructs Arm B's
direct union through the Arm-B builder, requires byte-exact canonical equality
with Arm B's persisted dataset manifest, and adds exactly 458 task-disjoint
gold rows selected from the remaining clean typed TRAIN universe by a sealed
seeded hash order after a corpus-wide production-verifier eligibility gate.
Gold implementations are decoder supervision only; the encoder still receives
only the opaque typed contract and F2 text.

The eligibility gate is the explicit Arm-C2 amendment.  The original Arm-C
selection admitted a trusted gold program that the production evaluator can
never score because it imports ``dart:io``.  C2 applies the byte-identical
production verifier uniformly to every ranked, source-unique replay candidate
before taking 458 rows; it never substitutes a hand-picked target.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_contract_sft as typed_sft
from scripts.training import t5gemma2_typed_direct_rs_sft as pass1
from scripts.training import t5gemma2_typed_fold_rs_sft_union_v1 as fold


RUN_SCHEMA = "t5gemma2-typed-fold-gold-replay-run-v2"
CHECKPOINT_SCHEMA = "t5gemma2-typed-fold-gold-replay-checkpoint-v2"
DATASET_SCHEMA = "t5gemma2-typed-fold-gold-replay-dataset-v2"
# Preserve the originally preregistered ordering so C2 is a minimal
# production-eligibility amendment (457 retained rows plus one replacement),
# not an unrelated reseed caused by changing the schema string.
REPLAY_RANKING_SCHEMA = "t5gemma2-typed-fold-gold-replay-dataset-v1"
ARM_C2_AMENDMENT_SHA256 = (
    "8226d0ebd55476088d2e2a5cbfb06e573e92539012c4dc4ba551417158e261ed"
)
ARM_C2_AMENDMENT_SEAL_SHA256 = (
    "a15d5b9f42a4df410dadda677e3ceba262bc7ff5f743c566e1370f41cedb2cb7"
)

EXPECTED_DIRECT_ROWS = 458
EXPECTED_GOLD_REPLAY_ROWS = 458
EXPECTED_TOTAL_ROWS = EXPECTED_DIRECT_ROWS + EXPECTED_GOLD_REPLAY_ROWS
EXPECTED_C002_TAIL_ROWS = EXPECTED_DIRECT_ROWS - fold.BASE_ROWS
EXPECTED_PLANNED_UPDATES = 58
EXPECTED_CHECKPOINT_INTERVAL = 20
ARM_C2_PRODUCTION_VERIFY_WORKERS = 64
EXPECTED_DIRECT_SOURCE_SHA256S = 457
EXPECTED_REPLAY_ELIGIBLE_ROWS = 2315
EXPECTED_REPLAY_ELIGIBLE_UNIQUE_SOURCES = 2314
EXPECTED_REPLAY_PRODUCTION_ELIGIBLE_UNIQUE_SOURCES = 2312
EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS = frozenset(
    {
        "sigless_67bb88ce699e",
        "sigless_bfde11b99b84",
    }
)
EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS_SHA256 = canonical_sha256(
    sorted(EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS)
)
EXPECTED_REPLAY_TASK_IDS_SHA256 = (
    "6da49d120c902fde194c09fa14f7718bb379d8e676da8071211e1ac95da8e9df"
)
EXPECTED_REPLAY_SOURCE_SHA256S_SHA256 = (
    "1c818f33808c4142eb7b148733ce6879a779f795f1855e66533832baa99b31d6"
)
EXPECTED_REPLAY_TARGET_SHA256S_SHA256 = (
    "c7031487c72a2edba0baca1d8fe9eadc76232136c439a0f1fc95b25e2044e8f6"
)

VerifyFn = Callable[[str, str, str], bool]
_MIXED_RUNTIME_CONTRACT = mixed._runtime_contract  # noqa: SLF001


def _production_eligible_replay_pool(
    pairs: Sequence[base_sft.TextPair],
    *,
    tests_by_id: Mapping[str, str],
    verify: VerifyFn,
    workers: int,
) -> tuple[list[base_sft.TextPair], dict[str, Any]]:
    """Filter every source-unique replay candidate with the deployed verifier.

    This is deliberately a corpus-wide gate.  It prevents the original
    failure from being repaired by an outcome-aware one-row substitution and
    ensures every retained decoder target is capable of scoring under the
    exact evaluator used after training.
    """

    if workers <= 0:
        raise ValueError("production eligibility workers must be positive")

    def one(position: int) -> tuple[int, bool]:
        pair = pairs[position]
        return position, bool(
            verify(
                pair.target,
                tests_by_id[pair.task_id],
                f"typed-arm-c2-eligibility-{position:04d}",
            )
        )

    if workers == 1:
        outcomes = [one(index) for index in range(len(pairs))]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as pool:
            outcomes = list(pool.map(one, range(len(pairs))))

    eligible = [pairs[index] for index, passed in outcomes if passed]
    rejected = [pairs[index] for index, passed in outcomes if not passed]
    rejected_ids = {pair.task_id for pair in rejected}
    if (
        len(pairs) != EXPECTED_REPLAY_ELIGIBLE_UNIQUE_SOURCES
        or len(eligible) != EXPECTED_REPLAY_PRODUCTION_ELIGIBLE_UNIQUE_SOURCES
        or rejected_ids != EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS
    ):
        raise ValueError(
            "Arm C2 corpus-wide production-verifier eligibility set differs"
        )
    return eligible, {
        "scope": "all_post_direct_disjoint_source_unique_replay_candidates",
        "verifier": "exact_production_complete_acceptance_verifier",
        "candidates_checked": len(pairs),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "rejected_task_ids_sha256": (
            EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS_SHA256
        ),
        "stability_runs": pass1.FULL_VERIFY_STABILITY_RUNS,
        "timeout_seconds": pass1.FULL_VERIFY_TIMEOUT_SECONDS,
        "tests_model_visible": False,
        "diagnostics_persisted": False,
        "uses_arm_b_predictions_or_scores": False,
        "selection_amendment_fixed_after_arm_b_result": True,
    }


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = __import__("json").loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_arm_b_manifest(
    direct_manifest: Mapping[str, Any],
    *,
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Arm B dataset manifest differs from its SHA-256 pin")
    persisted = _read_object(path, "Arm B dataset manifest")
    if (
        persisted.get("schema") != fold.DATASET_SCHEMA
        or persisted.get("arm") != "B_fold_only"
        or int(persisted.get("rows", -1)) != EXPECTED_DIRECT_ROWS
        or canonical_sha256(persisted) != canonical_sha256(direct_manifest)
    ):
        raise ValueError("reconstructed direct union differs from sealed Arm B")
    composition = persisted.get("composition")
    if (
        not isinstance(composition, Mapping)
        or int(composition.get("verified_direct", -1)) != EXPECTED_DIRECT_ROWS
        or int(composition.get("kimi_c002_tail", -1)) != EXPECTED_C002_TAIL_ROWS
        or int(composition.get("gold_replay", -1)) != 0
    ):
        raise ValueError("sealed Arm B composition differs")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "canonical_sha256": canonical_sha256(persisted),
        "rows": EXPECTED_DIRECT_ROWS,
        "schedule_sha256": str(persisted.get("schedule_sha256") or ""),
        "identity": "byte_pinned_manifest_and_canonical_reconstruction_match",
    }


def _equivalent_source_cross_verify(
    pairs: Sequence[mixed.MixedPair],
    *,
    tests_by_id: Mapping[str, str],
    verify: VerifyFn,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_source: dict[str, list[mixed.MixedPair]] = defaultdict(list)
    for pair in pairs:
        by_source[pair.source_sha256].append(pair)
    groups = {digest: group for digest, group in by_source.items() if len(group) > 1}
    checks: list[mixed.MixedPair] = []
    records: list[dict[str, Any]] = []
    for source_digest, group in sorted(groups.items()):
        records.append(
            {
                "source_sha256": source_digest,
                "task_ids": [pair.source_task_id for pair in group],
                "target_sha256s": [pair.target_sha256 for pair in group],
                "rows": len(group),
                "cross_acceptance_checks": len(group) ** 2,
            }
        )
        for target_pair in group:
            for test_pair in group:
                checks.append(
                    mixed._make_pair(  # noqa: SLF001
                        pair_id=(
                            "arm-c-equivalence-cross::"
                            f"{target_pair.source_task_id}::to::{test_pair.source_task_id}"
                        ),
                        source_task_id=test_pair.source_task_id,
                        kind=target_pair.kind,
                        source=test_pair.source,
                        target=target_pair.target,
                        provenance=(
                            ("equivalent_source_sha256", source_digest),
                            ("target_origin_task_id", target_pair.source_task_id),
                            ("test_owner_task_id", test_pair.source_task_id),
                        ),
                    )
                )
    if checks:
        verification = pass1._verify_all(  # noqa: SLF001
            checks,
            tests_by_id=tests_by_id,
            verify=verify,
            workers=workers,
        )
    else:
        verification = {
            "rows": 0,
            "passed": 0,
            "stability_runs": pass1.FULL_VERIFY_STABILITY_RUNS,
            "timeout_seconds": pass1.FULL_VERIFY_TIMEOUT_SECONDS,
            "diagnostics_persisted": False,
            "tests_model_visible": False,
        }
    return records, verification


def _compose_arm_c(
    *,
    direct_pairs: Sequence[mixed.MixedPair],
    direct_manifest: Mapping[str, Any],
    arm_b_manifest_path: Path,
    expected_arm_b_manifest_sha256: str,
    typed_pairs: Sequence[base_sft.TextPair],
    typed_manifest: Mapping[str, Any],
    tests_by_id: Mapping[str, str],
    seed: int,
    verify: VerifyFn,
    verification_workers: int,
) -> tuple[list[mixed.MixedPair], dict[str, Any]]:
    if len(direct_pairs) != EXPECTED_DIRECT_ROWS:
        raise ValueError("Arm C requires Arm B's exact 458 direct rows")
    arm_b_seal = _require_arm_b_manifest(
        direct_manifest,
        path=arm_b_manifest_path,
        expected_sha256=expected_arm_b_manifest_sha256,
    )
    direct_ids = [pair.source_task_id for pair in direct_pairs]
    if len(set(direct_ids)) != EXPECTED_DIRECT_ROWS:
        raise ValueError("Arm B direct identities are not unique")

    exclusions = typed_manifest.get("training_exclusions")
    heldout = typed_manifest.get("heldout")
    if (
        typed_manifest.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or not isinstance(exclusions, Mapping)
        or pass1.CONTAMINATED_TRAIN_TASK_ID not in (exclusions.get("task_ids") or [])
        or not isinstance(heldout, Mapping)
        or heldout.get("task_id_overlap") != 0
        or heldout.get("exact_gold_source_overlap") != 0
        or heldout.get("exact_acceptance_test_overlap") != 0
        or heldout.get("model_visible") is not False
    ):
        raise ValueError("typed TRAIN privacy/heldout contract differs")

    typed_by_id = {pair.task_id: pair for pair in typed_pairs}
    if len(typed_by_id) != len(typed_pairs):
        raise ValueError("clean typed TRAIN universe contains duplicate task IDs")
    if not set(direct_ids).issubset(typed_by_id):
        raise ValueError("Arm B direct identity lies outside clean typed TRAIN")
    direct_source_sha256s = {pair.source_sha256 for pair in direct_pairs}
    if len(direct_source_sha256s) != EXPECTED_DIRECT_SOURCE_SHA256S:
        raise ValueError("Arm B direct typed-source cardinality differs from 457")
    replay_pool = [
        pair
        for pair in typed_pairs
        if pair.task_id not in set(direct_ids)
        and pair.source_sha256 not in direct_source_sha256s
    ]
    replay_pool.sort(
        key=lambda pair: canonical_sha256(
            {
                "schema": REPLAY_RANKING_SCHEMA,
                "seed": seed,
                "kind": "gold_replay",
                "task_id": pair.task_id,
                "source_sha256": pair.source_sha256,
            }
        )
    )
    if len(replay_pool) != EXPECTED_REPLAY_ELIGIBLE_ROWS:
        raise ValueError("Arm C replay-eligible pool cardinality differs from 2315")
    # A model-visible typed prompt may be shared by distinct source tasks.  Arm
    # C admits at most one replay row per typed-source hash, preventing hidden
    # competing decoder targets and keeping replay disjoint from Arm B on both
    # task identity and the actual encoder input.
    unique_replay_pool: list[base_sft.TextPair] = []
    selected_source_sha256s: set[str] = set()
    for pair in replay_pool:
        if pair.source_sha256 in selected_source_sha256s:
            continue
        selected_source_sha256s.add(pair.source_sha256)
        unique_replay_pool.append(pair)
    if len(unique_replay_pool) != EXPECTED_REPLAY_ELIGIBLE_UNIQUE_SOURCES:
        raise ValueError(
            "Arm C replay-eligible unique-source cardinality differs from 2314"
        )
    production_eligible_pool, production_eligibility = (
        _production_eligible_replay_pool(
            unique_replay_pool,
            tests_by_id=tests_by_id,
            verify=verify,
            workers=verification_workers,
        )
    )
    selected_gold = production_eligible_pool[:EXPECTED_GOLD_REPLAY_ROWS]
    replay_pairs = [
        mixed._make_pair(  # noqa: SLF001
            pair_id=f"{pair.task_id}::typed-gold-replay-arm-c-v1",
            source_task_id=pair.task_id,
            kind="gold_replay",
            source=pair.source,
            target=pair.target,
            provenance=(
                ("dataset_schema", DATASET_SCHEMA),
                (
                    "gold_f2_sha256",
                    str(typed_manifest.get("f2", {}).get("sha256") or ""),
                ),
                (
                    "gold_train_sha256",
                    str(typed_manifest.get("dataset", {}).get("sha256") or ""),
                ),
                (
                    "selection",
                    "production_verifier_eligible_then_deterministic_seeded_hash_order",
                ),
                ("ranking_schema", REPLAY_RANKING_SCHEMA),
                ("typed_source_sha256", pair.source_sha256),
            ),
        )
        for pair in selected_gold
    ]
    replay_verification = pass1._verify_all(  # noqa: SLF001
        replay_pairs,
        tests_by_id=tests_by_id,
        verify=verify,
        workers=verification_workers,
    )

    combined = list(direct_pairs) + replay_pairs
    combined.sort(
        key=lambda pair: canonical_sha256(
            {
                "schema": DATASET_SCHEMA,
                "seed": seed,
                "pair_id": pair.pair_id,
                "kind": pair.kind,
                "source_sha256": pair.source_sha256,
                "target_sha256": pair.target_sha256,
            }
        )
    )
    task_ids = [pair.source_task_id for pair in combined]
    pair_ids = [pair.pair_id for pair in combined]
    if (
        len(combined) != EXPECTED_TOTAL_ROWS
        or len(set(task_ids)) != EXPECTED_TOTAL_ROWS
        or len(set(pair_ids)) != EXPECTED_TOTAL_ROWS
        or pass1.CONTAMINATED_TRAIN_TASK_ID in task_ids
    ):
        raise ValueError("Arm C combined schedule identity differs")
    equivalent_groups, cross_verification = _equivalent_source_cross_verify(
        combined,
        tests_by_id=tests_by_id,
        verify=verify,
        workers=verification_workers,
    )
    direct_verification = direct_manifest.get("full_acceptance_reverification")
    if (
        not isinstance(direct_verification, Mapping)
        or int(direct_verification.get("rows", -1)) != EXPECTED_DIRECT_ROWS
        or int(direct_verification.get("passed", -1)) != EXPECTED_DIRECT_ROWS
        or direct_verification.get("tests_model_visible") is not False
        or direct_verification.get("diagnostics_persisted") is not False
        or int(replay_verification.get("rows", -1)) != EXPECTED_GOLD_REPLAY_ROWS
        or int(replay_verification.get("passed", -1)) != EXPECTED_GOLD_REPLAY_ROWS
    ):
        raise ValueError("Arm C acceptance re-verification differs")

    schedule = [
        {
            "position": position,
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "kind": pair.kind,
            "source_sha256": pair.source_sha256,
            "target_sha256": pair.target_sha256,
            "provenance": dict(pair.provenance),
        }
        for position, pair in enumerate(combined)
    ]
    replay_ids = [pair.source_task_id for pair in replay_pairs]
    replay_task_ids_sha256 = canonical_sha256(replay_ids)
    replay_source_sha256s_sha256 = canonical_sha256(
        [pair.source_sha256 for pair in replay_pairs]
    )
    replay_target_sha256s_sha256 = canonical_sha256(
        [pair.target_sha256 for pair in replay_pairs]
    )
    if (
        replay_task_ids_sha256 != EXPECTED_REPLAY_TASK_IDS_SHA256
        or replay_source_sha256s_sha256 != EXPECTED_REPLAY_SOURCE_SHA256S_SHA256
        or replay_target_sha256s_sha256 != EXPECTED_REPLAY_TARGET_SHA256S_SHA256
    ):
        raise ValueError("Arm C2 replay selection differs from its amendment seal")
    replay_selection = {
        "requested_rows": EXPECTED_GOLD_REPLAY_ROWS,
        "selected_rows": EXPECTED_GOLD_REPLAY_ROWS,
        "ratio_to_direct": 1.0,
        "direct_tasks_excluded": True,
        "direct_typed_source_sha256s_excluded": True,
        "unique_typed_source_sha256_within_replay": True,
        "selection": (
            "corpus_wide_production_verifier_eligibility_then_"
            "deterministic_seeded_hash_order"
        ),
        "seed": seed,
        "ranking_schema": REPLAY_RANKING_SCHEMA,
        "eligible_rows_after_direct_task_and_source_exclusion": (
            EXPECTED_REPLAY_ELIGIBLE_ROWS
        ),
        "eligible_unique_typed_sources": EXPECTED_REPLAY_ELIGIBLE_UNIQUE_SOURCES,
        "production_verifier_eligible_unique_typed_sources": (
            EXPECTED_REPLAY_PRODUCTION_ELIGIBLE_UNIQUE_SOURCES
        ),
        "production_admissibility": production_eligibility,
        "direct_unique_typed_sources": EXPECTED_DIRECT_SOURCE_SHA256S,
        "selection_key_fields": [
            "schema",
            "seed",
            "kind",
            "task_id",
            "source_sha256",
        ],
        "selected_task_ids_sha256": replay_task_ids_sha256,
        "selected_source_sha256s_sha256": replay_source_sha256s_sha256,
        "selected_target_sha256s_sha256": replay_target_sha256s_sha256,
        "base_ranking_fixed_before_arm_b_result": True,
        "selection_fixed_before_arm_b_result": False,
        "selection_amended_after_arm_b_result": True,
    }
    manifest = {
        "schema": DATASET_SCHEMA,
        "rows": EXPECTED_TOTAL_ROWS,
        "architecture": "native_encoder_decoder",
        "arm": "C2_fold_plus_production_eligible_typed_gold_replay_1to1",
        "estimand": (
            "practical_B_plus_1to1_production_eligible_typed_gold_replay_recipe"
        ),
        "original_arm_c_status": "TERMINATED_PREFLIGHT_INFEASIBLE",
        "arm_c2_amendment": {
            "document_sha256": ARM_C2_AMENDMENT_SHA256,
            "seal_sha256": ARM_C2_AMENDMENT_SEAL_SHA256,
            "fixed_before_optimizer_step": True,
            "original_arm_c_optimizer_steps": 0,
        },
        "pure_gold_content_causal_claim_permitted": False,
        "mechanism_control_required": "duplicated_direct_GA16_without_gold_replay",
        "fresh_branch_from": "typed_sft_optstep348",
        "composition": {
            "verified_direct": EXPECTED_DIRECT_ROWS,
            "gold_replay": EXPECTED_GOLD_REPLAY_ROWS,
            "repair_conditioned": 0,
            "reasoning_rows": 0,
        },
        "direct_union": {
            "rows": EXPECTED_DIRECT_ROWS,
            "dataset_schema": fold.DATASET_SCHEMA,
            "dataset_manifest": arm_b_seal,
            "pair_ids_sha256": str(direct_manifest.get("pair_ids_sha256") or ""),
            "source_sha256s_sha256": str(
                direct_manifest.get("source_sha256s_sha256") or ""
            ),
            "target_sha256s_sha256": str(
                direct_manifest.get("target_sha256s_sha256") or ""
            ),
            "schedule_sha256": str(direct_manifest.get("schedule_sha256") or ""),
            "canonical_manifest_sha256": canonical_sha256(direct_manifest),
            "full_acceptance_reverification": dict(direct_verification),
        },
        "gold_replay": replay_selection,
        "typed_train": dict(typed_manifest),
        "heldout_overlap": 0,
        "heldout_175_model_visible": False,
        "known_contaminant_excluded": pass1.CONTAMINATED_TRAIN_TASK_ID,
        "task_identity_policy": "direct_and_gold_replay_are_globally_task_disjoint",
        "typed_source_identity_policy": (
            "replay_excludes_all_direct_source_sha256s_and_is_unique_within_replay"
        ),
        "model_visible_encoder_fields": ["opaque_typed_contract", "F2.text"],
        "decoder_supervision_fields": [
            "verified_direct_dart_source",
            "gold_replay_dart_source",
        ],
        "tests_model_visible": False,
        "private_feedback_model_visible": False,
        "repair_conditioned_prefixes_visible": False,
        "reasoning_model_visible": False,
        "gold_implementation_encoder_visible": False,
        "gold_replay_decoder_supervision": True,
        "gold_replay_acceptance_reverification": replay_verification,
        "full_acceptance_reverification": {
            "rows": EXPECTED_TOTAL_ROWS,
            "passed": EXPECTED_TOTAL_ROWS,
            "direct": dict(direct_verification),
            "gold_replay": replay_verification,
            "tests_model_visible": False,
            "diagnostics_persisted": False,
        },
        "equivalent_typed_source_policy": (
            "retain_distinct_tasks_only_after_all_targets_cross_pass_all_private_suites"
        ),
        "equivalent_typed_source_groups": len(equivalent_groups),
        "equivalent_typed_source_group_records": equivalent_groups,
        "equivalent_typed_source_groups_sha256": canonical_sha256(equivalent_groups),
        "equivalent_typed_source_cross_acceptance": cross_verification,
        "schedule": schedule,
        "schedule_sha256": canonical_sha256(schedule),
        "task_ids_sha256": canonical_sha256(task_ids),
        "pair_ids_sha256": canonical_sha256(pair_ids),
        "source_sha256s_sha256": canonical_sha256(
            [pair.source_sha256 for pair in combined]
        ),
        "target_sha256s_sha256": canonical_sha256(
            [pair.target_sha256 for pair in combined]
        ),
        "production_floor_eligible": True,
        "automatic_promotion_permitted": False,
        "promotion_status": "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS",
        "verpo_status": "HOLD",
    }
    return combined, manifest


def build_typed_fold_gold_replay_pairs(
    *,
    gold_train_jsonl: Path,
    gold_f2_jsonl: Path,
    expected_gold_train_sha256: str,
    expected_gold_f2_sha256: str,
    expected_gold_rows: int,
    heldout_jsonl: Path,
    expected_heldout_sha256: str,
    expected_heldout_rows: int,
    local_reports: Sequence[tuple[Path, str]],
    api_reports: Sequence[tuple[Path, str]],
    warmstart: mixed.WarmstartIdentity,
    gold_replay_ratio: float,
    gold_replay_rows: int,
    min_verified_direct_targets: int,
    min_repair_conditioned_targets: int,
    allow_exploratory_inputs: bool,
    require_local_production_floor: bool,
    seed: int,
    arm_b_dataset_manifest: Path,
    expected_arm_b_dataset_manifest_sha256: str,
    verify: VerifyFn | None = None,
    verification_workers: int = ARM_C2_PRODUCTION_VERIFY_WORKERS,
) -> tuple[list[mixed.MixedPair], dict[str, Any]]:
    del warmstart
    if (
        gold_replay_ratio != 1.0
        or gold_replay_rows != EXPECTED_GOLD_REPLAY_ROWS
        or min_verified_direct_targets != EXPECTED_DIRECT_ROWS
        or min_repair_conditioned_targets != 0
        or allow_exploratory_inputs
        or require_local_production_floor
    ):
        raise ValueError(
            "Arm C2 accepts only exact 458 direct + 458 production-eligible "
            "typed-gold profile"
        )
    runtime_verify = verify or pass1._runtime_verify  # noqa: SLF001
    direct_pairs, direct_manifest = fold.build_typed_fold_union_pairs(
        gold_train_jsonl=gold_train_jsonl,
        gold_f2_jsonl=gold_f2_jsonl,
        expected_gold_train_sha256=expected_gold_train_sha256,
        expected_gold_f2_sha256=expected_gold_f2_sha256,
        expected_gold_rows=expected_gold_rows,
        heldout_jsonl=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        local_reports=local_reports,
        api_reports=api_reports,
        warmstart=pass1._source_sft_identity(),  # noqa: SLF001
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=fold.BASE_ROWS,
        min_repair_conditioned_targets=0,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        seed=seed,
        verify=runtime_verify,
        verification_workers=verification_workers,
    )
    typed_pairs, typed_manifest = typed_sft.load_typed_text_pairs(
        gold_train_jsonl,
        gold_f2_jsonl,
        expected_dataset_sha256=expected_gold_train_sha256,
        expected_f2_sha256=expected_gold_f2_sha256,
        expected_rows=expected_gold_rows,
        heldout_path=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        exclude_train_task_ids=[pass1.CONTAMINATED_TRAIN_TASK_ID],
        allow_unpinned_inputs=False,
    )
    train_rows = base_sft._read_jsonl(gold_train_jsonl)  # noqa: SLF001
    tests_by_id: dict[str, str] = {}
    for index, row in enumerate(train_rows):
        task_id = base_sft._identity(row, index)  # noqa: SLF001
        if task_id != pass1.CONTAMINATED_TRAIN_TASK_ID:
            tests_by_id[task_id] = pass1._complete_tests(row, task_id)  # noqa: SLF001
    return _compose_arm_c(
        direct_pairs=direct_pairs,
        direct_manifest=direct_manifest,
        arm_b_manifest_path=arm_b_dataset_manifest,
        expected_arm_b_manifest_sha256=expected_arm_b_dataset_manifest_sha256,
        typed_pairs=typed_pairs,
        typed_manifest=typed_manifest,
        tests_by_id=tests_by_id,
        seed=seed,
        verify=runtime_verify,
        verification_workers=verification_workers,
    )


def _profile_runtime_contract() -> dict[str, str]:
    record = dict(_MIXED_RUNTIME_CONTRACT())
    record["mixed_training_engine_sha256"] = record["trainer_sha256"]
    record["trainer_sha256"] = base_sft.sha256_file(Path(__file__).resolve())
    record["fold_union_builder_sha256"] = base_sft.sha256_file(
        Path(fold.__file__).resolve()
    )
    record["typed_source_builder_sha256"] = base_sft.sha256_file(
        Path(typed_sft.__file__).resolve()
    )
    record["trainer_profile"] = (
        "typed_fold_plus_production_eligible_gold_replay_arm_c2_v2"
    )
    return record


def _validate_profile_args(args: argparse.Namespace) -> None:
    expected = {
        "gold_replay_ratio": 1.0,
        "gold_replay_rows": EXPECTED_GOLD_REPLAY_ROWS,
        "min_verified_direct_targets": EXPECTED_DIRECT_ROWS,
        "min_repair_conditioned_targets": 0,
        "expected_warmstart_update": 348,
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "max_updates": 0,
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "max_source_tokens": 32768,
        "max_target_tokens": 32768,
        "checkpoint_interval": EXPECTED_CHECKPOINT_INTERVAL,
        "seed": 42,
        "attn_implementation": "sdpa",
        "bf16": True,
        "gradient_checkpointing": True,
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        matches = (
            math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float)
            else observed == wanted
        )
        if not matches:
            raise ValueError(
                f"typed Arm C fixes --{name}={wanted}, observed={observed}"
            )
    if args.allow_exploratory_inputs or args.require_local_production_floor:
        raise ValueError("typed Arm C requires sealed aggregate inputs")
    if (
        len(args.local_report) != fold.LOCAL_REPORT_COUNT
        or len(args.api_report) != fold.API_REPORT_COUNT
    ):
        raise ValueError("typed Arm C requires 9 local and 19 API artifacts")
    digest = str(args.expected_arm_b_dataset_manifest_sha256 or "")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("Arm C requires a valid Arm B dataset-manifest SHA-256")
    if not str(args.arm_b_dataset_manifest or ""):
        raise ValueError("Arm C requires the sealed Arm B dataset manifest")


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_profile_args(args)
    validate_dart_binary()
    originals = {
        "run_schema": mixed.RUN_SCHEMA,
        "checkpoint_schema": mixed.CHECKPOINT_SCHEMA,
        "dataset_schema": mixed.DATASET_SCHEMA,
        "builder": mixed.build_mixed_pairs,
        "warmstart": mixed.validate_warmstart,
        "runtime": mixed._runtime_contract,  # noqa: SLF001
    }

    def profile_builder(**kwargs: Any):
        return build_typed_fold_gold_replay_pairs(
            **kwargs,
            arm_b_dataset_manifest=Path(args.arm_b_dataset_manifest).resolve(),
            expected_arm_b_dataset_manifest_sha256=(
                args.expected_arm_b_dataset_manifest_sha256
            ),
        )

    mixed.RUN_SCHEMA = RUN_SCHEMA
    mixed.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    mixed.DATASET_SCHEMA = DATASET_SCHEMA
    mixed.build_mixed_pairs = profile_builder
    mixed.validate_warmstart = pass1.validate_typed_warmstart
    mixed._runtime_contract = _profile_runtime_contract  # noqa: SLF001
    try:
        result = mixed.train(args)
        if not args.preflight_only and (
            int(result.get("rows", -1)) != EXPECTED_TOTAL_ROWS
            or int(result.get("planned_updates", -1)) != EXPECTED_PLANNED_UPDATES
            or int(result.get("updates", -1)) != EXPECTED_PLANNED_UPDATES
        ):
            raise ValueError("Arm C did not complete the sealed 916-row/58-update plan")
        return result
    finally:
        mixed.RUN_SCHEMA = originals["run_schema"]
        mixed.CHECKPOINT_SCHEMA = originals["checkpoint_schema"]
        mixed.DATASET_SCHEMA = originals["dataset_schema"]
        mixed.build_mixed_pairs = originals["builder"]
        mixed.validate_warmstart = originals["warmstart"]
        mixed._runtime_contract = originals["runtime"]  # noqa: SLF001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--arm_b_dataset_manifest", required=True)
    extra.add_argument("--expected_arm_b_dataset_manifest_sha256", required=True)
    known, remaining = extra.parse_known_args(argv)
    args = mixed.parse_args(remaining)
    args.arm_b_dataset_manifest = known.arm_b_dataset_manifest
    args.expected_arm_b_dataset_manifest_sha256 = (
        known.expected_arm_b_dataset_manifest_sha256
    )
    try:
        _validate_profile_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(__import__("json").dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
