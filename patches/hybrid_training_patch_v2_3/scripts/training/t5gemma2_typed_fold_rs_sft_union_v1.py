#!/usr/bin/env python3
"""Fresh typed RS-SFT over the exact union of every accepted direct target.

This is the fold-only Arm B.  It branches from the two-epoch typed SFT
checkpoint (optstep 348), not from any RS-SFT checkpoint, and trains for one
epoch over the task-disjoint union

    pass1-225 + pass2-209 + Kimi-c001-12 + c002-prefix-1 + c002-tail-T.

The historical pass-1/pass-2 rows are reconstructed from their original
SHA-pinned producer reports and compared with the published dataset manifests;
the manifests are never treated as a source of target text.  Every target is
then rebound through the current opaque typed-source builder and the complete
union is re-executed against private TRAIN acceptance tests.  Tests,
diagnostics, semantic parameter names, reasoning, and gold implementations are
never serialized into a model-visible row.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_contract_sft as typed_sft
from scripts.training import t5gemma2_typed_direct_rs_sft as pass1
from scripts.training import t5gemma2_typed_direct_rs_sft_pass2 as pass2
from scripts.training import t5gemma2_typed_direct_rs_sft_pass3 as pass3


RUN_SCHEMA = "t5gemma2-typed-fold-rs-sft-union-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-fold-rs-sft-union-checkpoint-v1"
DATASET_SCHEMA = "t5gemma2-typed-fold-rs-sft-union-dataset-v1"

EXPECTED_PASS1_ROWS = 225
EXPECTED_PASS2_ROWS = 209
EXPECTED_C001_ROWS = 12
EXPECTED_PREFIX_ROWS = 1
BASE_ROWS = EXPECTED_PASS1_ROWS + EXPECTED_PASS2_ROWS + EXPECTED_C001_ROWS + EXPECTED_PREFIX_ROWS
MAX_C002_ROWS = 47

# local_report layout is deliberately positional and fail-closed:
#   0:4  pass-1 local producer reports
#   4    published pass-1 dataset manifest
#   5:8  pass-2 local report/journal/targets
#   8    published pass-2 dataset manifest
LOCAL_REPORT_COUNT = 9
PASS1_LOCAL_SLICE = slice(0, 4)
PASS1_MANIFEST_POSITION = 4
PASS2_LOCAL_SLICE = slice(5, 8)
PASS2_MANIFEST_POSITION = 8

# api_report layout:
#   0:7   pass-1 API producer reports
#   7:10  pass-2 orchestration report/manifest/targets
#   10:19 c001, c002-tail, and c002-prefix report/manifest/targets
API_REPORT_COUNT = 19
PASS1_API_SLICE = slice(0, 7)
PASS2_API_SLICE = slice(7, 10)
CONTINUATION_SLICE = slice(10, 19)

LOCAL_BASENAMES = (
    "harvest_report.json",
    "harvest_report.json",
    "harvest_report.json",
    "harvest_report.json",
    "dataset_manifest.json",
    "harvest_report.json",
    "harvest.journal.jsonl",
    "direct_targets.jsonl",
    "dataset_manifest.json",
)
API_BASENAMES = (
    "api_rescue_report.json",
    "api_rescue_report.json",
    "api_rescue_report.json",
    "api_rescue_report.json",
    "api_rescue_report.json",
    "api_rescue_report.json",
    "api_rescue_report.json",
    "orchestration_report.json",
    "direct_manifest.json",
    "direct_targets.jsonl",
    *pass3.SOURCE_BASENAMES,
)

VerifyFn = Callable[[str, str, str], bool]
_MIXED_RUNTIME_CONTRACT = mixed._runtime_contract  # noqa: SLF001


def _read_object(path: Path, label: str) -> dict[str, Any]:
    return pass2._read_object(path, label)  # noqa: SLF001


def _require_layout(
    specs: Sequence[tuple[Path, str]], basenames: Sequence[str], label: str
) -> tuple[tuple[Path, str], ...]:
    if len(specs) != len(basenames):
        raise ValueError(f"{label} requires exactly {len(basenames)} pinned artifacts")
    result = tuple(specs)
    for position, ((path, digest), basename) in enumerate(
        zip(result, basenames, strict=True)
    ):
        if path.name != basename or sha256_file(path) != digest:
            raise ValueError(f"{label} artifact {position} binding differs")
    return result


def _require_manifest_reconstruction(
    observed: Mapping[str, Any], manifest_spec: tuple[Path, str], label: str
) -> dict[str, Any]:
    path, digest = manifest_spec
    published = _read_object(path, f"{label} published dataset manifest")
    if canonical_sha256(observed) != canonical_sha256(published):
        raise ValueError(f"{label} reconstruction differs from its published manifest")
    return {
        "path": str(path),
        "sha256": digest,
        "canonical_sha256": canonical_sha256(published),
        "rows": int(published.get("rows", -1)),
        "target_text_source": "sealed_producer_reports_not_dataset_manifest",
    }


def _rebind_union(
    *,
    collected: Sequence[tuple[str, mixed.MixedPair]],
    typed_pairs: Sequence[base_sft.TextPair],
    heldout_ids: set[str],
    tests_by_id: Mapping[str, str],
    gold_target_by_id: Mapping[str, str],
    seed: int,
    source_audits: Mapping[str, Any],
    typed_manifest: Mapping[str, Any],
    heldout_record: Mapping[str, Any],
    verify: VerifyFn,
    verification_workers: int,
) -> tuple[list[mixed.MixedPair], dict[str, Any]]:
    """Apply global identity/privacy gates and build the deterministic fold."""

    opaque = typed_manifest.get("opaque_contract")
    exclusions = typed_manifest.get("training_exclusions")
    typed_heldout = typed_manifest.get("heldout")
    if (
        typed_manifest.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or not isinstance(opaque, Mapping)
        or opaque.get("parameter_name_policy") != "p{zero_based_index}"
        or opaque.get("semantic_function_name_exposed") is not False
        or opaque.get("semantic_parameter_names_exposed") is not False
        or not isinstance(exclusions, Mapping)
        or pass1.CONTAMINATED_TRAIN_TASK_ID not in (exclusions.get("task_ids") or [])
        or not isinstance(typed_heldout, Mapping)
        or typed_heldout.get("task_id_overlap") != 0
        or typed_heldout.get("exact_gold_source_overlap") != 0
        or typed_heldout.get("exact_acceptance_test_overlap") != 0
        or typed_heldout.get("model_visible") is not False
    ):
        raise ValueError("opaque typed TRAIN privacy/heldout contract differs")

    typed_by_id = {pair.task_id: pair for pair in typed_pairs}
    if len(typed_by_id) != len(typed_pairs):
        raise ValueError("clean typed TRAIN universe contains duplicate task IDs")

    task_ids = [pair.source_task_id for _category, pair in collected]
    if any(not task_id for task_id in task_ids):
        raise ValueError("fold union contains an empty task identity")
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise ValueError("fold union contains duplicate task IDs: " + duplicates[0])
    if len(task_ids) < BASE_ROWS or len(task_ids) > BASE_ROWS + MAX_C002_ROWS:
        raise ValueError("fold union row count is outside 447+T, 0<=T<=47")
    if set(task_ids) & heldout_ids:
        raise ValueError("held-out task entered fold union")
    if pass1.CONTAMINATED_TRAIN_TASK_ID in task_ids:
        raise ValueError("known train/heldout contaminant entered fold union")
    unknown = sorted(set(task_ids) - set(typed_by_id))
    if unknown:
        raise ValueError("fold target is outside clean typed TRAIN: " + unknown[0])

    rebound: list[mixed.MixedPair] = []
    category_by_id: dict[str, str] = {}
    for category, historical in collected:
        canonical = typed_by_id[historical.source_task_id]
        if historical.source_sha256 != canonical.source_sha256:
            raise ValueError("fold target is bound to a different typed source")
        if not historical.target.strip() or historical.target_sha256 != mixed.sha256_text(
            historical.target
        ):
            raise ValueError("fold target text/hash binding differs")
        category_by_id[historical.source_task_id] = category
        rebound.append(
            mixed._make_pair(  # noqa: SLF001
                pair_id=(
                    f"{historical.source_task_id}::typed-fold-union-v1::{category}"
                ),
                source_task_id=historical.source_task_id,
                kind="verified_direct",
                source=canonical.source,
                target=historical.target,
                provenance=tuple(
                    sorted(
                        set(historical.provenance)
                        | {
                            ("dataset_schema", DATASET_SCHEMA),
                            ("fold_source_category", category),
                            ("upstream_pair_id", historical.pair_id),
                            ("typed_source_sha256", canonical.source_sha256),
                        }
                    )
                ),
            )
        )

    # This is intentionally task-only deduplication.  Identical accepted code
    # for two distinct tasks remains two rows; target-text dedup would silently
    # change the requested task distribution.
    target_to_tasks: dict[str, list[str]] = defaultdict(list)
    for pair in rebound:
        target_to_tasks[pair.target_sha256].append(pair.source_task_id)
    shared_groups = {
        digest: ids for digest, ids in target_to_tasks.items() if len(ids) > 1
    }

    # Distinct task identities can also have byte-identical opaque typed input.
    # Such a group is not silently treated as a conflict or deduplicated.  It is
    # admitted only after the Cartesian product of group targets and private
    # task acceptance suites passes, proving the retained alternatives are
    # equivalent under all available task-specific evidence.
    source_to_pairs: dict[str, list[mixed.MixedPair]] = defaultdict(list)
    for pair in rebound:
        source_to_pairs[pair.source_sha256].append(pair)
    equivalent_source_groups = {
        digest: group for digest, group in source_to_pairs.items() if len(group) > 1
    }
    cross_pairs: list[mixed.MixedPair] = []
    for source_digest, group in sorted(equivalent_source_groups.items()):
        for target_pair in group:
            for test_pair in group:
                cross_pairs.append(
                    mixed._make_pair(  # noqa: SLF001
                        pair_id=(
                            "equivalence-cross::"
                            f"{target_pair.source_task_id}::to::{test_pair.source_task_id}"
                        ),
                        source_task_id=test_pair.source_task_id,
                        kind="verified_direct",
                        source=test_pair.source,
                        target=target_pair.target,
                        provenance=(
                            ("equivalent_source_sha256", source_digest),
                            ("target_origin_task_id", target_pair.source_task_id),
                            ("test_owner_task_id", test_pair.source_task_id),
                        ),
                    )
                )
    if cross_pairs:
        cross_verification = pass1._verify_all(  # noqa: SLF001
            cross_pairs,
            tests_by_id=tests_by_id,
            verify=verify,
            workers=verification_workers,
        )
    else:
        cross_verification = {
            "rows": 0,
            "passed": 0,
            "stability_runs": pass1.FULL_VERIFY_STABILITY_RUNS,
            "timeout_seconds": pass1.FULL_VERIFY_TIMEOUT_SECONDS,
            "diagnostics_persisted": False,
            "tests_model_visible": False,
        }
    equivalent_source_record = [
        {
            "source_sha256": digest,
            "task_ids": [pair.source_task_id for pair in group],
            "target_sha256s": [pair.target_sha256 for pair in group],
            "rows": len(group),
            "cross_acceptance_checks": len(group) ** 2,
        }
        for digest, group in sorted(equivalent_source_groups.items())
    ]

    verification = pass1._verify_all(  # noqa: SLF001
        rebound,
        tests_by_id=tests_by_id,
        verify=verify,
        workers=verification_workers,
    )
    rebound.sort(
        key=lambda pair: canonical_sha256(
            {
                "schema": DATASET_SCHEMA,
                "seed": seed,
                "pair_id": pair.pair_id,
                "source_sha256": pair.source_sha256,
                "target_sha256": pair.target_sha256,
            }
        )
    )

    category_counts = Counter(category_by_id.values())
    expected = {
        "pass1_225": EXPECTED_PASS1_ROWS,
        "pass2_209": EXPECTED_PASS2_ROWS,
        "kimi_c001": EXPECTED_C001_ROWS,
        "kimi_c002_prefix": EXPECTED_PREFIX_ROWS,
    }
    for category, count in expected.items():
        if category_counts.get(category, 0) != count:
            raise ValueError(f"fold union {category} count differs")
    c002_rows = category_counts.get("kimi_c002_tail", 0)
    if not 0 <= c002_rows <= MAX_C002_ROWS or len(rebound) != BASE_ROWS + c002_rows:
        raise ValueError("fold union c002 late-bound count differs")

    exact_gold = sum(
        pair.target.strip() == gold_target_by_id[pair.source_task_id].strip()
        for pair in rebound
    )
    schedule = [
        {
            "position": position,
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "kind": "verified_direct",
            "source_category": category_by_id[pair.source_task_id],
            "source_sha256": pair.source_sha256,
            "target_sha256": pair.target_sha256,
            "provenance": dict(pair.provenance),
        }
        for position, pair in enumerate(rebound)
    ]
    manifest = {
        "schema": DATASET_SCHEMA,
        "rows": len(rebound),
        "architecture": "native_encoder_decoder",
        "arm": "B_fold_only",
        "fresh_branch_from": "typed_sft_optstep348",
        "composition": {
            "verified_direct": len(rebound),
            "pass1_225": EXPECTED_PASS1_ROWS,
            "pass2_209": EXPECTED_PASS2_ROWS,
            "kimi_c001": EXPECTED_C001_ROWS,
            "kimi_c002_prefix": EXPECTED_PREFIX_ROWS,
            "kimi_c002_tail": c002_rows,
            "gold_replay": 0,
            "repair_conditioned": 0,
            "reasoning_rows": 0,
            "independently_generated_exact_gold_matches": exact_gold,
        },
        "row_count_policy": "exact_447_plus_late_bound_c002_tail_T_0_to_47",
        "source_audits": dict(source_audits),
        "typed_train": dict(typed_manifest),
        "heldout_identity_audit": dict(heldout_record),
        "heldout_overlap": 0,
        "heldout_175_model_visible": False,
        "known_contaminant_excluded": pass1.CONTAMINATED_TRAIN_TASK_ID,
        "task_id_deduplication": "reject_any_duplicate_across_all_source_cohorts",
        "target_code_deduplication": "none_retain_same_code_for_distinct_tasks",
        "shared_target_code_groups": len(shared_groups),
        "shared_target_code_rows": sum(len(ids) for ids in shared_groups.values()),
        "shared_target_code_groups_sha256": canonical_sha256(shared_groups),
        "equivalent_typed_source_policy": (
            "retain_distinct_tasks_only_after_all_targets_cross_pass_all_private_suites"
        ),
        "equivalent_typed_source_groups": len(equivalent_source_record),
        "equivalent_typed_source_rows": sum(
            int(group["rows"]) for group in equivalent_source_record
        ),
        "equivalent_typed_source_group_records": equivalent_source_record,
        "equivalent_typed_source_groups_sha256": canonical_sha256(
            equivalent_source_record
        ),
        "equivalent_typed_source_cross_acceptance": cross_verification,
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "tests_model_visible": False,
        "private_feedback_model_visible": False,
        "repair_conditioned_prefixes_visible": False,
        "reasoning_model_visible": False,
        "gold_implementation_model_visible": False,
        "gold_replay": {"selected_rows": 0, "forbidden": True},
        "full_acceptance_reverification": verification,
        "schedule": schedule,
        "schedule_sha256": canonical_sha256(schedule),
        "task_ids_sha256": canonical_sha256(
            [pair.source_task_id for pair in rebound]
        ),
        "pair_ids_sha256": canonical_sha256([pair.pair_id for pair in rebound]),
        "source_sha256s_sha256": canonical_sha256(
            [pair.source_sha256 for pair in rebound]
        ),
        "target_sha256s_sha256": canonical_sha256(
            [pair.target_sha256 for pair in rebound]
        ),
        "production_floor_eligible": True,
        "automatic_promotion_permitted": False,
        "promotion_status": "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS",
        "verpo_status": "HOLD",
    }
    return rebound, manifest


def build_typed_fold_union_pairs(
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
    verify: VerifyFn | None = None,
    verification_workers: int = pass1.FULL_VERIFY_WORKERS,
) -> tuple[list[mixed.MixedPair], dict[str, Any]]:
    del warmstart
    if (
        gold_replay_ratio != 0.0
        or gold_replay_rows != 0
        or min_verified_direct_targets != BASE_ROWS
        or min_repair_conditioned_targets != 0
        or allow_exploratory_inputs
        or require_local_production_floor
    ):
        raise ValueError("typed fold accepts only exact direct-only/no-replay Arm B")

    local = _require_layout(local_reports, LOCAL_BASENAMES, "fold local input")
    api = _require_layout(api_reports, API_BASENAMES, "fold API input")

    # Reconstruct pass 1 solely from its original producer reports.
    pass1_pairs, pass1_manifest = pass1.build_typed_direct_pairs(
        gold_train_jsonl=gold_train_jsonl,
        gold_f2_jsonl=gold_f2_jsonl,
        expected_gold_train_sha256=expected_gold_train_sha256,
        expected_gold_f2_sha256=expected_gold_f2_sha256,
        expected_gold_rows=expected_gold_rows,
        heldout_jsonl=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        local_reports=local[PASS1_LOCAL_SLICE],
        api_reports=api[PASS1_API_SLICE],
        warmstart=pass1._source_sft_identity(),  # noqa: SLF001
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=EXPECTED_PASS1_ROWS,
        min_repair_conditioned_targets=0,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        seed=seed,
        verify=verify or pass1._runtime_verify,  # noqa: SLF001
        verification_workers=verification_workers,
    )
    if len(pass1_pairs) != EXPECTED_PASS1_ROWS:
        raise ValueError("pass-1 reconstruction is not exactly 225 rows")
    pass1_seal = _require_manifest_reconstruction(
        pass1_manifest, local[PASS1_MANIFEST_POSITION], "pass-1"
    )

    # Reconstruct pass 2 from its local/API source evidence.  The pass-1
    # manifest is supplied only to its historical exclusion audit.
    pass2_pairs, pass2_manifest = pass2.build_typed_direct_pass2_pairs(
        gold_train_jsonl=gold_train_jsonl,
        gold_f2_jsonl=gold_f2_jsonl,
        expected_gold_train_sha256=expected_gold_train_sha256,
        expected_gold_f2_sha256=expected_gold_f2_sha256,
        expected_gold_rows=expected_gold_rows,
        heldout_jsonl=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        local_reports=(*local[PASS2_LOCAL_SLICE], local[PASS1_MANIFEST_POSITION]),
        api_reports=api[PASS2_API_SLICE],
        warmstart=pass1._source_sft_identity(),  # noqa: SLF001
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=pass2.EXPECTED_LOCAL_NEW_ROWS,
        min_repair_conditioned_targets=0,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        seed=seed,
    )
    if len(pass2_pairs) != EXPECTED_PASS2_ROWS:
        raise ValueError("pass-2 reconstruction is not exactly 209 rows")
    pass2_seal = _require_manifest_reconstruction(
        pass2_manifest, local[PASS2_MANIFEST_POSITION], "pass-2"
    )

    continuation = pass3._require_sources(api[CONTINUATION_SLICE])  # noqa: SLF001
    c001_rows, c001_audit = pass3._audit_c001(*continuation[0:3])  # noqa: SLF001
    c002_rows, c002_audit = pass3._audit_resume47(*continuation[3:6])  # noqa: SLF001
    prefix_rows, prefix_audit = pass3._audit_prefix(*continuation[6:9])  # noqa: SLF001

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
    typed_by_id = {pair.task_id: pair for pair in typed_pairs}

    def continuation_pair(category: str, row: Mapping[str, Any]) -> mixed.MixedPair:
        task_id = str(row.get("task_id") or "")
        canonical = typed_by_id.get(task_id)
        target = str(row.get("dart_source") or "")
        if canonical is None or row.get("source_sha256") != canonical.source_sha256:
            raise ValueError(f"{category} target typed-source binding differs")
        return mixed._make_pair(  # noqa: SLF001
            pair_id=f"{task_id}::source::{category}",
            source_task_id=task_id,
            kind="verified_direct",
            source=canonical.source,
            target=target,
            provenance=(
                ("source_category", category),
                ("source_sha256", canonical.source_sha256),
            ),
        )

    collected = [
        *(("pass1_225", pair) for pair in pass1_pairs),
        *(("pass2_209", pair) for pair in pass2_pairs),
        *(("kimi_c001", continuation_pair("kimi_c001", row)) for row in c001_rows),
        *(("kimi_c002_tail", continuation_pair("kimi_c002_tail", row)) for row in c002_rows),
        *(("kimi_c002_prefix", continuation_pair("kimi_c002_prefix", row)) for row in prefix_rows),
    ]

    train_rows = base_sft._read_jsonl(gold_train_jsonl)  # noqa: SLF001
    tests_by_id: dict[str, str] = {}
    gold_target_by_id: dict[str, str] = {}
    for index, row in enumerate(train_rows):
        task_id = base_sft._identity(row, index)  # noqa: SLF001
        if task_id == pass1.CONTAMINATED_TRAIN_TASK_ID:
            continue
        tests_by_id[task_id] = pass1._complete_tests(row, task_id)  # noqa: SLF001
        gold_target_by_id[task_id] = base_sft._target_source(row, task_id)  # noqa: SLF001
    heldout_ids, heldout_record = mixed._load_heldout_ids(  # noqa: SLF001
        heldout_jsonl,
        expected_sha256=expected_heldout_sha256,
        expected_rows=expected_heldout_rows,
    )
    return _rebind_union(
        collected=collected,
        typed_pairs=typed_pairs,
        heldout_ids=heldout_ids,
        tests_by_id=tests_by_id,
        gold_target_by_id=gold_target_by_id,
        seed=seed,
        source_audits={
            "pass1_reconstruction": pass1_seal,
            "pass2_reconstruction": pass2_seal,
            "kimi_c001": c001_audit,
            "kimi_c002_tail": c002_audit,
            "kimi_c002_prefix": prefix_audit,
        },
        typed_manifest=typed_manifest,
        heldout_record=heldout_record,
        verify=verify or pass1._runtime_verify,  # noqa: SLF001
        verification_workers=verification_workers,
    )


def _profile_runtime_contract() -> dict[str, str]:
    record = dict(_MIXED_RUNTIME_CONTRACT())
    record["mixed_training_engine_sha256"] = record["trainer_sha256"]
    record["trainer_sha256"] = base_sft.sha256_file(Path(__file__).resolve())
    record["typed_source_builder_sha256"] = base_sft.sha256_file(
        Path(typed_sft.__file__).resolve()
    )
    record["pass1_reconstruction_validator_sha256"] = base_sft.sha256_file(
        Path(pass1.__file__).resolve()
    )
    record["pass2_reconstruction_validator_sha256"] = base_sft.sha256_file(
        Path(pass2.__file__).resolve()
    )
    record["continuation_validator_sha256"] = base_sft.sha256_file(
        Path(pass3.__file__).resolve()
    )
    record["trainer_profile"] = "typed_fold_only_union_arm_b_v1"
    return record


def _validate_profile_args(args: argparse.Namespace) -> None:
    expected = {
        "gold_replay_ratio": 0.0,
        "gold_replay_rows": 0,
        "min_verified_direct_targets": BASE_ROWS,
        "min_repair_conditioned_targets": 0,
        "expected_warmstart_update": 348,
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_updates": 0,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.0,
        "seed": 42,
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        matches = (
            math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float)
            else observed == wanted
        )
        if not matches:
            raise ValueError(f"typed fold Arm B fixes --{name}={wanted}, observed={observed}")
    if args.allow_exploratory_inputs or args.require_local_production_floor:
        raise ValueError("typed fold Arm B requires sealed aggregate inputs")
    if len(args.local_report) != LOCAL_REPORT_COUNT or len(args.api_report) != API_REPORT_COUNT:
        raise ValueError("typed fold Arm B requires 9 local and 19 API artifacts")


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
    mixed.RUN_SCHEMA = RUN_SCHEMA
    mixed.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    mixed.DATASET_SCHEMA = DATASET_SCHEMA
    mixed.build_mixed_pairs = build_typed_fold_union_pairs
    mixed.validate_warmstart = pass1.validate_typed_warmstart
    mixed._runtime_contract = _profile_runtime_contract  # noqa: SLF001
    try:
        return mixed.train(args)
    finally:
        mixed.RUN_SCHEMA = originals["run_schema"]
        mixed.CHECKPOINT_SCHEMA = originals["checkpoint_schema"]
        mixed.DATASET_SCHEMA = originals["dataset_schema"]
        mixed.build_mixed_pairs = originals["builder"]
        mixed.validate_warmstart = originals["warmstart"]
        mixed._runtime_contract = originals["runtime"]  # noqa: SLF001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = mixed.parse_args(argv)
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
