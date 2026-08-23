#!/usr/bin/env python3
"""Build sealed direct-compact Monte Carlo sequence forward-KL/NLL data.

Every completion-attested, nonempty K=8 teacher final-content draw is joined
back to the exact compact source row through the hash-bound API-readable prompt
artifact. The only target transform is deterministic outer-whitespace trimming.
Every draw is emitted once and therefore receives equal per-draw sequence-NLL
weight. No correctness, confidence, parseability, or logprob filter is applied.
An optional deterministic gold replay fraction may be mixed in as ordinary
unit-weight SFT rows.

This is sampled-sequence Monte Carlo forward-KL/NLL, not dense token KL. The
verified-only RS-SFT artifact is deliberately not consumed here.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.direct_compact_causal import (  # noqa: E402
    CONTRACT_SCHEMA_V3,
    JOIN_SEAL_SCHEMA_V1,
    JOIN_SEAL_SCHEMA_V2,
    POOL_ALIGNMENT_SCHEMA_V1,
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    AUDIT_SCHEMA,
    DEFAULT_MODEL,
    PARSEABLE_SCHEMA,
    ArtifactError,
    JournalState,
    StudentTokenizerBinding,
    atomic_write_json,
    atomic_write_jsonl,
    backend_identity_sha256,
    file_record,
    load_verified_prompt_rows,
    read_jsonl,
    sha256_text,
    stable_sha256,
    validate_mc_teacher_sampling,
    validate_qwen38_sequence_sampling,
    target_length_evidence,
)


BUILD_SCHEMA = "direct-compact-mc-sequence-forward-kl-nll-build-v1"
SCHEDULE_SCHEMA = "direct-compact-mc-sequence-forward-kl-nll-schedule-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_FIELDS = ("supervised_target", "dart_source", "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--compact-train-jsonl", required=True, type=Path)
    parser.add_argument("--compact-train-seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--teacher-parseable-jsonl", required=True, type=Path)
    parser.add_argument("--expected-teacher-parseable-sha256", required=True)
    parser.add_argument("--teacher-journal", required=True, type=Path)
    parser.add_argument("--expected-teacher-journal-sha256", required=True)
    parser.add_argument("--teacher-audit-json", required=True, type=Path)
    parser.add_argument("--expected-teacher-audit-sha256", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-seal", required=True, type=Path)
    parser.add_argument("--schedule-output", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument(
        "--student-tokenizer-json",
        type=Path,
        help="Required for a v3 contract so the repeated pool-use seal is exact.",
    )
    parser.add_argument(
        "--expected-student-tokenizer-sha256",
        default="",
        help="Required with --student-tokenizer-json.",
    )
    parser.add_argument(
        "--gold-replay-fraction",
        type=float,
        default=0.0,
        help="Target fraction of final rows that are original gold SFT rows.",
    )
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def require_file_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    normalized = expected.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ArtifactError(f"{label} expected hash is not lowercase SHA-256")
    if record["sha256"] != normalized:
        raise ArtifactError(
            f"{label} hash mismatch: expected {normalized}, got {record['sha256']}"
        )
    return record


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: expected a JSON object")
    return value


def compact_ids_sha256(row: Mapping[str, Any], identity: str) -> str:
    raw = row.get("compact_input_ids")
    if (
        not isinstance(raw, list)
        or not raw
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
    ):
        raise ArtifactError(f"{identity}: invalid compact_input_ids")
    return stable_sha256([int(value) for value in raw])


def target_text(row: Mapping[str, Any], identity: str) -> str:
    for field in TARGET_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ArtifactError(f"{identity}: compact train row has no gold target")


def replace_target(row: Mapping[str, Any], code: str) -> dict[str, Any]:
    result = dict(row)
    present = [field for field in TARGET_FIELDS if field in result]
    result["dart_source"] = code
    for field in present:
        result[field] = code
    return result


def load_student_tokenizer(
    contract: DirectCompactContract,
    path: Path | None,
    expected_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    if path is None:
        raise ArtifactError(
            "--student-tokenizer-json is required for target-length validation"
        )
    record = require_file_hash(
        path, expected_sha256, "student tokenizer"
    )
    if record["sha256"] != contract.tokenizer_json_sha256:
        raise ArtifactError("student tokenizer does not match compact contract")
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - dependency failure
        raise ArtifactError("the tokenizers package is required for v3") from exc
    return Tokenizer.from_file(record["path"]), record


def exact_output_seal(
    *,
    output_path: Path,
    contract_path: Path,
    contract: DirectCompactContract,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any | None,
) -> dict[str, Any]:
    seal: dict[str, Any] = {
        "schema": (
            JOIN_SEAL_SCHEMA_V2
            if contract.schema == CONTRACT_SCHEMA_V3
            else JOIN_SEAL_SCHEMA_V1
        ),
        "contract_schema": contract.schema,
        "selected_role": "fit",
        "rows": len(rows),
        "source_rows": len(rows),
        "skipped_rows": 0,
        "output_sha256": sha256_file(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "contract_sha256": sha256_file(contract_path),
        "model_visible_fields": sorted(
            set().union(*(set(row) for row in rows))
        ),
        "withheld_from_model": [
            "teacher request IDs and logprobs",
            "teacher confidence",
            "hidden verifier tests",
            "verification results",
            "teacher/gold schedule labels",
        ],
    }
    if contract.schema == CONTRACT_SCHEMA_V3:
        if tokenizer is None:
            raise AssertionError("v3 output seal requires tokenizer")
        pool_rows: list[dict[str, Any]] = []
        total_uses = 0
        for index, row in enumerate(rows):
            compact_ids = [int(value) for value in row["compact_input_ids"]]
            pool = contract.validate_v3_pool_payload(
                compact_ids, tokenizer, f"output-row-{index}"
            )
            use_count = len(pool["uses"])
            total_uses += use_count
            pool_rows.append(
                {
                    "row": index,
                    "compact_ids_sha256": stable_sha256(compact_ids),
                    "use_count": use_count,
                    "source_blind": True,
                }
            )
        seal["pool_metadata"] = {
            "schema": POOL_ALIGNMENT_SCHEMA_V1,
            "rows": len(rows),
            "source_blind_rows": len(rows),
            "target_function": contract.target_function,
            "projection_sha256": stable_sha256(pool_rows),
            "total_use_count": total_uses,
        }
    return seal


def _find_audit_shard(
    audit: Mapping[str, Any], teacher_record: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [
        shard
        for shard in audit.get("homogeneous_backend_shards") or []
        if isinstance(shard, Mapping)
        and (shard.get("parseable_output") or {}).get("sha256")
        == teacher_record["sha256"]
    ]
    if len(matches) != 1:
        raise ArtifactError(
            "teacher parseable artifact is not exactly one sealed audit shard"
        )
    return dict(matches[0])


def _gold_count(teacher_count: int, requested_fraction: float) -> int:
    if not 0.0 <= requested_fraction < 1.0:
        raise ArtifactError("--gold-replay-fraction must be in [0, 1)")
    if teacher_count <= 0:
        raise ArtifactError("teacher parseable artifact contains no rows")
    if requested_fraction == 0.0:
        return 0
    exact = teacher_count * requested_fraction / (1.0 - requested_fraction)
    return int(math.floor(exact + 0.5))


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.compact_train_jsonl.expanduser().resolve()
    train_seal_path = args.compact_train_seal.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    for path in (train_path, train_seal_path, contract_path):
        if not path.is_file():
            raise ArtifactError(f"required artifact does not exist: {path}")

    contract = DirectCompactContract.load(contract_path)
    tokenizer, tokenizer_record = load_student_tokenizer(
        contract,
        (
            args.student_tokenizer_json.expanduser().resolve()
            if args.student_tokenizer_json
            else None
        ),
        args.expected_student_tokenizer_sha256,
    )
    header_target_contract: Mapping[str, Any] | None = None
    source_seal = validate_join_seal(
        train_path, train_seal_path, contract_path, expected_role="fit"
    )
    train_rows = read_jsonl(train_path)
    if len(train_rows) != int(source_seal["rows"]):
        raise ArtifactError("compact train row count changed after seal validation")
    train_by_hash: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(train_rows):
        identity = f"compact-train-row-{index}"
        contract.validate_row(row, identity)
        target_text(row, identity)
        digest = compact_ids_sha256(row, identity)
        train_by_hash.setdefault(digest, []).append((index, row))

    prompt_path = args.prompt_jsonl.expanduser().resolve()
    prompts, prompt_record = load_verified_prompt_rows(
        prompt_path, expected_sha256=args.expected_prompt_sha256
    )
    raw_prompts = read_jsonl(prompt_path)
    if len(raw_prompts) != len(prompts):
        raise AssertionError("verified prompt loader changed row count")
    prompt_by_task: dict[str, dict[str, Any]] = {}
    compact_row_by_task: dict[str, tuple[int, dict[str, Any]]] = {}
    for prompt, raw in zip(prompts, raw_prompts, strict=True):
        expected_compact = str(raw.get("compact_ids_sha256") or "")
        if not SHA256_RE.fullmatch(expected_compact):
            raise ArtifactError(
                f"prompt {prompt.task_id} lacks compact_ids_sha256"
            )
        candidates = train_by_hash.get(expected_compact, [])
        if len(candidates) > 1:
            exact_task_matches = [
                item
                for item in candidates
                if str(item[1].get("task_id") or item[1].get("id") or "")
                == prompt.task_id
            ]
            candidates = exact_task_matches
        if len(candidates) != 1:
            raise ArtifactError(
                f"prompt {prompt.task_id} does not map bijectively to one exact "
                f"compact train row (matches={len(candidates)})"
            )
        row_index, source_row = candidates[0]
        if compact_ids_sha256(source_row, prompt.task_id) != expected_compact:
            raise AssertionError("compact prompt join hash drifted")
        prompt_by_task[prompt.task_id] = raw
        compact_row_by_task[prompt.task_id] = (row_index, source_row)

    journal_path = args.teacher_journal.expanduser().resolve()
    journal_record = require_file_hash(
        journal_path,
        args.expected_teacher_journal_sha256,
        "teacher journal",
    )
    journal = JournalState.load(journal_path)
    if journal.header is None:
        raise ArtifactError("teacher journal has no run header")
    header_payload = journal.header.get("payload") or {}
    if (
        header_payload.get("requested_model") != DEFAULT_MODEL
        or header_payload.get("returned_model_must_equal_requested") is not True
    ):
        raise ArtifactError(
            "teacher journal is not sealed to the exact "
            f"{DEFAULT_MODEL} model identity"
        )
    seed_contract = header_payload.get("sampling_seed_contract")
    if (
        not isinstance(seed_contract, Mapping)
        or seed_contract.get("unique_seed_required_per_task_draw") is not True
        or seed_contract.get("provider_seed_honor_not_assumed") is not True
        or seed_contract.get(
            "response_seed_echo_required_to_attest_honor"
        )
        is not True
    ):
        raise ArtifactError(
            "teacher journal lacks the fail-closed requested-seed contract"
        )
    generation_parameters = header_payload.get("generation_parameters")
    if not isinstance(generation_parameters, Mapping):
        raise ArtifactError("teacher journal has no sealed generation parameters")
    validate_mc_teacher_sampling(generation_parameters)
    objective_mode = str(
        header_payload.get("objective_mode") or "require_top5"
    )
    if objective_mode == "sequence_only":
        validate_qwen38_sequence_sampling(
            str(header_payload.get("requested_model") or ""),
            generation_parameters,
        )
    header_target_contract = header_payload.get("target_length_contract")
    if not isinstance(header_target_contract, Mapping):
        raise ArtifactError("teacher journal lacks the target-length contract")
    if (
        (header_target_contract.get("trainer_contract") or {}).get("sha256")
        != sha256_file(contract_path)
        or int(header_target_contract.get("max_target_tokens", -1))
        != contract.max_target_tokens
        or (header_target_contract.get("student_tokenizer") or {}).get("sha256")
        != tokenizer_record["sha256"]
    ):
        raise ArtifactError(
            "teacher target-length gate differs from the supplied "
            "trainer/tokenizer contract"
        )
    tokenizer_binding = StudentTokenizerBinding(
        tokenizer,
        eos_token_id=int(
            header_target_contract.get("student_eos_token_id", -1)
        ),
        tokenizer_record=tokenizer_record,
    )
    header_prompt = header_payload.get("prompt_artifact") or {}
    if header_prompt.get("sha256") != prompt_record["sha256"]:
        raise ArtifactError("teacher journal was collected from a different prompt file")
    header_prompt_manifest = header_payload.get("prompt_manifest")
    f2_prompt_contract = header_payload.get("f2_prompt_contract")
    if (
        not isinstance(header_prompt_manifest, Mapping)
        or not SHA256_RE.fullmatch(
            str(header_prompt_manifest.get("sha256") or "")
        )
        or not isinstance(f2_prompt_contract, Mapping)
        or f2_prompt_contract.get("representation_schema")
        != "lossless-semantic-f2"
        or not SHA256_RE.fullmatch(
            str(f2_prompt_contract.get("system_prompt_sha256") or "")
        )
        or any(
            prompt.system_prompt_sha256
            != f2_prompt_contract.get("system_prompt_sha256")
            for prompt in prompts
        )
    ):
        raise ArtifactError("teacher journal lacks the sealed F2 prompt contract")
    if set(header_payload.get("task_ids") or []) != set(prompt_by_task):
        raise ArtifactError("teacher journal task set differs from prompt artifact")

    audit_path = args.teacher_audit_json.expanduser().resolve()
    audit_record = require_file_hash(
        audit_path, args.expected_teacher_audit_sha256, "teacher audit"
    )
    audit = strict_json(audit_path)
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ArtifactError("teacher audit schema mismatch")
    if (audit.get("journal") or {}).get("sha256") != journal_record["sha256"]:
        raise ArtifactError("teacher audit does not bind the supplied journal")
    chain_head_path = Path(str(journal_path) + ".chain-head.json")
    if (
        not chain_head_path.is_file()
        or audit.get("journal_chain_head") != file_record(chain_head_path)
    ):
        raise ArtifactError(
            "teacher audit does not bind the durable journal chain head"
        )
    if (
        audit.get("prompt_manifest") != header_prompt_manifest
        or audit.get("f2_prompt_contract") != f2_prompt_contract
    ):
        raise ArtifactError("teacher audit changed the F2 prompt contract")
    readiness = audit.get("production_readiness") or {}
    if readiness.get("mc_sequence_forward_kl_nll") is not True:
        raise ArtifactError("teacher audit is not production-ready for sequence NLL")
    if (audit.get("capabilities") or {}).get("dense_full_vocabulary_kl") is not False:
        raise ArtifactError("teacher audit makes an invalid dense-KL claim")
    sampling_audit = audit.get("sampling")
    if (
        not isinstance(sampling_audit, Mapping)
        or sampling_audit.get("provider_seed_honor_assumed") is not False
        or sampling_audit.get("duplicates_filtered_from_sequence_training")
        is not False
        or sampling_audit.get(
            "pathological_all_tasks_have_identical_k8_draws"
        )
        is not False
    ):
        raise ArtifactError("teacher audit has an invalid sampling/diversity seal")
    objective_mode = str(audit.get("objective_mode") or "require_top5")
    if objective_mode not in {"require_top5", "sequence_only"}:
        raise ArtifactError("teacher audit has an unsupported objective mode")
    if (
        objective_mode == "sequence_only"
        and float(args.gold_replay_fraction) != 0.0
    ):
        raise ArtifactError(
            "sequence_only must be a pure teacher-distribution objective; "
            "gold adaptation is a separate initialization stage"
        )
    header_mode = str(header_payload.get("objective_mode") or "require_top5")
    if header_mode != objective_mode:
        raise ArtifactError("teacher audit objective mode differs from journal header")
    target_gate = audit.get("target_length_gate")
    if (
        not isinstance(target_gate, Mapping)
        or target_gate.get("passed") is not True
        or int(target_gate.get("targets_checked", -1))
        != int((audit.get("coverage") or {}).get("candidates", -2))
        or int(target_gate.get("overflow_count", -1)) != 0
        or (
            objective_mode == "require_top5"
            and int(target_gate.get("non_code_target_count", -1)) != 0
        )
        or target_gate.get("final_dart_code_only_required")
        != (objective_mode == "require_top5")
        or not SHA256_RE.fullmatch(
            str(target_gate.get("evidence_sha256") or "")
        )
        or target_gate.get("target_contract") != header_target_contract
        or (target_gate.get("failure_policy") or {}).get("truncate") is not False
        or (target_gate.get("failure_policy") or {}).get("filter_draw") is not False
        or (target_gate.get("failure_policy") or {}).get("resample_draw") is not False
        or (target_gate.get("failure_policy") or {}).get("continue_to_gpu")
        is not False
    ):
        raise ArtifactError(
            "teacher audit target-length/code-only gate is not passed"
        )

    teacher_path = args.teacher_parseable_jsonl.expanduser().resolve()
    teacher_record = require_file_hash(
        teacher_path,
        args.expected_teacher_parseable_sha256,
        "teacher parseable artifact",
    )
    audit_shard = _find_audit_shard(audit, teacher_record)
    shard_identity_sha = str(audit_shard["backend_identity_sha256"])
    teacher_rows = read_jsonl(teacher_path)
    expected_sequence_rows = int(
        audit_shard.get("sequence_rows", audit_shard.get("parseable_rows", -1))
    )
    if len(teacher_rows) != expected_sequence_rows:
        raise ArtifactError("teacher sequence row count differs from audit shard")

    seen_candidates: set[str] = set()
    expected_candidates = {
        candidate_id
        for candidate_id, candidate in journal.candidates.items()
        if backend_identity_sha256(candidate) == shard_identity_sha
        and candidate.get("completion_attested") is True
    }
    teacher_schedule: list[dict[str, Any]] = []
    for index, row in enumerate(teacher_rows):
        if row.get("schema") != PARSEABLE_SCHEMA:
            raise ArtifactError(f"teacher row {index} schema mismatch")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_candidates:
            raise ArtifactError(f"teacher row {index} has duplicate/missing candidate_id")
        seen_candidates.add(candidate_id)
        candidate = journal.candidates.get(candidate_id)
        if candidate is None:
            raise ArtifactError(f"teacher row {index} is absent from journal")
        task_id = str(row.get("task_id") or "")
        if task_id not in compact_row_by_task:
            raise ArtifactError(f"teacher row {index} task is outside prompt/train join")
        eligibility = row.get("eligibility") or {}
        if (
            eligibility.get("sft") is not True
            or eligibility.get("mc_sequence_forward_kl") is not True
            or eligibility.get("dense_full_vocabulary_kl") is not False
        ):
            raise ArtifactError(f"teacher row {index} eligibility contract failed")
        target = str(row.get("sequence_target") or "")
        target_sha256 = str(row.get("sequence_target_sha256") or "")
        if (
            not target
            or target != target.strip()
            or sha256_text(target) != target_sha256
        ):
            raise ArtifactError(f"teacher row {index} sequence target hash mismatch")
        observed_length = target_length_evidence(
            target,
            binding=tokenizer_binding,
            max_target_tokens=contract.max_target_tokens,
        )
        if (
            observed_length.get("within_contract") is not True
            or row.get("target_length_evidence") != observed_length
        ):
            raise ArtifactError(
                f"teacher row {index} target-length evidence mismatch: "
                f"task={task_id} draw={row.get('sample_index')} "
                f"tokens={observed_length['eos_inclusive_target_token_count']} "
                f"limit={contract.max_target_tokens}"
            )
        content_contract = row.get("target_content_contract")
        if (
            not isinstance(content_contract, Mapping)
            or content_contract.get("reasoning_excluded") is not True
            or (
                objective_mode == "require_top5"
                and content_contract.get("final_dart_code_only") is not True
            )
        ):
            raise ArtifactError(
                f"teacher row {index} violates the target-content contract"
            )
        parse = candidate.get("parse") or {}
        if (
            candidate.get("completion_attested") is not True
            or str((candidate.get("response") or {}).get("raw_content") or "").strip()
            != target
            or str(candidate.get("prompt_sha256") or "")
            != str(row.get("prompt_sha256") or "")
            or backend_identity_sha256(candidate) != shard_identity_sha
        ):
            raise ArtifactError(f"teacher row {index} journal binding mismatch")
        base_index, base_row = compact_row_by_task[task_id]
        teacher_schedule.append(
            {
                "kind": "teacher_draw",
                "task_id": task_id,
                "candidate_id": candidate_id,
                "sample_index": int(row["sample_index"]),
                "base_row_index": base_index,
                "compact_ids_sha256": compact_ids_sha256(base_row, task_id),
                "target_sha256": target_sha256,
                "target": target,
                "draw_weight": 1.0,
            }
        )
    if seen_candidates != expected_candidates:
        missing = sorted(expected_candidates.difference(seen_candidates))
        extra = sorted(seen_candidates.difference(expected_candidates))
        raise ArtifactError(
            "teacher parseable shard is not the complete parseable journal subset: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    gold_count = _gold_count(
        len(teacher_schedule), float(args.gold_replay_fraction)
    )
    rng = random.Random(int(args.seed))
    gold_indices = list(range(len(train_rows)))
    rng.shuffle(gold_indices)
    gold_schedule: list[dict[str, Any]] = []
    for replay_index in range(gold_count):
        base_index = gold_indices[replay_index % len(gold_indices)]
        base_row = train_rows[base_index]
        gold = target_text(base_row, f"gold-row-{base_index}")
        gold_schedule.append(
            {
                "kind": "gold_replay",
                "task_id": str(
                    base_row.get("task_id")
                    or base_row.get("id")
                    or f"sealed-row-{base_index}"
                ),
                "candidate_id": None,
                "sample_index": None,
                "base_row_index": base_index,
                "compact_ids_sha256": compact_ids_sha256(
                    base_row, f"gold-row-{base_index}"
                ),
                "target_sha256": sha256_text(gold),
                "target": gold,
                "draw_weight": 1.0,
            }
        )

    schedule = teacher_schedule + gold_schedule
    rng.shuffle(schedule)
    output_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    for position, item in enumerate(schedule):
        base_row = train_rows[int(item["base_row_index"])]
        if item["kind"] == "teacher_draw":
            output_row = replace_target(base_row, str(item["target"]))
        else:
            output_row = dict(base_row)
        contract.validate_row(output_row, f"output-row-{position}")
        if compact_ids_sha256(output_row, f"output-row-{position}") != item[
            "compact_ids_sha256"
        ]:
            raise AssertionError("output compact source changed during target join")
        output_rows.append(output_row)
        schedule_rows.append(
            {
                "schema": SCHEDULE_SCHEMA,
                "position": position,
                **{key: value for key, value in item.items() if key != "target"},
            }
        )

    output_path = args.output_jsonl.expanduser().resolve()
    output_seal_path = args.output_seal.expanduser().resolve()
    schedule_path = args.schedule_output.expanduser().resolve()
    manifest_path = args.build_manifest.expanduser().resolve()
    atomic_write_jsonl(output_path, output_rows)
    atomic_write_jsonl(schedule_path, schedule_rows)
    output_seal = exact_output_seal(
        output_path=output_path,
        contract_path=contract_path,
        contract=contract,
        rows=output_rows,
        tokenizer=tokenizer,
    )
    atomic_write_json(output_seal_path, output_seal)
    validate_join_seal(
        output_path, output_seal_path, contract_path, expected_role="fit"
    )

    realized_gold_fraction = gold_count / len(output_rows)
    manifest = {
        "schema": BUILD_SCHEMA,
        "objective": {
            "name": "monte_carlo_sequence_forward_kl_nll",
            "display_name": "Monte Carlo sequence forward-KL/NLL",
            "sequence_monte_carlo_forward_kl_nll_primary": True,
            "implementation": (
                "equal-weight mean of EOS-inclusive summed causal NLL over "
                "K=8 independently sampled teacher final-content sequences "
                "after trim_outer_whitespace"
            ),
            "loss_reduction": (
                "sum_target_token_nll_within_each_sequence_then_mean_draws"
            ),
            "base_token_mean_loss_forbidden": True,
            "teacher_distribution": (
                "pushforward of the exactly configured teacher sampling "
                "distribution through trim_outer_whitespace"
            ),
            "teacher_sampling": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 101,
                "tempered": False,
                "truncated": False,
                "generation_parameters_sha256": stable_sha256(
                    generation_parameters
                ),
                "hidden_reasoning_tokens_enabled": bool(
                    (generation_parameters.get("extra_body") or {}).get(
                        "enable_thinking"
                    )
                ),
                "provider_seed_honor_attested": bool(
                    sampling_audit.get("provider_seed_honor_attested")
                ),
                "provider_seed_honor_assumed": False,
                "unique_final_sequences_per_task": dict(
                    sampling_audit.get("unique_final_sequences_per_task") or {}
                ),
                "duplicate_teacher_draws_retained": True,
                "pathological_all_tasks_have_identical_k8_draws": False,
            },
            "objective_mode": objective_mode,
            "all_k8_draws_required_and_emitted": True,
            "parseability_filtering": False,
            "correctness_filtering": False,
            "target_transform": "trim_outer_whitespace",
            "every_teacher_draw_emitted_exactly_once": True,
            "teacher_draw_weight": 1.0,
            "confidence_weighting": False,
            "teacher_logprob_weighting": False,
            "provider_top5_logprobs_required": (
                objective_mode == "require_top5"
            ),
            "content_logprob_prefix_fully_visible_to_student": bool(
                objective_mode == "require_top5"
                and (generation_parameters.get("extra_body") or {}).get(
                    "enable_thinking"
                )
                is False
            ),
            "sparse_top5_tail_auxiliary_eligible": bool(
                (audit.get("production_readiness") or {}).get(
                    "sparse_top5_plus_tail"
                )
            ),
            "dense_token_kl": False,
            "dense_full_vocabulary_kl": False,
            "verified_only_rs_sft_consumed": False,
            "gold_targets_mixed_into_sequence_objective": gold_count > 0,
        },
        "seed": int(args.seed),
        "gold_replay": {
            "requested_final_fraction": float(args.gold_replay_fraction),
            "realized_final_fraction": realized_gold_fraction,
            "rows": gold_count,
            "sampling": "deterministic_shuffled_cycle_over_sealed_gold_train_rows",
            "row_weight": 1.0,
            "required_zero_for_sequence_only": (
                objective_mode == "sequence_only"
            ),
        },
        "target_length_gate": {
            "passed": True,
            "teacher_rows_revalidated": len(teacher_schedule),
            "student_tokenizer": tokenizer_record,
            "student_eos_token_id": tokenizer_binding.eos_token_id,
            "trainer_contract": file_record(contract_path),
            "max_target_tokens": contract.max_target_tokens,
            "audit_evidence_sha256": target_gate["evidence_sha256"],
            "tokenization": {
                "add_special_tokens": False,
                "eos_policy": (
                    "append_exactly_once_if_final_token_is_not_eos"
                ),
                "truncation_permitted": False,
                "overflow_filtering_permitted": False,
                "overflow_resampling_permitted": False,
            },
            "final_dart_code_only_required": (
                objective_mode == "require_top5"
            ),
            "final_dart_code_only_rows": int(
                target_gate.get("targets_final_dart_code_only", 0)
            ),
            "non_code_target_rows_retained": int(
                target_gate.get("non_code_target_count", 0)
            ),
            "reasoning_content_excluded": True,
            "gpu_launch_authorized_only_after_this_manifest": True,
        },
        "counts": {
            "teacher_draw_rows": len(teacher_schedule),
            "gold_replay_rows": gold_count,
            "output_rows": len(output_rows),
            "unique_teacher_candidate_ids": len(seen_candidates),
        },
        "inputs": {
            "compact_train": file_record(train_path),
            "compact_train_seal": file_record(train_seal_path),
            "contract": file_record(contract_path),
            "prompt_artifact": prompt_record,
            "prompt_manifest": dict(header_prompt_manifest),
            "f2_prompt_contract": dict(f2_prompt_contract),
            "teacher_parseable": teacher_record,
            "teacher_journal": journal_record,
            "teacher_audit": audit_record,
            "student_tokenizer": tokenizer_record,
            "backend_identity_sha256": shard_identity_sha,
        },
        "outputs": {
            "dataset": file_record(output_path),
            "standard_direct_compact_seal": file_record(output_seal_path),
            "schedule": file_record(schedule_path),
        },
        "schedule_sha256": stable_sha256(schedule_rows),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build(args)
    print(
        "QWEN_SEQUENCE_KD_BUILD "
        f"teacher_draws={manifest['counts']['teacher_draw_rows']} "
        f"gold_replay={manifest['counts']['gold_replay_rows']} "
        f"rows={manifest['counts']['output_rows']} "
        f"dense_token_kl=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
