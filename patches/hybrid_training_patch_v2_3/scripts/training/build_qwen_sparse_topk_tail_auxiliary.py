#!/usr/bin/env python3
"""Attach a sealed coarsened top-k+tail KL auxiliary to sequence-SFT rows.

The input must already be the sealed direct-compact Monte Carlo sequence
forward-KL/NLL build. A teacher draw receives sparse auxiliary metadata only if:

* the raw response differs from the SFT target only by trailing whitespace;
* that trailing trim lands on an exact provider-token byte boundary;
* the retained chosen-token byte sequence maps to the exact local target IDs;
* every top-5 byte string maps to one unique local token ID; and
* every raw top-5 probability plus the unmodified residual tail is valid.

The API does not expose a final EOS distribution. EOS remains supervised only
by primary sequence NLL and is never synthesized for the sparse auxiliary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.direct_compact_causal import DirectCompactContract, validate_join_seal
from scripts.training.build_qwen_sequence_kd import exact_output_seal, target_text
from scripts.training.direct_compact_sparse_topk_tail import (
    PROBABILITY_TOLERANCE,
    SPARSE_FIELD,
    SPARSE_MANIFEST_SCHEMA,
    SPARSE_ROW_SCHEMA,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (
    AUDIT_SCHEMA,
    PARSEABLE_SCHEMA,
    ArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    read_jsonl,
    sha256_file,
    stable_sha256,
)


class IneligibleSparseDraw(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--sequence-train-jsonl", required=True, type=Path)
    parser.add_argument("--sequence-train-seal", required=True, type=Path)
    parser.add_argument("--sequence-schedule-jsonl", required=True, type=Path)
    parser.add_argument("--sequence-build-manifest", required=True, type=Path)
    parser.add_argument("--expected-sequence-build-manifest-sha256", required=True)
    parser.add_argument("--teacher-parseable-jsonl", required=True, type=Path)
    parser.add_argument("--expected-teacher-parseable-sha256", required=True)
    parser.add_argument("--teacher-audit-json", required=True, type=Path)
    parser.add_argument("--expected-teacher-audit-sha256", required=True)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--student-eos-token-id", required=True, type=int)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-seal", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--minimum-eligible-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of teacher draws that must support the auxiliary. "
            "Default 1.0 fails rather than silently reducing coverage."
        ),
    )
    return parser.parse_args()


def _expected_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected.strip().lower():
        raise ArtifactError(
            f"{label} hash mismatch: expected {expected}, got {record['sha256']}"
        )
    return record


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: expected a JSON object")
    return value


def _target(row: Mapping[str, Any], identity: str) -> str:
    return target_text(row, identity)


def _position_payload(
    teacher_position: Mapping[str, Any],
    audit_position: Mapping[str, Any],
    *,
    observed_id: int,
    vocab_size: int,
    eos_token_id: int,
    identity: str,
) -> dict[str, Any]:
    top = teacher_position.get("top_logprobs")
    mappings = audit_position.get("top_alternative_mappings")
    if not isinstance(top, list) or len(top) != 5:
        raise IneligibleSparseDraw(f"{identity}: provider did not return exact top-5")
    if not isinstance(mappings, list) or len(mappings) != len(top):
        raise IneligibleSparseDraw(f"{identity}: top-5 mapping audit is incomplete")
    top_ids: list[int] = []
    top_logprobs: list[float] = []
    for index, (entry, mapping) in enumerate(zip(top, mappings, strict=True)):
        if not isinstance(entry, Mapping) or not isinstance(mapping, Mapping):
            raise IneligibleSparseDraw(f"{identity}: malformed top-5 entry")
        token_id = mapping.get("student_token_id")
        if (
            mapping.get("mapping_error") is not None
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
        ):
            raise IneligibleSparseDraw(
                f"{identity}: top-5 token {index} lacks exact shared-token mapping"
            )
        logprob = entry.get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise IneligibleSparseDraw(f"{identity}: top-5 logprob is not numeric")
        logprob = float(logprob)
        if not math.isfinite(logprob) or logprob > 0:
            raise IneligibleSparseDraw(f"{identity}: invalid top-5 logprob")
        top_ids.append(int(token_id))
        top_logprobs.append(logprob)
    if len(set(top_ids)) != len(top_ids):
        raise IneligibleSparseDraw(
            f"{identity}: shared tokenizer maps top-5 to duplicate IDs"
        )
    tail = audit_position.get("tail_probability_mass_raw")
    if isinstance(tail, bool) or not isinstance(tail, (int, float)):
        raise IneligibleSparseDraw(f"{identity}: missing raw residual tail")
    tail = float(tail)
    if not math.isfinite(tail) or tail < 0.0 or tail > 1.0:
        raise IneligibleSparseDraw(
            f"{identity}: raw residual tail is outside [0,1]"
        )
    total = math.fsum(math.exp(value) for value in top_logprobs) + tail
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise IneligibleSparseDraw(
            f"{identity}: top-5 plus raw tail sums to {total:.17g}"
        )
    if observed_id == eos_token_id:
        raise IneligibleSparseDraw(f"{identity}: chosen content token is EOS")
    return {
        "observed_token_id": int(observed_id),
        "top_token_ids": top_ids,
        "top_logprobs": top_logprobs,
        "tail_probability_mass": tail,
    }


def _sparse_payload(
    teacher_row: Mapping[str, Any],
    target: str,
    *,
    tokenizer: Any,
    vocab_size: int,
    eos_token_id: int,
) -> dict[str, Any]:
    candidate_id = str(teacher_row.get("candidate_id") or "")
    identity = candidate_id[:12] or "teacher-row"
    if teacher_row.get("schema") != PARSEABLE_SCHEMA:
        raise ArtifactError(f"{identity}: teacher row schema mismatch")
    raw_content = teacher_row.get("raw_content")
    if (
        not isinstance(raw_content, str)
        or not target
        or raw_content.strip() != target
    ):
        raise IneligibleSparseDraw(
            f"{identity}: response normalization is not trim_outer_whitespace"
        )
    mapping_audit = teacher_row.get("student_token_mapping_audit")
    if not isinstance(mapping_audit, Mapping):
        raise IneligibleSparseDraw(f"{identity}: no student mapping audit")
    summary = mapping_audit.get("summary") or {}
    if summary.get("chosen_bytes_reconstruct_raw_content") is not True:
        raise IneligibleSparseDraw(f"{identity}: chosen bytes do not reconstruct content")
    if summary.get("chosen_mapping_complete") is not True:
        raise IneligibleSparseDraw(f"{identity}: chosen-token mapping is incomplete")
    if summary.get("top_mapping_complete") is not True:
        raise IneligibleSparseDraw(f"{identity}: top-5 mapping is incomplete")
    if summary.get("top5_count_complete") is not True:
        raise IneligibleSparseDraw(f"{identity}: provider top-5 count is incomplete")
    if summary.get("materially_negative_tail_positions") != 0:
        raise IneligibleSparseDraw(f"{identity}: materially negative tail observed")
    if summary.get("logged_eos_covered") is True:
        raise IneligibleSparseDraw(
            f"{identity}: provider EOS cannot be aligned as content bytes"
        )

    teacher_tokens = teacher_row.get("chosen_tokens_with_top_logprobs")
    audit_positions = mapping_audit.get("tokens")
    if (
        not isinstance(teacher_tokens, list)
        or not isinstance(audit_positions, list)
        or len(teacher_tokens) != len(audit_positions)
    ):
        raise IneligibleSparseDraw(
            f"{identity}: teacher token and mapping-audit sequences do not align"
        )

    # Sequence NLL deliberately trims outer whitespace. Sparse token KL is
    # stricter: removing leading teacher tokens would change every subsequent
    # conditioning prefix, so only a trailing trim can preserve the exact
    # teacher/student prefix. That trailing trim must also remove whole
    # provider tokens; slicing through one destroys the categorical event.
    left_trimmed = raw_content.lstrip()
    target_start_character = len(raw_content) - len(left_trimmed)
    target_end_character = target_start_character + len(target)
    if raw_content[target_start_character:target_end_character] != target:
        raise IneligibleSparseDraw(
            f"{identity}: trim target character interval is ambiguous"
        )
    target_start_byte = len(
        raw_content[:target_start_character].encode("utf-8")
    )
    target_end_byte = len(
        raw_content[:target_end_character].encode("utf-8")
    )
    raw_bytes = raw_content.encode("utf-8")
    boundaries = [0]
    reconstructed_parts: list[bytes] = []
    for position_index, teacher_position in enumerate(teacher_tokens):
        if not isinstance(teacher_position, Mapping):
            raise IneligibleSparseDraw(
                f"{identity}:{position_index}: malformed chosen token"
            )
        token_bytes = teacher_position.get("bytes")
        if (
            not isinstance(token_bytes, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in token_bytes
            )
        ):
            raise IneligibleSparseDraw(
                f"{identity}:{position_index}: invalid chosen-token bytes"
            )
        token_bytes_value = bytes(token_bytes)
        reconstructed_parts.append(token_bytes_value)
        boundaries.append(boundaries[-1] + len(token_bytes_value))
    if b"".join(reconstructed_parts) != raw_bytes:
        raise IneligibleSparseDraw(
            f"{identity}: chosen-token bytes do not reconstruct content"
        )
    boundary_to_index = {
        byte_offset: token_index
        for token_index, byte_offset in enumerate(boundaries)
    }
    first_position = boundary_to_index.get(target_start_byte)
    end_position = boundary_to_index.get(target_end_byte)
    if first_position is None or end_position is None:
        raise IneligibleSparseDraw(
            f"{identity}: trim_outer_whitespace cuts a provider token"
        )
    if first_position != 0:
        raise IneligibleSparseDraw(
            f"{identity}: leading whitespace changes the teacher prefix"
        )
    if first_position >= end_position:
        raise IneligibleSparseDraw(
            f"{identity}: trim_outer_whitespace retained no provider tokens"
        )
    teacher_tokens = teacher_tokens[first_position:end_position]
    audit_positions = audit_positions[first_position:end_position]

    encoded = tokenizer.encode(target, add_special_tokens=False)
    target_ids = [int(value) for value in encoded.ids]
    if not target_ids or len(target_ids) != len(teacher_tokens):
        raise IneligibleSparseDraw(
            f"{identity}: retained teacher/student token sequences do not align"
        )
    chosen_ids = [
        position.get("chosen_student_token_id")
        if isinstance(position, Mapping)
        else None
        for position in audit_positions
    ]
    if chosen_ids != target_ids:
        raise IneligibleSparseDraw(
            f"{identity}: full local tokenization differs from mapped chosen tokens"
        )
    decoded = tokenizer.decode(target_ids, skip_special_tokens=False)
    if decoded != target:
        raise IneligibleSparseDraw(
            f"{identity}: local target token IDs do not byte-roundtrip"
        )
    if eos_token_id in target_ids:
        raise IneligibleSparseDraw(f"{identity}: target content contains EOS")

    positions = [
        _position_payload(
            teacher_position,
            audit_position,
            observed_id=target_id,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            identity=f"{identity}:{position_index}",
        )
        for position_index, (teacher_position, audit_position, target_id) in enumerate(
            zip(teacher_tokens, audit_positions, target_ids, strict=True)
        )
    ]
    return {
        "schema": SPARSE_ROW_SCHEMA,
        "candidate_id": candidate_id,
        "target_token_ids": target_ids,
        "teacher_positions": positions,
        "target_alignment": {
            "transform": "trim_trailing_outer_whitespace",
            "trim_on_provider_token_boundaries": True,
            "leading_provider_tokens_omitted": first_position,
            "trailing_provider_tokens_omitted": (
                len(boundaries) - 1 - end_position
            ),
        },
        "eos_policy": {
            "teacher_eos_distribution_available": False,
            "sparse_auxiliary_applied_to_eos": False,
            "student_eos_supervised_by_primary_sequence_nll": True,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.minimum_eligible_fraction <= 1.0:
        raise ArtifactError("--minimum-eligible-fraction must be in (0,1]")
    contract_path = args.contract.expanduser().resolve()
    contract = DirectCompactContract.load(contract_path)
    sequence_path = args.sequence_train_jsonl.expanduser().resolve()
    sequence_seal_path = args.sequence_train_seal.expanduser().resolve()
    validate_join_seal(
        sequence_path, sequence_seal_path, contract_path, expected_role="fit"
    )
    sequence_rows = read_jsonl(sequence_path)
    schedule_path = args.sequence_schedule_jsonl.expanduser().resolve()
    schedule = read_jsonl(schedule_path)
    if len(sequence_rows) != len(schedule):
        raise ArtifactError("sequence dataset/schedule row count mismatch")

    sequence_manifest_path = args.sequence_build_manifest.expanduser().resolve()
    sequence_manifest_record = _expected_file(
        sequence_manifest_path,
        args.expected_sequence_build_manifest_sha256,
        "sequence build manifest",
    )
    sequence_manifest = _json(sequence_manifest_path)
    if sequence_manifest.get("schema") != (
        "direct-compact-mc-sequence-forward-kl-nll-build-v1"
    ):
        raise ArtifactError("sequence build manifest schema mismatch")
    outputs = sequence_manifest.get("outputs") or {}
    if (outputs.get("dataset") or {}).get("sha256") != sha256_file(sequence_path):
        raise ArtifactError("sequence manifest does not bind training JSONL")
    if (outputs.get("schedule") or {}).get("sha256") != sha256_file(schedule_path):
        raise ArtifactError("sequence manifest does not bind schedule")
    if (sequence_manifest.get("objective") or {}).get("dense_token_kl") is not False:
        raise ArtifactError("sequence build makes an invalid dense-KL claim")

    teacher_path = args.teacher_parseable_jsonl.expanduser().resolve()
    teacher_record = _expected_file(
        teacher_path,
        args.expected_teacher_parseable_sha256,
        "teacher parseable artifact",
    )
    expected_teacher = (
        (sequence_manifest.get("inputs") or {}).get("teacher_parseable") or {}
    )
    if expected_teacher.get("sha256") != teacher_record["sha256"]:
        raise ArtifactError("sequence build used a different teacher artifact")
    teachers = {
        str(row.get("candidate_id") or ""): row for row in read_jsonl(teacher_path)
    }
    if "" in teachers or len(teachers) != len(read_jsonl(teacher_path)):
        raise ArtifactError("teacher artifact has duplicate/missing candidate IDs")

    audit_path = args.teacher_audit_json.expanduser().resolve()
    audit_record = _expected_file(
        audit_path, args.expected_teacher_audit_sha256, "teacher audit"
    )
    audit = _json(audit_path)
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ArtifactError("teacher audit schema mismatch")
    if audit.get("objective_mode") != "require_top5":
        raise ArtifactError("sparse auxiliary requires objective_mode=require_top5")
    if (audit.get("production_readiness") or {}).get(
        "sparse_top5_plus_tail"
    ) is not True:
        raise ArtifactError("teacher audit is not ready for sparse top5+tail KL")
    if (audit.get("capabilities") or {}).get(
        "content_logprob_prefix_fully_visible_to_student"
    ) is not True:
        raise ArtifactError(
            "teacher content logprobs were conditioned on a prefix unavailable "
            "to the student"
        )
    shard_matches = [
        shard
        for shard in audit.get("homogeneous_backend_shards") or []
        if (shard.get("parseable_output") or {}).get("sha256")
        == teacher_record["sha256"]
    ]
    if len(shard_matches) != 1:
        raise ArtifactError("teacher artifact is not one homogeneous audit shard")
    shard = shard_matches[0]
    backend_identity_sha = str(shard["backend_identity_sha256"])
    backend_identity = shard["backend_identity"]

    student_tokenizer_path = args.student_tokenizer_json.expanduser().resolve()
    student_tokenizer_record = _expected_file(
        student_tokenizer_path,
        args.expected_student_tokenizer_sha256,
        "student tokenizer",
    )
    if student_tokenizer_record["sha256"] != contract.tokenizer_json_sha256:
        raise ArtifactError("student tokenizer differs from compact contract")
    if (audit.get("student_tokenizer") or {}).get("sha256") != (
        student_tokenizer_record["sha256"]
    ):
        raise ArtifactError("teacher audit used a different student tokenizer")
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover
        raise ArtifactError("the tokenizers package is required") from exc
    tokenizer = Tokenizer.from_file(student_tokenizer_record["path"])
    tokenizer_vocab_size = int(tokenizer.get_vocab_size())
    output_vocab_size = int(contract.base_vocab_size or 0)
    if output_vocab_size <= 1:
        raise ArtifactError("compact contract has no output-vocabulary binding")
    if not 0 <= args.student_eos_token_id < output_vocab_size:
        raise ArtifactError("student EOS is outside model output vocabulary")
    if args.student_eos_token_id >= tokenizer_vocab_size:
        raise ArtifactError("student EOS is absent from tokenizer vocabulary")

    enriched: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    teacher_draws = 0
    eligible = 0
    sparse_positions = 0
    tails: list[float] = []
    for row_index, (row, scheduled) in enumerate(
        zip(sequence_rows, schedule, strict=True)
    ):
        output = dict(row)
        if scheduled.get("kind") != "teacher_draw":
            enriched.append(output)
            continue
        teacher_draws += 1
        candidate_id = str(scheduled.get("candidate_id") or "")
        teacher = teachers.get(candidate_id)
        if teacher is None:
            raise ArtifactError(
                f"schedule row {row_index} references unknown teacher candidate"
            )
        try:
            sparse = _sparse_payload(
                teacher,
                _target(output, f"sequence-row-{row_index}"),
                tokenizer=tokenizer,
                vocab_size=output_vocab_size,
                eos_token_id=args.student_eos_token_id,
            )
        except IneligibleSparseDraw as exc:
            rejection_counts[str(exc).split(":", 1)[-1].strip()] += 1
        else:
            output[SPARSE_FIELD] = sparse
            eligible += 1
            sparse_positions += len(sparse["teacher_positions"])
            tails.extend(
                float(position["tail_probability_mass"])
                for position in sparse["teacher_positions"]
            )
        enriched.append(output)
    if teacher_draws <= 0:
        raise ArtifactError("sequence schedule contains no teacher draws")
    eligible_fraction = eligible / teacher_draws
    if eligible_fraction + 1e-15 < args.minimum_eligible_fraction:
        raise ArtifactError(
            "sparse auxiliary eligibility below required floor: "
            f"{eligible}/{teacher_draws}={eligible_fraction:.3%}; "
            f"rejections={dict(rejection_counts)}"
        )
    if eligible == 0 or sparse_positions == 0:
        raise ArtifactError("no mathematically valid sparse auxiliary positions")

    output_path = args.output_jsonl.expanduser().resolve()
    output_seal_path = args.output_seal.expanduser().resolve()
    output_manifest_path = args.output_manifest.expanduser().resolve()
    atomic_write_jsonl(output_path, enriched)
    output_seal = exact_output_seal(
        output_path=output_path,
        contract_path=contract_path,
        contract=contract,
        rows=enriched,
        tokenizer=tokenizer,
    )
    atomic_write_json(output_seal_path, output_seal)
    validate_join_seal(
        output_path, output_seal_path, contract_path, expected_role="fit"
    )
    manifest = {
        "schema": SPARSE_MANIFEST_SCHEMA,
        "dataset_sha256": sha256_file(output_path),
        "dataset_seal_sha256": sha256_file(output_seal_path),
        "contract_sha256": sha256_file(contract_path),
        "student_tokenizer_json_sha256": student_tokenizer_record["sha256"],
        "backend_identity_sha256": backend_identity_sha,
        "backend_identity": backend_identity,
        "student_output_vocab_size": output_vocab_size,
        "rows": len(enriched),
        "teacher_draw_rows": teacher_draws,
        "rows_with_sparse_auxiliary": eligible,
        "sparse_positions": sparse_positions,
        "eligible_fraction": eligible_fraction,
        "minimum_eligible_fraction": float(args.minimum_eligible_fraction),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "objective": "coarsened_topk_plus_tail_forward_kl",
        "mathematical_scope": (
            "KL on each observed raw-byte-mapped top-k singleton category plus "
            "one aggregate complement at the exact observed teacher prefix; "
            "not full-vocabulary KL"
        ),
        "mapping_contract": (
            "every chosen and top-k provider raw-byte token at every used "
            "position maps one-to-one to a unique sealed student token ID; "
            "only whole trailing provider whitespace tokens may be omitted; "
            "leading whitespace is rejected because it changes the prefix"
        ),
        "target_transform": (
            "trim_trailing_outer_whitespace_on_provider_token_boundaries"
        ),
        "global_provider_tokenizer_identity_claimed": False,
        "sequence_monte_carlo_forward_kl_nll_primary": True,
        "dense_full_vocabulary_kl": False,
        "full_vocabulary_kd": False,
        "probability_temperature_transform": "none",
        "teacher_logprob_semantics": (
            "natural-log probabilities returned by provider under sealed request"
        ),
        "negative_teacher_tail_policy": "reject_never_clamp",
        "eos_token_id": int(args.student_eos_token_id),
        "eos_policy": {
            "teacher_eos_distribution_available": False,
            "sparse_auxiliary_applied_to_eos": False,
            "student_eos_supervised_by_primary_sequence_nll": True,
        },
        "tail_statistics": {
            "minimum": min(tails),
            "maximum": max(tails),
            "mean": math.fsum(tails) / len(tails),
        },
        "inputs": {
            "sequence_dataset": file_record(sequence_path),
            "sequence_dataset_seal": file_record(sequence_seal_path),
            "sequence_schedule": file_record(schedule_path),
            "sequence_build_manifest": sequence_manifest_record,
            "teacher_parseable": teacher_record,
            "teacher_audit": audit_record,
            "student_tokenizer": student_tokenizer_record,
        },
        "sparse_payload_sha256": stable_sha256(
            [
                row.get(SPARSE_FIELD)
                for row in enriched
                if row.get(SPARSE_FIELD) is not None
            ]
        ),
    }
    atomic_write_json(output_manifest_path, manifest)
    return manifest


def main() -> int:
    manifest = build(parse_args())
    print(
        "QWEN_SPARSE_TOPK_TAIL_AUX "
        f"eligible={manifest['rows_with_sparse_auxiliary']}/"
        f"{manifest['teacher_draw_rows']} "
        f"positions={manifest['sparse_positions']} dense_kl=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
