#!/usr/bin/env python3
"""Offline audit/materializer for a Qwen direct-compact teacher journal.

This command performs no network calls. It replays the append-only journal,
checks the sealed K=8/request/backend contract, audits raw provider-token bytes
against the pinned local student tokenizer, and emits two deliberately separate
artifacts:

* every completion-attested draw for equal-draw auditing; production readiness
  additionally requires every target to be final Dart code and to fit the
  exact EOS-inclusive trainer target contract;
* only completion-attested, independently verifier-passing draws for RS-SFT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    NEGATIVE_TAIL_TOLERANCE,
    ArtifactError,
    StudentTokenizerBinding,
    materialize_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--student-eos-token-id", required=True, type=int)
    parser.add_argument("--parseable-output", required=True, type=Path)
    parser.add_argument("--rs-sft-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument(
        "--negative-tail-tolerance",
        type=float,
        default=NEGATIVE_TAIL_TOLERANCE,
        help="Only floating-point overshoot at or above -tolerance is tolerated.",
    )
    parser.add_argument(
        "--split-homogeneous-shards",
        action="store_true",
        help=(
            "Explicitly permit more than one returned backend identity and "
            "materialize each identity into a distinct shard."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.negative_tail_tolerance < 0:
        raise ArtifactError("--negative-tail-tolerance cannot be negative")
    binding = StudentTokenizerBinding.from_file(
        args.student_tokenizer_json,
        expected_sha256=args.expected_student_tokenizer_sha256,
        eos_token_id=args.student_eos_token_id,
    )
    audit = materialize_artifacts(
        journal_path=args.journal,
        binding=binding,
        parseable_output=args.parseable_output,
        rs_sft_output=args.rs_sft_output,
        audit_output=args.audit_output,
        allow_homogeneous_shards=args.split_homogeneous_shards,
        negative_tail_tolerance=args.negative_tail_tolerance,
    )
    print(
        json.dumps(
            {
                "production_ready": audit["production_ready"],
                "production_readiness": audit["production_readiness"],
                "production_failures": audit["production_failures"],
                "coverage": audit["coverage"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if audit["production_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
