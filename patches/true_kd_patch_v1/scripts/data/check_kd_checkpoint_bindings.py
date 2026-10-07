#!/usr/bin/env python3
"""Validate dense-KD checkpoint/contract bindings without loading either model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from true_distribution_kd_antigravity import (  # noqa: E402
    _canonical_json_sha256,
    _checkpoint_layout,
    _hash_directory,
    _validate_checkpoint_binding,
)
from validate_qwen_kd_artifacts import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    args = parser.parse_args()

    tokenizer_sha256 = sha256_file(args.tokenizer_json)
    teacher = _checkpoint_layout(args.teacher_checkpoint, identity="teacher")
    student = _checkpoint_layout(args.student_checkpoint, identity="student")
    _validate_checkpoint_binding(
        teacher,
        identity="teacher",
        contract_path=args.contract,
        tokenizer_sha256=tokenizer_sha256,
    )
    _validate_checkpoint_binding(
        student,
        identity="student",
        contract_path=args.contract,
        tokenizer_sha256=tokenizer_sha256,
    )

    teacher_adapter = _hash_directory(teacher["adapter"])
    student_adapter = _hash_directory(student["adapter"])
    teacher_overlay = sha256_file(teacher["overlay"])
    student_overlay = sha256_file(student["overlay"])
    if teacher_adapter == student_adapter and teacher_overlay == student_overlay:
        raise SystemExit(
            "teacher and student checkpoints are identical; dense KL would "
            "start as a no-op"
        )

    print(
        json.dumps(
            {
                "schema": "dense-kd-checkpoint-preflight-v1",
                "status": "compatible",
                "contract_file_sha256": sha256_file(args.contract),
                "contract_canonical_json_sha256": _canonical_json_sha256(
                    args.contract
                ),
                "tokenizer_json_sha256": tokenizer_sha256,
                "teacher": {
                    "path": str(teacher["root"]),
                    "adapter_sha256": teacher_adapter,
                    "overlay_sha256": teacher_overlay,
                    "saved_contract_file_sha256": sha256_file(
                        teacher["contract"]
                    ),
                    "saved_tokenizer_json_sha256": sha256_file(
                        teacher["tokenizer_json"]
                    ),
                },
                "student": {
                    "path": str(student["root"]),
                    "adapter_sha256": student_adapter,
                    "overlay_sha256": student_overlay,
                    "saved_contract_file_sha256": sha256_file(
                        student["contract"]
                    ),
                    "saved_tokenizer_json_sha256": sha256_file(
                        student["tokenizer_json"]
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
