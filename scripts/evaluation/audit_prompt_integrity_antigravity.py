"""Audit that benchmark scoring tests never enter policy prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.provenance_antigravity import file_record, write_json
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
    PROMPT_SCHEMA_VERSION,
    _build_test_call_hint,
    build_decoder_prompt,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    prompt_digest = hashlib.sha256()
    rows_with_tests = 0
    rows_with_candidate_assertions = 0
    leaked_rows: list[dict] = []

    for index, row in enumerate(rows):
        tests = str(row.get("tests", "") or "")
        hint = _build_test_call_hint(tests)
        prompt = build_decoder_prompt(row)
        prompt_digest.update(prompt.encode("utf-8"))
        rows_with_tests += int(bool(tests.strip()))
        rows_with_candidate_assertions += int(bool(hint))
        if hint and hint in prompt:
            leaked_rows.append(
                {
                    "row_index": index,
                    "task_id": row.get("task_id", row.get("id", index)),
                    "hint_preview": hint[:200],
                }
            )

    report = {
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "dataset": file_record(args.dataset),
        "rows": len(rows),
        "rows_with_tests": rows_with_tests,
        "rows_with_candidate_assertions": rows_with_candidate_assertions,
        "rows_leaking_scoring_tests": len(leaked_rows),
        "leaked_rows": leaked_rows,
        "prompt_stream_sha256": prompt_digest.hexdigest(),
        "passed": not leaked_rows,
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if leaked_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
