#!/usr/bin/env python3
"""Split a private compact evaluation JSONL into strict inference/test views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PUBLIC_FIELDS = (
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--role",
        choices=["fit", "measure"],
        default="measure",
        help="Private partition role written only to the alignment sidecar.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    existing_nonempty = output_dir.exists() and any(output_dir.iterdir())
    source_rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{input_path}:{line_number}: blank row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_number}: row is not an object")
            source_rows.append(row)
    if not source_rows:
        raise ValueError("evaluation input is empty")

    public = []
    alignment = []
    tests = []
    seen: set[str] = set()
    for index, row in enumerate(source_rows):
        task_id = str(row.get("task_id") or "")
        test_code = str(
            row.get("acceptance_tests")
            or row.get("tests")
            or row.get("feedback_tests")
            or ""
        )
        if not task_id or task_id in seen or not test_code:
            raise ValueError(f"row {index}: missing/duplicate task or tests")
        missing = [field for field in PUBLIC_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{task_id}: missing compact fields {missing}")
        seen.add(task_id)
        public.append({field: row[field] for field in PUBLIC_FIELDS})
        alignment.append(
            {"model_row": index, "role": args.role, "task_id": task_id}
        )
        tests.append({"task_id": task_id, "tests": test_code})

    outputs = {
        "public": output_dir / "public.jsonl",
        "alignment": output_dir / "alignment.jsonl",
        "tests": output_dir / "tests.jsonl",
    }
    if existing_nonempty:
        expected_rows = {
            "public": public,
            "alignment": alignment,
            "tests": tests,
        }
        for name, path in outputs.items():
            if not path.is_file():
                raise ValueError(
                    f"existing evaluation view is incomplete: {path}"
                )
            observed: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ValueError(f"{path}:{line_number}: blank row")
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{path}:{line_number}: row is not an object"
                        )
                    observed.append(value)
            if observed != expected_rows[name]:
                raise ValueError(
                    f"existing {name} view differs from deterministic input split"
                )
        report_path = output_dir / "report.json"
        if not report_path.is_file():
            raise ValueError("existing evaluation views have no report.json")
        expected_report = {
            "schema": "direct-compact-eval-views-v1",
            "role": args.role,
            "input": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "rows": len(source_rows),
            },
            "outputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in outputs.items()
            },
            "task_ids_sha256": hashlib.sha256(
                json.dumps(sorted(seen), separators=(",", ":")).encode()
            ).hexdigest(),
        }
        observed_report = json.loads(report_path.read_text(encoding="utf-8"))
        if observed_report != expected_report:
            raise ValueError(
                "existing evaluation-view report differs from deterministic split"
            )
        print("DIRECT_COMPACT_EVAL_VIEWS_REUSED", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs["public"], public)
    write_jsonl(outputs["alignment"], alignment)
    write_jsonl(outputs["tests"], tests)
    report = {
        "schema": "direct-compact-eval-views-v1",
        "role": args.role,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows": len(source_rows),
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
        "task_ids_sha256": hashlib.sha256(
            json.dumps(sorted(seen), separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
