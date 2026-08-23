"""Audit exact and high-similarity source overlap between train and benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

KEYWORDS = {
    "abstract", "as", "assert", "async", "await", "break", "case", "catch",
    "class", "const", "continue", "default", "do", "dynamic", "else", "enum",
    "extends", "external", "factory", "false", "final", "finally", "for", "get",
    "if", "implements", "import", "in", "int", "is", "late", "List", "Map",
    "mixin", "new", "null", "num", "on", "operator", "part", "required",
    "rethrow", "return", "set", "static", "String", "super", "switch", "sync",
    "this", "throw", "true", "try", "typedef", "var", "void", "while", "with",
    "yield", "bool", "double",
}
TOKEN_RE = re.compile(
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|[A-Za-z_]\w*|\d+(?:\.\d+)?|"
    r"=>|==|!=|<=|>=|&&|\|\||\+\+|--|\?\?|\.\.|[^\s]"
)


def source_of(row: dict) -> str:
    return str(row.get("source", row.get("dart_source", "")) or "")


def normalize_source(source: str) -> str:
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"\s+", " ", source).strip()


def normalized_tokens(source: str) -> list[str]:
    out = []
    for token in TOKEN_RE.findall(normalize_source(source)):
        if token.startswith(("'", '"')):
            out.append("STR")
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            out.append("NUM")
        elif re.fullmatch(r"[A-Za-z_]\w*", token) and token not in KEYWORDS:
            out.append("ID")
        else:
            out.append(token)
    return out


def shingles(source: str, width: int = 7) -> set[str]:
    tokens = normalized_tokens(source)
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--near_threshold", type=float, default=0.8)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--filtered_train", type=Path)
    parser.add_argument("--expected_filtered_rows", type=int)
    args = parser.parse_args()

    train_rows = load(args.train)
    benchmark_rows = load(args.benchmark)
    train_norm = [normalize_source(source_of(row)) for row in train_rows]
    benchmark_norm = [normalize_source(source_of(row)) for row in benchmark_rows]
    train_hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(train_norm):
        train_hash_to_indices[hashlib.sha256(source.encode()).hexdigest()].append(index)
    duplicate_train_source_rows = sum(
        len(indices) - 1
        for indices in train_hash_to_indices.values()
        if len(indices) > 1
    )

    exact = []
    for bench_index, source in enumerate(benchmark_norm):
        digest = hashlib.sha256(source.encode()).hexdigest()
        for train_index in train_hash_to_indices.get(digest, []):
            exact.append({"benchmark_index": bench_index, "train_index": train_index})

    train_shingles = [shingles(source) for source in train_norm]
    inverted: dict[str, list[int]] = defaultdict(list)
    for train_index, values in enumerate(train_shingles):
        for value in values:
            inverted[value].append(train_index)

    near = []
    for bench_index, source in enumerate(benchmark_norm):
        values = shingles(source)
        intersections: dict[int, int] = defaultdict(int)
        for value in values:
            for train_index in inverted.get(value, []):
                intersections[train_index] += 1
        for train_index, intersection in intersections.items():
            union = len(values) + len(train_shingles[train_index]) - intersection
            score = intersection / union if union else 0.0
            if score >= args.near_threshold:
                near.append(
                    {
                        "benchmark_index": bench_index,
                        "benchmark_task_id": benchmark_rows[bench_index].get("task_id", bench_index),
                        "train_index": train_index,
                        "jaccard_7gram": score,
                    }
                )
    near.sort(key=lambda item: item["jaccard_7gram"], reverse=True)
    dropped_train_indices = sorted(
        {item["train_index"] for item in exact}
        | {item["train_index"] for item in near}
    )
    dropped_train_set = set(dropped_train_indices)
    report = {
        "train": str(args.train),
        "train_sha256": file_sha256(args.train),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_sha256(args.benchmark),
        "train_rows": len(train_rows),
        "benchmark_rows": len(benchmark_rows),
        "duplicate_train_source_rows": duplicate_train_source_rows,
        "normalization": "comments/whitespace removed; identifiers/literals normalized; token 7-gram Jaccard",
        "exact_overlap_pairs": len(exact),
        "exact_overlaps": exact[: args.top],
        "near_threshold": args.near_threshold,
        "near_overlap_pairs": len(near),
        "near_overlaps": near[: args.top],
        "dropped_train_indices": dropped_train_indices,
        "dropped_train_rows": len(dropped_train_indices),
    }
    if args.filtered_train is not None:
        filtered_rows = [
            row for index, row in enumerate(train_rows)
            if index not in dropped_train_set
        ]
        if (
            args.expected_filtered_rows is not None
            and len(filtered_rows) != args.expected_filtered_rows
        ):
            raise SystemExit(
                f"expected {args.expected_filtered_rows} filtered rows, "
                f"found {len(filtered_rows)}"
            )
        write_jsonl_atomic(args.filtered_train, filtered_rows)
        report["filtered_train"] = str(args.filtered_train)
        report["filtered_train_rows"] = len(filtered_rows)
        report["filtered_train_sha256"] = file_sha256(args.filtered_train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "train_rows": len(train_rows),
        "benchmark_rows": len(benchmark_rows),
        "duplicate_train_source_rows": duplicate_train_source_rows,
        "exact_overlap_pairs": len(exact),
        "near_overlap_pairs": len(near),
    }, indent=2))


if __name__ == "__main__":
    main()
