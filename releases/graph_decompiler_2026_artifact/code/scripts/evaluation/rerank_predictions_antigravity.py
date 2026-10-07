"""
Rerank Antigravity multi-sample predictions and optionally promote one candidate.

This is the deployable counterpart to analyze_rerank_antigravity.py:
it writes a new prediction JSON with candidates reordered, plus an optional
single-candidate JSON for measuring selected pass@1 with the existing harness.

Modes:
  heuristic      No Dart execution; code-shape score only.
  stats_compile  Offline replay using an existing *_stats.csv compile flag.
  stats_compile_vote
                 stats_compile plus self-consistency: exact-duplicate
                 candidates vote for their shared implementation instead of
                 being penalized. Picks the largest compiling cluster.
  compile_cluster_vote
                 compile plus self-consistency: compile candidates still get
                 the hard compile bonus, and exact duplicate clusters get a
                 small score bonus.
  compile        Run the Dart compile harness for each candidate.
  test           Run the stored test harness for each candidate. This is only a
                 fair inference-time reranker if those tests are public tests.
                 On benchmark/eval tests it is an oracle diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any


_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)
_FUNC_RE = re.compile(
    r"(?m)^\s*(?:@pragma\([^)]*\)\s*)*"
    r"(?P<return>[A-Za-z_][\w.<>, ?\[\]]*)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^)]*)\)\s*(?:async\s*)?\{"
)


def _safe_console(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def _extract_code(text: str) -> str:
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    starters = ("@pragma", "import ", "library ", "void ", "Future", "main(")
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(starters) or re.match(r"^[\w<>\[\],\?\s]+\s+\w+\s*\(", stripped):
            return "\n".join(lines[idx:]).strip()
    return text.strip()


def _normalize_code(code: str) -> str:
    code = _extract_code(code)
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"\s+", "", code)


def _parse_bool(value: Any) -> int:
    if value in (1, "1", True, "true", "True"):
        return 1
    return 0


def _resolve_dart_binary(preferred: str | None) -> str:
    if preferred:
        return preferred
    candidates = [
        "/home/zeus/dart-sdk/bin/dart",
        os.path.join(os.path.expanduser("~"), "dart-sdk", "bin", "dart"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("dart") or "dart"


def validate_dart_binary(dart_bin: str) -> None:
    try:
        result = subprocess.run(
            [dart_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: Dart binary is not runnable: {dart_bin!r} ({exc})")
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"ERROR: Dart binary failed: {dart_bin!r}\n{diagnostic}")


def extract_signature(code: str) -> dict[str, Any] | None:
    match = _FUNC_RE.search(_extract_code(code))
    if not match:
        return None
    params = [param.strip() for param in match.group("params").split(",") if param.strip()]
    return {
        "return": " ".join(match.group("return").split()),
        "name": match.group("name"),
        "params": params,
        "arity": len(params),
        "normalized": re.sub(r"\s+", "", match.group(0).rstrip("{").strip()),
    }


def extract_target_name(row: dict[str, Any]) -> str | None:
    tests = row.get("tests", "") or ""
    match = re.search(r"\bfinal\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;", tests)
    if match:
        return match.group(1)
    signature = extract_signature(row.get("reference", "") or "")
    if signature:
        return signature["name"]
    return None


def strip_main_and_imports(code: str) -> str:
    """Remove generated top-level boilerplate before appending tests."""
    code = re.sub(r"^import\s+.*;\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^@pragma\(.*\)\s*$", "", code, flags=re.MULTILINE)

    main_match = re.search(r"void\s+main\s*\([^)]*\)\s*\{", code)
    if main_match:
        start = main_match.start()
        depth = 0
        idx = main_match.end() - 1
        while idx < len(code):
            if code[idx] == "{":
                depth += 1
            elif code[idx] == "}":
                depth -= 1
                if depth == 0:
                    code = code[:start] + code[idx + 1 :]
                    break
            idx += 1

    return code.strip()


def wrap_like_legacy_compile(code: str) -> str:
    if "void main()" not in code and "main(" not in code:
        return f"void main() {{\n{code}\n}}"
    return code


def run_dart_compile(raw: str, dart_bin: str, timeout: int) -> tuple[bool, str]:
    code = _extract_code(raw)
    if len(code.strip()) < 5:
        return False, "empty_or_tiny_candidate"

    with tempfile.TemporaryDirectory(prefix="ag_compile_", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "main.dart"
        snapshot_path = Path(tmp) / "main.aot"
        path.write_text(wrap_like_legacy_compile(code), encoding="utf-8")
        try:
            result = subprocess.run(
                [dart_bin, "compile", "aot-snapshot", str(path), "-o", str(snapshot_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "compile_timeout"
        except FileNotFoundError as exc:
            return False, f"dart_not_found: {exc}"
    return result.returncode == 0, (result.stderr or result.stdout or "").strip()


def run_dart_tests(raw: str, test_code: str, task_id: str, dart_bin: str, timeout: int) -> tuple[bool, str]:
    generated_code = _extract_code(raw)
    if not generated_code.strip() or not test_code.strip():
        return False, "empty_candidate_or_tests"

    clean_code = strip_main_and_imports(generated_code)
    imports = sorted(set(re.findall(r"^import\s+.*;\s*$", generated_code, re.MULTILINE)))
    imports_section = "\n".join(imports)
    full_source = (imports_section + "\n\n" if imports_section else "") + clean_code + "\n\n" + test_code

    with tempfile.TemporaryDirectory(prefix=f"ag_test_{task_id}_", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "test.dart"
        path.write_text(full_source, encoding="utf-8")
        try:
            result = subprocess.run(
                [dart_bin, "--disable-dart-dev", "run", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "test_timeout"
        except FileNotFoundError as exc:
            return False, f"dart_not_found: {exc}"
    return result.returncode == 0, (result.stderr or result.stdout or "").strip()


def structural_features(raw: str, target_name: str | None, median_length: int) -> dict[str, Any]:
    code = _extract_code(raw)
    normalized = _normalize_code(code)
    signature = extract_signature(code)
    has_markdown = "```" in raw or bool(re.search(r"(?i)\b(here is|explanation|solution:)\b", raw))
    has_main = bool(re.search(r"\bvoid\s+main\s*\(", code))
    has_balanced_braces = code.count("{") == code.count("}")
    name_match = bool(signature and target_name and signature["name"] == target_name)
    length = len(code)

    score = 0.0
    score += 12.0 if name_match else 0.0
    score += 3.0 if has_balanced_braces else -8.0
    score += -8.0 if has_main else 2.0
    score += -6.0 if has_markdown else 1.0
    if median_length:
        ratio = length / median_length
        if ratio < 0.35 or ratio > 2.5:
            score -= 6.0
        elif 0.65 <= ratio <= 1.65:
            score += 2.0
    if not normalized:
        score -= 20.0

    return {
        "score": score,
        "normalized": normalized,
        "signature": signature,
        "name_match": name_match,
        "has_main": has_main,
        "has_markdown": has_markdown,
        "has_balanced_braces": has_balanced_braces,
        "length": length,
    }


def load_stats(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stats_flag(stats_rows: list[dict[str, str]] | None, row_idx: int, cand_idx: int, field: str) -> int:
    if not stats_rows:
        return 0
    return _parse_bool(stats_rows[row_idx].get(f"cand_{cand_idx + 1}_{field}", "0"))


def evaluate_candidate(
    mode: str,
    row_idx: int,
    cand_idx: int,
    row: dict[str, Any],
    candidate: str,
    target_name: str | None,
    median_length: int,
    duplicate_number: int,
    duplicate_cluster_size: int,
    stats_rows: list[dict[str, str]] | None,
    dart_bin: str,
    timeout: int,
    test_compile_fallback: bool,
    cluster_vote_bonus: float,
) -> dict[str, Any]:
    feat = structural_features(candidate, target_name, median_length)
    compile_ok: bool | None = None
    pass_ok: bool | None = None
    diagnostic = ""

    # Exact-duplicate completions are penalized for ranking diversity, but in
    # vote mode duplication IS the signal (self-consistency): identical
    # compiling candidates vote for their shared implementation.
    if mode == "stats_compile_vote":
        score = feat["score"] + 6.0 * duplicate_number
    elif mode == "compile_cluster_vote":
        score = feat["score"] - 4.0 * duplicate_number
        score += cluster_vote_bonus * max(0, duplicate_cluster_size - 1)
    else:
        score = feat["score"] - 4.0 * duplicate_number
    if mode in {"stats_compile", "stats_compile_vote"}:
        compile_ok = bool(stats_flag(stats_rows, row_idx, cand_idx, "compile"))
        score += 100.0 if compile_ok else 0.0
    elif mode == "stats_pass_oracle":
        compile_ok = bool(stats_flag(stats_rows, row_idx, cand_idx, "compile"))
        pass_ok = bool(stats_flag(stats_rows, row_idx, cand_idx, "pass"))
        score += 1000.0 if pass_ok else 0.0
        score += 100.0 if compile_ok else 0.0
    elif mode in {"compile", "compile_cluster_vote"}:
        compile_ok, diagnostic = run_dart_compile(candidate, dart_bin, timeout)
        score += 100.0 if compile_ok else 0.0
    elif mode == "test":
        pass_ok, diagnostic = run_dart_tests(
            candidate,
            row.get("tests", "") or "",
            str(row.get("id", row_idx)),
            dart_bin,
            timeout,
        )
        if pass_ok:
            compile_ok = True
            score += 1000.0
        elif test_compile_fallback:
            compile_ok, compile_diag = run_dart_compile(candidate, dart_bin, timeout)
            if compile_ok:
                score += 100.0
            if not diagnostic:
                diagnostic = compile_diag
    elif mode != "heuristic":
        raise ValueError(f"Unsupported rerank mode: {mode}")

    return {
        "index": cand_idx,
        "score": score,
        "compile": compile_ok,
        "pass": pass_ok,
        "diagnostic": diagnostic[:500],
        "features": {
            key: value for key, value in feat.items()
            if key not in {"normalized", "signature"}
        },
    }


def process_row(
    mode: str,
    row_idx: int,
    row: dict[str, Any],
    stats_rows: list[dict[str, str]] | None,
    dart_bin: str,
    timeout: int,
    workers: int,
    test_compile_fallback: bool,
    cluster_vote_bonus: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = row.get("predictions", []) or [row.get("prediction", "")]
    target_name = extract_target_name(row)
    lengths = [len(_extract_code(candidate)) for candidate in candidates if _extract_code(candidate)]
    median_length = sorted(lengths)[len(lengths) // 2] if lengths else 0

    seen: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    normalized_candidates = [_normalize_code(candidate) for candidate in candidates]
    for normalized in normalized_candidates:
        if normalized:
            cluster_counts[normalized] = cluster_counts.get(normalized, 0) + 1

    duplicate_numbers = []
    duplicate_cluster_sizes = []
    for normalized in normalized_candidates:
        duplicate_numbers.append(seen.get(normalized, 0))
        duplicate_cluster_sizes.append(cluster_counts.get(normalized, 0) if normalized else 0)
        if normalized:
            seen[normalized] = seen.get(normalized, 0) + 1

    evals: list[dict[str, Any]] = []
    if workers > 1 and mode in {"compile", "compile_cluster_vote", "test"}:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    evaluate_candidate,
                    mode,
                    row_idx,
                    cand_idx,
                    row,
                    candidate,
                    target_name,
                    median_length,
                    duplicate_numbers[cand_idx],
                    duplicate_cluster_sizes[cand_idx],
                    stats_rows,
                    dart_bin,
                    timeout,
                    test_compile_fallback,
                    cluster_vote_bonus,
                ): cand_idx
                for cand_idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                evals.append(future.result())
        evals.sort(key=lambda item: item["index"])
    else:
        for cand_idx, candidate in enumerate(candidates):
            evals.append(
                evaluate_candidate(
                    mode,
                    row_idx,
                    cand_idx,
                    row,
                    candidate,
                    target_name,
                    median_length,
                    duplicate_numbers[cand_idx],
                    duplicate_cluster_sizes[cand_idx],
                    stats_rows,
                    dart_bin,
                    timeout,
                    test_compile_fallback,
                    cluster_vote_bonus,
                )
            )

    ranked = sorted(evals, key=lambda item: (-float(item["score"]), int(item["index"])))
    order = [int(item["index"]) for item in ranked]
    best_idx = order[0] if order else 0

    reranked_candidates = [candidates[idx] for idx in order]
    reranked_row = dict(row)
    reranked_row["predictions"] = reranked_candidates
    reranked_row["prediction"] = reranked_candidates[0] if reranked_candidates else row.get("prediction", "")
    reranked_row["_rerank"] = {
        "mode": mode,
        "best_original_index": best_idx,
        "order": order,
        "scores": [evals[idx]["score"] for idx in order],
    }

    selected_row = dict(row)
    selected = candidates[best_idx] if candidates else row.get("prediction", "")
    selected_row["predictions"] = [selected]
    selected_row["prediction"] = selected
    selected_row["_rerank"] = dict(reranked_row["_rerank"])

    row_report = {
        "row_index": row_idx,
        "problem_id": row.get("id", row_idx),
        "filename": row.get("filename"),
        "target_name": target_name,
        "best_original_index": best_idx,
        "moved": best_idx != 0,
        "original": evals[0] if evals else {},
        "selected": evals[best_idx] if evals else {},
        "candidate_evaluations": evals,
    }
    return reranked_row, selected_row, row_report


def build_summary(mode: str, row_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(row_reports), 1)
    moved = sum(1 for row in row_reports if row.get("moved"))

    def known_rate(which: str, field: str) -> float | None:
        values = [
            row.get(which, {}).get(field)
            for row in row_reports
            if row.get(which, {}).get(field) is not None
        ]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    def known_count(which: str, field: str) -> int:
        return sum(
            1 for row in row_reports
            if row.get(which, {}).get(field) is not None
        )

    selected_scores = [
        float(row.get("selected", {}).get("score", 0.0))
        for row in row_reports
    ]
    return {
        "mode": mode,
        "total_problems": len(row_reports),
        "moved_count": moved,
        "moved_rate": moved / total,
        "original_compile_rate_observed": known_rate("original", "compile"),
        "original_compile_known_count": known_count("original", "compile"),
        "selected_compile_rate_observed": known_rate("selected", "compile"),
        "selected_compile_known_count": known_count("selected", "compile"),
        "original_pass_rate_observed": known_rate("original", "pass"),
        "original_pass_known_count": known_count("original", "pass"),
        "selected_pass_rate_observed": known_rate("selected", "pass"),
        "selected_pass_known_count": known_count("selected", "pass"),
        "mean_selected_score": mean(selected_scores) if selected_scores else 0.0,
        "test_mode_is_oracle_unless_tests_are_public": mode == "test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Writes all candidates reordered by reranker score.")
    parser.add_argument("--selected_output", type=Path,
                        help="Optional one-candidate-per-problem JSON for selected pass@1 evaluation.")
    parser.add_argument("--report", type=Path,
                        help="Optional JSON report with per-row scores and observed rates.")
    parser.add_argument("--stats", type=Path,
                        help="Required for stats_compile/stats_pass_oracle modes.")
    parser.add_argument("--mode", choices=["heuristic", "stats_compile", "stats_compile_vote",
                                           "stats_pass_oracle", "compile", "compile_cluster_vote",
                                           "test"],
                        default="compile")
    parser.add_argument("--dart", default=None)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cluster_vote_bonus", type=float, default=5.0,
                        help="Score bonus per exact duplicate in compile_cluster_vote mode.")
    parser.add_argument("--test_compile_fallback", action="store_true",
                        help="In test mode, AOT-compile candidates that fail tests to improve nonpassing ordering. Slower.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    stats_rows = load_stats(args.stats)
    if args.mode in {"stats_compile", "stats_compile_vote", "stats_pass_oracle"} and stats_rows is None:
        raise SystemExit(f"ERROR: --stats is required for mode {args.mode}")
    if stats_rows is not None and args.limit and args.limit > 0:
        stats_rows = stats_rows[: args.limit]
    if stats_rows is not None and len(stats_rows) != len(rows):
        raise SystemExit(f"ERROR: predictions rows ({len(rows)}) and stats rows ({len(stats_rows)}) differ")

    dart_bin = _resolve_dart_binary(args.dart)
    if args.mode in {"compile", "compile_cluster_vote", "test"}:
        validate_dart_binary(dart_bin)
        print(f"Using Dart binary: {dart_bin}", file=sys.stderr)

    reranked_rows = []
    selected_rows = []
    row_reports = []
    workers = max(1, args.workers)

    for row_idx, row in enumerate(rows):
        reranked_row, selected_row, row_report = process_row(
            args.mode,
            row_idx,
            row,
            stats_rows,
            dart_bin,
            args.timeout,
            workers,
            args.test_compile_fallback,
            args.cluster_vote_bonus,
        )
        reranked_rows.append(reranked_row)
        selected_rows.append(selected_row)
        row_reports.append(row_report)
        if row_idx % 10 == 0 or row_idx + 1 == len(rows):
            print(_safe_console(f"[{row_idx + 1}/{len(rows)}] best_idx={row_report['best_original_index']} moved={row_report['moved']}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reranked_rows, indent=2), encoding="utf-8")

    if args.selected_output:
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_output.write_text(json.dumps(selected_rows, indent=2), encoding="utf-8")

    report = {
        "predictions": str(args.predictions),
        "output": str(args.output),
        "selected_output": str(args.selected_output) if args.selected_output else None,
        "stats": str(args.stats) if args.stats else None,
        "summary": build_summary(args.mode, row_reports),
        "rows": row_reports,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved reranked predictions to: {args.output}")
    if args.selected_output:
        print(f"Saved selected predictions to: {args.selected_output}")
    if args.report:
        print(f"Saved rerank report to: {args.report}")


if __name__ == "__main__":
    main()
