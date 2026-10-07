"""
Analyze Antigravity multi-sample candidate files and simulate reranking.

This script is intentionally offline: it consumes the prediction JSON plus the
per-candidate stats CSV already produced by compile_statistical_results_antigravity.py.
It does not run Dart again. Compile flags may be used by deployable rerankers
because a real inference-time system can cheaply compile candidates; pass flags
and CodeBLEU are used only for evaluation/oracle diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
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
    code = re.sub(r"\s+", "", code)
    return code


def _parse_bool(value: Any) -> int:
    if value in (1, "1", True, "true", "True"):
        return 1
    return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    product = 1.0
    for idx in range(k):
        product *= (n - c - idx) / (n - idx)
    return 1.0 - product


def extract_target_name(row: dict[str, Any]) -> str | None:
    tests = row.get("tests", "") or ""
    match = re.search(r"\bfinal\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;", tests)
    if match:
        return match.group(1)

    reference_sig = extract_signature(row.get("reference", "") or "")
    if reference_sig:
        return reference_sig["name"]
    return None


def extract_signature(code: str) -> dict[str, Any] | None:
    match = _FUNC_RE.search(_extract_code(code))
    if not match:
        return None
    params = [p.strip() for p in match.group("params").split(",") if p.strip()]
    return {
        "return": " ".join(match.group("return").split()),
        "name": match.group("name"),
        "params": params,
        "arity": len(params),
        "normalized": re.sub(r"\s+", "", match.group(0).rstrip("{").strip()),
    }


def candidate_features(raw: str, target_name: str | None) -> dict[str, Any]:
    code = _extract_code(raw)
    normalized = _normalize_code(code)
    signature = extract_signature(code)
    has_markdown = "```" in raw or bool(re.search(r"(?i)\b(here is|explanation|solution:)\b", raw))
    has_main = bool(re.search(r"\bvoid\s+main\s*\(", code))
    has_balanced_braces = code.count("{") == code.count("}")
    name_match = bool(signature and target_name and signature["name"] == target_name)
    return {
        "code": code,
        "normalized": normalized,
        "signature": signature,
        "has_markdown": has_markdown,
        "has_main": has_main,
        "has_balanced_braces": has_balanced_braces,
        "name_match": name_match,
        "length": len(code),
    }


def infer_k(stats_row: dict[str, str]) -> int:
    k_values = []
    for key in stats_row:
        match = re.fullmatch(r"cand_(\d+)_pass", key)
        if match:
            k_values.append(int(match.group(1)))
    return max(k_values) if k_values else 0


def get_candidate_stat(stats_row: dict[str, str], index: int, field: str) -> float | int:
    value = stats_row.get(f"cand_{index + 1}_{field}", "")
    if field in {"compile", "pass"}:
        return _parse_bool(value)
    return _safe_float(value)


def original_order(row: dict[str, Any], stats_row: dict[str, str]) -> list[int]:
    return list(range(len(row.get("predictions", []) or [row.get("prediction", "")])))


def compile_first_order(row: dict[str, Any], stats_row: dict[str, str]) -> list[int]:
    order = original_order(row, stats_row)
    return sorted(order, key=lambda i: (-int(get_candidate_stat(stats_row, i, "compile")), i))


def heuristic_order(row: dict[str, Any], stats_row: dict[str, str]) -> list[int]:
    candidates = row.get("predictions", []) or [row.get("prediction", "")]
    target_name = extract_target_name(row)
    features = [candidate_features(cand, target_name) for cand in candidates]
    lengths = [feat["length"] for feat in features if feat["length"] > 0]
    median_length = sorted(lengths)[len(lengths) // 2] if lengths else 0
    seen: dict[str, int] = {}
    scored = []

    for idx, feat in enumerate(features):
        compile_ok = int(get_candidate_stat(stats_row, idx, "compile"))
        score = 100.0 * compile_ok
        score += 12.0 if feat["name_match"] else 0.0
        score += 3.0 if feat["has_balanced_braces"] else -8.0
        score += -8.0 if feat["has_main"] else 2.0
        score += -6.0 if feat["has_markdown"] else 1.0
        if median_length:
            ratio = feat["length"] / median_length
            if ratio < 0.35 or ratio > 2.5:
                score -= 6.0
            elif 0.65 <= ratio <= 1.65:
                score += 2.0
        if feat["normalized"]:
            duplicate_number = seen.get(feat["normalized"], 0)
            score -= 4.0 * duplicate_number
            seen[feat["normalized"]] = duplicate_number + 1
        else:
            score -= 20.0
        scored.append((score, idx))

    return [idx for _, idx in sorted(scored, key=lambda item: (-item[0], item[1]))]


def codebleu_oracle_order(row: dict[str, Any], stats_row: dict[str, str]) -> list[int]:
    order = original_order(row, stats_row)
    return sorted(
        order,
        key=lambda i: (
            -int(get_candidate_stat(stats_row, i, "compile")),
            -float(get_candidate_stat(stats_row, i, "codebleu")),
            i,
        ),
    )


def pass_oracle_order(row: dict[str, Any], stats_row: dict[str, str]) -> list[int]:
    order = original_order(row, stats_row)
    return sorted(
        order,
        key=lambda i: (
            -int(get_candidate_stat(stats_row, i, "pass")),
            -int(get_candidate_stat(stats_row, i, "compile")),
            i,
        ),
    )


RANKERS = {
    "original_order": original_order,
    "compile_first": compile_first_order,
    "compile_shape_heuristic": heuristic_order,
    "compile_codebleu_oracle": codebleu_oracle_order,
    "pass_oracle_upper_bound": pass_oracle_order,
}


def summarize_ranker(
    name: str,
    rows: list[dict[str, Any]],
    stats_rows: list[dict[str, str]],
    k_values: list[int],
) -> dict[str, Any]:
    selected_compile = []
    selected_pass = []
    selected_codebleu = []
    topk_pass_hits = {k: [] for k in k_values}
    topk_compile_hits = {k: [] for k in k_values}
    selected_indices = []

    for row, stats_row in zip(rows, stats_rows):
        order = RANKERS[name](row, stats_row)
        if not order:
            continue
        selected = order[0]
        selected_indices.append(selected + 1)
        selected_compile.append(int(get_candidate_stat(stats_row, selected, "compile")))
        selected_pass.append(int(get_candidate_stat(stats_row, selected, "pass")))
        selected_codebleu.append(float(get_candidate_stat(stats_row, selected, "codebleu")))
        for k in k_values:
            top = order[: min(k, len(order))]
            topk_pass_hits[k].append(any(int(get_candidate_stat(stats_row, i, "pass")) for i in top))
            topk_compile_hits[k].append(any(int(get_candidate_stat(stats_row, i, "compile")) for i in top))

    total = max(len(selected_pass), 1)
    return {
        "selected_compile_at_1": sum(selected_compile) / total,
        "selected_pass_at_1": sum(selected_pass) / total,
        "selected_mean_codebleu": mean(selected_codebleu) if selected_codebleu else 0.0,
        "mean_selected_rank": mean(selected_indices) if selected_indices else 0.0,
        "topk_hit_rate": {
            f"pass_top_{k}": sum(topk_pass_hits[k]) / total for k in k_values
        }
        | {
            f"compile_top_{k}": sum(topk_compile_hits[k]) / total for k in k_values
        },
    }


def row_candidate_counts(stats_row: dict[str, str]) -> tuple[int, int, int, list[float]]:
    k = infer_k(stats_row)
    compile_count = 0
    pass_count = 0
    codebleus = []
    for idx in range(k):
        compile_count += int(get_candidate_stat(stats_row, idx, "compile"))
        pass_count += int(get_candidate_stat(stats_row, idx, "pass"))
        codebleus.append(float(get_candidate_stat(stats_row, idx, "codebleu")))
    return k, compile_count, pass_count, codebleus


def exact_unique_count(candidates: list[str]) -> int:
    normalized = [_normalize_code(cand) for cand in candidates]
    return len({code for code in normalized if code})


def max_pair_similarity(candidates: list[str]) -> float:
    normalized = [_normalize_code(cand) for cand in candidates if _normalize_code(cand)]
    if len(normalized) < 2:
        return 1.0 if normalized else 0.0
    best = 0.0
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            best = max(best, SequenceMatcher(a=normalized[i], b=normalized[j]).ratio())
    return best


def example_record(
    row: dict[str, Any],
    stats_row: dict[str, str],
    idx: int,
    compile_count: int,
    pass_count: int,
    codebleus: list[float],
) -> dict[str, Any]:
    candidates = row.get("predictions", []) or [row.get("prediction", "")]
    passes = [int(get_candidate_stat(stats_row, i, "pass")) for i in range(len(candidates))]
    compiles = [int(get_candidate_stat(stats_row, i, "compile")) for i in range(len(candidates))]
    first_pass = passes.index(1) + 1 if 1 in passes else None
    return {
        "row_index": idx,
        "problem_id": row.get("id", stats_row.get("problem_id", idx)),
        "filename": row.get("filename"),
        "source_line": row.get("source_line"),
        "target_name": extract_target_name(row),
        "compile_count": compile_count,
        "pass_count": pass_count,
        "first_pass_rank": first_pass,
        "max_codebleu": max(codebleus) if codebleus else 0.0,
        "candidate_1_codebleu": codebleus[0] if codebleus else 0.0,
        "candidate_1_compile": compiles[0] if compiles else 0,
        "candidate_1_pass": passes[0] if passes else 0,
        "unique_candidates": exact_unique_count(candidates),
        "max_pair_similarity": round(max_pair_similarity(candidates), 4),
    }


def analyze_failures(
    rows: list[dict[str, Any]],
    stats_rows: list[dict[str, str]],
    top_examples: int,
    high_codebleu_threshold: float,
) -> dict[str, Any]:
    pass_count_hist: dict[str, int] = {}
    compile_count_hist: dict[str, int] = {}
    first_pass_hist: dict[str, int] = {}
    compiles_but_no_pass = []
    high_codebleu_zero_pass = []
    low_diversity = []
    late_first_pass = []
    unique_counts = []
    similarities = []

    for idx, (row, stats_row) in enumerate(zip(rows, stats_rows)):
        k, compile_count, pass_count, codebleus = row_candidate_counts(stats_row)
        candidates = row.get("predictions", []) or [row.get("prediction", "")]
        unique_count = exact_unique_count(candidates)
        similarity = max_pair_similarity(candidates)
        unique_counts.append(unique_count)
        similarities.append(similarity)
        pass_count_hist[str(pass_count)] = pass_count_hist.get(str(pass_count), 0) + 1
        compile_count_hist[str(compile_count)] = compile_count_hist.get(str(compile_count), 0) + 1

        passes = [int(get_candidate_stat(stats_row, i, "pass")) for i in range(k)]
        first_pass = passes.index(1) + 1 if 1 in passes else None
        if first_pass is not None:
            first_pass_hist[str(first_pass)] = first_pass_hist.get(str(first_pass), 0) + 1

        record = example_record(row, stats_row, idx, compile_count, pass_count, codebleus)
        if compile_count >= max(3, k // 2) and pass_count == 0:
            compiles_but_no_pass.append(record)
        if pass_count == 0 and codebleus and max(codebleus) >= high_codebleu_threshold:
            high_codebleu_zero_pass.append(record)
        if unique_count <= max(2, k // 3) or similarity >= 0.985:
            low_diversity.append(record)
        if first_pass is not None and first_pass > 1:
            late_first_pass.append(record)

    compiles_but_no_pass.sort(key=lambda item: (-item["compile_count"], -item["max_codebleu"]))
    high_codebleu_zero_pass.sort(key=lambda item: (-item["max_codebleu"], -item["compile_count"]))
    low_diversity.sort(key=lambda item: (item["unique_candidates"], -item["max_pair_similarity"]))
    late_first_pass.sort(key=lambda item: (-item["first_pass_rank"], -item["pass_count"]))

    return {
        "pass_count_histogram": pass_count_hist,
        "compile_count_histogram": compile_count_hist,
        "first_pass_rank_histogram": first_pass_hist,
        "mean_unique_candidates": mean(unique_counts) if unique_counts else 0.0,
        "mean_max_pair_similarity": mean(similarities) if similarities else 0.0,
        "compiles_but_no_pass": {
            "count": len(compiles_but_no_pass),
            "examples": compiles_but_no_pass[:top_examples],
        },
        "high_codebleu_zero_pass": {
            "threshold": high_codebleu_threshold,
            "count": len(high_codebleu_zero_pass),
            "examples": high_codebleu_zero_pass[:top_examples],
        },
        "low_diversity": {
            "count": len(low_diversity),
            "examples": low_diversity[:top_examples],
        },
        "late_first_pass": {
            "count": len(late_first_pass),
            "examples": late_first_pass[:top_examples],
        },
    }


def aggregate_unbiased_metrics(stats_rows: list[dict[str, str]], k_values: list[int]) -> dict[str, Any]:
    pass_sums = {k: 0.0 for k in k_values}
    compile_sums = {k: 0.0 for k in k_values}
    candidate_1_pass = 0
    candidate_1_compile = 0
    total = len(stats_rows)

    for stats_row in stats_rows:
        n, compile_count, pass_count, _ = row_candidate_counts(stats_row)
        candidate_1_pass += int(get_candidate_stat(stats_row, 0, "pass"))
        candidate_1_compile += int(get_candidate_stat(stats_row, 0, "compile"))
        for k in k_values:
            effective_k = min(k, n)
            pass_sums[k] += pass_at_k_estimator(n, pass_count, effective_k)
            compile_sums[k] += pass_at_k_estimator(n, compile_count, effective_k)

    divisor = max(total, 1)
    return {
        "candidate_1_pass_rate": candidate_1_pass / divisor,
        "candidate_1_compile_rate": candidate_1_compile / divisor,
        "unbiased_pass_at_k": {f"pass_at_{k}": pass_sums[k] / divisor for k in k_values},
        "unbiased_compile_at_k": {f"compile_at_{k}": compile_sums[k] / divisor for k in k_values},
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    with Path(args.stats).open("r", encoding="utf-8", newline="") as handle:
        stats_rows = list(csv.DictReader(handle))

    if len(rows) != len(stats_rows):
        raise SystemExit(f"Prediction rows ({len(rows)}) and stats rows ({len(stats_rows)}) differ.")

    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    candidate_counts = [len(row.get("predictions", []) or [row.get("prediction", "")]) for row in rows]

    report = {
        "predictions": str(args.predictions),
        "stats": str(args.stats),
        "total_problems": len(rows),
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "mean": mean(candidate_counts) if candidate_counts else 0.0,
        },
        "baseline": aggregate_unbiased_metrics(stats_rows, k_values),
        "rankers": {
            name: summarize_ranker(name, rows, stats_rows, k_values)
            for name in RANKERS
        },
        "failure_modes": analyze_failures(
            rows,
            stats_rows,
            args.top_examples,
            args.high_codebleu_threshold,
        ),
        "notes": [
            "compile_first and compile_shape_heuristic use compile flags plus code-shape checks only; they do not inspect pass flags.",
            "compile_codebleu_oracle is leaky because it uses reference CodeBLEU; use it only as an analysis upper bound.",
            "pass_oracle_upper_bound is the best possible reranker over the generated candidates and equals whether any candidate passed.",
        ],
    }
    return report


def print_summary(report: dict[str, Any]) -> None:
    baseline = report["baseline"]
    print(json.dumps({
        "total_problems": report["total_problems"],
        "candidate_count": report["candidate_count"],
        "candidate_1_pass_rate": baseline["candidate_1_pass_rate"],
        "candidate_1_compile_rate": baseline["candidate_1_compile_rate"],
        "unbiased_pass_at_k": baseline["unbiased_pass_at_k"],
        "unbiased_compile_at_k": baseline["unbiased_compile_at_k"],
    }, indent=2))

    print("\nReranker simulation:")
    for name, metrics in report["rankers"].items():
        print(
            f"- {name}: selected_pass@1={metrics['selected_pass_at_1']:.4f}, "
            f"selected_compile@1={metrics['selected_compile_at_1']:.4f}, "
            f"mean_selected_rank={metrics['mean_selected_rank']:.2f}, "
            f"pass_top_5={metrics['topk_hit_rate'].get('pass_top_5', 0.0):.4f}, "
            f"pass_top_10={metrics['topk_hit_rate'].get('pass_top_10', 0.0):.4f}"
        )

    failures = report["failure_modes"]
    print("\nFailure modes:")
    print(f"- compiles_but_no_pass: {failures['compiles_but_no_pass']['count']}")
    print(f"- high_codebleu_zero_pass: {failures['high_codebleu_zero_pass']['count']}")
    print(f"- low_diversity: {failures['low_diversity']['count']}")
    print(f"- late_first_pass: {failures['late_first_pass']['count']}")
    print(f"- mean_unique_candidates: {failures['mean_unique_candidates']:.2f}")
    print(f"- mean_max_pair_similarity: {failures['mean_max_pair_similarity']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k_values", default="1,5,10")
    parser.add_argument("--top_examples", type=int, default=10)
    parser.add_argument("--high_codebleu_threshold", type=float, default=0.65)
    args = parser.parse_args()

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nSaved analysis JSON to: {args.output}")


if __name__ == "__main__":
    main()
