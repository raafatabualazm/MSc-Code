"""
Batch compile@k evaluator with a shared Dart compile cache.

This is useful when auditing many prediction pools on the same benchmark:
exact duplicate candidates across arms should count as separate samples, but
they should not spawn another Dart compiler process.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.graph_compile_at_k_antigravity import (
    DART_BIN,
    _extract_code,
    compile_at_k_estimator,
    compile_dart_detail,
    compile_dart_jit_tests_detail,
    compile_dart_tests_detail,
    compile_swift,
    validate_dart_binary,
)


def _last_json_object(text: str) -> dict | None:
    start = text.rfind("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pass
    matches = list(re.finditer(r"\{[\s\S]*?\}", text))
    for match in reversed(matches):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.count(b"\x00") > max(8, len(raw) // 10):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _complete_output(path: Path, expected_total: int = 154) -> bool:
    if not path.is_file():
        return False
    data = _last_json_object(_read_text(path))
    return bool(data and data.get("total_problems") == expected_total and "compile_at_1" in data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="JSON list with label/path entries")
    parser.add_argument("--output_dir", default="results/compile154_passharness")
    parser.add_argument("--k_values", default="1,5,10")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--compile_mode", choices=["legacy", "tests", "jit_tests"], default="tests")
    parser.add_argument("--skip_existing", type=int, default=1)
    args = parser.parse_args()

    validate_dart_binary()
    print(f"Using Dart binary: {DART_BIN} | workers={args.workers}")

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_list = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    compile_cache: dict[tuple[str, str, str], tuple[bool, str, str]] = {}
    compile_cache_lock = Lock()

    def compile_cached(raw: str, test_code: str, task_id: str) -> tuple[bool, str, str]:
        code = _extract_code(raw)
        key = (args.compile_mode, code, test_code if args.compile_mode in {"tests", "jit_tests"} else "")
        with compile_cache_lock:
            cached = compile_cache.get(key)
            if cached is not None:
                return cached
        if args.compile_mode == "jit_tests" and test_code:
            result = compile_dart_jit_tests_detail(raw, test_code, task_id, timeout=args.timeout)
        elif args.compile_mode == "tests" and test_code:
            result = compile_dart_tests_detail(raw, test_code, task_id, timeout=args.timeout)
        else:
            result = compile_dart_detail(raw, timeout=args.timeout)
        with compile_cache_lock:
            compile_cache.setdefault(key, result)
            return compile_cache[key]

    def score_row(item: tuple[int, dict]) -> tuple[int, int, int]:
        idx, row = item
        lang = row.get("language", "dart").lower()
        candidates = row.get("predictions", [row.get("prediction", "")]) or [row.get("prediction", "")]
        test_code = row.get("tests", "")
        task_id = str(row.get("id", idx))
        compiling = 0
        for candidate in candidates:
            if lang == "dart":
                ok, _, _ = compile_cached(candidate, test_code, task_id)
            else:
                ok = compile_swift(candidate)
            if ok:
                compiling += 1
        return idx, len(candidates), compiling

    summary_rows: list[dict] = []
    for job in jobs:
        label = job["label"]
        pred_path = Path(job["path"])
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        out_path = output_dir / f"{label}_compile_at_k.json"

        if args.skip_existing and _complete_output(out_path):
            data = _last_json_object(_read_text(out_path)) or {}
            data = {"label": label, **data, "source": str(out_path), "skipped_existing": True}
            summary_rows.append(data)
            print(f"=== SKIP COMPLETE {label} ===")
            continue

        print(f"=== COMPILE154-PASSHARNESS {label} ===")
        rows = json.loads(pred_path.read_text(encoding="utf-8"))
        indexed_rows = list(enumerate(rows))
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            iterator = pool.map(score_row, indexed_rows)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(indexed_rows), desc=label, unit="row", dynamic_ncols=True)
            row_results = list(iterator)

        compile_sums = {k: 0.0 for k in k_list}
        for _, n, compiling in row_results:
            for k in k_list:
                eval_k = k if n >= k else n
                compile_sums[k] += compile_at_k_estimator(n, compiling, eval_k)

        result = {f"compile_at_{k}": compile_sums[k] / max(len(rows), 1) for k in k_list}
        result["total_problems"] = len(rows)
        result["cache_entries"] = len(compile_cache)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary_rows.append({"label": label, **result, "source": str(out_path), "skipped_existing": False})
        print(json.dumps({"label": label, **result}, indent=2))

    summary_json = output_dir / "summary_compile_at_k_154_passharness.json"
    summary_csv = output_dir / "summary_compile_at_k_154_passharness.csv"
    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    fields = ["label", "compile_at_1", "compile_at_5", "compile_at_10", "total_problems", "cache_entries", "skipped_existing", "source"]
    lines = [",".join(fields)]
    for row in summary_rows:
        lines.append(",".join(str(row.get(field, "")) for field in fields))
    summary_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_json}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
