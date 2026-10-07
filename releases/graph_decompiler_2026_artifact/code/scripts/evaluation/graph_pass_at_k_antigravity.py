"""
Graph-aware decompiler pass@k evaluation script (Antigravity version).
Computes unbiased pass@k for multiple candidate predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# Match a fenced code block, optionally tagged ```dart / ```Dart, etc.
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
    """Pull runnable Dart out of a raw model generation (see compile@k)."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = text.splitlines()
    starters = ("@pragma", "import ", "library ", "void ", "Future", "main(")
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(starters) or re.match(r"^[\w<>\[\],\?\s]+\s+\w+\s*\(", s):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def _resolve_dart_binary() -> str:
    candidates = [
        '/home/zeus/dart-sdk/bin/dart',
        os.path.join(os.path.expanduser('~'), 'dart-sdk', 'bin', 'dart'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    found = shutil.which('dart')
    return found if found else 'dart'


DART_BIN = _resolve_dart_binary()
FORCED_ZERO_SOURCE_LINE = 152
FORCED_ZERO_FILENAME = "160.dart"


def validate_dart_binary() -> None:
    try:
        result = subprocess.run(
            [DART_BIN, '--version'],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: Dart binary is not runnable: {DART_BIN!r} ({exc})")
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"ERROR: Dart binary failed: {DART_BIN!r}\n{diagnostic}")


def should_force_zero(row: dict) -> bool:
    return (
        row.get('source_line') == FORCED_ZERO_SOURCE_LINE
        and row.get('filename') == FORCED_ZERO_FILENAME
    )


def strip_main_and_imports(code: str) -> str:
    """Remove generated top-level boilerplate before appending legacy tests."""
    code = re.sub(r"^import\s+.*;\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^@pragma\(.*\)\s*$", "", code, flags=re.MULTILINE)

    main_match = re.search(r"void\s+main\s*\([^)]*\)\s*\{", code)
    if main_match:
        start = main_match.start()
        depth = 0
        i = main_match.end() - 1
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    code = code[:start] + code[i + 1:]
                    break
            i += 1

    return code.strip()


def run_dart_test_detail(raw: str, test_code: str, task_id: str, timeout: int = 30) -> tuple[bool, str, str]:
    """Run the parent-folder pass@k harness: candidate code + stored tests."""
    generated_code = _extract_code(raw)
    if not generated_code.strip() or not test_code.strip():
        return False, "empty_candidate_or_tests", generated_code

    clean_code = strip_main_and_imports(generated_code)
    imports = set(re.findall(r"^import\s+.*;\s*$", generated_code, re.MULTILINE))
    full_source = "\n".join(sorted(imports)) + "\n\n" + clean_code + "\n\n" + test_code

    with tempfile.TemporaryDirectory(prefix=f"dart_passk_{task_id}_") as tmp:
        path = Path(tmp) / 'test.dart'
        path.write_text(full_source, encoding='utf-8')
        try:
            result = subprocess.run(
                [DART_BIN, 'run', str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "pass_timeout", full_source
        except FileNotFoundError as exc:
            return False, f"dart_not_found: {exc}", full_source
        return result.returncode == 0, (result.stderr or result.stdout or "").strip(), full_source


def run_dart_test(raw: str, test_code: str, task_id: str, timeout: int = 30) -> bool:
    ok, _, _ = run_dart_test_detail(raw, test_code, task_id, timeout=timeout)
    return ok


def run_dart(raw: str, timeout: int = 30) -> bool:
    """Pass == the generated `main()` program runs to completion (exit 0).

    The test set (test-set.jsonl) targets are self-contained `void main()`
    programs with no unit-test harness, so functional success is defined as
    successful execution rather than assertion-based unit tests.
    """
    code = _extract_code(raw)
    if len(code.strip()) < 5:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        path.write_text(code, encoding='utf-8')
        try:
            result = subprocess.run(
                [DART_BIN, '--disable-dart-dev', 'run', str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return result.returncode == 0


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    # Compute C(n-c, k) / C(n, k)
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--k_values', default='1,5')
    parser.add_argument('--workers', type=int, default=int(os.environ.get("EVAL_DART_WORKERS", "1")),
                        help="Parallel Dart pass@k row workers. Use 64-128 on high-core CPU boxes")
    parser.add_argument('--timeout', type=int, default=30,
                        help="Per-candidate Dart run timeout in seconds")
    parser.add_argument('--debug_failures', type=int, default=0,
                        help="Print diagnostics for the first N failed Dart candidates")
    args = parser.parse_args()
    validate_dart_binary()
    workers = max(1, int(args.workers))
    print(f"Using Dart binary: {DART_BIN} | workers={workers}", file=sys.stderr)

    rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    k_list = [int(x.strip()) for x in args.k_values.split(',')]

    pass_sums = {k: 0.0 for k in k_list}
    total = len(rows)
    def score_row(item: tuple[int, dict]) -> tuple[int, int, int, bool, list[tuple[str, str]]]:
        idx, row = item
        candidates = row.get('predictions', [row.get('prediction', '')])
        test_code = row.get('tests', '')
        task_id = str(row.get('id', idx))

        if not candidates:
            candidates = [row.get('prediction', '')]

        if should_force_zero(row):
            return idx, len(candidates), 0, True, []

        c = 0
        failures: list[tuple[str, str]] = []
        for cand in candidates:
            if test_code:
                ok, diagnostic, full_source = run_dart_test_detail(cand, test_code, task_id, timeout=args.timeout)
                if not ok and len(failures) < 1:
                    failures.append((diagnostic, full_source[:600].replace("\n", "\\n")))
            else:
                ok = run_dart(cand, timeout=args.timeout)
            if ok:
                c += 1
        return idx, len(candidates), c, False, failures

    # Bar on stderr: the runner captures this script's stdout for the JSON
    # result, so stderr is the only live channel.
    indexed_rows = list(enumerate(rows))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            row_results = list(
                tqdm(pool.map(score_row, indexed_rows), total=len(indexed_rows), desc="pass@k", unit="row", dynamic_ncols=True)
                if tqdm is not None else pool.map(score_row, indexed_rows)
            )
    else:
        row_iter = tqdm(indexed_rows, desc="pass@k", unit="row", dynamic_ncols=True) if tqdm is not None else indexed_rows
        row_results = [score_row(item) for item in row_iter]

    debug_printed = 0
    for idx, n, c, forced_zero, failures in row_results:
        if forced_zero:
            for k in k_list:
                pass_sums[k] += 0.0
            print(f'[{idx + 1}/{len(rows)}] n={n}, passed=0, forced_zero=True')
            continue

        if failures and debug_printed < args.debug_failures:
            diagnostic, preview = failures[0]
            debug_printed += 1
            print(f"\n--- Pass failure #{debug_printed} row={idx} ---")
            print(diagnostic[:1600])
            print(f"source_preview={preview}")

        for k in k_list:
            if n >= k:
                val = pass_at_k_estimator(n, c, k)
                pass_sums[k] += val
            else:
                val = pass_at_k_estimator(n, c, n)
                pass_sums[k] += val

        print(f'[{idx + 1}/{len(rows)}] n={n}, passed={c}, pass@1={c/n:.4f}')

    results = {}
    for k in k_list:
        results[f'pass_at_{k}'] = pass_sums[k] / max(total, 1)

    results['total_problems'] = total
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
