"""
Statistical Results Compiler for Neural Decompiler (Antigravity version).
Evaluates each individual candidate prediction on CodeBLEU, compilation, and test passing,
and outputs a per-problem, per-candidate CSV for statistical analysis.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast
    _old_add_tokens = PreTrainedTokenizerFast._add_tokens
    
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        dict_or_attr = lambda o, k, d: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
        conv_tokens = []
        for t in new_tokens:
            if isinstance(t, str):
                conv_tokens.append(t)
            else:
                conv_tokens.append(tokenizers.AddedToken(
                    dict_or_attr(t, 'content', str(t)),
                    single_word=dict_or_attr(t, 'single_word', False),
                    lstrip=dict_or_attr(t, 'lstrip', False),
                    rstrip=dict_or_attr(t, 'rstrip', False),
                    normalized=dict_or_attr(t, 'normalized', True),
                    special=dict_or_attr(t, 'special', False)
                ))
        if special_tokens:
            return self._tokenizer.add_special_tokens(conv_tokens)
        return self._tokenizer.add_tokens(conv_tokens)
        
    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens
except Exception as e:
    pass

import argparse
import json
import csv
import subprocess
import tempfile
import re
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.codebleu import CodeBLEUCalculator
from scripts.evaluation.graph_compile_at_k_antigravity import (
    compile_dart_tests_detail,
    evaluate_dart_jit_tests_detail,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
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
    return shutil.which('dart') or 'dart'


DART_BIN = _resolve_dart_binary()


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


def _wrap_like_legacy_compile(code: str) -> str:
    if "void main()" not in code and "main(" not in code:
        return f"void main() {{\n{code}\n}}"
    return code


def compile_dart(raw: str, timeout: int = 30) -> bool:
    code = _extract_code(raw)
    if len(code.strip()) < 5:
        return False

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        snapshot_path = Path(tmp) / 'main.aot'
        try:
            path.write_text(_wrap_like_legacy_compile(code), encoding='utf-8')
            result = subprocess.run(
                [DART_BIN, 'compile', 'aot-snapshot', str(path), '-o', str(snapshot_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False

def compile_swift(code: str) -> bool:
    return bool(code.strip())

def run_dart(raw: str, timeout: int = 30) -> bool:
    code = _extract_code(raw)
    if len(code.strip()) < 5:
        return False

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        try:
            path.write_text(code, encoding='utf-8')
            result = subprocess.run(
                [DART_BIN, '--disable-dart-dev', 'run', str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False


def strip_main_and_imports(code: str) -> str:
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


def run_sandbox_pass(solution_code: str, test_code: str, timeout: int = 30) -> bool:
    """Check if the code passes actual functional unit tests using Dart sandbox execution"""
    solution_code = _extract_code(solution_code)
    if not solution_code.strip() or not test_code.strip():
        return False

    imports = sorted(set(re.findall(r"^import\s+.*;\s*$", solution_code, re.MULTILINE)))
    imports_section = '\n'.join(imports) if imports else ''
    function_section = strip_main_and_imports(solution_code)
    full_code = (imports_section + "\n\n" if imports_section else "") + function_section + "\n\n" + test_code
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filepath = os.path.join(temp_dir, 'temp_test.dart')
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(full_code)
                
            test_proc = subprocess.run(
                [DART_BIN, '--disable-dart-dev', 'run', test_filepath],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout,
            )
            return test_proc.returncode == 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help="Path to predictions JSON file")
    parser.add_argument('--output', required=True, help="Path to write statistical CSV file")
    parser.add_argument('--workers', type=int, default=int(os.environ.get("EVAL_DART_WORKERS", "1")),
                        help="Parallel row workers for Dart compile/pass checks")
    parser.add_argument('--timeout', type=int, default=30,
                        help="Per-candidate Dart compile/run timeout in seconds")
    parser.add_argument('--compile_mode', choices=['legacy', 'tests', 'jit_tests'], default='legacy',
                        help="Compile classification stored in cand_*_compile. Use jit_tests for pass-aligned statistics")
    args = parser.parse_args()

    rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    if not rows:
        print("Empty predictions file.")
        return
    if any(row.get('language', 'dart').lower() == 'dart' for row in rows):
        validate_dart_binary()
        print(f"Using Dart binary: {DART_BIN}", file=sys.stderr)

    calculators = {
        'dart': CodeBLEUCalculator('dart'),
        'swift': CodeBLEUCalculator('swift'),
    }

    # Determine maximum candidate count (K) to dynamically format CSV headers
    max_k = 0
    for row in rows:
        candidates = row.get('predictions', [row.get('prediction', '')])
        if not candidates:
            candidates = [row.get('prediction', '')]
        max_k = max(max_k, len(candidates))

    # Construct CSV headers
    headers = ['problem_id', 'language', 'reference_length']
    for i in range(max_k):
        headers.extend([
            f'cand_{i+1}_codebleu',
            f'cand_{i+1}_compile',
            f'cand_{i+1}_pass'
        ])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    print(f"Stats CSV workers={workers}", file=sys.stderr)

    def score_row(item: tuple[int, dict]) -> list:
        idx, row = item
        lang = row.get('language', 'dart').lower()
        calc = CodeBLEUCalculator(lang if lang in calculators else 'dart')
        reference = row.get('reference', '')
        test_code = row.get('tests', '')
        task_id = str(row.get('task_id', row.get('id', idx)))

        candidates = row.get('predictions', [row.get('prediction', '')])
        if not candidates:
            candidates = [row.get('prediction', '')]

        row_data = [
            row.get('id', idx),
            lang,
            len(reference)
        ]

        for i in range(max_k):
            if i < len(candidates):
                cand = candidates[i]
                code = _extract_code(cand)

                try:
                    score = calc.compute_codebleu(reference, code)['codebleu']
                except Exception:
                    score = 0.0

                if lang == 'dart' and args.compile_mode == 'jit_tests' and test_code:
                    compiled, did_pass, _diagnostic, _source = evaluate_dart_jit_tests_detail(
                        cand, test_code, task_id, timeout=args.timeout
                    )
                    comp = int(compiled)
                    passed = int(did_pass)
                elif lang == 'dart' and args.compile_mode == 'tests' and test_code:
                    compiled, _diagnostic, _source = compile_dart_tests_detail(
                        cand, test_code, task_id, timeout=args.timeout
                    )
                    comp = int(compiled)
                    passed = 1 if run_sandbox_pass(cand, test_code, timeout=args.timeout) else 0
                else:
                    comp = 1 if (compile_dart(cand, timeout=args.timeout) if lang == 'dart' else compile_swift(cand)) else 0
                    if test_code:
                        passed = 1 if run_sandbox_pass(cand, test_code, timeout=args.timeout) else 0
                    else:
                        passed = 1 if run_dart(cand, timeout=args.timeout) else 0

                row_data.extend([round(score, 5), comp, passed])
            else:
                row_data.extend(['', '', ''])
        return row_data

    indexed_rows = list(enumerate(rows))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored_rows = list(
                tqdm(pool.map(score_row, indexed_rows), total=len(indexed_rows), desc="stats CSV", unit="row", dynamic_ncols=True)
                if tqdm is not None else pool.map(score_row, indexed_rows)
            )
    else:
        row_iter = tqdm(indexed_rows, desc="stats CSV", unit="row", dynamic_ncols=True) if tqdm is not None else indexed_rows
        scored_rows = [score_row(item) for item in row_iter]

    with open(out_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row_data in scored_rows:
            writer.writerow(row_data)

    print(f"Successfully compiled per-problem statistics to: {args.output}")

if __name__ == '__main__':
    main()
