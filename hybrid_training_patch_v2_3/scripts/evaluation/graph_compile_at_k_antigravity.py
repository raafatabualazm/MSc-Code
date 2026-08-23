"""
Graph-aware decompiler compile@k evaluation script (Antigravity version).
Computes unbiased compile@k for multiple candidate predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# Match a fenced code block, optionally tagged ```dart / ```Dart, etc.
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
    """Pull runnable Dart out of a raw model generation.

    Models often emit markdown fences and/or prose around the code. We:
      1. Prefer the first fenced block if present.
      2. Otherwise, trim leading prose up to the first plausible Dart token,
         including directives, metadata, top-level type declarations, or a
         `main(` / type+ident `(` signature.
    Falls back to the stripped original if nothing matches.
    """
    if not text:
        return ""

    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    lines = text.splitlines()
    # Dart programs do not have to start with an import or function.  In
    # particular, class-first gold/model outputs are common in this corpus.
    # Omitting declaration starters silently trimmed their enclosing type and
    # began at the first method, turning valid programs into compile failures.
    starters = (
        "@",
        "import ",
        "export ",
        "library ",
        "part ",
        "class ",
        "abstract class ",
        "abstract base class ",
        "abstract final class ",
        "abstract interface class ",
        "base class ",
        "final class ",
        "interface class ",
        "sealed class ",
        "enum ",
        "mixin ",
        "extension ",
        "typedef ",
        "void ",
        "Future",
        "main(",
    )
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(starters) or re.match(r"^[\w<>\[\],\?\s]+\s+\w+\s*\(", s):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def _resolve_dart_binary() -> str:
    """Find the dart binary, checking known locations first."""
    candidates = [
        '/home/zeus/dart-sdk/bin/dart',
        os.path.join(os.path.expanduser('~'), 'dart-sdk', 'bin', 'dart'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Fallback to PATH lookup
    found = shutil.which('dart')
    if found:
        # On Windows, PATH commonly resolves to flutter/bin/dart.bat. Calling
        # the batch wrapper via subprocess timeout can leave dart.exe children
        # holding tempdirs open, so prefer the SDK executable next to Flutter.
        if found.lower().endswith(('.bat', '.cmd')):
            candidate = Path(found).parent / 'cache' / 'dart-sdk' / 'bin' / 'dart.exe'
            if candidate.is_file():
                return str(candidate)
        return found
    return 'dart'


DART_BIN = _resolve_dart_binary()


def _dart_subprocess_env(workdir: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    home = Path(workdir) / ".dart_home"
    appdata = home / "AppData" / "Roaming"
    localappdata = home / "AppData" / "Local"
    pub_cache = home / ".pub-cache"
    for path in (home, appdata, localappdata, pub_cache):
        path.mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home), "USERPROFILE": str(home),
        "APPDATA": str(appdata), "LOCALAPPDATA": str(localappdata),
        "PUB_CACHE": str(pub_cache), "CI": "true",
        "DART_SUPPRESS_ANALYTICS": "1",
    })
    return env


def validate_dart_binary() -> None:
    """Fail loudly if Dart cannot be executed.

    Returning all-zero compile@k because `dart` is missing is much worse than
    stopping the run: it makes a model-quality problem indistinguishable from an
    environment problem.
    """
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


def _safe_console(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def _wrap_like_legacy_compile(code: str) -> str:
    """Match the old parent-folder compile harness."""
    if "void main()" not in code and "main(" not in code:
        return f"void main() {{\n{code}\n}}"
    return code


def strip_main_and_imports(code: str) -> str:
    """Remove generated top-level boilerplate before appending stored tests."""
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
                    code = code[:start] + code[idx + 1:]
                    break
            idx += 1

    return code.strip()


def compile_dart_detail(raw: str, timeout: int = 30) -> tuple[bool, str, str]:
    """Return True if the extracted Dart code passes the old compile harness.

    The legacy evaluator wrapped snippets without `main` and ran
    `dart compile aot-snapshot`, so this intentionally keeps that stricter
    behavior for apples-to-apples compile@k with the parent-folder results.
    """
    code = _extract_code(raw)
    if len(code.strip()) < 5:
        return False, "empty_or_tiny_candidate", code

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        snapshot_path = Path(tmp) / 'main.aot'
        path.write_text(_wrap_like_legacy_compile(code), encoding='utf-8')

        try:
            result = subprocess.run(
                [DART_BIN, 'compile', 'aot-snapshot', str(path), '-o', str(snapshot_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                env=_dart_subprocess_env(tmp),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "compile_timeout_or_dart_not_found", code

        return result.returncode == 0, (result.stderr or result.stdout or "").strip(), code


def _candidate_plus_tests_source(raw: str, test_code: str) -> tuple[bool, str, str]:
    generated_code = _extract_code(raw)
    if not generated_code.strip() or not test_code.strip():
        return False, "empty_candidate_or_tests", generated_code

    clean_code = strip_main_and_imports(generated_code)
    # The scrubbed-dataset builder certifies every stored harness after adding
    # these two imports to the combined program, but stores the harness without
    # them.  Recreate that trusted harness prelude here.  Otherwise the 387
    # runZoned/differential harnesses in the sealed train/dev split compile only
    # when the *candidate* happens to supply their private test dependencies.
    # Candidate imports remain part of the evaluated compilation unit.
    trusted_harness_imports = {
        "import 'dart:async';",
        "import 'dart:convert';",
    }
    imports = sorted(
        trusted_harness_imports
        | set(
            re.findall(
                r"^import\s+.*;\s*$",
                generated_code,
                re.MULTILINE,
            )
        )
    )
    imports_section = "\n".join(imports)
    full_source = (imports_section + "\n\n" if imports_section else "") + clean_code + "\n\n" + test_code
    return True, "", full_source


def compile_dart_tests_detail(raw: str, test_code: str, task_id: str, timeout: int = 30) -> tuple[bool, str, str]:
    """Strict AOT compile candidate + stored tests without running tests.

    This is intentionally stricter than the pass@k harness because it uses
    `dart compile aot-snapshot`; it can reject programs that `dart run` accepts.
    Use `jit_tests` when comparing compile@k directly against pass@k.
    """
    ok_source, diagnostic, full_source = _candidate_plus_tests_source(raw, test_code)
    if not ok_source:
        return False, diagnostic, full_source

    with tempfile.TemporaryDirectory(prefix=f"dart_compilek_{task_id}_") as tmp:
        path = Path(tmp) / 'test.dart'
        snapshot_path = Path(tmp) / 'test.aot'
        path.write_text(full_source, encoding='utf-8')
        try:
            result = subprocess.run(
                [DART_BIN, 'compile', 'aot-snapshot', str(path), '-o', str(snapshot_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                env=_dart_subprocess_env(tmp),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "compile_timeout_or_dart_not_found", full_source

        return result.returncode == 0, (result.stderr or result.stdout or "").strip(), full_source


_JIT_STATIC_ERROR_RE = re.compile(
    r"(^|\n)(?:.+\.dart:\d+:\d+: )?Error:\s+"
    r"(?:"
    r"Expected|"
    r"Undefined name|"
    r"Method not found|"
    r"Type '|"
    r"The (?:method|getter|setter|operator|argument type)|"
    r"A value of type|"
    r"The return type|"
    r"Can't return|"
    r"Too (?:few|many)|"
    r"No named parameter|"
    r"Required named parameter|"
    r"Couldn't resolve|"
    r"Not found|"
    r"Error when reading|"
    r"Variables must be declared|"
    r"Expected an identifier|"
    r"The non-abstract class|"
    r"Can't assign|"
    r"This expression has type|"
    r"Compilation failed"
    r")",
    re.IGNORECASE,
)


def _is_dart_jit_static_error(diagnostic: str) -> bool:
    """Return True for Dart front-end errors, False for runtime/test failures."""
    if not diagnostic:
        return False
    if "Compilation failed" in diagnostic:
        return True
    if _JIT_STATIC_ERROR_RE.search(diagnostic):
        return True
    # Kernel/frontend failures sometimes end with these summaries even when the
    # individual diagnostic lines were elided by the tool wrapper.
    lowered = diagnostic.lower()
    return any(
        marker in lowered
        for marker in (
            "couldn't resolve the package",
            "error when reading",
            "target kernel_snapshot_program failed",
            "generate_kernel_snapshot failed",
            "failed to load",
            # A front-end crash never produced a runnable program; counting it
            # as "compiled" inflates compile@k (review finding F4).
            "crash when compiling",
        )
    )


_DART_TEST_MAIN_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)"
    r"(?P<prefix>(?:(?:void|dynamic|Future(?:Or)?\s*<[^>\r\n]+>)\s+)?)"
    r"(?P<name>main)"
    r"(?P<params>\s*\((?P<arguments>[^)\r\n]*)\))"
    r"(?P<suffix>\s*(?:async\s*)?\{)"
)
_DART_MAIN_ARGS_RE = re.compile(
    r"^(?:final\s+)?List\s*<\s*String\s*>\s+\w+$"
)
_DART_LIBRARY_DIRECTIVE_RE = re.compile(
    r"(?ms)^[ \t]*(?:import|export)\s+(?P<body>.*?);"
)
_DART_URI_LITERAL_RE = re.compile(
    r"""(?P<raw>[rR]?)(?P<quote>['"])(?P<value>.*?)(?P=quote)""",
    re.S,
)
_DART_URI_ESCAPE_RE = re.compile(
    r"\\(?:x(?P<x>[0-9a-fA-F]{2})|u(?P<u>[0-9a-fA-F]{4})|u\{(?P<ub>[0-9a-fA-F]+)\})"
)
_DISALLOWED_JIT_LIBRARIES = frozenset(
    {
        "dart:ffi",
        "dart:io",
        "dart:mirrors",
        "package:ffi/ffi.dart",
    }
)
_COMPLETION_MARKER_PREFIX = "__ANTIGRAVITY_DART_TEST_COMPLETED_"
# Stable verifier-contract identity for provenance consumers.  Changing any
# nonce, wrapper, or exactly-once acceptance semantics requires a new value.
COMPLETION_ATTESTATION_ID = "per-run-256-bit-marker-exactly-once-v1"


def _decode_dart_uri_literal(raw_prefix: str, value: str) -> str:
    if raw_prefix:
        return value

    def replace_escape(match: re.Match[str]) -> str:
        digits = match.group("x") or match.group("u") or match.group("ub")
        try:
            return chr(int(digits, 16))
        except (TypeError, ValueError, OverflowError):
            return ""

    return _DART_URI_ESCAPE_RE.sub(replace_escape, value)


def disallowed_dart_test_runtime_library(source: str) -> str:
    """Return a process-control/introspection import URI, if present.

    Completion attestation is an in-process contract, not an OS sandbox.  Code
    with IO, FFI, or mirrors capabilities could inspect/spoof the generated
    marker (or invoke native process termination).  The RS-SFT/VeRPO verifier
    contract is for deterministic pure functions, so these capabilities are
    outside its accepted language subset and are rejected before execution.
    ``dart:isolate`` remains allowed: isolate termination still cannot pass
    because it prevents the trusted wrapper from emitting the marker.
    """
    for directive in _DART_LIBRARY_DIRECTIVE_RE.finditer(source):
        body = directive.group("body")
        for literal in _DART_URI_LITERAL_RE.finditer(body):
            uri = _decode_dart_uri_literal(
                literal.group("raw"), literal.group("value")
            ).strip().lower()
            if uri in _DISALLOWED_JIT_LIBRARIES:
                return uri
    return ""


def prepare_dart_test_completion_attestation(
    source: str,
) -> tuple[bool, str, str, str]:
    """Wrap a Dart test ``main`` and attest that it returned normally.

    A zero process exit code is not sufficient evidence that a candidate
    passed: candidate code can call ``dart:io``'s ``exit(0)`` (and older Dart
    runtimes can terminate successfully through isolate APIs) before the
    trusted assertions finish.  Rename the trusted test entry point, invoke it
    through a wrapper that handles both synchronous and asynchronous mains,
    and emit a per-execution 256-bit completion marker only after it returns.

    The caller must require both exit code zero and exactly one marker line.
    Ambiguous/missing/unsupported entry points fail closed rather than silently
    falling back to return-code-only scoring.
    """
    disallowed_library = disallowed_dart_test_runtime_library(source)
    if disallowed_library:
        return (
            False,
            f"completion_attestation_disallowed_library:{disallowed_library}",
            source,
            "",
        )

    matches = list(_DART_TEST_MAIN_RE.finditer(source))
    if len(matches) != 1:
        return (
            False,
            f"completion_attestation_requires_one_main:found={len(matches)}",
            source,
            "",
        )

    match = matches[0]
    arguments = (match.group("arguments") or "").strip()
    if not arguments:
        applied_arguments = "const <dynamic>[]"
    elif _DART_MAIN_ARGS_RE.fullmatch(arguments):
        applied_arguments = "<dynamic>[__antigravity_program_args]"
    else:
        return (
            False,
            "completion_attestation_unsupported_main_parameters",
            source,
            "",
        )

    nonce = secrets.token_hex(32)
    renamed_main = f"_antigravity_test_main_{nonce}"
    marker = f"{_COMPLETION_MARKER_PREFIX}{nonce}__"
    instrumented = (
        source[: match.start("name")]
        + renamed_main
        + source[match.end("name") :]
        + "\n\n"
        + "Future<void> main(List<String> __antigravity_program_args) async {\n"
        + "  final dynamic __antigravity_result = Function.apply(\n"
        + f"    {renamed_main},\n"
        + f"    {applied_arguments},\n"
        + "  );\n"
        + "  if (__antigravity_result is Future) {\n"
        + "    await __antigravity_result;\n"
        + "  }\n"
        + f"  print('{marker}');\n"
        + "}\n"
    )
    return True, "", instrumented, marker


def dart_test_completion_observed(stdout: str | None, marker: str) -> bool:
    """Return True only for one exact completion-marker output line."""
    if not marker:
        return False
    return sum(line.strip() == marker for line in (stdout or "").splitlines()) == 1


def strip_dart_test_completion_marker(text: str | None, marker: str) -> str:
    """Remove the private marker from user-facing diagnostics."""
    if not text:
        return ""
    return "\n".join(
        line for line in str(text).splitlines() if line.strip() != marker
    ).strip()


def evaluate_dart_jit_tests_detail(
    raw: str,
    test_code: str,
    task_id: str,
    timeout: int = 30,
    stability_runs: int | None = None,
) -> tuple[bool, bool, str, str]:
    """Return (compiled, passed, diagnostic, source) from one JIT test run.

    This uses the same `dart run` path as pass@k. A zero exit code both compiles
    and passes. A nonzero exit is counted as compile failure only if Dart reports
    a static/front-end diagnostic; runtime exceptions and assertion failures are
    counted as compiled-but-not-passing. This is the metric to compare directly
    with pass@k because pass@k cannot exceed it for the same candidate pool.
    """
    ok_source, diagnostic, full_source = _candidate_plus_tests_source(raw, test_code)
    if not ok_source:
        return False, False, diagnostic, full_source
    (
        attestation_ok,
        attestation_diagnostic,
        full_source,
        completion_marker,
    ) = prepare_dart_test_completion_attestation(full_source)
    if not attestation_ok:
        return False, False, attestation_diagnostic, full_source

    if stability_runs is None:
        stability_runs = int(os.environ.get("EVAL_PASS_STABILITY_RUNS", "1"))
    stability_runs = max(1, int(stability_runs))

    with tempfile.TemporaryDirectory(prefix=f"dart_jit_compilek_{task_id}_", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / 'test.dart'
        path.write_text(full_source, encoding='utf-8')
        last_diagnostic = ""
        for run_index in range(stability_runs):
            try:
                result = subprocess.run(
                    [DART_BIN, 'run', str(path)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout,
                    env=_dart_subprocess_env(tmp),
                )
            except subprocess.TimeoutExpired:
                # The pass-harness JIT metric separates front-end acceptance
                # from runtime correctness. A timeout after source acceptance
                # is a compiled-but-failing candidate.
                return True, False, (
                    f"stability_run_{run_index + 1}/{stability_runs}: "
                    "jit_runtime_timeout_after_frontend"
                ), full_source
            except FileNotFoundError:
                return False, False, "dart_not_found", full_source

            stdout_without_marker = strip_dart_test_completion_marker(
                result.stdout, completion_marker
            )
            diagnostic = (result.stderr or stdout_without_marker or "").strip()
            last_diagnostic = diagnostic
            completed = dart_test_completion_observed(
                result.stdout, completion_marker
            )
            if result.returncode == 0 and completed:
                continue
            if result.returncode == 0:
                return True, False, (
                    f"stability_run_{run_index + 1}/{stability_runs}: "
                    "missing_or_duplicate_completion_attestation"
                ), full_source
            tagged = f"stability_run_{run_index + 1}/{stability_runs}: {diagnostic}"
            if _is_dart_jit_static_error(diagnostic):
                return False, False, tagged, full_source
            return True, False, tagged or "runtime_or_test_failure_after_jit_compile", full_source
        return True, True, last_diagnostic, full_source


def compile_dart_jit_tests_detail(raw: str, test_code: str, task_id: str, timeout: int = 30) -> tuple[bool, str, str]:
    """JIT/pass-harness compile classification, compatible with legacy callers."""
    compiled, _passed, diagnostic, full_source = evaluate_dart_jit_tests_detail(
        raw, test_code, task_id, timeout
    )
    return compiled, diagnostic, full_source


def compile_dart(raw: str, timeout: int = 30) -> bool:
    ok, _, _ = compile_dart_detail(raw, timeout=timeout)
    return ok


def compile_swift(code: str) -> bool:
    return bool(code.strip())


def compile_at_k_estimator(n: int, c: int, k: int) -> float:
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
                        help="Parallel Dart compile workers. Use 64-128 on high-core CPU boxes; too high can hurt via tempdir/disk contention")
    parser.add_argument('--timeout', type=int, default=30,
                        help="Per-candidate Dart compile timeout in seconds")
    parser.add_argument('--debug_failures', type=int, default=0,
                        help="Print compiler diagnostics for the first N failed Dart candidates")
    parser.add_argument('--compile_mode', choices=['legacy', 'tests', 'jit_tests'], default='legacy',
                        help="legacy: old standalone wrapper compile. tests: strict AOT compile candidate+tests. jit_tests: pass-harness Dart run compile classifier")
    args = parser.parse_args()
    validate_dart_binary()
    workers = max(1, int(args.workers))
    print(f"Using Dart binary: {DART_BIN} | workers={workers}", file=sys.stderr)

    rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    k_list = [int(x.strip()) for x in args.k_values.split(',')]

    compile_sums = {k: 0.0 for k in k_list}
    total = len(rows)

    # Many multi-sample model outputs contain exact duplicates or repeated empty
    # candidates. Cache by extracted code so duplicate candidates still count
    # toward n/c, but do not pay for another Dart AOT subprocess.
    compile_cache: dict[tuple[str, str, str], tuple[bool, str, str]] = {}
    compile_cache_lock = Lock()

    def compile_dart_cached(raw: str, test_code: str, task_id: str) -> tuple[bool, str, str]:
        code = _extract_code(raw)
        key = (args.compile_mode, code, test_code if args.compile_mode in {'tests', 'jit_tests'} else "")
        with compile_cache_lock:
            cached = compile_cache.get(key)
            if cached is not None:
                return cached
        if args.compile_mode == 'jit_tests' and test_code:
            ok, diagnostic, normalized_code = compile_dart_jit_tests_detail(raw, test_code, task_id, timeout=args.timeout)
        elif args.compile_mode == 'tests' and test_code:
            ok, diagnostic, normalized_code = compile_dart_tests_detail(raw, test_code, task_id, timeout=args.timeout)
        else:
            ok, diagnostic, normalized_code = compile_dart_detail(raw, timeout=args.timeout)
        with compile_cache_lock:
            compile_cache.setdefault(key, (ok, diagnostic, normalized_code))
            return compile_cache[key]

    def score_row(item: tuple[int, dict]) -> tuple[int, int, int, list[tuple[str, str]]]:
        idx, row = item
        lang = row.get('language', 'dart').lower()
        candidates = row.get('predictions', [row.get('prediction', '')])
        test_code = row.get('tests', '')
        task_id = str(row.get('id', idx))

        if not candidates:
            candidates = [row.get('prediction', '')]

        c = 0
        failures: list[tuple[str, str]] = []
        for cand in candidates:
            if lang == 'dart':
                ok, diagnostic, code = compile_dart_cached(cand, test_code, task_id)
                if not ok and len(failures) < 1:
                    failures.append((diagnostic, code[:600].replace("\n", "\\n")))
            else:
                ok = compile_swift(cand)
            if ok:
                c += 1
        return idx, len(candidates), c, failures

    # Bar on stderr: the runner captures this script's stdout for the JSON
    # result, so stderr is the only live channel.
    indexed_rows = list(enumerate(rows))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            row_results = list(
                tqdm(pool.map(score_row, indexed_rows), total=len(indexed_rows), desc="compile@k", unit="row", dynamic_ncols=True)
                if tqdm is not None else pool.map(score_row, indexed_rows)
            )
    else:
        row_iter = tqdm(indexed_rows, desc="compile@k", unit="row", dynamic_ncols=True) if tqdm is not None else indexed_rows
        row_results = [score_row(item) for item in row_iter]

    debug_printed = 0
    for idx, n, c, failures in row_results:
        if failures and debug_printed < args.debug_failures:
            diagnostic, preview = failures[0]
            debug_printed += 1
            print(_safe_console(f"\n--- Compile failure #{debug_printed} row={idx} ---"))
            print(_safe_console(diagnostic[:1600]))
            print(_safe_console(f"candidate_preview={preview}"))

        # Compute compile@k for each k
        for k in k_list:
            if n >= k:
                val = compile_at_k_estimator(n, c, k)
                compile_sums[k] += val
            else:
                # If we generated fewer than k samples, compile@k defaults to compile@n
                val = compile_at_k_estimator(n, c, n)
                compile_sums[k] += val

        print(f'[{idx + 1}/{len(rows)}] n={n}, compiling={c}, compile@1={c/n:.4f}')

    results = {}
    for k in k_list:
        results[f'compile_at_{k}'] = compile_sums[k] / max(total, 1)

    results['total_problems'] = total
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
