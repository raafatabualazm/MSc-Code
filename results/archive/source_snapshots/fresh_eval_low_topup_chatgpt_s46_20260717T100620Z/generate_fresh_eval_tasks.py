"""Generate FRESH evaluation tasks for the Dart AOT decompilation benchmark.

Motivation (clean-study audit, 2026-07-15): the 154-task benchmark is
HumanEval-Dart — task identity is recoverable from the exact signature alone
(signature-only pass@10 0.234), so architecture knobs compete over a handful of
marginal tasks. This script builds a NEW evaluation pool whose tasks are
original (not HumanEval paraphrases), leak-checked against BOTH existing
corpora, and drawn from the SAME DISTRIBUTION as the benchmark: the shape
targets (return types, parameter counts, implementation length, CFG
block-count strata) are MEASURED from the seed corpora, not hardcoded.

Improvements over generate_synthetic_tasks_parallel.py:
  1. Empirical distribution matching — return-type and parameter-count
     sampling weights come from the 154-task benchmark's signatures; per-
     stratum implementation-length bands come from the benchmark sources
     grouped by their real CFG block counts; a distribution-match report
     (fresh vs benchmark) is written next to the output.
  2. Mutation-testing gate (Gate M) — every accepted task's tests must KILL
     mutated variants of the reference (comparison flips, operator swaps,
     boundary shifts). Tasks whose tests cannot discriminate broken
     implementations are rejected; the kill rate is recorded per row.
  3. Stability gate — the reference must pass its own tests on N independent
     runs (defends against the one-in-3450 stochastic false positive found by
     the clean-study stability audit).
  4. Stratum quotas — the ACCEPTED set is forced to the target complexity mix,
     not just the attempted mix.
  5. DeepSeek direct API support alongside OpenRouter and Azure Foundry, with
     JSON-mode when available; per-provider acceptance stats to expose
     single-generator style bias.

Seeds (read-only, never copied into outputs):
  * data/testing/grpo_data_graphv2.jsonl          (154-task benchmark, with CFG)
      -> banned names, dedup pool, CFG strata, type/length distribution targets.
  * data/datasets/synthetic_pool_graphv2.jsonl    (1,726 synthetic tasks, with CFG)
      -> banned names, dedup pool, style exemplars (signature + expects only).

Pipeline per accepted task:
  prompt (category x flavor x measured rtype/params/length x CFG-stratum)
    -> Gate A  structural JSON contract     (reused from generate_synthetic_tasks_parallel)
    -> Gate N  novelty/leakage: banned names + token-Jaccard/SequenceMatcher
               dedup vs benchmark, synthetic pool, and already-accepted tasks
    -> Gate B  functional + stability: reference passes its own tests
               --stability_runs times (local Dart)
    -> Gate M  mutation: >= --mutation_min_kill of viable mutants must fail
    -> Gate Q  stratum quota
    -> Gate E  optional (--with_assembly): AOT compile + GDB dump (needs gdb)
    -> row in the benchmark schema + flat provenance fields.

Providers (OpenAI-compatible /chat/completions):
  openrouter : OPENROUTER_API_KEY. --openrouter_models takes raw slugs
               ("openai/gpt-5.5") or or_* aliases from
               generate_synthetic_tasks_parallel.OPENROUTER_MODELS.
  azure      : AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (Azure AI Foundry /
               Azure OpenAI). --azure_models are DEPLOYMENT names.
                 v1 (recommended): AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/openai/v1
                                   (leave --azure_api_version empty)
                 classic:          AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/openai/deployments/<dep>
                                   --azure_api_version 2024-12-01-preview
               GPT-5-class / o-series reasoning deployments reject
               temperature/top_p; they are OMITTED unless --azure_send_sampling,
               and provenance then records temperature="deployment_default".
  deepseek   : DEEPSEEK_API_KEY, direct https://api.deepseek.com API.
               --deepseek_models (default deepseek-chat; deepseek-reasoner also
               works — it silently ignores sampling parameters). JSON mode
               (response_format json_object) is requested and dropped
               automatically if the model rejects it.

Usage:
  # 1) offline sanity check (no network, runs local Dart incl. mutation gate):
  python generate_fresh_eval_tasks.py --self_test

  # 2) inspect one rendered prompt + the attempt plan + measured targets:
  python generate_fresh_eval_tasks.py --num_tasks 60 --dry_run

  # 3) real run, all three providers:
  export OPENROUTER_API_KEY=sk-or-...
  export AZURE_OPENAI_API_KEY=...
  export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1
  export DEEPSEEK_API_KEY=sk-...
  python generate_fresh_eval_tasks.py --num_tasks 60 \
      --providers openrouter,azure,deepseek \
      --openrouter_models "anthropic/claude-sonnet-4.6" \
      --azure_models gpt-5.5 --deepseek_models deepseek-chat,deepseek-reasoner \
      --out data/testing/fresh_eval_llm.jsonl

  # 4) afterwards, on a machine with gdb (the graph pipeline pod):
  python generate_fresh_eval_tasks.py --resume_assembly --out data/testing/fresh_eval_llm.jsonl
  python scripts/data/build_graph_v2_jsonl.py \
      --input data/testing/fresh_eval_llm.jsonl \
      --output data/testing/fresh_eval_llm_graphv2.jsonl \
      --rejected data/testing/fresh_eval_llm_graphv2.rejected.jsonl \
      --summary data/testing/fresh_eval_llm_graphv2.summary.json \
      --drop_invalid --max_block_instrs 20
  # and for the recognition-free variant feed the graphv2 file to
  # scripts/data/build_signature_scrubbed_eval.py --benchmark_kind fresh_holdout

Freeze-rule support: a manifest (config + SHA-256 of both seed files + prompt
template hash + rng seed) is written BEFORE the first request, matching the
comprehensive suite's "hash the holdout manifest before its tasks are
generated" rule. Every reject is logged with its gate and reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ======================================================================
# Vendored VERBATIM from generate_synthetic_tasks_parallel.py (2026-07-15)
# so this file is fully self-contained when copied alone to a pod. If the
# donor file's gates change, re-vendor rather than editing these copies.
# ======================================================================
GEN_SYSTEM = 'You generate self-contained Dart programming tasks used to build a compiler-verified dataset. Output STRICT JSON only — no markdown fences, no prose.'

CATEGORIES = [
    "string parsing and tokenization",
    "list/collection transformations (map, fold, dedupe, partition)",
    "numeric algorithms (gcd, primes, base conversion, modular arithmetic)",
    "bit manipulation (masks, popcount, rotations, flags)",
    "recursion (trees, divide and conquer, backtracking on small inputs)",
    "Map/Set based frequency counting and grouping",
    "sorting and ordering with custom comparators",
    "two-pointer and sliding-window techniques",
    "stack/queue simulation (matching brackets, RPN evaluation, undo logs)",
    "matrix / 2D grid traversal and transformation",
    "integer date and duration arithmetic (day indices, intervals, no DateTime)",
    "string encoding/decoding (run-length, caesar, simple checksums)",
    "geometry on integer coordinates (areas, overlaps, manhattan distance)",
    "state machine simulation (parsers for tiny grammars, token scanners)",
    "dynamic programming on 1D/2D arrays (small, iterative)",
    "search problems (binary search variants, kth element)"
]

FLAVORS = [
    "warehouse inventory",
    "music playlists",
    "chess board squares",
    "DNA base strings",
    "sensor telemetry samples",
    "server log lines",
    "election vote tallies",
    "shipping manifests",
    "RGB pixel values",
    "morse code symbols",
    "theater seating charts",
    "bank ledger entries",
    "dice game rounds",
    "satellite pass windows",
    "soil moisture grid",
    "barcode digits",
    "elevator floor requests",
    "wifi signal strength bins",
    "recipe ingredient scaling",
    "traffic light phases",
    "playing card decks",
    "password rule checks",
    "maze grid cells",
    "thermostat schedules",
    "tournament brackets",
    "ASCII art rows",
    "hash bucket counts",
    "train timetable minutes",
    "battery charge cycles",
    "QR code modules",
    "library shelf codes",
    "network packet sizes",
    "tide height readings",
    "spell-checker edits",
    "currency cent rounding",
    "game item inventories"
]

OPENROUTER_BASE = 'https://openrouter.ai/api/v1'
OPENROUTER_MODELS = {
    "fable": "anthropic/claude-fable-5",
    "opus": "anthropic/claude-opus-4.8",
    "sonnet": "anthropic/claude-sonnet-4.6",
    "gpt": "openai/gpt-5.5",
    "gpt_pro": "openai/gpt-5.5-pro",
    "deepseek": "deepseek/deepseek-v4-pro",
    "kimi": "moonshotai/kimi-k2.6",
    "qwen_max": "qwen/qwen3.7-max",
    "qwen_plus": "qwen/qwen3.7-plus",
    "glm": "z-ai/glm-5.1",
    "glm_turbo": "z-ai/glm-5-turbo",
    "minimax": "minimax/minimax-m3",
    "gemini_pro": "google/gemini-3.1-pro-preview",
    "gemini_flash": "google/gemini-3.5-flash",
    "grok": "x-ai/grok-4.3",
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b:free"
}

EXPECT_HARNESS = "\nvoid expect(dynamic a, dynamic b) {\n  if (a == b) return;\n\n  if (a is List && b is List) {\n    expectList(a, b);\n  } else if (a is Map && b is Map) {\n    expectMap(a, b);\n  } else {\n    throw '$a != $b';\n  }\n}\n\nvoid expectList(List a, List b) {\n  if (a.length != b.length) throw 'list lengths are not equal';\n\n  for (var i = 0; i < a.length; i++) {\n    expect(a[i], b[i]);\n  }\n}\n\nvoid expectMap(Map a, Map b) {\n  if (a.length != b.length) throw 'map lengths are not equal';\n\n  for (var key in a.keys) {\n    expect(a[key], b[key]);\n  }\n}"

CAMEL_RE = re.compile('^[a-z][a-zA-Z0-9]*$')
BANNED_SUBSTRINGS = ('dart:io', 'dart:async', 'Future<', 'await ', 'Stream<', 'print(', 'Random', 'DateTime', 'Stopwatch', 'stdin', 'stdout')
TOKEN_RE = re.compile('[A-Za-z_][A-Za-z0-9_]*|\\d+|[^\\sA-Za-z0-9_]')

GATE_TMP_ROOT = Path("gate_tmp")
DEBUG_DIR = Path("gate_failures")
_fail_counts: Counter = Counter()
_log_lock = threading.Lock()
FAIL_LOG_CAP = 25


def normalize(code: str) -> list[str]:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return [t.lower() for t in TOKEN_RE.findall(code)]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def too_similar(code: str, pool: list[dict], jac_thr: float, seq_thr: float) -> bool:
    toks = normalize(code)
    tset = set(toks)
    joined = " ".join(toks)
    for entry in pool:
        if joined == entry["joined"]:
            return True
        j = jaccard(tset, entry["tset"])
        # Short Dart functions share a small vocabulary, so token-set overlap
        # alone is not evidence of copying. Require both overlap and ordered
        # similarity unless the normalized token stream is exactly equal.
        if j >= jac_thr and SequenceMatcher(
                None, joined, entry["joined"]).ratio() >= seq_thr:
            return True
    return False


def pool_entry(code: str, name: str) -> dict:
    toks = normalize(code)
    return {"tset": set(toks), "joined": " ".join(toks), "name": name}


def parse_candidate(raw: str) -> dict | None:
    if raw is None:
        return None
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    required = {"function_name", "signature", "dart_function", "main_asserts",
                "test_expects"}
    if not required.issubset(obj):
        return None
    name = obj["function_name"]
    if not isinstance(name, str) or not CAMEL_RE.match(name):
        return None
    if name not in obj["dart_function"] or name not in obj["signature"]:
        return None
    if not isinstance(obj["main_asserts"], list) or not isinstance(obj["test_expects"], list):
        return None
    if not 8 <= len(obj["test_expects"]) <= 12 or len(obj["main_asserts"]) < 1:
        return None
    if any(
        not isinstance(test, str)
        or re.match(r"^\s*expect\s*\(\s*candidate\s*\(", test) is None
        for test in obj["test_expects"]
    ):
        return None
    if len({re.sub(r"\s+", "", test) for test in obj["test_expects"]}) != len(obj["test_expects"]):
        return None
    if "void main" in obj["dart_function"]:
        return None
    if any(b in obj["dart_function"] for b in BANNED_SUBSTRINGS):
        return None
    return obj


def build_tests_field(name: str, expects: list[str]) -> str:
    lines = "\n".join(f"  {e.strip()}" for e in expects)
    return (f"void main() {{\n  final candidate = {name};\n\n{lines}\n}}\n"
            + EXPECT_HARNESS + "\n")


def build_dart_source(func: str, asserts: list[str]) -> str:
    a = "\n".join(f"  {s.strip()}" for s in asserts)
    return (f"@pragma('vm:entry-point')\n{func.strip()}\n\n"
            f"@pragma('vm:entry-point')\nvoid main() {{\n{a}\n"
            f"  print('All tests passed!');\n}}")


def run(
    cmd: list[str],
    cwd: str,
    timeout: int = 90,
    output_limit: int = 4000,
) -> tuple[int, str]:
    # Keep Dart telemetry and pub state inside the disposable gate directory.
    # Ephemeral Windows/Linux workers often have a read-only profile; without
    # this isolation a valid compile can return 1 only because analytics could
    # not write its session file.
    env = os.environ.copy()
    sandbox_home = (Path(cwd).resolve() / ".dart_home")
    appdata = sandbox_home / "AppData" / "Roaming"
    localappdata = sandbox_home / "AppData" / "Local"
    pub_cache = sandbox_home / ".pub-cache"
    for path in (sandbox_home, appdata, localappdata, pub_cache):
        path.mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(sandbox_home),
        "USERPROFILE": str(sandbox_home),
        "APPDATA": str(appdata),
        "LOCALAPPDATA": str(localappdata),
        "PUB_CACHE": str(pub_cache),
        "CI": "true",
        "DART_SUPPRESS_ANALYTICS": "1",
    })
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        output = p.stdout + p.stderr
        if output_limit > 0:
            output = output[-output_limit:]
        return p.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def gate_tmpdir():
    GATE_TMP_ROOT.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="dartgen_", dir=GATE_TMP_ROOT)


def log_gate_failure(gate: str, output: str, source: str) -> None:
    with _log_lock:
        _fail_counts[gate] += 1
        n = _fail_counts[gate]
        if n > FAIL_LOG_CAP:
            return
        DEBUG_DIR.mkdir(exist_ok=True)
        p = DEBUG_DIR / f"{gate}_{n:02d}.log"
        p.write_text(f"=== GATE: {gate} ===\n\n--- tool output ---\n{output}\n\n"
                     f"--- source ---\n{source}\n")


def _gdb_dump(target: Path, name: str, cwd: str) -> tuple[int, str]:
    # GDB parses the text after `-ex file` itself, so subprocess argument
    # quoting is not enough for Windows paths containing spaces.
    gdb_target = target.resolve().as_posix().replace('"', '\\"')
    # GDB output is the training input. Never apply the generic diagnostic
    # tail limit here: it used to remove the start of large functions while
    # retaining `End of assembler dump.`, producing apparently complete but
    # control-flow-open records.
    return run([
        "gdb", "-batch", "-q",
        "-ex", "set disassembly-flavor intel",
        "-ex", f'file "{gdb_target}"',
        "-ex", f"info functions {name}",
        "-ex", f"disassemble {name}",
    ], cwd, timeout=120, output_limit=0)


def gate_assembly(name: str, dart_source: str, build_dir: Path,
                  filename: str) -> str | None:
    """Compile and disassemble. `dart compile exe` appends the AOT snapshot to
    a stripped runtime ELF, so gdb sees no Dart symbols there. The
    aot-snapshot output is a bare ELF with Dart symbols + DWARF (file:// URI
    comp units) — matching the eval set's dumps — so we use it first and keep
    exe only as a fallback for SDKs where it remains symbol-visible."""
    build_dir.mkdir(parents=True, exist_ok=True)
    src_path = build_dir / filename
    src_path.write_text(dart_source)
    fail_outputs = []
    for kind, ext in (("aot-snapshot", ".aot"), ("exe", ".exe")):
        out_path = build_dir / (filename.replace(".dart", "") + ext)
        rc, out = run(["dart", "compile", kind, str(src_path), "-o", str(out_path)],
                      str(build_dir), timeout=180)
        if rc != 0:
            fail_outputs.append(f"[compile {kind}] rc={rc}\n{out}")
            continue
        rc, out = _gdb_dump(out_path, name, str(build_dir))
        out_path.unlink(missing_ok=True)  # keep .dart sources, drop binaries
        if rc == 0 and "Dump of assembler code" in out:
            idx = out.find("All functions matching")
            if idx == -1:
                idx = out.find("Dump of assembler code")
            asm = out[idx:].strip()
            if not asm.endswith("End of assembler dump."):
                asm += "\nEnd of assembler dump."
            return asm
        fail_outputs.append(f"[gdb on {kind}] rc={rc}\n{out}")
    log_gate_failure("assembly", "\n\n".join(fail_outputs), dart_source)
    return None


ROOT = Path(__file__).resolve().parent

DEFAULT_BENCHMARK = "data/testing/grpo_data_graphv2.jsonl"
DEFAULT_SYNTHETIC = "data/datasets/synthetic_pool_graphv2.jsonl"
DEEPSEEK_BASE = "https://api.deepseek.com/v1"

# --------------------------------------------------------------------------
# Prompt: same strict JSON contract as the synthetic generator (so Gate A is
# reusable verbatim); shape targets are filled from MEASURED seed statistics.
# --------------------------------------------------------------------------
EVAL_GEN_PROMPT = """Create ONE original Dart function task for a COMPILER-LEVEL evaluation set. Requirements:

- Category: {category}
- Theme the problem around: {flavor}
- Return type: {rtype}
- Use exactly {param_count} parameter(s).
- HARD SIZE LIMIT: the function body must be AT MOST {loc_hi} non-blank lines
  (target {loc_lo}-{loc_hi}) and contain AT MOST {branch_hi} branch points total
  (counting every if / else if / for / while / case / catch / && / || / ?: and
  every early return, break, continue). This is a strict ceiling — a longer or
  more branch-heavy function is REJECTED automatically, not scored. Prefer the
  simplest algorithm that satisfies the contract.
- CONTROL-FLOW TARGET: {stratum_desc} When compiled ahead-of-time this
  function should produce roughly {block_lo}-{block_hi} basic blocks, so shape
  the algorithm's branching accordingly. Do not pad with dead code, do not add
  defensive input validation beyond what the contract needs, and do not unroll
  loops or split simple conditions into many statements.
- Pure function: deterministic, no I/O, no dart:io, no async, no Random,
  no DateTime, no Stopwatch, no external packages.
- Use only dart:core / dart:math (math only for sqrt/pow/min/max if needed).
- Parameters and return types must be concrete (int, double, bool, String,
  List<...>, Map<...>, Set<...>) — no dynamic, no generics in the signature.
- If using double, only values exactly representable in binary floating point
  (halves, quarters) so == comparisons in tests are exact.
- ORIGINALITY IS MANDATORY: this is an anti-memorization holdout. Do NOT
  recreate HumanEval, MBPP, LeetCode, or interview classics in any disguise,
  and do NOT reuse or trivially reword any of these existing function names:
  {banned_sample}.
  Invent a problem whose SEMANTICS (not just its name) differ from textbook
  tasks. Combining two mundane requirements from the theme is a good tactic.
- Function name: lowerCamelCase, descriptive, specific to the theme.
- TESTS MUST DISCRIMINATE: your expect() cases will be mutation-tested. Cover
  boundary values (empty input, exact thresholds, off-by-one neighbors, sign
  flips) so that a reference with a flipped comparison or shifted constant
  FAILS at least one test.

Style calibration — the evaluation set uses compact library functions shaped
like these (signatures and test style only; your task must be semantically
unrelated to them):
{exemplars}

Return JSON with EXACTLY these keys:
{{
  "function_name": "camelCaseName",
  "signature": "ReturnType camelCaseName(ArgType a, ...)",
  "dart_function": "<the complete function definition, no main(), no imports unless dart:math>",
  "main_asserts": ["assert(camelCaseName(...) == ...);", "..."],
  "test_expects": ["expect(candidate(...), ...);", "..."]
}}

main_asserts: 2-3 simple assert statements on scalar-returning calls
(for List/Map returns, compare .toString() or a scalar property instead of ==).
test_expects: 8-12 unique expect(...) lines covering normal cases, edge cases
(empty input, single element, boundary values). Each must be a complete Dart
statement referencing `candidate`. Expected values must be CORRECT — compute
them carefully step by step before writing them."""

STRATA = ("low", "mid", "high")
STRATUM_PROMPTS = {
    "low": ("Keep control flow MINIMAL: a single loop and at most one or two "
            "simple conditional decisions. No nested loops. This is the most "
            "important constraint — err on the side of too simple."),
    "mid": ("Use moderate control flow: one loop containing a conditional, or "
            "two short sequential loops. At most one level of nesting."),
    "high": ("Use rich control flow: multiple interacting loops (at least one "
             "nested), several distinct conditional branches, and early "
             "returns or continues where natural."),
}
STRATUM_TO_DIFFICULTY = {"low": "easy", "mid": "medium", "high": "hard"}

# Branch-point family used both in the prompt ceiling and the pre-assembly
# shape gate. Kept deliberately simple/regex-countable so the number the model
# is told to respect is the number the gate measures.
_BRANCH_RE = re.compile(
    r"\b(if|for|while|case|catch)\b|&&|\|\||\?(?!\.)|\breturn\b|\bbreak\b|\bcontinue\b"
)


# --------------------------------------------------------------------------
# Signature / source statistics (the same-distribution machinery)
# --------------------------------------------------------------------------
def split_signature(signature: str) -> tuple[str, int] | None:
    """Return (return_type, param_count) from a Dart signature, counting only
    TOP-LEVEL commas so generics like Map<String, int> are one parameter."""
    sig = signature.strip().rstrip(";").strip()
    paren = sig.find("(")
    if paren <= 0:
        return None
    head = sig[:paren].strip()
    name_match = re.search(r"([A-Za-z_]\w*)\s*$", head)
    if not name_match:
        return None
    rtype = head[: name_match.start()].strip()
    if not rtype:
        return None
    inner = sig[paren + 1 : sig.rfind(")")] if ")" in sig else sig[paren + 1 :]
    inner = inner.strip()
    if not inner:
        return rtype, 0
    depth = 0
    count = 1
    for ch in inner:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return rtype, count


def reference_body(dart_source: str) -> str:
    """The reference function text (everything before the generated main())."""
    return re.split(r"@pragma\('vm:entry-point'\)\s*\nvoid\s+main\s*\(", dart_source)[0]


def reference_loc(dart_source: str) -> int:
    """Non-empty lines of the reference function (text before the main())."""
    return sum(1 for line in reference_body(dart_source).splitlines() if line.strip())


def count_branches(function_text: str) -> int:
    """Regex proxy for CFG branch density: if/for/while/case/catch, boolean
    connectives, ternaries, and early exits. Same family the prompt ceiling
    names, so the model is told to respect the number the gate measures."""
    stripped = re.sub(r"//.*", "", function_text)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    return len(_BRANCH_RE.findall(stripped))


def percentile(sorted_values: list, frac: float):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round(frac * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def build_seed_context(benchmark_rows: list[dict], synthetic_rows: list[dict],
                       rng: random.Random, n_exemplars: int,
                       decontam_rows: list[dict] | None = None):
    """Banned names + dedup pool from BOTH corpora; distribution targets
    (strata ranges, return types, param counts, per-stratum LOC bands) from
    the BENCHMARK, which is the distribution the fresh set must match."""
    banned: set[str] = set()
    dedup_pool: list[dict] = []
    for row in benchmark_rows + synthetic_rows + list(decontam_rows or []):
        for key in ("function", "camel_case_function_name", "python_function_name"):
            name = row.get(key)
            if name:
                banned.add(name.strip().lower())
                banned.add(snakeify(name.strip()))
        src = row.get("dart_source", "")
        if src:
            # Candidates contain only the reference function. Compare like with
            # like so generated test harnesses cannot dilute similarity.
            dedup_pool.append(pool_entry(reference_body(src), row.get("function", "?")))

    # CFG block-count terciles from the benchmark define the strata targets.
    blocks = sorted(len(r.get("cfg") or []) for r in benchmark_rows if r.get("cfg"))
    if blocks:
        t1 = blocks[len(blocks) // 3]
        t2 = blocks[2 * len(blocks) // 3]
        strata_ranges = {"low": (max(blocks[0], 2), t1),
                         "mid": (t1, t2),
                         "high": (t2, blocks[-1])}
    else:
        strata_ranges = {"low": (3, 14), "mid": (14, 24), "high": (25, 60)}

    def stratum_of(block_count: int) -> str:
        if block_count <= strata_ranges["low"][1]:
            return "low"
        if block_count <= strata_ranges["mid"][1]:
            return "mid"
        return "high"

    # Measured return-type and param-count weights.
    rtypes: Counter = Counter()
    param_counts: Counter = Counter()
    loc_by_stratum: dict[str, list[int]] = {s: [] for s in STRATA}
    branch_by_stratum: dict[str, list[int]] = {s: [] for s in STRATA}
    for row in benchmark_rows:
        parsed = split_signature(row.get("dart_function_signature") or "")
        if parsed:
            rtype, n_params = parsed
            rtypes[rtype] += 1
            param_counts[n_params] += 1
        if row.get("cfg") and row.get("dart_source"):
            s = stratum_of(len(row["cfg"]))
            loc_by_stratum[s].append(reference_loc(row["dart_source"]))
            branch_by_stratum[s].append(count_branches(reference_body(row["dart_source"])))
    loc_bands = {}
    branch_bands = {}
    for stratum in STRATA:
        vals = sorted(loc_by_stratum[stratum])
        lo = max(3, percentile(vals, 0.25))
        hi = max(lo + 4, percentile(vals, 0.75))
        loc_bands[stratum] = (lo, hi)
        bvals = sorted(branch_by_stratum[stratum])
        # ceiling = 90th percentile of the benchmark stratum (a generous but
        # real cap); low/mid get a hard lid, high is effectively unbounded.
        branch_bands[stratum] = max(3, percentile(bvals, 0.90)) if bvals else 6

    # Drop the benchmark's few dynamic/Object signatures (2/154): the prompt
    # forbids them, so they cannot be sampled as targets without contradiction.
    sampleable_rtypes = {t: c for t, c in rtypes.items()
                         if "dynamic" not in t and "Object" not in t}
    dist = {
        "strata_ranges": strata_ranges,
        "rtype_weights": sampleable_rtypes or {"int": 4, "bool": 3, "String": 3,
                                               "List<int>": 2, "double": 1},
        "param_count_weights": dict(param_counts) or {1: 5, 2: 4, 3: 1},
        "loc_bands": loc_bands or {s: (5, 25) for s in STRATA},
        "branch_bands": branch_bands or {"low": 5, "mid": 9, "high": 40},
        "benchmark_loc": sorted(v for vals in loc_by_stratum.values() for v in vals),
    }

    # Style exemplars come from the SYNTHETIC pool only (original tasks); the
    # benchmark's HumanEval signatures must never steer generation.
    exemplars = []
    for row in rng.sample(synthetic_rows, min(n_exemplars, len(synthetic_rows))):
        sig = (row.get("dart_function_signature") or "").strip()
        expects = re.findall(r"expect\(candidate\([^\n]*\);", row.get("tests", ""))[:2]
        if sig:
            lines = [f"  - {sig}"] + [f"      {e}" for e in expects]
            exemplars.append("\n".join(lines))
    return banned, dedup_pool, dist, exemplars


def snakeify(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class ProviderSpec:
    def __init__(self, kind: str, model: str):
        self.kind = kind      # "openrouter" | "azure" | "deepseek"
        self.model = model    # OpenRouter slug / Azure deployment / DeepSeek model
        self.label = f"{kind}:{model}"

    def __repr__(self) -> str:
        return self.label


def resolve_openrouter_models(spec: str) -> list[str]:
    models = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        alias = item[3:] if item.startswith("or_") else item
        if "/" not in item and alias in OPENROUTER_MODELS:
            models.append(OPENROUTER_MODELS[alias])
        else:
            models.append(item)
    return models


def azure_chat_url(api_version: str | None) -> str:
    base = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if not base:
        raise SystemExit("ERROR: set AZURE_OPENAI_ENDPOINT for the azure provider.")
    url = base + "/chat/completions"
    if api_version:
        url += f"?api-version={api_version}"
    return url


# Statuses/markers that mean "this provider cannot serve ANY further request"
# (out of credit, invalid key, subscription disabled) rather than a transient
# fault: OpenRouter and DeepSeek report empty credit as 402; Azure reports a
# disabled subscription as 403 and exhausted quota as 429 with a telltale body.
HARD_FAIL_STATUSES = (401, 402, 403, 404)  # 404 = wrong model/deployment name
HARD_FAIL_MARKERS = ("insufficient", "quota", "balance", "billing", "credit")
# 400 bodies that mean the configured model name is permanently wrong
BAD_MODEL_MARKERS = ("model not exist", "does not exist", "modelnotfound",
                     "deploymentnotfound", "no such model", "unknown model",
                     "invalid model")


def call_provider(spec: ProviderSpec, prompt: str, args,
                  max_retries: int = 4) -> tuple[str | None, bool]:
    """Return (completion_text, provider_dead). provider_dead=True means the
    failure is billing/auth-level and the provider should be circuit-broken
    for the rest of the run. Mirrors the retry/fallback conventions of
    openrouter_baseline.py and llm_baseline_updated.py."""
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    if spec.kind == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return None, True
        url = OPENROUTER_BASE.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/raafatabualazm/MSc-Code",
            "X-Title": "fresh-eval-task-generation",
        }
        payload = {
            "model": spec.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
    elif spec.kind == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            return None, True
        url = DEEPSEEK_BASE.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        # deepseek-chat honors sampling; deepseek-reasoner silently ignores it.
        # JSON mode improves Gate-A yield; the 400-fallback below drops it if
        # the selected model rejects response_format.
        payload = {
            "model": spec.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "response_format": {"type": "json_object"},
        }
    else:  # azure
        key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not key:
            return None, True
        url = azure_chat_url(args.azure_api_version or None)
        headers = {"api-key": key, "Content-Type": "application/json"}
        # Reasoning deployments reject max_tokens AND temperature/top_p; start
        # with the modern parameter and only add sampling when asked.
        payload = {
            "model": spec.model,
            "messages": messages,
            "max_completion_tokens": args.max_tokens,
        }
        if args.azure_send_sampling:
            payload["temperature"] = args.temperature
            payload["top_p"] = args.top_p

    token_keys = ("max_tokens", "max_completion_tokens")
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers,
                              timeout=args.request_timeout)
            if r.status_code == 429:
                # plain rate limit -> back off; exhausted credit/quota -> dead
                if any(m in r.text.lower() for m in HARD_FAIL_MARKERS):
                    print(f"    [dead] {spec}: HTTP 429 quota/credit: "
                          f"{r.text[:300]}", file=sys.stderr)
                    return None, True
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code in HARD_FAIL_STATUSES:
                print(f"    [dead] {spec}: HTTP {r.status_code}: {r.text[:300]}",
                      file=sys.stderr)
                return None, True
            if r.status_code == 400:
                if "response_format" in payload and "response_format" in r.text:
                    payload.pop("response_format")
                    continue
                # Swap max_tokens <-> max_completion_tokens once if the server
                # names the parameter in its complaint.
                for a, b in (token_keys, token_keys[::-1]):
                    if a in payload and a in r.text:
                        payload[b] = payload.pop(a)
                        break
                else:
                    if any(m in r.text.lower() for m in BAD_MODEL_MARKERS):
                        print(f"    [dead] {spec}: bad model/deployment name: "
                              f"{r.text[:300]}", file=sys.stderr)
                        return None, True
                    print(f"    [error] {spec}: HTTP 400: {r.text[:300]}", file=sys.stderr)
                    return None, False
                continue
            if 400 < r.status_code < 500:
                print(f"    [error] {spec}: HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
                return None, False
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], False
        except Exception as exc:  # network / 5xx / malformed body
            print(f"    [warn] {spec} attempt {attempt + 1}: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None, False


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_reference(dart_function: str, tests: str, dart_bin: str,
                  timeout: int = 60) -> tuple[int, str]:
    with gate_tmpdir() as workdir:
        src = Path(workdir) / "validate.dart"
        src.write_text(dart_function.strip() + "\n\n" + tests, encoding="utf-8")
        return run([dart_bin, "run", "--enable-asserts", str(src)],
                   workdir, timeout=timeout)


def functional_gate(dart_function: str, tests: str, dart_bin: str,
                    stability_runs: int = 1) -> tuple[bool, str]:
    """Reference must pass its own tests on EVERY run (stability gate)."""
    for _ in range(max(1, stability_runs)):
        rc, out = run_reference(dart_function, tests, dart_bin)
        if rc != 0:
            return False, out[-1500:]
    return True, ""


def novelty_gate(candidate: dict, banned: set[str], dedup_pool: list[dict],
                 jac_thr: float, seq_thr: float) -> str | None:
    """Return a rejection reason or None if the task is acceptably novel."""
    name = candidate["function_name"].strip()
    if name.lower() in banned or snakeify(name) in banned:
        return f"banned function name: {name}"
    if too_similar(candidate["dart_function"], dedup_pool, jac_thr, seq_thr):
        return "source too similar to an existing corpus/accepted task"
    return None


# Mutation operators chosen to usually stay compilable (spaced comparison and
# arithmetic forms avoid generics like List<int>); stillborn mutants (compile
# errors) are excluded from the kill-rate denominator, standard practice.
MUTATION_OPS = [
    (r"(?<=[\w\)\]])\s*==\s*(?=[\w\(\[\-'\"])", " != "),
    (r"(?<=[\w\)\]])\s*!=\s*(?=[\w\(\[\-'\"])", " == "),
    (r"(?<=[\w\)\]]) <= (?=[\w\(\[\-])", " < "),
    (r"(?<=[\w\)\]]) >= (?=[\w\(\[\-])", " > "),
    (r"(?<=[\w\)\]]) < (?=[\w\(\[\-])", " <= "),
    (r"(?<=[\w\)\]]) > (?=[\w\(\[\-])", " >= "),
    (r"(?<=[\w\)\]]) \+ (?=[\w\(\[\-])", " - "),
    (r"(?<=[\w\)\]]) - (?=[\w\(\[])", " + "),
    (r"&&", "||"),
    (r"\|\|", "&&"),
    (r"return true;", "return false;"),
    (r"return false;", "return true;"),
    (r"= 0;", "= 1;"),
    (r"= 1;", "= 0;"),
]


def make_mutants(dart_function: str, cap: int) -> list[str]:
    """One mutant per (operator, rotating site), first `cap` distinct sources."""
    mutants: list[str] = []
    seen = {dart_function}
    for site_round in range(3):
        for pattern, replacement in MUTATION_OPS:
            matches = list(re.finditer(pattern, dart_function))
            if len(matches) <= site_round:
                continue
            m = matches[site_round]
            # splice at the span directly — the lookarounds are zero-width, so
            # the span covers exactly the operator text being replaced
            mutated = dart_function[: m.start()] + replacement + dart_function[m.end():]
            if mutated not in seen:
                seen.add(mutated)
                mutants.append(mutated)
            if len(mutants) >= cap:
                return mutants
    return mutants


def mutation_gate(dart_function: str, tests: str, dart_bin: str,
                  cap: int, min_kill: float) -> tuple[bool, dict]:
    """Tests must fail (kill) at least `min_kill` of the viable mutants.
    Classification: pass -> SURVIVED (bad); test throw/timeout -> KILLED;
    compile error -> stillborn (excluded)."""
    killed = survived = stillborn = 0
    for mutant in make_mutants(dart_function, cap):
        rc, out = run_reference(mutant, tests, dart_bin)
        if rc == 0:
            survived += 1
        elif "Unhandled exception" in out or rc == -1:  # test failure / timeout
            killed += 1
        else:  # compile / analyzer error
            stillborn += 1
    viable = killed + survived
    stats = {"mutants_viable": viable, "mutants_killed": killed,
             "mutants_stillborn": stillborn,
             "kill_rate": (killed / viable) if viable else None}
    if viable == 0:
        # nothing mutable usually means trivially short code; treat as weak
        return False, stats
    return stats["kill_rate"] >= min_kill, stats


# --------------------------------------------------------------------------
# Attempt planning / prompt rendering
# --------------------------------------------------------------------------
def weighted_choice(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def render_prompt(plan: dict, banned: set[str], exemplars: list[str],
                  rng: random.Random) -> str:
    banned_sample = ", ".join(rng.sample(sorted(banned), min(24, len(banned))))
    block_lo, block_hi = plan["block_range"]
    loc_lo, loc_hi = plan["loc_band"]
    return EVAL_GEN_PROMPT.format(
        category=plan["category"],
        flavor=plan["flavor"],
        rtype=plan["rtype"],
        param_count=plan["param_count"],
        loc_lo=loc_lo, loc_hi=loc_hi,
        branch_hi=plan["branch_cap"],
        stratum_desc=STRATUM_PROMPTS[plan["stratum"]],
        block_lo=block_lo, block_hi=block_hi,
        banned_sample=banned_sample,
        exemplars="\n".join(exemplars) if exemplars else "  (none)",
    )


def shape_gate(dart_function: str, plan: dict, loc_tol: float,
               branch_tol: float) -> tuple[bool, str]:
    """Pre-assembly proxy filter: reject candidates whose LOC or branch count
    overshoots the target stratum, since realized CFG block count tracks both
    and cannot be measured before compilation. Returns (ok, reason)."""
    loc = sum(1 for ln in dart_function.splitlines() if ln.strip())
    branches = count_branches(dart_function)
    loc_hi = plan["loc_band"][1]
    branch_hi = plan["branch_cap"]
    loc_cap = int(round(loc_hi * loc_tol))
    branch_cap = int(round(branch_hi * branch_tol))
    if loc > loc_cap:
        return False, f"loc {loc} > cap {loc_cap} (stratum {plan['stratum']})"
    if branches > branch_cap:
        return False, f"branches {branches} > cap {branch_cap} (stratum {plan['stratum']})"
    return True, ""


def make_attempt_plans(args, dist: dict, providers: list[ProviderSpec],
                       rng: random.Random) -> list[dict]:
    """Deterministic attempt grid. Prompt-axis diversity (category x flavor)
    fights mode collapse; shape axes (rtype, params, LOC, stratum) are drawn
    from the MEASURED benchmark distribution."""
    mix = {}
    for part in args.strata_mix.split(","):
        k, v = part.split(":")
        mix[k.strip()] = float(v)
    plans = []
    for i in range(args.num_tasks * args.oversample):
        stratum = rng.choices(STRATA, weights=[mix.get(s, 0.0) for s in STRATA])[0]
        plans.append({
            "index": i,
            "category": CATEGORIES[i % len(CATEGORIES)],
            "flavor": rng.choice(FLAVORS),
            "rtype": weighted_choice(rng, dist["rtype_weights"]),
            "param_count": weighted_choice(rng, dist["param_count_weights"]),
            "stratum": stratum,
            "block_range": dist["strata_ranges"][stratum],
            "loc_band": dist["loc_bands"][stratum],
            "branch_cap": dist["branch_bands"][stratum],
            "provider": providers[i % len(providers)],
        })
    rng.shuffle(plans)
    return plans


# --------------------------------------------------------------------------
# Distribution-match report
# --------------------------------------------------------------------------
def distribution_report(accepted: list[dict], dist: dict) -> dict:
    fresh_rtypes = Counter()
    fresh_params = Counter()
    fresh_loc = []
    for row in accepted:
        parsed = split_signature(row["dart_function_signature"])
        if parsed:
            fresh_rtypes[parsed[0]] += 1
            fresh_params[parsed[1]] += 1
        fresh_loc.append(reference_loc(row["dart_source"]))
    fresh_loc.sort()
    bench_loc = dist["benchmark_loc"]
    return {
        "return_types": {"benchmark": dist["rtype_weights"],
                         "fresh": dict(fresh_rtypes)},
        "param_counts": {"benchmark": dist["param_count_weights"],
                         "fresh": dict(fresh_params)},
        "reference_loc_quartiles": {
            "benchmark": [percentile(bench_loc, f) for f in (0.25, 0.5, 0.75)],
            "fresh": [percentile(fresh_loc, f) for f in (0.25, 0.5, 0.75)],
        },
        "target_strata": dict(Counter(r.get("target_stratum") for r in accepted)),
        "generator_models": dict(Counter(r.get("generator_model") for r in accepted)),
        "note": ("realized CFG strata require the assembly stage; run "
                 "build_graph_v2_jsonl.py and compare block counts per stratum."),
    }


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Self-test (offline; exercises every gate incl. mutation testing)
# --------------------------------------------------------------------------
def self_test(args) -> int:
    parsed = split_signature("Map<String, int> tallyZones(List<int> depths, int cap)")
    assert parsed == ("Map<String, int>", 2), f"signature parser broken: {parsed}"

    candidate = {
        "function_name": "ledgerDriftTotal",
        "signature": "int ledgerDriftTotal(List<int> entries, int cap)",
        "dart_function": (
            "int ledgerDriftTotal(List<int> entries, int cap) {\n"
            "  var drift = 0;\n  var previous = 0;\n"
            "  for (final e in entries) {\n"
            "    var step = e - previous;\n"
            "    if (step < 0) step = -step;\n"
            "    if (step > cap) step = cap;\n"
            "    drift += step;\n    previous = e;\n  }\n"
            "  return drift;\n}"
        ),
        "main_asserts": ["assert(ledgerDriftTotal([2, 5, 1], 10) == 9);"],
        "test_expects": [
            "expect(candidate([], 5), 0);",
            "expect(candidate([3], 5), 3);",
            "expect(candidate([2, 5, 1], 10), 9);",
            "expect(candidate([2, 5, 1], 2), 6);",
            "expect(candidate([0, 0, 0], 4), 0);",
            "expect(candidate([-3], 10), 3);",
            "expect(candidate([-3, 3], 4), 7);",
            "expect(candidate([10, 10, 10], 1), 1);",
        ],
    }
    reparsed = parse_candidate(json.dumps(candidate))
    assert reparsed is not None, "Gate A rejected the self-test candidate"
    tests = build_tests_field(reparsed["function_name"], reparsed["test_expects"])
    ok, diag = functional_gate(reparsed["dart_function"], tests, args.dart_bin,
                               stability_runs=2)
    assert ok, f"Gate B failed: {diag}"
    pool = [pool_entry(reparsed["dart_function"], "self")]
    assert too_similar(reparsed["dart_function"], pool, args.jac_thr, args.seq_thr), \
        "dedup should flag an identical source"
    overlap_only = "int compute(int b, int a) { if (a > b) return b; return a; }"
    common_pool = [pool_entry("int add(int a, int b) { return a + b; }", "add")]
    assert not too_similar(overlap_only, common_pool, args.jac_thr, args.seq_thr), \
        "dedup must not reject common Dart vocabulary without sequence similarity"
    wrapped_source = build_dart_source(
        reparsed["dart_function"], reparsed["main_asserts"] + ["assert(true);"] * 20)
    wrapped_pool = [pool_entry(reference_body(wrapped_source), "wrapped-self")]
    assert too_similar(reparsed["dart_function"], wrapped_pool,
                       args.jac_thr, args.seq_thr), \
        "dedup must compare reference functions without generated main harnesses"
    reason = novelty_gate(reparsed, {"ledgerdrifttotal"}, [], args.jac_thr, args.seq_thr)
    assert reason and "banned" in reason, "banned-name gate failed"
    ok_m, stats = mutation_gate(reparsed["dart_function"], tests, args.dart_bin,
                                cap=args.mutation_max, min_kill=0.0)
    assert stats["mutants_viable"] >= 2 and stats["mutants_killed"] >= 1, \
        f"mutation gate produced no signal: {stats}"

    # shape gate: branch counting + LOC/branch ceilings
    assert count_branches("if(a){for(x in y){if(b&&c)return 1;}}return 0;") >= 4, \
        "branch counter under-counts"
    low_plan = {"stratum": "low", "loc_band": (4, 11), "branch_cap": 5}
    ok_short, _ = shape_gate(reparsed["dart_function"], low_plan, 1.15, 1.15)
    # the 11-line, ~4-branch self-test function should pass a low plan
    assert ok_short, "shape gate wrongly rejected a compact function"
    bloated = "int f(List<int> x){\n" + "\n".join(
        f"  if(x[{i}]>{i}) return {i};" for i in range(20)) + "\n  return 0;\n}"
    ok_big, why = shape_gate(bloated, low_plan, 1.15, 1.15)
    assert not ok_big, "shape gate failed to reject an oversized function"

    print(f"self_test OK: signature parser, Gate A, Gate B x2 runs, dedup, "
          f"banned-name gate, shape gate (LOC/branch), mutation gate "
          f"(viable={stats['mutants_viable']} killed={stats['mutants_killed']} "
          f"stillborn={stats['mutants_stillborn']} kill_rate={stats['kill_rate']:.2f}).")
    return 0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 allow_abbrev=False)
    ap.add_argument("--num_tasks", type=int, default=60)
    ap.add_argument("--oversample", type=int, default=4,
                    help="attempts budget = num_tasks * oversample")
    ap.add_argument("--providers", default="openrouter,azure,deepseek")
    ap.add_argument("--openrouter_models",
                    default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--azure_models", default="gpt-5.5",
                    help="comma-separated Azure deployment names")
    ap.add_argument("--azure_api_version", default="",
                    help="only for the classic deployment endpoint; empty = v1")
    ap.add_argument("--azure_send_sampling", action="store_true")
    ap.add_argument("--deepseek_models", default="deepseek-chat",
                    help="comma-separated DeepSeek model names "
                         "(deepseek-chat, deepseek-reasoner)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--request_timeout", type=int, default=240)
    ap.add_argument("--benchmark", type=Path, default=ROOT / DEFAULT_BENCHMARK)
    ap.add_argument("--synthetic", type=Path, default=ROOT / DEFAULT_SYNTHETIC)
    ap.add_argument(
        "--decontam_jsonl",
        "--decontam-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Additional source/test corpus to exclude from generation; repeatable",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data/testing/fresh_eval_llm.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jac_thr", type=float, default=0.55)
    ap.add_argument("--seq_thr", type=float, default=0.70)
    ap.add_argument("--style_examples", type=int, default=2)
    ap.add_argument("--strata_mix", default="low:0.34,mid:0.33,high:0.33")
    ap.add_argument("--stability_runs", type=int, default=2,
                    help="reference must pass its own tests this many times")
    ap.add_argument("--mutation_max", type=int, default=8,
                    help="max mutants per task (0 disables Gate M)")
    ap.add_argument("--mutation_min_kill", type=float, default=0.5,
                    help="min fraction of viable mutants the tests must kill")
    ap.add_argument("--shape_gate", type=int, default=1,
                    help="1 = enforce pre-assembly LOC/branch caps (Gate S); 0 = off")
    ap.add_argument("--loc_tol", type=float, default=1.15,
                    help="Gate S: allowed LOC overshoot over the stratum's loc-band ceiling")
    ap.add_argument("--branch_tol", type=float, default=1.15,
                    help="Gate S: allowed branch-count overshoot over the stratum cap")
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--dart_bin", default=shutil.which("dart") or "dart")
    ap.add_argument("--with_assembly", action="store_true",
                    help="also AOT-compile + GDB-dump each accepted task (needs gdb)")
    ap.add_argument("--resume_assembly", action="store_true",
                    help="fill missing 'assembly' fields of an existing --out file, then exit")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test(args))

    rng = random.Random(args.rng_seed)
    required_inputs = [
        ("--benchmark", args.benchmark),
        ("--synthetic", args.synthetic),
        *[("--decontam-jsonl", path) for path in args.decontam_jsonl],
    ]
    for label, seed_path in required_inputs:
        if not seed_path.is_file():
            raise SystemExit(
                f"ERROR: seed file not found: {seed_path}\n"
                f"This script needs BOTH seed corpora next to it (or pass {label}):\n"
                "  data/testing/grpo_data_graphv2.jsonl          (154-task benchmark)\n"
                "  data/datasets/synthetic_pool_graphv2.jsonl    (synthetic pool)\n"
                "Copy them from the workspace with the same relative layout, or point\n"
                "--benchmark/--synthetic at their actual locations.")
    benchmark_rows = load_jsonl(args.benchmark)
    synthetic_rows = load_jsonl(args.synthetic)
    decontam_rows = [
        row
        for path in args.decontam_jsonl
        for row in load_jsonl(path)
    ]
    banned, dedup_pool, dist, exemplars = build_seed_context(
        benchmark_rows, synthetic_rows, rng, args.style_examples, decontam_rows)
    print(f"seeds: benchmark={len(benchmark_rows)} synthetic={len(synthetic_rows)} "
          f"extra_decontam={len(decontam_rows)} banned_names={len(banned)} "
          f"dedup_pool={len(dedup_pool)}")
    print(f"measured targets: strata={ {k: list(v) for k, v in dist['strata_ranges'].items()} } "
          f"loc_bands={ {k: list(v) for k, v in dist['loc_bands'].items()} }")
    print(f"  rtype_weights={dist['rtype_weights']}")
    print(f"  param_count_weights={dist['param_count_weights']}")

    # ------------------------------------------------------------------ #
    # assembly backfill mode (run on the gdb-equipped pod)
    # ------------------------------------------------------------------ #
    if args.resume_assembly:
        rows = load_jsonl(args.out)
        # must be absolute: run() executes dart/gdb with cwd=build_dir, so a
        # relative path here would be re-resolved against itself and fail
        build_dir = (args.out.parent / (args.out.stem + "_build")).resolve()
        filled = 0
        for row in rows:
            if row.get("assembly"):
                continue
            asm = gate_assembly(row["function"], row["dart_source"], build_dir,
                                row["filename"])
            if asm:
                row["assembly"] = asm
                filled += 1
            else:
                print(f"  [warn] assembly failed for {row['task_id']}", file=sys.stderr)
        args.out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")
        print(f"assembly backfill: filled {filled}, "
              f"missing {sum(1 for r in rows if not r.get('assembly'))} / {len(rows)}")
        print("next: scripts/data/build_graph_v2_jsonl.py to attach CFG/DFG "
              "(see module docstring).")
        return

    # ------------------------------------------------------------------ #
    # provider plan
    # ------------------------------------------------------------------ #
    providers: list[ProviderSpec] = []
    wanted = {p.strip() for p in args.providers.split(",") if p.strip()}
    if "openrouter" in wanted:
        for m in resolve_openrouter_models(args.openrouter_models):
            providers.append(ProviderSpec("openrouter", m))
    if "azure" in wanted:
        for m in [s.strip() for s in args.azure_models.split(",") if s.strip()]:
            providers.append(ProviderSpec("azure", m))
    if "deepseek" in wanted:
        for m in [s.strip() for s in args.deepseek_models.split(",") if s.strip()]:
            providers.append(ProviderSpec("deepseek", m))
    if not providers:
        raise SystemExit("ERROR: no providers configured "
                         "(--providers openrouter,azure,deepseek).")

    plans = make_attempt_plans(args, dist, providers, rng)

    if args.dry_run:
        print(f"\nattempt plan: {len(plans)} attempts over "
              f"{len(providers)} provider-models: {[p.label for p in providers]}")
        print("\n----- sample rendered prompt -----\n")
        print(render_prompt(plans[0], banned, exemplars, rng))
        return

    missing_keys = []
    if any(p.kind == "openrouter" for p in providers) and not os.environ.get("OPENROUTER_API_KEY"):
        missing_keys.append("OPENROUTER_API_KEY")
    if any(p.kind == "deepseek" for p in providers) and not os.environ.get("DEEPSEEK_API_KEY"):
        missing_keys.append("DEEPSEEK_API_KEY")
    if any(p.kind == "azure" for p in providers):
        missing_keys += [k for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
                         if not os.environ.get(k)]
    if missing_keys:
        raise SystemExit(f"ERROR: missing environment: {', '.join(missing_keys)}")

    # ------------------------------------------------------------------ #
    # freeze-rule manifest, written BEFORE the first request
    # ------------------------------------------------------------------ #
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest = {
        "created_unix": int(time.time()),
        "num_tasks": args.num_tasks,
        "oversample": args.oversample,
        "rng_seed": args.rng_seed,
        "providers": [p.label for p in providers],
        "temperature_openrouter_deepseek": args.temperature,
        "temperature_azure": (args.temperature if args.azure_send_sampling
                              else "deployment_default"),
        "jac_thr": args.jac_thr, "seq_thr": args.seq_thr,
        "strata_mix": args.strata_mix,
        "stability_runs": args.stability_runs,
        "mutation_max": args.mutation_max,
        "mutation_min_kill": args.mutation_min_kill,
        "shape_gate": {"enabled": bool(args.shape_gate),
                       "loc_tol": args.loc_tol, "branch_tol": args.branch_tol},
        "measured_distribution": {
            "strata_block_ranges": {k: list(v) for k, v in dist["strata_ranges"].items()},
            "loc_bands": {k: list(v) for k, v in dist["loc_bands"].items()},
            "branch_bands": dist["branch_bands"],
            "rtype_weights": dist["rtype_weights"],
            "param_count_weights": dist["param_count_weights"],
        },
        "prompt_template_sha256": sha256_text(EVAL_GEN_PROMPT),
        "benchmark": {"path": str(args.benchmark), "sha256": sha256_file(args.benchmark)},
        "synthetic": {"path": str(args.synthetic), "sha256": sha256_file(args.synthetic)},
        "additional_decontamination": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.decontam_jsonl
        ],
        "banned_names": len(banned),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {manifest_path}")

    # stratum quotas over the ACCEPTED set
    mix = {k.strip(): float(v) for k, v in
           (part.split(":") for part in args.strata_mix.split(","))}
    quotas = {s: int(round(mix.get(s, 0.0) * args.num_tasks)) for s in STRATA}
    while sum(quotas.values()) < args.num_tasks:
        quotas["mid"] += 1
    while sum(quotas.values()) > args.num_tasks:
        quotas[max(quotas, key=quotas.get)] -= 1

    # resume: accepted rows count toward the target, quotas, and dedup pool
    accepted: list[dict] = []
    if args.out.is_file():
        accepted = load_jsonl(args.out)
        for row in accepted:
            dedup_pool.append(pool_entry(row["dart_source"], row["function"]))
            banned.add(row["function"].lower())
            quotas[row.get("target_stratum", "mid")] = max(
                0, quotas.get(row.get("target_stratum", "mid"), 0) - 1)
        print(f"resume: {len(accepted)} already accepted in {args.out}; "
              f"remaining quotas={quotas}")

    rejects_path = args.out.with_suffix(args.out.suffix + ".rejects.jsonl")
    lock = threading.Lock()
    counters = {"attempted": 0, "gate_a": 0, "gate_n": 0, "gate_s": 0, "gate_b": 0,
                "gate_m": 0, "gate_q": 0, "gate_asm": 0, "provider_none": 0,
                "provider_reassigned": 0}
    dead_providers: set[str] = set()

    def mark_dead(provider: ProviderSpec) -> None:
        with lock:
            if provider.label not in dead_providers:
                dead_providers.add(provider.label)
                remaining = [p.label for p in providers if p.label not in dead_providers]
                print(f"  [circuit-breaker] {provider.label} disabled "
                      f"(credit/auth exhausted); continuing with {remaining or 'NOTHING'}",
                      file=sys.stderr)

    def pick_alive(index: int) -> ProviderSpec | None:
        with lock:
            alive = [p for p in providers if p.label not in dead_providers]
        return alive[index % len(alive)] if alive else None

    def record_reject(stage: str, reason: str, plan: dict, extra: dict | None = None):
        with lock:
            counters[stage] = counters.get(stage, 0) + 1
            with rejects_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "stage": stage, "reason": reason[:400],
                    "provider": plan["provider"].label,
                    "category": plan["category"], "stratum": plan["stratum"],
                    **(extra or {}),
                }, ensure_ascii=False) + "\n")

    def attempt(plan: dict) -> dict | None:
        # per-attempt RNG keeps banned-sample selection thread-safe
        arng = random.Random((args.rng_seed, plan["index"]).__hash__())
        with lock:  # quota preflight saves API spend on full strata
            if quotas.get(plan["stratum"], 0) <= 0:
                counters["gate_q"] += 1
                return None
        prompt = render_prompt(plan, banned, exemplars, arng)
        # Circuit breaker: skip dead providers, and on a billing/auth failure
        # retry the SAME attempt on the next surviving provider so the attempt
        # slot (and its stratum quota headroom) is not wasted.
        provider = plan["provider"]
        raw = None
        for hop in range(len(providers)):
            if provider is None:
                break
            with lock:
                is_dead = provider.label in dead_providers
            if is_dead:
                provider = pick_alive(plan["index"] + hop)
                continue
            raw, provider_dead = call_provider(provider, prompt, args)
            if provider_dead:
                mark_dead(provider)
                with lock:
                    counters["provider_reassigned"] += 1
                provider = pick_alive(plan["index"] + hop)
                continue
            break
        plan["provider"] = provider  # provenance records the ACTUAL producer
        with lock:
            counters["attempted"] += 1
        if raw is None or provider is None:
            with lock:
                counters["provider_none"] += 1
            return None
        cand = parse_candidate(raw)
        if cand is None:
            record_reject("gate_a", "structural contract violated", plan,
                          {"raw_head": raw[:200]})
            return None
        with lock:  # novelty gate reads shared state
            reason = novelty_gate(cand, banned, dedup_pool, args.jac_thr, args.seq_thr)
        if reason:
            record_reject("gate_n", reason, plan, {"name": cand["function_name"]})
            return None
        # Gate S: pre-assembly shape filter (LOC/branch proxy for CFG blocks).
        # Cheap (no Dart), so it runs before the functional/mutation gates.
        if args.shape_gate:
            ok_s, why = shape_gate(cand["dart_function"], plan,
                                   args.loc_tol, args.branch_tol)
            if not ok_s:
                record_reject("gate_s", why, plan, {"name": cand["function_name"]})
                return None
        tests = build_tests_field(cand["function_name"], cand["test_expects"])
        ok, diag = functional_gate(cand["dart_function"], tests, args.dart_bin,
                                   stability_runs=args.stability_runs)
        if not ok:
            record_reject("gate_b", diag, plan, {"name": cand["function_name"]})
            return None
        mstats = {"mutants_viable": None, "mutants_killed": None,
                  "mutants_stillborn": None, "kill_rate": None}
        if args.mutation_max > 0:
            ok_m, mstats = mutation_gate(cand["dart_function"], tests, args.dart_bin,
                                         cap=args.mutation_max,
                                         min_kill=args.mutation_min_kill)
            if not ok_m:
                record_reject("gate_m",
                              f"weak tests: kill_rate={mstats['kill_rate']} "
                              f"viable={mstats['mutants_viable']}",
                              plan, {"name": cand["function_name"]})
                return None
        dart_source = build_dart_source(cand["dart_function"], cand["main_asserts"])
        assembly = ""
        if args.with_assembly:
            build_dir = (args.out.parent / (args.out.stem + "_build")).resolve()
            fname = f"pending_{plan['index']:04d}.dart"
            assembly = gate_assembly(cand["function_name"], dart_source,
                                     build_dir, fname) or ""
            if not assembly:
                record_reject("gate_asm", "AOT/gdb dump failed", plan,
                              {"name": cand["function_name"]})
                return None
        return {"plan": plan, "cand": cand, "tests": tests,
                "dart_source": dart_source, "assembly": assembly,
                "mutation": mstats}

    print(f"generating: target={args.num_tasks} quotas={quotas} "
          f"attempts<={len(plans)} workers={args.workers}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(attempt, p): p for p in plans}
        for future in as_completed(pending):
            if future.cancelled():
                continue  # queued attempt cancelled after the target was reached
            result = future.result()
            if result is None:
                continue
            with lock:
                cand, plan = result["cand"], result["plan"]
                if len(accepted) >= args.num_tasks or quotas.get(plan["stratum"], 0) <= 0:
                    counters["gate_q"] += 1
                    continue
                # final same-lock dedup vs concurrently accepted tasks
                if novelty_gate(cand, banned, dedup_pool, args.jac_thr, args.seq_thr):
                    counters["gate_n"] += 1
                    continue
                quotas[plan["stratum"]] -= 1
                idx = len(accepted) + 1
                task_id = "fresh-eval-" + sha256_text(
                    " ".join(normalize(result["dart_source"])))[:12]
                row = {
                    "task_id": task_id,
                    "filename": f"fresh_{idx:03d}.dart",
                    "function": cand["function_name"],
                    "camel_case_function_name": cand["function_name"],
                    "dart_function_signature": cand["signature"].strip(),
                    "dart_source": result["dart_source"],
                    "tests": result["tests"],
                    "assembly": result["assembly"],
                    "lang": "dart",
                    # flat provenance, same style as the synthetic pool rows
                    "category": plan["category"],
                    "difficulty": STRATUM_TO_DIFFICULTY[plan["stratum"]],
                    "flavor": plan["flavor"],
                    "target_stratum": plan["stratum"],
                    "target_param_count": plan["param_count"],
                    "target_loc_band": list(plan["loc_band"]),
                    "mutation_kill_rate": result["mutation"]["kill_rate"],
                    "mutants_viable": result["mutation"]["mutants_viable"],
                    "stability_runs": args.stability_runs,
                    "generator_provider": plan["provider"].kind,
                    "generator_model": plan["provider"].model,
                    "generator_temperature": (
                        args.temperature
                        if plan["provider"].kind in ("openrouter", "deepseek")
                        or args.azure_send_sampling
                        else "deployment_default"),
                    "prompt_template_sha256": manifest["prompt_template_sha256"],
                    "created_unix": int(time.time()),
                }
                accepted.append(row)
                dedup_pool.append(pool_entry(
                    reference_body(row["dart_source"]), row["function"]))
                banned.add(row["function"].lower())
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kill = row["mutation_kill_rate"]
                print(f"  [{len(accepted)}/{args.num_tasks}] {row['function']} "
                      f"({plan['stratum']}, kill={kill if kill is None else round(kill, 2)}, "
                      f"{plan['provider'].label})")
                if len(accepted) >= args.num_tasks:
                    for f in pending:
                        f.cancel()

    manifest["finished_unix"] = int(time.time())
    manifest["accepted"] = len(accepted)
    manifest["counters"] = counters
    manifest["disabled_providers"] = sorted(dead_providers)
    if args.out.is_file():
        manifest["output_sha256"] = sha256_file(args.out)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = distribution_report(accepted, dist)
    report_path = args.out.with_suffix(args.out.suffix + ".distribution.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ndone: accepted={len(accepted)}/{args.num_tasks} counters={counters}")
    if dead_providers:
        print(f"PROVIDERS DISABLED MID-RUN (credit/auth): {sorted(dead_providers)} — "
              "top up and re-run the same command with only that provider "
              "(--providers <name>) to rebalance the generator mix.")
    print(f"accepted per provider-model: {report['generator_models']}")
    print(f"target-strata mix: {report['target_strata']} (remaining quotas={quotas})")
    print(f"LOC quartiles benchmark={report['reference_loc_quartiles']['benchmark']} "
          f"fresh={report['reference_loc_quartiles']['fresh']}")
    print(f"output: {args.out}\nrejects: {rejects_path}\nmanifest: {manifest_path}\n"
          f"distribution report: {report_path}")
    if not args.with_assembly:
        print("\nassembly/CFG are still pending. On a gdb-equipped machine run:\n"
              f"  python {Path(__file__).name} --resume_assembly --out {args.out}\n"
              "then attach graph-v2 CFG/DFG:\n"
              "  python scripts/data/build_graph_v2_jsonl.py --input "
              f"{args.out} --output {args.out.with_name(args.out.stem + '_graphv2.jsonl')} "
              "--rejected ... --summary ... --drop_invalid --max_block_instrs 20\n"
              "and validate before use:\n"
              "  python scripts/data/validate_synthetic_pool.py --pool "
              f"{args.out.with_name(args.out.stem + '_graphv2.jsonl')} --run_tests -1")


if __name__ == "__main__":
    main()
