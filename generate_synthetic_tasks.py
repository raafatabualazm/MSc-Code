#!/usr/bin/env python3
"""
generate_synthetic_tasks.py  (v2 — reviewed)

Multi-LLM synthetic task generator for Dart neural decompilation (GRPO training pool).

Pipeline per candidate:
  1. Generate task via an OpenAI-compatible chat endpoint (12 providers supported)
  2. Gate A — structural: JSON parses, required fields, signature/name/determinism sanity
  3. Gate B — reference validation: function + generated tests must `dart run` clean (exit 0)
  4. Gate C — dedup: token-Jaccard + sequence similarity vs. accepted pool
  5. Gate D — decontamination: similarity + exact-name match vs. the 154 eval tasks
  6. Gate E — AOT compile + gdb disassembly (intel, eval-format-matched), symbol must survive
  7. (optional) Gate F — difficulty band via your policy model (vLLM endpoint)

Diversity levers: (category x difficulty x domain-flavor x return-type) stratified
prompts per provider, deterministic per-provider RNG, post-hoc diversity report.

Output: JSONL, schema-compatible with the HumanEval-Dart eval format
(python_source left empty for synthetic tasks).

Usage:
  export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... DEEPSEEK_API_KEY=... etc.
  python3 generate_synthetic_tasks.py \
      --eval-jsonl humaneval_dart_154.jsonl \
      --out synthetic_pool.jsonl \
      --per-provider 250

Requirements: dart SDK + gdb on PATH, `pip install requests`.
"""

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
import time
import zlib
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Provider registry — all via OpenAI-compatible /chat/completions.
# EDIT model strings to the exact IDs you have access to.
# --------------------------------------------------------------------------
PROVIDERS = {
    "anthropic_fable":  {"base": "https://api.anthropic.com/v1",                           "model": "claude-fable-5",            "key_env": "ANTHROPIC_API_KEY"},
    "anthropic_opus":   {"base": "https://api.anthropic.com/v1",                           "model": "claude-opus-4-8",           "key_env": "ANTHROPIC_API_KEY"},
    "anthropic_sonnet": {"base": "https://api.anthropic.com/v1",                           "model": "claude-sonnet-4-6",         "key_env": "ANTHROPIC_API_KEY"},
    "openai_gpt":       {"base": "https://api.openai.com/v1",                              "model": "gpt-5.5",                   "key_env": "OPENAI_API_KEY"},
    "deepseek":         {"base": "https://api.deepseek.com/v1",                            "model": "deepseek-chat",             "key_env": "DEEPSEEK_API_KEY"},
    "kimi":             {"base": "https://api.moonshot.ai/v1",                             "model": "kimi-k2.6",                 "key_env": "MOONSHOT_API_KEY"},
    "qwen_max":         {"base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "model": "qwen3.7-max",               "key_env": "DASHSCOPE_API_KEY"},
    "qwen_plus":        {"base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "model": "qwen3.7-plus",              "key_env": "DASHSCOPE_API_KEY"},
    "glm":              {"base": "https://open.bigmodel.cn/api/paas/v4",                   "model": "glm-5.1",                   "key_env": "ZHIPU_API_KEY"},
    "minimax":          {"base": "https://api.minimax.io/v1",                              "model": "minimax-m3",                "key_env": "MINIMAX_API_KEY"},
    "gemini_pro":       {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3.1-pro",           "key_env": "GEMINI_API_KEY"},
    "gemini_flash":     {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3.5-flash-extended", "key_env": "GEMINI_API_KEY"},
    "nvidia_nemotron":  {"base": "https://integrate.api.nvidia.com/v1",                     "model": "nvidia/nemotron-3-ultra",   "key_env": "NVIDIA_API_KEY"},
}

# --------------------------------------------------------------------------
# OpenRouter: same models through one key (OPENROUTER_API_KEY).
# Registered with an "or_" prefix, e.g.  --providers or_fable,or_gpt,or_deepseek
# EDIT the model slugs to the exact IDs from https://openrouter.ai/models
# (OpenRouter uses vendor/model format). Avoid ":free" variants for bulk
# generation — their daily caps are far below a 200-attempt run.
# --------------------------------------------------------------------------
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS = {
    "fable":        "anthropic/claude-fable-5",   # pinned; "anthropic/claude-fable-latest" also exists
    "opus":         "anthropic/claude-opus-4.8",
    "sonnet":       "anthropic/claude-sonnet-4.6",
    "gpt":          "openai/gpt-5.5",
    "gpt_pro":      "openai/gpt-5.5-pro",
    "deepseek":     "deepseek/deepseek-v4-pro",
    "kimi":         "moonshotai/kimi-k2.6",
    "qwen_max":     "qwen/qwen3.7-max",
    "qwen_plus":    "qwen/qwen3.7-plus",
    "glm":          "z-ai/glm-5.1",
    "glm_turbo":    "z-ai/glm-5-turbo",
    "minimax":      "minimax/minimax-m3",
    "gemini_pro":   "google/gemini-3.1-pro-preview",
    "gemini_flash": "google/gemini-3.5-flash",
    "grok":         "x-ai/grok-4.3",
    "nemotron":     "nvidia/nemotron-3-ultra-550b-a55b:free",
}
for _k, _m in OPENROUTER_MODELS.items():
    PROVIDERS[f"or_{_k}"] = {"base": OPENROUTER_BASE, "model": _m,
                             "key_env": "OPENROUTER_API_KEY"}

# --------------------------------------------------------------------------
# Stratification axes. Diversity comes from the CROSS PRODUCT of these in the
# prompt, not from the number of models — identical prompts make all models
# converge on their per-category favorite task.
# --------------------------------------------------------------------------
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
    "search problems (binary search variants, kth element)",
]
DIFFICULTIES = [
    ("easy",   "8-15 lines, single loop or simple recursion, one core idea"),
    ("medium", "15-30 lines, nested control flow or two interacting ideas"),
    ("hard",   "25-45 lines, non-obvious algorithm or tricky edge cases"),
]
# Domain flavors force surface variety even within one (category, difficulty)
# cell — this is the main defense against per-model mode collapse.
FLAVORS = [
    "warehouse inventory", "music playlists", "chess board squares", "DNA base strings",
    "sensor telemetry samples", "server log lines", "election vote tallies",
    "shipping manifests", "RGB pixel values", "morse code symbols",
    "theater seating charts", "bank ledger entries", "dice game rounds",
    "satellite pass windows", "soil moisture grid", "barcode digits",
    "elevator floor requests", "wifi signal strength bins", "recipe ingredient scaling",
    "traffic light phases", "playing card decks", "password rule checks",
    "maze grid cells", "thermostat schedules", "tournament brackets",
    "ASCII art rows", "hash bucket counts", "train timetable minutes",
    "battery charge cycles", "QR code modules", "library shelf codes",
    "network packet sizes", "tide height readings", "spell-checker edits",
    "currency cent rounding", "game item inventories",
]
RETURN_TYPES = ["int", "bool", "String", "List<int>", "List<String>",
                "Map<String, int>", "double", "Set<int>"]

GEN_SYSTEM = ("You generate self-contained Dart programming tasks used to build a "
              "compiler-verified dataset. Output STRICT JSON only — no markdown "
              "fences, no prose.")

GEN_PROMPT = """Create ONE original Dart function task. Requirements:

- Category: {category}
- Difficulty: {difficulty_name} ({difficulty_desc})
- Theme the problem around: {flavor}
- Preferred return type: {rtype}
- Pure function: deterministic, no I/O, no dart:io, no async, no Random,
  no DateTime, no Stopwatch, no external packages.
- Use only dart:core / dart:math (math only for sqrt/pow/min/max if needed).
- Parameters and return types must be concrete (int, double, bool, String,
  List<...>, Map<...>, Set<...>) — no dynamic, no generics in the signature.
- If using double, only values exactly representable in binary floating point
  (halves, quarters) so == comparisons in tests are exact.
- Do NOT recreate well-known interview classics (fizzbuzz, palindrome,
  fibonacci, reverse string, two-sum) or HumanEval-style problems.
- Function name: lowerCamelCase, descriptive, specific to the theme.

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
test_expects: 6-9 expect(...) lines covering normal cases, edge cases
(empty input, single element, boundary values). Each must be a complete Dart
statement referencing `candidate`. Expected values must be CORRECT — compute
them carefully step by step before writing them."""

EXPECT_HARNESS = """
void expect(dynamic a, dynamic b) {
  if (a == b) return;

  if (a is List && b is List) {
    expectList(a, b);
  } else if (a is Map && b is Map) {
    expectMap(a, b);
  } else {
    throw '$a != $b';
  }
}

void expectList(List a, List b) {
  if (a.length != b.length) throw 'list lengths are not equal';

  for (var i = 0; i < a.length; i++) {
    expect(a[i], b[i]);
  }
}

void expectMap(Map a, Map b) {
  if (a.length != b.length) throw 'map lengths are not equal';

  for (var key in a.keys) {
    expect(a[key], b[key]);
  }
}"""

# --------------------------------------------------------------------------
# LLM call (with max_tokens -> max_completion_tokens fallback for providers
# that reject the legacy parameter, and dual auth headers for Anthropic).
# --------------------------------------------------------------------------
def call_llm(provider_cfg: dict, prompt: str, temperature: float = 1.0,
             max_retries: int = 3, timeout: int = 180) -> str | None:
    key = os.environ.get(provider_cfg["key_env"], "")
    if not key:
        return None
    url = provider_cfg["base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if "anthropic" in provider_cfg["base"]:
        headers["x-api-key"] = key
    payload = {
        "model": provider_cfg["model"],
        "temperature": temperature,
        "max_tokens": 8192,  # reasoning models spend output budget on thinking
        "messages": [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    token_key = "max_tokens"
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 400 and "max_tokens" in r.text and token_key == "max_tokens":
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                token_key = "max_completion_tokens"
                continue  # retry immediately with the new parameter name
            if 400 <= r.status_code < 500:
                # Deterministic client error (bad slug, invalid param, no access):
                # print the body — it says exactly what is wrong — and don't retry.
                print(f"    [error] {provider_cfg['model']} HTTP {r.status_code}: "
                      f"{r.text[:300]}", file=sys.stderr)
                return None
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    [warn] {provider_cfg['model']} attempt {attempt+1}: {e}",
                  file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------
# Gate A — structural validation
# --------------------------------------------------------------------------
CAMEL_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
BANNED_SUBSTRINGS = ("dart:io", "dart:async", "Future<", "await ", "Stream<",
                     "print(", "Random", "DateTime", "Stopwatch", "stdin", "stdout")

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
    if len(obj["test_expects"]) < 4 or len(obj["main_asserts"]) < 1:
        return None
    if "void main" in obj["dart_function"]:
        return None
    if any(b in obj["dart_function"] for b in BANNED_SUBSTRINGS):
        return None
    return obj


# --------------------------------------------------------------------------
# Gate B — reference validation: tests must pass against the source
# --------------------------------------------------------------------------
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
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        output = p.stdout + p.stderr
        if output_limit > 0:
            output = output[-output_limit:]
        return p.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "timeout"

def dart_tests_pass(dart_function: str, tests: str, workdir: str,
                    tag: str = "validate", log_fail: bool = True) -> bool:
    f = Path(workdir) / f"{tag}.dart"
    src_text = dart_function.strip() + "\n\n" + tests
    f.write_text(src_text)
    # NOTE: VM flags go AFTER the `run` subcommand.
    rc, out = run(["dart", "run", "--enable-asserts", str(f)], workdir, timeout=60)
    if rc != 0 and log_fail and tag == "validate":
        log_gate_failure("reference", out, src_text)
    return rc == 0


# --------------------------------------------------------------------------
# Gates C & D — dedup and decontamination
# --------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")

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
        j = jaccard(tset, entry["tset"])
        if j >= jac_thr:
            return True
        if j >= jac_thr * 0.7 and SequenceMatcher(
                None, joined, entry["joined"]).ratio() >= seq_thr:
            return True
    return False

def pool_entry(code: str, name: str) -> dict:
    toks = normalize(code)
    return {"tset": set(toks), "joined": " ".join(toks), "name": name}


# --------------------------------------------------------------------------
# Gate-failure forensics: keep the first N failures per gate (command output
# + offending source) so a 0% survival run explains itself.
# Tempdirs live under the CWD, not /tmp: snap-packaged dart cannot read /tmp.
# --------------------------------------------------------------------------
GATE_TMP_ROOT = Path("gate_tmp")
DEBUG_DIR = Path("gate_failures")
_fail_counts: Counter = Counter()
FAIL_LOG_CAP = 25

def gate_tmpdir():
    GATE_TMP_ROOT.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="dartgen_", dir=GATE_TMP_ROOT)

def log_gate_failure(gate: str, output: str, source: str) -> None:
    _fail_counts[gate] += 1
    n = _fail_counts[gate]
    if n > FAIL_LOG_CAP:
        return
    DEBUG_DIR.mkdir(exist_ok=True)
    p = DEBUG_DIR / f"{gate}_{n:02d}.log"
    p.write_text(f"=== GATE: {gate} ===\n\n--- tool output ---\n{output}\n\n"
                 f"--- source ---\n{source}\n")


# --------------------------------------------------------------------------
# Gate E — AOT compile + gdb disassembly.
# Compiled in a FIXED build dir so the `File <path>:` line in the dump is a
# stable, eval-like path rather than per-sample tempdir noise. The
# `info functions` query is UNANCHORED to reproduce the eval set's header:
#   All functions matching regular expression "name":
# --------------------------------------------------------------------------
def gate_assembly(name: str, dart_source: str, build_dir: Path,
                  filename: str) -> str | None:
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    src_path = build_dir / filename
    src_path.write_text(dart_source)
    failures = []
    for kind, extension in (("aot-snapshot", ".aot"), ("exe", ".exe")):
        output_path = build_dir / (filename.replace(".dart", "") + extension)
        rc, out = run(
            ["dart", "compile", kind, str(src_path), "-o", str(output_path)],
            str(build_dir),
            timeout=180,
        )
        if rc != 0:
            failures.append(f"[compile {kind}] rc={rc}\n{out}")
            continue
        gdb_target = output_path.resolve().as_posix().replace('"', '\\"')
        rc, out = run([
            "gdb", "-batch", "-q",
            "-ex", "set disassembly-flavor intel",
            "-ex", f'file "{gdb_target}"',
            "-ex", f"info functions {name}",
            "-ex", f"disassemble {name}",
        ], str(build_dir), timeout=120, output_limit=0)
        output_path.unlink(missing_ok=True)
        if rc == 0 and "Dump of assembler code" in out:
            idx = out.find("All functions matching")
            if idx == -1:
                idx = out.find("Dump of assembler code")
            asm = out[idx:].strip()
            if not asm.endswith("End of assembler dump."):
                asm += "\nEnd of assembler dump."
            return asm
        failures.append(f"[gdb on {kind}] rc={rc}\n{out}")
    log_gate_failure("assembly", "\n\n".join(failures), dart_source)
    return None


# --------------------------------------------------------------------------
# Gate F (optional) — difficulty band via your policy model
# --------------------------------------------------------------------------
MAIN_SPLIT_RE = re.compile(r"\n\s*(?:@pragma\([^)]*\)\s*\n)?void\s+main\s*\(")

def strip_main(code: str) -> str:
    """Drop any main() a prediction appended, so it can't collide with tests."""
    m = MAIN_SPLIT_RE.search(code)
    return code[:m.start()] if m else code

def gate_difficulty(assembly: str, tests: str, endpoint: str, model: str,
                    k: int, band: tuple[float, float], workdir: str) -> bool:
    """
    Samples k decompilations from your fine-tuned policy and keeps the task
    only if the empirical pass rate falls strictly inside `band`.

    TODO(raafat): replace PROMPT_TEMPLATE with the exact inference prompt from
    rerank_predictions_antigravity.py so the measurement matches your
    training-time distribution.
    """
    PROMPT_TEMPLATE = (
        "Decompile the following x86-64 assembly of a Dart function back "
        "into idiomatic Dart source code. Output only the Dart function.\n\n"
        "{assembly}\n"
    )
    payload = {
        "model": model,
        "temperature": 0.8,
        "n": k,
        "max_tokens": 2048,
        "messages": [{"role": "user",
                      "content": PROMPT_TEMPLATE.format(assembly=assembly)}],
    }
    try:
        r = requests.post(endpoint.rstrip("/") + "/chat/completions",
                          json=payload, timeout=600)
        r.raise_for_status()
        choices = [c["message"]["content"] for c in r.json()["choices"]]
    except Exception as e:
        print(f"    [warn] policy endpoint: {e}", file=sys.stderr)
        return True  # fail-open: don't discard data because the gate is down
    passes = 0
    for i, pred in enumerate(choices):
        m = re.search(r"```(?:dart)?\s*(.*?)```", pred, flags=re.S)
        code = strip_main((m.group(1) if m else pred).strip())
        if dart_tests_pass(code, tests, workdir, tag=f"policy_{i}"):
            passes += 1
    rate = passes / max(1, len(choices))
    return band[0] < rate < band[1]


# --------------------------------------------------------------------------
# Diversity report
# --------------------------------------------------------------------------
def diversity_report(records: list[dict], rng: random.Random) -> None:
    if not records:
        return
    print("\n---------- DIVERSITY REPORT ----------")
    for field in ("category", "difficulty", "generator_provider"):
        hist = Counter(r.get(field, "?") for r in records)
        print(f"  {field}:")
        for k, v in hist.most_common():
            print(f"    {v:4d}  {k}")
    stems = Counter(r["camel_case_function_name"][:4].lower() for r in records)
    print("  top name stems:", ", ".join(f"{s}({c})" for s, c in stems.most_common(8)))
    entries = [pool_entry(r["dart_source"], r["camel_case_function_name"])
               for r in records]
    if len(entries) >= 2:
        n_pairs = min(300, len(entries) * (len(entries) - 1) // 2)
        sims = []
        for _ in range(n_pairs):
            a, b = rng.sample(entries, 2)
            sims.append(jaccard(a["tset"], b["tset"]))
        sims.sort()
        print(f"  pairwise token-jaccard (n={len(sims)} sampled pairs): "
              f"mean={sum(sims)/len(sims):.3f}  p90={sims[int(0.9*len(sims))]:.3f}  "
              f"max={sims[-1]:.3f}")
        print("  (mean below ~0.45 and max below your dedup threshold = healthy)")


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

def load_eval_pool(path: str) -> tuple[list[dict], set[str]]:
    pool, names = [], set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            code = obj.get("dart_source", "") or obj.get("python_source", "")
            name = obj.get("camel_case_function_name") or obj.get("function", "")
            pool.append(pool_entry(code, name))
            if name:
                names.add(name.lower())
                names.add(camel_to_snake(name))
    return pool, names

SELFTEST_CANDIDATE = {
    "function_name": "tallyCargoWeights",
    "signature": "int tallyCargoWeights(List<int> crates, int limit)",
    "dart_function": (
        "int tallyCargoWeights(List<int> crates, int limit) {\n"
        "  var total = 0;\n"
        "  for (final w in crates) {\n"
        "    if (w <= limit) {\n"
        "      total += w;\n"
        "    }\n"
        "  }\n"
        "  return total;\n"
        "}"),
    "main_asserts": ["assert(tallyCargoWeights([3, 9, 4], 5) == 7);"],
    "test_expects": [
        "expect(candidate([], 10), 0);",
        "expect(candidate([3, 9, 4], 5), 7);",
        "expect(candidate([1, 2, 3], 0), 0);",
        "expect(candidate([5], 5), 5);",
    ],
}

def self_test() -> int:
    """Run one hardcoded valid candidate through every local gate, verbosely.
    No API calls, no eval file needed. Exit 0 only if all gates pass."""
    print("========== SELF TEST ==========")
    for tool in ("dart", "gdb"):
        loc = shutil.which(tool)
        print(f"[{tool}] -> {loc}")
        if loc is None:
            print(f"FAIL: `{tool}` not on PATH"); return 1
        rc, out = run([tool, "--version"], ".", timeout=30)
        print(f"  version: {out.strip().splitlines()[0] if out.strip() else rc}")
        if tool == "dart" and "/snap/" in (loc or ""):
            print("  [!] dart is snap-packaged: it cannot read /tmp. This script "
                  "now uses ./gate_tmp, which avoids that — but verify below.")
    cand = SELFTEST_CANDIDATE
    name = cand["function_name"]
    tests = build_tests_field(name, cand["test_expects"])
    ok = True

    with gate_tmpdir() as wd:
        print(f"\n--- Gate B: reference validation (workdir {wd}) ---")
        f = Path(wd) / "validate.dart"
        f.write_text(cand["dart_function"] + "\n\n" + tests)
        for cmd in (["dart", "run", "--enable-asserts", str(f)],
                    ["dart", str(f)]):
            rc, out = run(cmd, wd, timeout=120)
            print(f"  $ {' '.join(cmd[:3])} ...  -> rc={rc}")
            if out.strip():
                print("    " + "\n    ".join(out.strip().splitlines()[:8]))
            if cmd[1] == "run":
                gate_b_rc = rc
        if gate_b_rc != 0:
            ok = False
            print("  GATE B FAILED — the line(s) above are the real error.")
        else:
            print("  GATE B OK")

        print("\n--- Gate E: AOT compile + gdb ---")
        dart_source = build_dart_source(cand["dart_function"], cand["main_asserts"])
        src_path = Path(wd) / "selftest.dart"
        src_path.write_text(dart_source)
        exe_path = Path(wd) / "selftest.exe"
        rc, out = run(["dart", "compile", "exe", str(src_path), "-o", str(exe_path)],
                      wd, timeout=300)
        print(f"  $ dart compile exe ...  -> rc={rc}")
        if out.strip():
            print("    " + "\n    ".join(out.strip().splitlines()[:8]))
        if rc != 0:
            print("  GATE E FAILED at compile."); return 1
        rc, out = run(["gdb", "-batch", "-q",
                       "-ex", "set disassembly-flavor intel",
                       "-ex", f"file {exe_path}",
                       "-ex", f"info functions {name}",
                       "-ex", f"disassemble {name}"], wd, timeout=120)
        print(f"  $ gdb -batch ... disassemble {name}  -> rc={rc}")
        lines = out.strip().splitlines()
        print("    " + "\n    ".join(lines[:10]))
        if len(lines) > 10:
            print(f"    ... ({len(lines)} lines total)")
        if rc != 0 or "Dump of assembler code" not in out:
            ok = False
            print("  GATE E FAILED — gdb cannot disassemble the Dart symbol. "
                  "Compare with the exact compile/gdb commands used to build "
                  "your 154-task eval set.")
        else:
            print("  GATE E OK")

    print("\n========== SELF TEST " + ("PASSED" if ok else "FAILED") + " ==========")
    return 0 if ok else 1

def save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-jsonl", default=None,
                    help="Path to the 154-task eval JSONL (decontamination reference)")
    ap.add_argument("--self-test", action="store_true",
                    help="Run one known-good candidate through all local gates "
                         "verbosely (no API calls), then exit")
    ap.add_argument("--out", default="synthetic_pool.jsonl")
    ap.add_argument("--per-provider", type=int, default=200,
                    help="Generation attempts per provider (expect ~40-60%% survival)")
    ap.add_argument("--providers",
                    default=",".join(k for k in PROVIDERS if not k.startswith("or_")),
                    help="Comma-separated subset of provider keys")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--jaccard", type=float, default=0.75,
                    help="Dedup/decontam token-Jaccard threshold "
                         "(distinct short tasks land ~0.4-0.65; renamed dups ~0.85)")
    ap.add_argument("--seqratio", type=float, default=0.85)
    ap.add_argument("--build-dir", default="dart_build",
                    help="Fixed dir for AOT builds so asm 'File <path>' lines are stable")
    ap.add_argument("--policy-endpoint", default=None,
                    help="OpenAI-compatible vLLM endpoint for the difficulty gate")
    ap.add_argument("--policy-model", default="decompiler-v6")
    ap.add_argument("--policy-k", type=int, default=8)
    ap.add_argument("--band", default="0.0,0.8",
                    help="Keep tasks with policy pass-rate strictly inside (lo,hi)")
    ap.add_argument("--task-prefix", default="syn")
    args = ap.parse_args()

    for tool in ("dart", "gdb"):
        if shutil.which(tool) is None:
            sys.exit(f"error: `{tool}` not found on PATH")

    if args.self_test:
        sys.exit(self_test())

    band = tuple(float(x) for x in args.band.split(","))
    if not args.eval_jsonl or not Path(args.eval_jsonl).is_file():
        sys.exit(f"error: --eval-jsonl '{args.eval_jsonl}' not found.\n"
                 "Point it at your 154-task HumanEval-Dart JSONL (the file whose "
                 "lines contain dart_source/assembly/tests fields). It is required "
                 "for decontamination — do not run without it.")
    eval_pool, eval_names = load_eval_pool(args.eval_jsonl)
    print(f"[*] Loaded {len(eval_pool)} eval tasks for decontamination")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
        print("[*] tip: `pip install tqdm` for a progress bar")

    build_dir = Path(args.build_dir).resolve()
    accepted_pool: list[dict] = []
    accepted_names: set[str] = set()
    accepted_records: list[dict] = []
    n_accepted = 0
    stats = Counter()

    out_path = Path(args.out)
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                obj = json.loads(line)
                accepted_pool.append(pool_entry(obj["dart_source"],
                                                obj["camel_case_function_name"]))
                accepted_names.add(obj["camel_case_function_name"].lower())
                accepted_records.append(obj)
                n_accepted += 1
        print(f"[*] Resuming: {n_accepted} samples already in {args.out}")

    # Per-provider attempt cursor: rerunning the same command continues where
    # it stopped instead of re-spending API budget from attempt 0.
    state_path = out_path.with_suffix(out_path.suffix + ".state.json")
    state = {"attempts": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            done = {k: v for k, v in state.get("attempts", {}).items() if v}
            if done:
                print("[*] Attempt cursors: " +
                      ", ".join(f"{k}={v}" for k, v in done.items()))
        except json.JSONDecodeError:
            print("[!] Corrupt state file, starting cursors fresh", file=sys.stderr)

    provider_keys = [p.strip() for p in args.providers.split(",") if p.strip()]
    out_fh = out_path.open("a")
    report_rng = random.Random(0xDA47)
    interrupted = False

    def attempt_once(pkey: str, cfg: dict, i: int) -> str:
        """Run one generation attempt through all gates; returns the stat key."""
        nonlocal n_accepted
        # Deterministic per-attempt stratification: resume-safe because nothing
        # depends on sequential RNG state, only on (provider, attempt index).
        arng = random.Random(zlib.crc32(pkey.encode()) ^ (0xDA47 + i))
        category = CATEGORIES[(i + zlib.crc32(pkey.encode())) % len(CATEGORIES)]
        dname, ddesc = DIFFICULTIES[i % len(DIFFICULTIES)]
        flavor = arng.choice(FLAVORS)
        rtype = arng.choice(RETURN_TYPES)
        prompt = GEN_PROMPT.format(category=category, difficulty_name=dname,
                                   difficulty_desc=ddesc, flavor=flavor,
                                   rtype=rtype)
        raw = call_llm(cfg, prompt, temperature=args.temperature)
        if raw is None:
            return "gen_fail"
        cand = parse_candidate(raw)
        if cand is None:
            return "structural"
        name = cand["function_name"]
        if name.lower() in accepted_names:
            return "dedup"
        if name.lower() in eval_names or camel_to_snake(name) in eval_names:
            return "decontam"

        with gate_tmpdir() as wd:
            tests = build_tests_field(name, cand["test_expects"])
            if not dart_tests_pass(cand["dart_function"], tests, wd):
                return "reference"
            if too_similar(cand["dart_function"], accepted_pool,
                           args.jaccard, args.seqratio):
                return "dedup"
            if too_similar(cand["dart_function"], eval_pool,
                           args.jaccard, args.seqratio):
                return "decontam"

            task_id = f"{args.task_prefix}_{n_accepted:04d}"
            filename = f"{task_id}.dart"
            dart_source = build_dart_source(cand["dart_function"],
                                            cand["main_asserts"])
            asm = gate_assembly(name, dart_source, build_dir, filename)
            if asm is None:
                return "assembly"
            if args.policy_endpoint and not gate_difficulty(
                    asm, tests, args.policy_endpoint, args.policy_model,
                    args.policy_k, band, wd):
                return "difficulty"

        record = {
            "filename": filename,
            "function": name,
            "python_function_name": camel_to_snake(name),
            "camel_case_function_name": name,
            "dart_function_signature": cand["signature"].strip(),
            "python_source": "",
            "dart_source": dart_source,
            "assembly": asm,
            "lang": "Dart",
            "task_id": task_id,
            "tests": tests,
            "generator_model": cfg["model"],
            "generator_provider": pkey,
            "category": category,
            "difficulty": dname,
            "flavor": flavor,
            "sha256": hashlib.sha256(dart_source.encode()).hexdigest()[:16],
        }
        out_fh.write(json.dumps(record) + "\n")
        out_fh.flush()
        accepted_pool.append(pool_entry(cand["dart_function"], name))
        accepted_names.add(name.lower())
        accepted_records.append(record)
        n_accepted += 1
        return "ok"

    try:
        for pkey in provider_keys:
            cfg = PROVIDERS.get(pkey)
            if cfg is None:
                print(f"[!] Unknown provider key: {pkey}", file=sys.stderr)
                continue
            if not os.environ.get(cfg["key_env"]):
                print(f"[!] Skipping {pkey}: ${cfg['key_env']} not set")
                continue
            start_i = int(state["attempts"].get(pkey, 0))
            if start_i >= args.per_provider:
                print(f"=== {pkey}: already completed "
                      f"({start_i}/{args.per_provider}), skipping ===")
                continue
            print(f"\n=== {pkey} ({cfg['model']}) — attempts "
                  f"{start_i + 1}..{args.per_provider} ===")
            pbar = (tqdm(total=args.per_provider, initial=start_i, unit="try",
                         desc=pkey, dynamic_ncols=True) if tqdm else None)
            note = pbar.write if pbar else print
            for i in range(start_i, args.per_provider):
                key = attempt_once(pkey, cfg, i)
                stats[key] += 1
                if key == "ok":
                    note(f"  [+] {accepted_records[-1]['task_id']} "
                         f"{accepted_records[-1]['camel_case_function_name']} "
                         f"({accepted_records[-1]['category'].split(' ')[0]} / "
                         f"{accepted_records[-1]['difficulty']} / "
                         f"{accepted_records[-1]['flavor']})")
                state["attempts"][pkey] = i + 1
                save_state(state, state_path)
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(accepted=n_accepted, ok=stats["ok"])
                elif (i + 1) % 5 == 0 or i + 1 == args.per_provider:
                    print(f"  [{pkey} {i + 1}/{args.per_provider}] "
                          f"total accepted: {n_accepted}")
            if pbar:
                pbar.close()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[!] Interrupted — progress saved. Rerun the SAME command to "
              "resume from this exact attempt.", file=sys.stderr)
    finally:
        out_fh.close()
        print("\n========== GATE SUMMARY ==========")
        total = sum(stats.values())
        for k in ("gen_fail", "structural", "reference", "dedup", "decontam",
                  "assembly", "difficulty", "ok"):
            v = stats[k]
            print(f"  {k:>11}: {v:5d}" + (f"  ({100*v/total:.1f}%)" if total else ""))
        diversity_report(accepted_records, report_rng)
        print(f"\n[*] {n_accepted} total accepted samples in {args.out}")
        if sum(_fail_counts.values()):
            print(f"[*] Gate failure details (first {FAIL_LOG_CAP} per gate) "
                  f"saved in ./{DEBUG_DIR}/")
        if interrupted:
            print(f"[*] Cursors in {state_path} — delete it to restart from zero, "
                  "or edit a provider's count to re-run/skip it.")

if __name__ == "__main__":
    main()
