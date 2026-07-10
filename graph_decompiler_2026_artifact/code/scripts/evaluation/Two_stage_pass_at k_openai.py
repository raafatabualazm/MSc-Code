from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "raafatabualazm/decompiler-v1"  # adjust to your deployed model
DATA_FILE = "grpo_data"
LANGUAGE = "dart"

N_REASONINGS = 4
N_CODES_PER_REASONING = 3

ENABLE_REPAIR = True
MAX_REPAIRS_PER_TASK = 4
REPAIR_SAMPLES_PER_CANDIDATE = 2

REASONING_TEMPERATURE = 0.7
REASONING_TOP_P = 0.95
REASONING_MAX_NEW_TOKENS = 2048

CODE_TEMPERATURE = 0.35
CODE_TOP_P = 0.9
CODE_MAX_NEW_TOKENS = 2048

REPAIR_TEMPERATURE = 0.3
REPAIR_TOP_P = 0.9
REPAIR_MAX_NEW_TOKENS = 1536

CONCURRENCY = 4       # max parallel API requests
DART_TIMEOUT = 30
SEED = 1337
DEBUG_JSONL = f"dart_two_stage_debug_{MODEL_NAME.replace('/', '_')}.jsonl"

REPAIR_ERROR_POLICY = "compiler_only"

BASE_URL = None        # set for vLLM/TGI, e.g. "http://localhost:8000/v1"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
# ────────────────────────────────────────────────────────────────────────

RAW_K = N_REASONINGS * N_CODES_PER_REASONING

client_kwargs: dict[str, Any] = {"api_key": API_KEY}
if BASE_URL:
    client_kwargs["base_url"] = BASE_URL
client = OpenAI(**client_kwargs)


def log_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Prompting ───────────────────────────────────────────────────────────
REASONING_SYSTEM = (
    "You are an expert reverse engineer. Recover the semantics of the assembly precisely. "
    "Be concise, structured, and deterministic."
)

REASONING_USER = """Analyze the following assembly and infer its semantics.

Return ONLY this XML structure:
<analysis>
<signature>brief guess of parameters and return type</signature>
<variables>key registers/stack slots and their roles</variables>
<control_flow>branches, loops, and exits</control_flow>
<algorithm>the high-level algorithm in 3-6 short bullet points</algorithm>
<edge_cases>important edge cases if any</edge_cases>
</analysis>

Rules:
- Do NOT write Dart code.
- Keep it under 220 words.
- Do NOT restate the assembly line by line.

Assembly:
{assembly}
"""

CODE_SYSTEM = (
    "You are an expert reverse engineer and Dart programmer. "
    "Use the semantic analysis to reconstruct clean, valid Dart code."
)

CODE_USER = """Convert the following assembly to Dart using the provided analysis.

Assembly:
{assembly}

Analysis:
{analysis}

Return ONLY valid Dart code inside a ```dart code fence.
Rules:
- No explanation outside the code fence.
- No main().
- Include imports only if essential.
- Prefer one top-level function plus small helpers only if required.
- The output must be syntactically valid Dart.
"""

REPAIR_SYSTEM = (
    "You are repairing a nearly-correct Dart decompilation candidate. "
    "Make the smallest change needed to produce valid, testable Dart code."
)

REPAIR_USER = """Repair the Dart code below.

Assembly:
{assembly}

Analysis:
{analysis}

Broken candidate:
```dart
{code}
```

Observed failure:
{error_summary}

Return ONLY corrected Dart code inside a ```dart code fence.
Rules:
- No explanation outside the code fence.
- No main().
- Keep the original algorithm unless the failure clearly implies it is wrong.
"""


# ── EvalResult ──────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    passed: bool
    compiled: bool
    timed_out: bool
    exit_code: int | None
    compile_stderr: str
    compile_stdout: str
    run_stderr: str
    run_stdout: str
    failure_type: str  # pass | compile | runtime | timeout | empty


# ── Code extraction helpers ─────────────────────────────────────────────
def extract_xml_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


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


_DART_DECL_PATTERN = re.compile(
    r"^(?:(?:int|double|String|bool|void|List|Map|Set|dynamic|Future|Stream|num|Iterable|BigInt)\b"
    r"|(?:class|enum|typedef|extension|mixin)\b"
    r"|(?:[A-Z][a-zA-Z0-9]*(?:<[^>]+>)?\s+\w+\s*[\(=;]))",
    re.MULTILINE,
)


def extract_dart_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    for tag in ["final_code", "code", "answer", "response"]:
        tagged = extract_xml_tag(text, tag)
        if tagged:
            text = tagged
            break

    if "```dart" in text:
        text = text.split("```dart", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    else:
        if "</analysis>" in text:
            text = text.split("</analysis>", 1)[1]
        m = _DART_DECL_PATTERN.search(text)
        if m:
            text = text[m.start():]

    if "</analysis>" in text:
        text = text.split("</analysis>", 1)[1]

    return text.strip()


def normalize_code(code: str) -> str:
    code = strip_main_and_imports(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def canonical_key(code: str) -> str:
    imports = sorted(set(re.findall(r"^import\s+.*;\s*$", code, re.MULTILINE)))
    body = normalize_code(code)
    return "\n".join(imports) + "\n\n" + body


# ── OpenAI generation ──────────────────────────────────────────────────
def _single_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """One chat completion API call → one output string."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=SEED,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  API error: {e}")
        return ""


def generate_texts(
    messages: list[dict[str, str]],
    n_samples: int,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    use_repetition_penalty: bool = False,  # kept for signature compat; ignored by API
) -> list[str]:
    """Generate n_samples completions via concurrent API calls."""
    samples: list[str] = [""] * n_samples

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        future_to_idx = {
            pool.submit(
                _single_completion,
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            ): i
            for i in range(n_samples)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            samples[idx] = fut.result()

    return samples


# ── Two-phase Dart evaluation ──────────────────────────────────────────
def run_dart_test(generated_code: str, test_code: str, task_id: str) -> EvalResult:
    clean_code = normalize_code(generated_code)

    gen_imports = set(re.findall(r"^import\s+.*;\s*$", generated_code, re.MULTILINE))
    test_imports = set(re.findall(r"^import\s+.*;\s*$", test_code, re.MULTILINE))
    all_imports = gen_imports | test_imports
    test_body = re.sub(r"^import\s+.*;\s*$", "", test_code, flags=re.MULTILINE).strip()

    full_source = "\n".join(sorted(all_imports)) + "\n\n" + clean_code + "\n\n" + test_body

    if not clean_code.strip():
        return EvalResult(
            passed=False, compiled=False, timed_out=False, exit_code=None,
            compile_stderr="", compile_stdout="",
            run_stderr="", run_stdout="", failure_type="empty",
        )

    tmp_dir = tempfile.mkdtemp(prefix=f"dart_passk_{task_id}_")
    dart_file = os.path.join(tmp_dir, "test.dart")
    dill_file = os.path.join(tmp_dir, "test.dill")

    with open(dart_file, "w", encoding="utf-8") as f:
        f.write(full_source)

    phase = "compile"
    compile_stderr = ""
    compile_stdout = ""

    try:
        # Phase 1: compilation
        compile_result = subprocess.run(
            ["dart", "compile", "kernel", dart_file, "-o", dill_file],
            capture_output=True, text=True, timeout=DART_TIMEOUT,
        )
        compile_stderr = compile_result.stderr[-2000:]
        compile_stdout = compile_result.stdout[-2000:]

        if compile_result.returncode != 0:
            return EvalResult(
                passed=False, compiled=False, timed_out=False,
                exit_code=compile_result.returncode,
                compile_stderr=compile_stderr, compile_stdout=compile_stdout,
                run_stderr="", run_stdout="", failure_type="compile",
            )

        # Phase 2: execution
        phase = "run"
        run_result = subprocess.run(
            ["dart", "run", dart_file],
            capture_output=True, text=True, timeout=DART_TIMEOUT,
        )
        passed = run_result.returncode == 0
        return EvalResult(
            passed=passed, compiled=True, timed_out=False,
            exit_code=run_result.returncode,
            compile_stderr=compile_stderr, compile_stdout=compile_stdout,
            run_stderr=run_result.stderr[-2000:], run_stdout=run_result.stdout[-2000:],
            failure_type="pass" if passed else "runtime",
        )

    except subprocess.TimeoutExpired as e:
        return EvalResult(
            passed=False, compiled=(phase == "run"), timed_out=True, exit_code=None,
            compile_stderr=compile_stderr if phase == "run" else "",
            compile_stdout=compile_stdout if phase == "run" else "",
            run_stderr=(e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
            run_stdout=(e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            failure_type="timeout",
        )
    except Exception as e:
        return EvalResult(
            passed=False, compiled=False, timed_out=False, exit_code=None,
            compile_stderr="", compile_stdout="",
            run_stderr=str(e), run_stdout="", failure_type="runtime",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── pass@k (Chen et al., 2021) ─────────────────────────────────────────
def pass_at_k(n: int, c: int, k: int) -> float:
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# ── Data loading ────────────────────────────────────────────────────────
def load_data() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for fname in [DATA_FILE, DATA_FILE + ".jsonl", DATA_FILE + ".json"]:
        if not os.path.exists(fname):
            continue
        with open(fname, encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            continue

        entries: list[dict[str, Any]]
        if raw.startswith("["):
            entries = json.loads(raw)
        else:
            entries = [json.loads(line) for line in raw.splitlines() if line.strip()]

        for entry in entries:
            lang = entry.get("lang", entry.get("language", "")).lower()
            if lang == LANGUAGE:
                data.append(entry)

        print(f"Loaded {len(data)} {LANGUAGE} entries from {fname}")
        return data

    raise FileNotFoundError(f"Cannot find {DATA_FILE}")


def unique_nonempty(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        norm = text.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


# ── Repair error sanitization ──────────────────────────────────────────
def summarize_error_for_repair(ev: EvalResult) -> str:
    if REPAIR_ERROR_POLICY == "generic_label":
        return f"failure_type={ev.failure_type}"

    if ev.failure_type == "compile":
        pieces = ["failure_type=compile"]
        if ev.compile_stderr.strip():
            pieces.append("compiler diagnostics:\n" + ev.compile_stderr.strip()[:1500])
        elif ev.compile_stdout.strip():
            pieces.append("compiler output:\n" + ev.compile_stdout.strip()[:1500])
        return "\n\n".join(pieces)
    else:
        return f"failure_type={ev.failure_type}"


def summarize_error_full(ev: EvalResult) -> str:
    pieces = [f"failure_type={ev.failure_type}"]
    if ev.exit_code is not None:
        pieces.append(f"exit_code={ev.exit_code}")
    if ev.compile_stderr.strip():
        pieces.append("compile_stderr:\n" + ev.compile_stderr.strip()[:1200])
    if ev.run_stderr.strip():
        pieces.append("run_stderr:\n" + ev.run_stderr.strip()[:1200])
    elif ev.run_stdout.strip():
        pieces.append("run_stdout:\n" + ev.run_stdout.strip()[:1200])
    return "\n\n".join(pieces)


# ── Task pipeline ───────────────────────────────────────────────────────

def task_pipeline(entry: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
    task_id = entry.get("task_id", str(idx))
    assembly = entry["assembly"]
    tests = entry["tests"]

    print(f"\n[{idx + 1}/{total}] Task {task_id} — raw candidate budget = {RAW_K}")

    # ── Stage 1: generate N_REASONINGS analyses ─────────────────────
    reasoning_messages = [
        {"role": "system", "content": REASONING_SYSTEM},
        {"role": "user", "content": REASONING_USER.format(assembly=assembly)},
    ]
    raw_reasonings = generate_texts(
        reasoning_messages,
        N_REASONINGS,
        temperature=REASONING_TEMPERATURE,
        top_p=REASONING_TOP_P,
        max_new_tokens=REASONING_MAX_NEW_TOKENS,
        use_repetition_penalty=True,
    )
    assert len(raw_reasonings) == N_REASONINGS

    all_reasonings: list[str] = []
    for r_idx, r in enumerate(raw_reasonings):
        analysis = extract_xml_tag(r, "analysis")
        if analysis:
            all_reasonings.append(analysis)
        else:
            log_jsonl(
                DEBUG_JSONL,
                {
                    "task_id": task_id,
                    "stage": "reasoning_extraction_fail",
                    "reasoning_index": r_idx,
                    "raw_reasoning_preview": r[:500],
                    "has_open_tag": "<analysis>" in r.lower(),
                    "has_close_tag": "</analysis>" in r.lower(),
                },
            )
            fallback = re.sub(r"<think>.*?</think>", "", r, flags=re.DOTALL).strip()
            all_reasonings.append(fallback if fallback else "")
    assert len(all_reasonings) == N_REASONINGS

    n_valid = sum(1 for a in all_reasonings if a)
    print(f"  analyses: {n_valid}/{N_REASONINGS} non-empty")

    unique_reasonings = unique_nonempty(all_reasonings)
    print(f"  unique analyses: {len(unique_reasonings)}/{N_REASONINGS}")

    # ── Stage 2: code generation from EACH reasoning ────────────────
    all_raw_candidates: list[dict[str, Any]] = []

    for r_idx, analysis_text in enumerate(all_reasonings):
        code_messages = [
            {"role": "system", "content": CODE_SYSTEM},
            {
                "role": "user",
                "content": CODE_USER.format(
                    assembly=assembly,
                    analysis=f"<analysis>\n{analysis_text}\n</analysis>" if analysis_text else "(no analysis available)",
                ),
            },
        ]

        raw_outputs = generate_texts(
            code_messages,
            N_CODES_PER_REASONING,
            temperature=CODE_TEMPERATURE,
            top_p=CODE_TOP_P,
            max_new_tokens=CODE_MAX_NEW_TOKENS,
            use_repetition_penalty=False,
        )
        assert len(raw_outputs) == N_CODES_PER_REASONING

        for c_idx, raw_output in enumerate(raw_outputs):
            code = extract_dart_code(raw_output).strip()
            all_raw_candidates.append(
                {
                    "origin": "raw",
                    "reasoning_index": r_idx,
                    "code_index": c_idx,
                    "analysis": analysis_text,
                    "raw_output": raw_output,
                    "code": code,
                }
            )

    assert len(all_raw_candidates) == RAW_K, (
        f"Expected {RAW_K} raw candidates, got {len(all_raw_candidates)}"
    )

    deduped_raw: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for cand in all_raw_candidates:
        if not cand["code"].strip():
            continue
        key = canonical_key(cand["code"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_raw.append(cand)

    print(f"  unique non-empty raw candidates: {len(deduped_raw)}/{RAW_K}")

    # ── Evaluate all RAW_K draws ────────────────────────────────────
    eval_cache: dict[str, EvalResult] = {}

    def eval_code(code: str) -> EvalResult:
        if not code.strip():
            return EvalResult(
                passed=False, compiled=False, timed_out=False, exit_code=None,
                compile_stderr="", compile_stdout="",
                run_stderr="", run_stdout="", failure_type="empty",
            )
        ck = canonical_key(code)
        if ck in eval_cache:
            return eval_cache[ck]
        ev = run_dart_test(code, tests, task_id)
        eval_cache[ck] = ev
        return ev

    raw_eval_results: list[dict[str, Any]] = []
    for cand_idx, cand in enumerate(all_raw_candidates):
        ev = eval_code(cand["code"])
        raw_eval_results.append({**cand, "eval": ev})
        status = "✓" if ev.passed else ("·" if not cand["code"] else "✗")
        print(f"    raw {cand_idx + 1}/{RAW_K}: {status} ({ev.failure_type})")
        log_jsonl(
            DEBUG_JSONL,
            {
                "task_id": task_id,
                "stage": "raw_eval",
                "candidate_index": cand_idx,
                "reasoning_index": cand["reasoning_index"],
                "analysis": cand["analysis"][:300],
                "code": cand["code"],
                "raw_output_preview": cand["raw_output"][:500],
                "failure_type": ev.failure_type,
                "passed": ev.passed,
                "compiled": ev.compiled,
                "compile_stderr": ev.compile_stderr,
                "run_stderr": ev.run_stderr,
                "run_stdout": ev.run_stdout,
            },
        )

    # ── Paper-comparable raw pass@k ─────────────────────────────────
    raw_n = RAW_K
    raw_c = sum(int(r["eval"].passed) for r in raw_eval_results)
    raw_compiled = sum(int(r["eval"].compiled) for r in raw_eval_results)
    raw_compile_rate = raw_compiled / raw_n if raw_n else 0.0
    raw_runtime_pass_rate = raw_c / raw_compiled if raw_compiled else 0.0

    deduped_n = len(deduped_raw)
    deduped_c = sum(int(eval_code(c["code"]).passed) for c in deduped_raw)
    deduped_compiled = sum(int(eval_code(c["code"]).compiled) for c in deduped_raw)

    # ── Stage 3: repair ─────────────────────────────────────────────
    pipeline_candidates = list(raw_eval_results)

    if ENABLE_REPAIR:
        repair_budget = 0
        repaired_seen = {canonical_key(c["code"]) for c in deduped_raw if c["code"].strip()}

        repair_queue: list[dict[str, Any]] = []
        for cand in deduped_raw:
            ev = eval_code(cand["code"])
            if not ev.passed:
                repair_queue.append({**cand, "eval": ev})

        repair_queue.sort(
            key=lambda r: {"compile": 0, "runtime": 1, "timeout": 2, "empty": 3}.get(
                r["eval"].failure_type, 4
            )
        )

        for row in repair_queue:
            if repair_budget >= MAX_REPAIRS_PER_TASK:
                break

            error_summary = summarize_error_for_repair(row["eval"])

            repair_messages = [
                {"role": "system", "content": REPAIR_SYSTEM},
                {
                    "role": "user",
                    "content": REPAIR_USER.format(
                        assembly=assembly,
                        analysis=f"<analysis>\n{row['analysis']}\n</analysis>",
                        code=row["code"],
                        error_summary=error_summary,
                    ),
                },
            ]

            n_repair_samples = min(
                REPAIR_SAMPLES_PER_CANDIDATE,
                MAX_REPAIRS_PER_TASK - repair_budget,
            )

            repaired_outputs = generate_texts(
                repair_messages,
                n_repair_samples,
                temperature=REPAIR_TEMPERATURE,
                top_p=REPAIR_TOP_P,
                max_new_tokens=REPAIR_MAX_NEW_TOKENS,
                use_repetition_penalty=False,
            )

            for repaired_output in repaired_outputs:
                if repair_budget >= MAX_REPAIRS_PER_TASK:
                    break

                repaired_code = extract_dart_code(repaired_output).strip()
                if not repaired_code:
                    continue
                rk = canonical_key(repaired_code)
                if rk in repaired_seen:
                    continue
                repaired_seen.add(rk)
                repair_budget += 1

                repaired_eval = eval_code(repaired_code)
                pipeline_candidates.append(
                    {
                        "origin": "repair",
                        "reasoning_index": row["reasoning_index"],
                        "analysis": row["analysis"],
                        "raw_output": repaired_output,
                        "code": repaired_code,
                        "eval": repaired_eval,
                    }
                )
                status = "✓" if repaired_eval.passed else "✗"
                print(
                    f"    repair {repair_budget}/{MAX_REPAIRS_PER_TASK}: {status} ({repaired_eval.failure_type})"
                )
                log_jsonl(
                    DEBUG_JSONL,
                    {
                        "task_id": task_id,
                        "stage": "repair_eval",
                        "repair_error_policy": REPAIR_ERROR_POLICY,
                        "error_summary_given": error_summary,
                        "original_code": row["code"],
                        "original_failure": row["eval"].failure_type,
                        "repaired_code": repaired_code,
                        "failure_type": repaired_eval.failure_type,
                        "passed": repaired_eval.passed,
                        "compiled": repaired_eval.compiled,
                        "compile_stderr": repaired_eval.compile_stderr,
                        "run_stderr": repaired_eval.run_stderr,
                    },
                )

    # ── Pipeline metrics ────────────────────────────────────────────
    pipe_n = len(pipeline_candidates)
    pipe_c = sum(int(r["eval"].passed) for r in pipeline_candidates)
    pipe_compiled = sum(int(r["eval"].compiled) for r in pipeline_candidates)
    pipe_compile_rate = pipe_compiled / pipe_n if pipe_n else 0.0

    failure_types: dict[str, int] = {}
    for r in raw_eval_results:
        ft = r["eval"].failure_type
        failure_types[ft] = failure_types.get(ft, 0) + 1

    result_row: dict[str, Any] = {
        "task_id": task_id,
        "raw_n": raw_n,
        "raw_c": raw_c,
        "raw_compile_rate": raw_compile_rate,
        "raw_runtime_pass_rate": raw_runtime_pass_rate,
        "deduped_n": deduped_n,
        "deduped_c": deduped_c,
        "deduped_compiled": deduped_compiled,
        "system_n": pipe_n,
        "system_c": pipe_c,
        "system_compile_rate": pipe_compile_rate,
        "failure_breakdown": json.dumps(failure_types),
    }

    for k_val in [1, 5, 10]:
        if k_val <= raw_n:
            result_row[f"raw_pass@{k_val}"] = pass_at_k(raw_n, raw_c, k_val)
        if k_val <= deduped_n:
            result_row[f"deduped_pass@{k_val}"] = pass_at_k(deduped_n, deduped_c, k_val)
        if k_val <= pipe_n:
            result_row[f"system_success@{k_val}"] = pass_at_k(pipe_n, pipe_c, k_val)

    print(
        f"  ⇒ raw (paper, n={raw_n}): "
        f"{raw_c}/{raw_n} passed, compile={raw_compile_rate:.3f}, "
        f"runtime_pass|compiled={raw_runtime_pass_rate:.3f}, "
        f"failures={failure_types}"
    )
    print(
        "    "
        + "  ".join(
            f"pass@{k}: {result_row.get(f'raw_pass@{k}', float('nan')):.4f}"
            for k in [1, 5, 10]
            if f"raw_pass@{k}" in result_row
        )
    )
    print(
        f"  ⇒ deduped (aux, n={deduped_n}): "
        f"{deduped_c}/{deduped_n} passed | "
        + "  ".join(
            f"pass@{k}: {result_row.get(f'deduped_pass@{k}', float('nan')):.4f}"
            for k in [1, 5, 10]
            if f"deduped_pass@{k}" in result_row
        )
    )
    print(
        f"  ⇒ system (raw+repair, n={pipe_n}): "
        f"{pipe_c}/{pipe_n} passed, compile={pipe_compile_rate:.3f} | "
        + "  ".join(
            f"pass@{k}: {result_row.get(f'system_success@{k}', float('nan')):.4f}"
            for k in [1, 5, 10]
            if f"system_success@{k}" in result_row
        )
    )

    return result_row


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    if os.path.exists(DEBUG_JSONL):
        os.remove(DEBUG_JSONL)

    data = load_data()
    results = []
    for idx, entry in enumerate(data):
        results.append(task_pipeline(entry, idx, len(data)))

    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)

    metrics_to_report = [
        "raw_pass@1", "raw_pass@5", "raw_pass@10",
        "raw_compile_rate", "raw_runtime_pass_rate",
        "deduped_pass@1", "deduped_pass@5", "deduped_pass@10",
        "system_success@1", "system_success@5", "system_success@10",
        "system_compile_rate",
    ]

    summary_rows = []
    for metric in metrics_to_report:
        values = [row[metric] for row in results if metric in row]
        if not values:
            continue
        avg = float(np.mean(values))
        label = (
            "(PAPER)" if metric.startswith("raw_")
            else "(aux)" if metric.startswith("deduped_")
            else "(system)"
        )
        print(f"  {metric:<28} = {avg:.4f}  {label}  (over {len(values)} tasks)")
        summary_rows.append([metric, f"{avg:.6f}", len(values)])

    total_failures: dict[str, int] = {}
    for row in results:
        fb = json.loads(row.get("failure_breakdown", "{}"))
        for ft, count in fb.items():
            total_failures[ft] = total_failures.get(ft, 0) + count
    print(f"\n  aggregate raw failure breakdown: {total_failures}")
    total_raw = sum(total_failures.values())
    if total_raw:
        for ft, count in sorted(total_failures.items(), key=lambda x: -x[1]):
            print(f"    {ft:<12}: {count:>5}  ({100 * count / total_raw:.1f}%)")
    summary_rows.append(["failure_breakdown", json.dumps(total_failures), len(results)])

    per_task_file = f"dart_two_stage_results_{MODEL_NAME.replace('/', '_')}.csv"
    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(per_task_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_file = f"dart_two_stage_summary_{MODEL_NAME.replace('/', '_')}.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "n_tasks"])
        writer.writerows(summary_rows)

    print(f"\nPer-task results  → {per_task_file}")
    print(f"Summary           → {summary_file}")
    print(f"Debug JSONL       → {DEBUG_JSONL}")


if __name__ == "__main__":
    main()