from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
MODEL_DIR = "raafatabualazm/decompiler-v1"
DATA_FILE = "grpo_data"
LANGUAGE = "dart"

# Raw paper-comparable candidate budget = N_REASONINGS * N_CODES_PER_REASONING
N_REASONINGS = 4
N_CODES_PER_REASONING = 3

# Optional repair budget (report separately from raw pass@k)
ENABLE_REPAIR = True
MAX_REPAIRS_PER_TASK = 4

REASONING_TEMPERATURE = 0.7
REASONING_TOP_P = 0.95
REASONING_MAX_NEW_TOKENS = 900

CODE_TEMPERATURE = 0.35
CODE_TOP_P = 0.9
CODE_MAX_NEW_TOKENS = 2048

REPAIR_TEMPERATURE = 0.2
REPAIR_TOP_P = 0.9
REPAIR_MAX_NEW_TOKENS = 1536

BATCH_SIZE = 4
DART_TIMEOUT = 30
SEED = 1337
DEBUG_JSONL = f"dart_two_stage_debug_{MODEL_DIR.replace('/', '_')}.jsonl"
# ────────────────────────────────────────────────────────────────────────

RAW_K = N_REASONINGS * N_CODES_PER_REASONING

accelerator = Accelerator()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.cuda.set_device(accelerator.local_process_index)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def log_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Load model & tokenizer ──────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(tokenizer, "padding_side", None) != "left":
    tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map={"": accelerator.local_process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = True
model.eval()

try:
    model = torch.compile(model, mode="reduce-overhead")
except Exception:
    pass


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

Return ONLY valid Dart code.
Rules:
- No explanation.
- No markdown fences unless unavoidable.
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
{code}

Observed failure:
{error_summary}

Return ONLY corrected Dart code.
Rules:
- No explanation.
- No markdown fences unless unavoidable.
- No main().
- Keep the original algorithm unless the failure clearly implies it is wrong.
"""


def build_prompt(messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for non-chat models.
    prompt_parts = []
    for m in messages:
        prompt_parts.append(f"[{m['role'].upper()}]\n{m['content']}\n")
    prompt_parts.append("[ASSISTANT]\n")
    return "\n".join(prompt_parts)


@dataclass
class EvalResult:
    passed: bool
    compiled: bool
    timed_out: bool
    exit_code: int | None
    stderr: str
    stdout: str
    failure_type: str  # pass | compile | runtime | timeout | empty


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
                    code = code[:start] + code[i + 1 :]
                    break
            i += 1

    return code.strip()


def extract_dart_code(text: str) -> str:
    text = text.strip()

    for tag in ["final_code", "code", "answer", "response"]:
        tagged = extract_xml_tag(text, tag)
        if tagged:
            text = tagged
            break

    if "```dart" in text:
        text = text.split("```dart", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    # If analysis leaked into the answer, drop it.
    if "</analysis>" in text:
        text = text.split("</analysis>", 1)[1]

    return text.strip()


def normalize_code(code: str) -> str:
    code = strip_main_and_imports(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def generate_texts(
    messages: list[dict[str, str]],
    n_samples: int,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    prompt = build_prompt(messages)
    samples: list[str] = []

    remaining = n_samples
    while remaining > 0:
        batch_n = min(remaining, BATCH_SIZE)
        batch_prompts = [prompt] * batch_n

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(
                getattr(tokenizer, "model_max_length", 32768),
                32768,
            ),
        ).to(model.device)

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                renormalize_logits=True,
            )

        generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        samples.extend(decoded)
        remaining -= batch_n

    return samples[:n_samples]


def classify_failure(stderr: str, stdout: str, returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "pass"
    if not stderr and not stdout:
        return "empty"
    lower = (stderr + "\n" + stdout).lower()
    if "unhandled exception" in lower or "exception:" in lower:
        return "runtime"
    if "error:" in lower or "failed to compile" in lower or "compilation failed" in lower:
        return "compile"
    return "runtime"


def run_dart_test(generated_code: str, test_code: str, task_id: str) -> EvalResult:
    clean_code = normalize_code(generated_code)
    imports = set(re.findall(r"^import\s+.*;\s*$", generated_code, re.MULTILINE))
    full_source = "\n".join(imports) + "\n\n" + clean_code + "\n\n" + test_code

    if not clean_code.strip():
        return EvalResult(
            passed=False,
            compiled=False,
            timed_out=False,
            exit_code=None,
            stderr="",
            stdout="",
            failure_type="empty",
        )

    tmp_dir = tempfile.mkdtemp(prefix=f"dart_passk_{task_id}_")
    dart_file = os.path.join(tmp_dir, "test.dart")

    with open(dart_file, "w", encoding="utf-8") as f:
        f.write(full_source)

    try:
        result = subprocess.run(
            ["dart", "run", dart_file],
            capture_output=True,
            text=True,
            timeout=DART_TIMEOUT,
        )
        failure_type = classify_failure(result.stderr, result.stdout, result.returncode, False)
        return EvalResult(
            passed=result.returncode == 0,
            compiled=(failure_type != "compile"),
            timed_out=False,
            exit_code=result.returncode,
            stderr=result.stderr[-2000:],
            stdout=result.stdout[-2000:],
            failure_type=failure_type,
        )
    except subprocess.TimeoutExpired as e:
        return EvalResult(
            passed=False,
            compiled=True,
            timed_out=True,
            exit_code=None,
            stderr=(e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
            stdout=(e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            failure_type="timeout",
        )
    except Exception as e:
        return EvalResult(
            passed=False,
            compiled=False,
            timed_out=False,
            exit_code=None,
            stderr=str(e),
            stdout="",
            failure_type="runtime",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── pass@k (Chen et al.) ───────────────────────────────────────────────
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
        if os.path.exists(fname):
            with open(fname, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    lang = entry.get("lang", entry.get("language", "")).lower()
                    if lang == LANGUAGE:
                        data.append(entry)
            print(f"Loaded {len(data)} {LANGUAGE} entries from {fname}")
            return data
    raise FileNotFoundError(f"Cannot find {DATA_FILE}")


def unique_nonempty(texts: list[str]) -> list[str]:
    seen = set()
    out = []
    for text in texts:
        norm = text.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def summarize_error(ev: EvalResult) -> str:
    pieces = [f"failure_type={ev.failure_type}"]
    if ev.exit_code is not None:
        pieces.append(f"exit_code={ev.exit_code}")
    if ev.stderr.strip():
        pieces.append("stderr:\n" + ev.stderr.strip()[:1200])
    elif ev.stdout.strip():
        pieces.append("stdout:\n" + ev.stdout.strip()[:1200])
    return "\n\n".join(pieces)


def task_pipeline(entry: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
    task_id = entry.get("task_id", str(idx))
    assembly = entry["assembly"]
    tests = entry["tests"]

    print(f"\n[{idx + 1}/{total}] Task {task_id} — raw candidate budget = {RAW_K}")

    # Stage 1: concise semantic analyses.
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
    )
    reasonings = unique_nonempty(
        [extract_xml_tag(r, "analysis") or r.strip() for r in raw_reasonings]
    )
    if not reasonings:
        reasonings = [""]

    print(f"  analyses: {len(reasonings)}/{N_REASONINGS} unique")

    # Stage 2: code-only generation from each reasoning.
    raw_candidates: list[dict[str, Any]] = []
    for r_idx, analysis_text in enumerate(reasonings[:N_REASONINGS]):
        code_messages = [
            {"role": "system", "content": CODE_SYSTEM},
            {
                "role": "user",
                "content": CODE_USER.format(
                    assembly=assembly,
                    analysis=f"<analysis>\n{analysis_text}\n</analysis>",
                ),
            },
        ]

        raw_outputs = generate_texts(
            code_messages,
            N_CODES_PER_REASONING,
            temperature=CODE_TEMPERATURE,
            top_p=CODE_TOP_P,
            max_new_tokens=CODE_MAX_NEW_TOKENS,
        )

        for c_idx, raw_output in enumerate(raw_outputs):
            code = normalize_code(extract_dart_code(raw_output))
            raw_candidates.append(
                {
                    "origin": "raw",
                    "reasoning_index": r_idx,
                    "code_index": c_idx,
                    "analysis": analysis_text,
                    "raw_output": raw_output,
                    "code": code,
                }
            )

    # Deduplicate by normalized code while preserving order.
    deduped_raw = []
    seen_code = set()
    for cand in raw_candidates:
        key = cand["code"]
        if not key or key in seen_code:
            continue
        seen_code.add(key)
        deduped_raw.append(cand)

    # Keep the first RAW_K unique raw candidates for paper-comparable pass@k.
    deduped_raw = deduped_raw[:RAW_K]
    print(f"  unique raw code candidates: {len(deduped_raw)}/{RAW_K}")

    raw_eval_results = []
    compile_count = 0
    pass_count = 0
    for cand_idx, cand in enumerate(deduped_raw):
        ev = run_dart_test(cand["code"], tests, task_id)
        compile_count += int(ev.compiled)
        pass_count += int(ev.passed)
        raw_eval_results.append({**cand, "eval": ev})
        status = "✓" if ev.passed else "✗"
        print(f"    raw {cand_idx + 1}/{len(deduped_raw)}: {status} ({ev.failure_type})")
        log_jsonl(
            DEBUG_JSONL,
            {
                "task_id": task_id,
                "stage": "raw_eval",
                "candidate_index": cand_idx,
                "reasoning_index": cand["reasoning_index"],
                "analysis": cand["analysis"],
                "code": cand["code"],
                "failure_type": ev.failure_type,
                "passed": ev.passed,
                "compiled": ev.compiled,
                "stderr": ev.stderr,
                "stdout": ev.stdout,
            },
        )

    pipeline_candidates = list(raw_eval_results)

    # Stage 3: optional minimal repair using compiler/runtime feedback only.
    if ENABLE_REPAIR:
        repair_budget = 0
        repaired_seen = {cand["code"] for cand in deduped_raw}
        failed_for_repair = [row for row in raw_eval_results if not row["eval"].passed]
        # Prefer compile failures first; they are often high-ROI fixes.
        failed_for_repair.sort(
            key=lambda row: {"compile": 0, "runtime": 1, "timeout": 2, "empty": 3}.get(row["eval"].failure_type, 4)
        )

        for row in failed_for_repair:
            if repair_budget >= MAX_REPAIRS_PER_TASK:
                break

            repair_messages = [
                {"role": "system", "content": REPAIR_SYSTEM},
                {
                    "role": "user",
                    "content": REPAIR_USER.format(
                        assembly=assembly,
                        analysis=f"<analysis>\n{row['analysis']}\n</analysis>",
                        code=row["code"],
                        error_summary=summarize_error(row["eval"]),
                    ),
                },
            ]

            repaired_output = generate_texts(
                repair_messages,
                1,
                temperature=REPAIR_TEMPERATURE,
                top_p=REPAIR_TOP_P,
                max_new_tokens=REPAIR_MAX_NEW_TOKENS,
            )[0]
            repaired_code = normalize_code(extract_dart_code(repaired_output))
            if not repaired_code or repaired_code in repaired_seen:
                continue
            repaired_seen.add(repaired_code)
            repair_budget += 1

            repaired_eval = run_dart_test(repaired_code, tests, task_id)
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
            print(f"    repair {repair_budget}/{MAX_REPAIRS_PER_TASK}: {status} ({repaired_eval.failure_type})")
            log_jsonl(
                DEBUG_JSONL,
                {
                    "task_id": task_id,
                    "stage": "repair_eval",
                    "analysis": row["analysis"],
                    "original_code": row["code"],
                    "repaired_code": repaired_code,
                    "failure_type": repaired_eval.failure_type,
                    "passed": repaired_eval.passed,
                    "compiled": repaired_eval.compiled,
                    "stderr": repaired_eval.stderr,
                    "stdout": repaired_eval.stdout,
                },
            )

    raw_n = len(raw_eval_results)
    raw_c = sum(int(row["eval"].passed) for row in raw_eval_results)
    raw_compile_rate = (sum(int(row["eval"].compiled) for row in raw_eval_results) / raw_n) if raw_n else 0.0

    pipe_n = len(pipeline_candidates)
    pipe_c = sum(int(row["eval"].passed) for row in pipeline_candidates)
    pipe_compile_rate = (sum(int(row["eval"].compiled) for row in pipeline_candidates) / pipe_n) if pipe_n else 0.0

    row = {
        "task_id": task_id,
        "raw_n": raw_n,
        "raw_c": raw_c,
        "pipeline_n": pipe_n,
        "pipeline_c": pipe_c,
        "raw_compile_rate": raw_compile_rate,
        "pipeline_compile_rate": pipe_compile_rate,
    }
    for k_val in [1, 5, 10]:
        if k_val <= raw_n:
            row[f"raw_pass@{k_val}"] = pass_at_k(raw_n, raw_c, k_val)
        if k_val <= pipe_n:
            row[f"pipeline_pass@{k_val}"] = pass_at_k(pipe_n, pipe_c, k_val)

    print(
        "  ⇒ raw: "
        f"{raw_c}/{raw_n} passed, compile={raw_compile_rate:.3f} | "
        + "  ".join(
            f"pass@{k}: {row.get(f'raw_pass@{k}', float('nan')):.4f}"
            for k in [1, 5, 10]
            if f"raw_pass@{k}" in row
        )
    )
    print(
        "  ⇒ pipeline: "
        f"{pipe_c}/{pipe_n} passed, compile={pipe_compile_rate:.3f} | "
        + "  ".join(
            f"pass@{k}: {row.get(f'pipeline_pass@{k}', float('nan')):.4f}"
            for k in [1, 5, 10]
            if f"pipeline_pass@{k}" in row
        )
    )

    torch.cuda.empty_cache()
    return row


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
        "raw_pass@1",
        "raw_pass@5",
        "raw_pass@10",
        "pipeline_pass@1",
        "pipeline_pass@5",
        "pipeline_pass@10",
        "raw_compile_rate",
        "pipeline_compile_rate",
    ]

    summary_rows = []
    for metric in metrics_to_report:
        values = [row[metric] for row in results if metric in row]
        if not values:
            continue
        avg = float(np.mean(values))
        print(f"  {metric:<22} = {avg:.4f}  (over {len(values)} tasks)")
        summary_rows.append([metric, f"{avg:.6f}", len(values)])

    per_task_file = f"dart_two_stage_results_{MODEL_DIR.replace('/', '_')}.csv"
    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(per_task_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_file = f"dart_two_stage_summary_{MODEL_DIR.replace('/', '_')}.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "n_tasks"])
        writer.writerows(summary_rows)

    print(f"\nPer-task results  → {per_task_file}")
    print(f"Summary           → {summary_file}")
    print(f"Debug JSONL       → {DEBUG_JSONL}")


if __name__ == "__main__":
    main()
