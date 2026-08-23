#!/usr/bin/env python3
"""
External-baseline candidate generation for Dart AOT neural decompilation.

Purpose
-------
This is the "computed external baseline" a reviewer asks for: what does a strong,
general, *off-the-shelf* LLM recover from the same x86-64 Dart AOT assembly, under
the same evaluation harness, WITHOUT any of the graph-conditioning / LoRA training
in the paper? It calls a model through the OpenRouter API and emits a predictions
file whose schema is byte-for-byte compatible with the local pass@k evaluator
(graph_pass_at_k_antigravity.py), so the numbers land in the same table.

Two prompt modes let you answer two distinct reviewer questions with one script:
  --mode asm        raw assembly only            -> "does a frontier LLM beat our 8B?"
  --mode asm_cfg    assembly + serialized CFG/DFG -> prompt-level graph vs our learned
                                                     graph-prefix (the CodeInverter contrast)

Design choices that make this a *fair* baseline, not a strawman:
  * The full assembly is sent (long context is the frontier model's natural
    advantage); we do NOT truncate to the paper's 2048-token decoder budget.
  * Sampling parity with the paper: temperature 0.7, top_p 0.95, 10 samples for
    pass@10 (configurable). Greedy single-sample is available via --temperature 0.
  * The system/user prompt is frozen here; tune it on a handful of tasks, then keep
    it fixed for the reported run. A lazy prompt invites a "you sandbagged the
    baseline" objection.
  * Model id, query date, prompt hash, and all raw generations are written into the
    output for provenance (API models drift; reviewers increasingly check this).

Output schema (JSON array; one object per task) -- matches the evaluator:
    [
      {
        "id":         <task_id>,          # evaluator reads row['id']
        "task_id":    <task_id>,          # kept for convenience
        "predictions":[<raw gen>, ...],   # evaluator reads row['predictions']
        "tests":      <dart test harness>,# evaluator reads row['tests']
        "function":   <name>,             # provenance only
        "prompt_mode":"asm" | "asm_cfg",  # provenance only
        "meta": {...}                     # model, params, prompt hash, timings
      },
      ...
    ]

Then score with the UNMODIFIED local harness, e.g.:
    python graph_pass_at_k_antigravity.py \
        --predictions preds_frontier_asm.json --k_values 1,5,10

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...
    python openrouter_baseline.py \
        --input grpo_data_cfg.jsonl \
        --output preds_frontier_asm.json \
        --model deepseek/deepseek-chat \
        --mode asm --samples 10 --temperature 0.7 --workers 8

Notes
-----
  * No third-party deps required (uses urllib). `tqdm` is used if present.
  * --limit N runs only the first N tasks (cheap smoke test before the full run).
  * --resume reuses any tasks already present in --output (crash-safe / cost-safe).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional
    tqdm = None


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are an expert reverse engineer specializing in Dart Ahead-of-Time (AOT) "
    "compiled binaries. You are given the x86-64 disassembly of a single Dart "
    "function compiled by the Dart AOT compiler (gen_snapshot). Reconstruct the "
    "original high-level Dart source for that one function.\n\n"
    "Rules:\n"
    "1. Output ONLY the Dart function, inside a single ```dart code block.\n"
    "2. Match the given function signature exactly (name, parameter types, return "
    "type).\n"
    "3. Do NOT include a main() function, tests, imports, or commentary.\n"
    "4. Produce idiomatic, compilable Dart that is functionally equivalent to the "
    "assembly -- same inputs must produce same outputs.\n"
    "5. Dart AOT code uses runtime-specific object layouts, boxed numbers, and "
    "bounds checks; recover the *program logic*, not those runtime artifacts."
)


def _format_cfg(cfg: list, edges: list, max_blocks: int = 64) -> str:
    """Serialize CFG/DFG structure compactly for the asm_cfg prompt mode.

    Mirrors the *information* the paper's graph channel encodes (basic blocks,
    control-flow edges, per-block instruction summaries), but as text in the
    prompt -- this is deliberately the CodeInverter-style serialized-structure
    baseline, to contrast against the paper's learned graph-prefix path.
    """
    lines = []
    lines.append(f"Control-flow graph: {len(cfg)} basic blocks.")
    shown = cfg[:max_blocks]
    for b in shown:
        bid = b.get("id")
        btype = b.get("block_type", "?")
        succ = b.get("successors", []) or []
        etypes = b.get("edge_types", []) or []
        n_ins = b.get("instruction_count", len(b.get("instructions", []) or []))
        if succ and etypes and len(succ) == len(etypes):
            edge_str = ", ".join(f"{s}[{t}]" for s, t in zip(succ, etypes))
        elif succ:
            edge_str = ", ".join(str(s) for s in succ)
        else:
            edge_str = "-"
        lines.append(f"  block {bid} ({btype}, {n_ins} instrs) -> {edge_str}")
    if len(cfg) > max_blocks:
        lines.append(f"  ... ({len(cfg) - max_blocks} more blocks omitted)")
    # A compact edge list can help the model see back-edges/loops at a glance.
    if edges:
        e_preview = "; ".join(
            f"{e.get('source')}->{e.get('target')}[{e.get('edge_type','?')}]"
            for e in edges[:48]
        )
        more = "" if len(edges) <= 48 else f" ; ... (+{len(edges) - 48} edges)"
        lines.append(f"Edges: {e_preview}{more}")
    return "\n".join(lines)


def build_user_prompt(row: dict, mode: str) -> str:
    sig = row.get("dart_function_signature", "").strip()
    asm = row.get("assembly", "")
    parts = []
    if sig:
        parts.append(f"Target Dart function signature:\n{sig}\n")
    parts.append("x86-64 AOT disassembly:\n```asm\n" + asm.strip() + "\n```")
    if mode == "asm_cfg":
        cfg = row.get("cfg", []) or []
        edges = row.get("edges", []) or []
        if cfg:
            parts.append(
                "\nExtracted control-/data-flow structure (for reference):\n"
                + _format_cfg(cfg, edges)
            )
    parts.append(
        "\nReconstruct the original Dart function. Output only a single "
        "```dart code block."
    )
    return "\n".join(parts)


def prompt_hash(system: str, mode: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(mode.encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# OpenRouter call with retries
# --------------------------------------------------------------------------- #

class RateLimit(Exception):
    pass


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    referer: str,
    title: str,
    max_retries: int = 5,
) -> tuple[str, dict]:
    """Return (text, raw_usage_dict). Retries on 429/5xx with backoff."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter attribution headers (optional but recommended):
        "HTTP-Referer": referer,
        "X-Title": title,
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_URL, data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Standard OpenAI-compatible shape
            choice = payload["choices"][0]
            text = choice.get("message", {}).get("content", "") or ""
            usage = payload.get("usage", {}) or {}
            usage["_model_reported"] = payload.get("model", model)
            return text, usage
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                err_body = e.read().decode("utf-8")[:400]
            except Exception:
                err_body = ""
            last_err = f"HTTP {code}: {err_body}"
            if code in (429, 500, 502, 503, 504):
                sleep = min(2 ** attempt + 0.5 * attempt, 30)
                time.sleep(sleep)
                continue
            # Non-retryable (400/401/403): stop early
            raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = f"network: {e}"
            time.sleep(min(2 ** attempt, 20))
            continue
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = f"malformed response: {e}"
            time.sleep(1.0 + attempt)
            continue
    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


# --------------------------------------------------------------------------- #
# Task loading / driving
# --------------------------------------------------------------------------- #

def load_tasks(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def load_existing(output_path: str) -> dict:
    """Return {task_id: record} for resume; empty if file absent/unreadable."""
    p = Path(output_path)
    if not p.is_file():
        return {}
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
        return {str(r.get("task_id", r.get("id"))): r for r in arr}
    except Exception:
        return {}


def generate_for_task(
    row: dict,
    *,
    api_key: str,
    model: str,
    mode: str,
    samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    referer: str,
    title: str,
) -> dict:
    tid = str(row.get("task_id", row.get("id")))
    user_prompt = build_user_prompt(row, mode)

    predictions: list[str] = []
    prompt_tok = comp_tok = 0
    errors = 0
    t0 = time.time()

    # Greedy single sample if temperature == 0, else `samples` stochastic draws.
    n_draws = 1 if temperature == 0 else samples
    for _ in range(n_draws):
        try:
            text, usage = call_openrouter(
                api_key, model, SYSTEM_PROMPT, user_prompt,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, timeout=timeout,
                referer=referer, title=title,
            )
            predictions.append(text)
            prompt_tok += int(usage.get("prompt_tokens", 0) or 0)
            comp_tok += int(usage.get("completion_tokens", 0) or 0)
        except Exception as e:  # keep going; record an empty candidate
            predictions.append("")
            errors += 1
            print(f"[warn] task {tid}: {e}", file=sys.stderr)

    return {
        "id": tid,                       # evaluator reads this
        "task_id": tid,
        "predictions": predictions,      # evaluator reads this
        "tests": row.get("tests", ""),   # evaluator reads this
        "function": row.get("function", ""),
        "dart_function_signature": row.get("dart_function_signature", ""),
        "prompt_mode": mode,
        "meta": {
            "model": model,
            "n_candidates": len(predictions),
            "n_errors": errors,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "seconds": round(time.time() - t0, 2),
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate an external LLM baseline via OpenRouter, "
                    "output-compatible with the local pass@k evaluator."
    )
    ap.add_argument("--input", required=True,
                    help="Path to the 154-task JSONL (grpo_data_cfg.jsonl).")
    ap.add_argument("--output", required=True,
                    help="Path to write the predictions JSON array.")
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id, e.g. deepseek/deepseek-chat, "
                         "qwen/qwen-2.5-coder-32b-instruct, anthropic/claude-..., "
                         "openai/gpt-...")
    ap.add_argument("--mode", choices=["asm", "asm_cfg"], default="asm",
                    help="asm = assembly only; asm_cfg = assembly + serialized "
                         "CFG/DFG (prompt-level graph baseline).")
    ap.add_argument("--samples", type=int, default=10,
                    help="Stochastic samples per task for pass@k (default 10).")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0 = greedy single sample; else nucleus sampling.")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1024,
                    help="Max completion tokens per candidate.")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-request timeout (seconds).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel tasks. Keep modest to respect rate limits.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run the first N tasks (0 = all). Cheap smoke test.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse tasks already present in --output.")
    ap.add_argument("--referer", default="https://example.org/dart-decompilation",
                    help="OpenRouter HTTP-Referer attribution header.")
    ap.add_argument("--title", default="Dart AOT Decompilation Baseline",
                    help="OpenRouter X-Title attribution header.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: set OPENROUTER_API_KEY in the environment.")

    tasks = load_tasks(args.input)
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    existing = load_existing(args.output) if args.resume else {}
    todo = [r for r in tasks
            if str(r.get("task_id", r.get("id"))) not in existing]

    print(
        f"model={args.model} mode={args.mode} "
        f"samples={'1(greedy)' if args.temperature == 0 else args.samples} "
        f"| {len(tasks)} tasks, {len(existing)} cached, {len(todo)} to run "
        f"| prompt_hash={prompt_hash(SYSTEM_PROMPT, args.mode)}",
        file=sys.stderr,
    )

    results: dict[str, dict] = dict(existing)
    write_lock = threading.Lock()

    def flush():
        """Atomically write the current results (crash/cost-safe)."""
        ordered = [results[str(r.get("task_id", r.get("id")))]
                   for r in tasks
                   if str(r.get("task_id", r.get("id"))) in results]
        tmp = args.output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2)
        os.replace(tmp, args.output)

    def run_one(row: dict) -> dict:
        rec = generate_for_task(
            row,
            api_key=api_key, model=args.model, mode=args.mode,
            samples=args.samples, temperature=args.temperature,
            top_p=args.top_p, max_tokens=args.max_tokens,
            timeout=args.timeout, referer=args.referer, title=args.title,
        )
        with write_lock:
            results[rec["task_id"]] = rec
            flush()  # persist after every task
        return rec

    if todo:
        if args.workers > 1:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(run_one, r) for r in todo]
                it = cf.as_completed(futs)
                if tqdm is not None:
                    it = tqdm(it, total=len(futs), desc="generate", unit="task")
                for _ in it:
                    pass
        else:
            it = tqdm(todo, desc="generate", unit="task") if tqdm else todo
            for r in it:
                run_one(r)

    flush()

    # Summary
    tot_prompt = sum(r["meta"]["prompt_tokens"] for r in results.values())
    tot_comp = sum(r["meta"]["completion_tokens"] for r in results.values())
    tot_err = sum(r["meta"]["n_errors"] for r in results.values())
    print(
        f"\nDone. {len(results)} tasks written to {args.output}\n"
        f"tokens: prompt={tot_prompt:,} completion={tot_comp:,} "
        f"| candidate errors={tot_err}\n"
        f"Next: python graph_pass_at_k_antigravity.py "
        f"--predictions {args.output} --k_values 1,5,10",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
