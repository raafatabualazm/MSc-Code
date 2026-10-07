#!/usr/bin/env python3
"""
External-baseline candidate generation for Dart AOT neural decompilation.

Purpose
-------
This is the "computed external baseline" a reviewer asks for: what does a strong,
general, *off-the-shelf* LLM recover from the same x86-64 Dart AOT assembly, under
the same evaluation harness, WITHOUT any of the graph-conditioning / LoRA training
in the paper? It calls a model through either the OpenRouter API or Azure OpenAI
and emits a predictions file whose schema is byte-for-byte compatible with the
local pass@k evaluator (graph_pass_at_k_antigravity.py), so the numbers land in the
same table.

Providers (--provider)
----------------------
  openrouter : OPENROUTER_API_KEY. Sends temperature/top_p/max_tokens. Use for the
               open-weights and proprietary models routed through OpenRouter
               (deepseek, qwen, glm, claude, ...).
  azure      : AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, 'api-key' header. Uses
               max_completion_tokens and, by default, OMITS temperature/top_p
               because GPT-5-class and o-series reasoning deployments reject them
               and run at a fixed default temperature. --model is the deployment
               name (e.g. gpt-5.5). Set --azure_api_version only for the classic
               deployment endpoint; leave empty for the v1 endpoint.

  IMPORTANT (methodology): when Azure omits temperature/top_p, the deployment
  samples at its own default temperature, not the paper's 0.7. pass@k is still
  well-defined (we draw `samples` stochastic completions), but the frontier rows
  produced this way are at the deployment default, not 0.7/0.95 -- state this in
  the paper's setup rather than claiming exact sampling parity. Each record's meta
  records temperature="deployment_default" so the provenance is explicit.

Two prompt modes let you answer two distinct reviewer questions with one script:
  --mode asm        raw assembly only            -> "does a frontier LLM beat our 8B?"
  --mode asm_cfg    assembly + serialized CFG/DFG -> prompt-level graph vs our learned
                                                     graph-prefix (the CodeInverter contrast)

Fairness ablations (address the "not architecture-controlled" reviewer objection):
  1. Same-budget frontier:  --max_asm_tokens 2048
     Runs a frontier model under the same ~2048-token prompt budget the specialized
     decoder sees, separating frontier reasoning from the full-assembly context
     advantage. (~46/154 tasks exceed this budget and get truncated.)
  2. Local base zero-shot:  --model qwen/qwen3-8b   (no code change; just pick the model)
     Separates local-base weakness from the training/graph effects.
  3. Graph-summary only:    --mode graph_only
     Frontier model gets the serialized CFG/DFG and NO raw assembly, testing whether
     the graph summary alone suffices / how lossy it is versus full assembly.

Design choices that make this a *fair* baseline, not a strawman:
  * Full assembly is sent by default (long context is the frontier model's natural
    advantage); truncation is opt-in via --max_asm_tokens for the ablation above.
  * The system/user prompt is frozen here; tune it on a handful of tasks, then keep
    it fixed for the reported run. A lazy prompt invites a "you sandbagged the
    baseline" objection.
  * Provider, model, params, prompt hash, token/reasoning-token counts, and all raw
    generations are written into the output for provenance (API models drift;
    reviewers increasingly check this).

Output schema (JSON array; one object per task) -- matches the evaluator:
    [
      {
        "id":         <task_id>,          # evaluator reads row['id']
        "task_id":    <task_id>,          # kept for convenience
        "predictions":[<raw gen>, ...],   # evaluator reads row['predictions']
        "tests":      <dart test harness>,# evaluator reads row['tests']
        "function":   <name>,             # provenance only
        "prompt_mode":"asm"|"asm_cfg"|"graph_only",
        "meta": {...}                     # provider, model, params, tokens, timings
      },
      ...
    ]

Then score with the UNMODIFIED local harness, e.g.:
    python graph_pass_at_k_antigravity.py \
        --predictions preds_frontier_asm.json --k_values 1,5,10

Usage
-----
  OpenRouter (open-weights / proprietary via router):
    export OPENROUTER_API_KEY=sk-or-...
    python llm_baseline.py --provider openrouter \
        --input grpo_data_cfg.jsonl --output preds_deepseek_asm.json \
        --model deepseek/deepseek-chat --mode asm \
        --samples 10 --temperature 0.7 --workers 8

  Azure OpenAI (GPT-5-class deployment; temperature not configurable):
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1
    python llm_baseline.py --provider azure \
        --input grpo_data_cfg.jsonl --output preds_gpt55_asm.json \
        --model gpt-5.5 --mode asm --samples 10 --max_tokens 8192 --workers 4
    # classic deployment endpoint instead of v1:
    #   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/deployments/gpt-5.5
    #   --azure_api_version 2024-12-01-preview

Notes
-----
  * No third-party deps required (uses urllib). `tqdm` is used if present.
  * --limit N runs only the first N tasks (cheap smoke test before the full run).
  * --resume reuses any tasks already present in --output (crash-safe / cost-safe).
  * Azure reasoning deployments spend part of --max_tokens on reasoning tokens; if
    you see empty completions, raise --max_tokens (the warning reports finish_reason
    and reasoning_tokens to confirm).
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


def _azure_url(api_version: str | None) -> str:
    """Build the Azure OpenAI chat/completions URL from AZURE_OPENAI_ENDPOINT.

    Supports both Azure surfaces:
      * v1 API (recommended): set AZURE_OPENAI_ENDPOINT to
        https://<resource>.openai.azure.com/openai/v1  and leave api_version unset.
      * Classic deployment API: set AZURE_OPENAI_ENDPOINT to
        https://<resource>.openai.azure.com/openai/deployments/<deployment>
        and pass an api_version (e.g. 2024-12-01-preview), which is appended as a
        query parameter.
    In both cases the request path is <endpoint>/chat/completions. The --model
    value is the deployment name (sent in the body; ignored by the classic surface,
    which takes the deployment from the URL).
    """
    base = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    url = f"{base}/chat/completions"
    if api_version:
        url += f"?api-version={api_version}"
    return url


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


def _truncate_asm(asm: str, max_asm_tokens: int | None, chars_per_token: float) -> tuple[str, bool]:
    """Truncate assembly to ~max_asm_tokens (char-estimated). Returns (text, was_truncated).

    Keeps the head of the listing (function prologue + entry blocks, the most
    informative region) and marks the cut so the model knows context is partial.
    Used for the same-budget fairness ablation: run a frontier model under the
    same ~2048-token prompt budget the specialized decoder sees, to separate
    frontier reasoning from the context-length advantage of full assembly.
    """
    asm = asm.strip()
    if not max_asm_tokens or max_asm_tokens <= 0:
        return asm, False
    budget_chars = int(max_asm_tokens * chars_per_token)
    if len(asm) <= budget_chars:
        return asm, False
    head = asm[:budget_chars].rstrip()
    return head + "\n; ... [assembly truncated to fit the prompt budget] ...", True


def build_user_prompt(
    row: dict,
    mode: str,
    *,
    max_asm_tokens: int | None = None,
    chars_per_token: float = 3.6,
) -> str:
    """Assemble the user prompt for one task.

    Modes:
      asm        : signature + full x86-64 assembly (default practitioner baseline)
      asm_cfg    : asm + serialized CFG/DFG (prompt-level structure, CodeInverter-style)
      graph_only : signature + serialized CFG/DFG, NO raw assembly. Ablation asking
                   whether the graph summary alone is sufficient / how lossy it is
                   versus full assembly.
    max_asm_tokens truncates the assembly (see _truncate_asm); inert in graph_only.
    """
    sig = row.get("dart_function_signature", "").strip()
    asm = row.get("assembly", "")
    cfg = row.get("cfg", []) or []
    edges = row.get("edges", []) or []
    parts = []
    if sig:
        parts.append(f"Target Dart function signature:\n{sig}\n")

    if mode == "graph_only":
        if cfg:
            parts.append(
                "Control-/data-flow structure extracted from the AOT assembly "
                "(no raw assembly is provided):\n" + _format_cfg(cfg, edges)
            )
        else:  # fall back to asm if a row somehow lacks a graph
            parts.append("x86-64 AOT disassembly:\n```asm\n" + asm.strip() + "\n```")
    else:
        asm_text, truncated = _truncate_asm(asm, max_asm_tokens, chars_per_token)
        header = "x86-64 AOT disassembly"
        if truncated:
            header += " (truncated to the prompt budget)"
        parts.append(header + ":\n```asm\n" + asm_text + "\n```")
        if mode == "asm_cfg" and cfg:
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
# --------------------------------------------------------------------------- #
# Chat-completions call with retries (OpenRouter or Azure OpenAI)
# --------------------------------------------------------------------------- #

class RateLimit(Exception):
    pass


def call_model(
    provider: str,
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
    azure_api_version: str | None = None,
    azure_send_sampling: bool = False,
    max_retries: int = 5,
) -> tuple[str, dict]:
    """Return (text, usage_dict). Retries on 429/5xx with backoff.

    provider = "openrouter" | "azure".

    Provider differences handled here:
      * URL      : OpenRouter fixed endpoint vs Azure <endpoint>/chat/completions.
      * Auth     : OpenRouter 'Authorization: Bearer' (+ attribution headers) vs
                   Azure 'api-key' header.
      * Body     : OpenRouter sends temperature/top_p/max_tokens. Azure GPT-5-class
                   and reasoning deployments reject temperature/top_p and use
                   max_completion_tokens instead of max_tokens, so those are omitted
                   by default (override with azure_send_sampling for deployments that
                   do accept them, e.g. some GPT-4o deployments).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if provider == "azure":
        body = {
            "model": model,
            "messages": messages,
            # Reasoning / GPT-5-class deployments require max_completion_tokens and
            # spend part of it on internal reasoning tokens before emitting content.
            "max_completion_tokens": max_tokens,
        }
        if azure_send_sampling:
            body["temperature"] = temperature
            body["top_p"] = top_p
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }
        url = _azure_url(azure_api_version)
    else:  # openrouter
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (optional but recommended):
            "HTTP-Referer": referer,
            "X-Title": title,
        }
        url = OPENROUTER_URL

    data = json.dumps(body).encode("utf-8")

    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Standard OpenAI-compatible shape (OpenRouter and Azure both use it)
            choice = (payload.get("choices") or [{}])[0]
            text = (choice.get("message", {}) or {}).get("content", "") or ""
            usage = payload.get("usage", {}) or {}
            usage["_model_reported"] = payload.get("model", model)
            usage["_finish_reason"] = choice.get("finish_reason")
            # Reasoning-token accounting for GPT-5-class/o-series (when reported)
            details = usage.get("completion_tokens_details") or {}
            usage["_reasoning_tokens"] = details.get("reasoning_tokens", 0)
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
    provider: str,
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
    max_asm_tokens: int | None = None,
    chars_per_token: float = 3.6,
    azure_api_version: str | None = None,
    azure_send_sampling: bool = False,
) -> dict:
    tid = str(row.get("task_id", row.get("id")))
    user_prompt = build_user_prompt(
        row, mode, max_asm_tokens=max_asm_tokens, chars_per_token=chars_per_token
    )

    predictions: list[str] = []
    prompt_tok = comp_tok = reasoning_tok = 0
    errors = 0
    empty = 0
    t0 = time.time()

    # Greedy single sample only when we actually control temperature: OpenRouter at
    # temperature 0. Azure deployments that reject temperature run at their fixed
    # default, so we always draw `samples` and rely on that default stochasticity
    # for pass@k (never collapse to one deterministic draw).
    controls_temp = not (provider == "azure" and not azure_send_sampling)
    greedy = controls_temp and temperature == 0
    n_draws = 1 if greedy else samples

    for _ in range(n_draws):
        try:
            text, usage = call_model(
                provider, api_key, model, SYSTEM_PROMPT, user_prompt,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, timeout=timeout,
                referer=referer, title=title,
                azure_api_version=azure_api_version,
                azure_send_sampling=azure_send_sampling,
            )
            predictions.append(text)
            prompt_tok += int(usage.get("prompt_tokens", 0) or 0)
            comp_tok += int(usage.get("completion_tokens", 0) or 0)
            reasoning_tok += int(usage.get("_reasoning_tokens", 0) or 0)
            if not text.strip():
                empty += 1
                fr = usage.get("_finish_reason")
                # Common on reasoning models when max_completion_tokens is too small:
                # the budget is consumed by reasoning tokens before any content.
                print(
                    f"[warn] task {tid}: empty content (finish_reason={fr}, "
                    f"reasoning_tokens={usage.get('_reasoning_tokens', 0)}); "
                    f"consider raising --max_tokens for reasoning deployments.",
                    file=sys.stderr,
                )
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
            "provider": provider,
            "model": model,
            "n_candidates": len(predictions),
            "n_errors": errors,
            "n_empty": empty,
            # For Azure without sampling params, the effective temperature is the
            # deployment default, not this value; recorded for provenance/honesty.
            "temperature": temperature if controls_temp else "deployment_default",
            "top_p": top_p if controls_temp else "deployment_default",
            "max_tokens": max_tokens,
            "max_asm_tokens": max_asm_tokens or 0,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "reasoning_tokens": reasoning_tok,
            "seconds": round(time.time() - t0, 2),
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate an external LLM baseline via OpenRouter or Azure "
                    "OpenAI, output-compatible with the local pass@k evaluator."
    )
    ap.add_argument("--input", required=True,
                    help="Path to the 154-task JSONL (grpo_data_cfg.jsonl).")
    ap.add_argument("--output", required=True,
                    help="Path to write the predictions JSON array.")
    ap.add_argument("--provider", choices=["openrouter", "azure"], default="openrouter",
                    help="openrouter: OPENROUTER_API_KEY, hosted router. "
                         "azure: AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, "
                         "api-key header, max_completion_tokens, temperature/top_p "
                         "omitted by default (GPT-5-class/reasoning deployments "
                         "reject them).")
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id (e.g. deepseek/deepseek-chat, "
                         "qwen/qwen3-8b, anthropic/claude-..., openai/gpt-...), OR "
                         "for --provider azure the deployment name (e.g. gpt-5.5).")
    ap.add_argument("--mode", choices=["asm", "asm_cfg", "graph_only"], default="asm",
                    help="asm = assembly only; asm_cfg = assembly + serialized "
                         "CFG/DFG (prompt-level graph baseline); graph_only = "
                         "signature + serialized CFG/DFG with NO raw assembly "
                         "(graph-summary ablation).")
    ap.add_argument("--max_asm_tokens", type=int, default=0,
                    help="Truncate assembly to ~N tokens (0 = full, the default). "
                         "Set to 2048 for the same-budget frontier ablation that "
                         "matches the specialized decoder's prompt budget. Inert "
                         "in graph_only mode.")
    ap.add_argument("--chars_per_token", type=float, default=3.6,
                    help="Chars-per-token estimate used to convert --max_asm_tokens "
                         "into a character budget (assembly text is ~3.5-3.7).")
    ap.add_argument("--samples", type=int, default=10,
                    help="Stochastic samples per task for pass@k (default 10).")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="OpenRouter: 0 = greedy single sample, else nucleus "
                         "sampling. Azure (default): ignored -- the deployment runs "
                         "at its fixed default temperature and `samples` draws are "
                         "always taken.")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="OpenRouter nucleus p. Ignored on Azure unless "
                         "--azure_send_sampling.")
    ap.add_argument("--max_tokens", type=int, default=1024,
                    help="Max completion tokens per candidate (sent as "
                         "max_completion_tokens on Azure). For Azure reasoning / "
                         "GPT-5-class deployments raise this (e.g. 4096-8192): "
                         "reasoning tokens are charged against this budget and can "
                         "otherwise leave no room for the answer.")
    ap.add_argument("--azure_api_version", default=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
                    help="Azure api-version query param (env AZURE_OPENAI_API_VERSION). "
                         "Leave empty for the v1 endpoint "
                         "(.../openai/v1); set e.g. 2024-12-01-preview for a classic "
                         "deployment endpoint (.../openai/deployments/<name>).")
    ap.add_argument("--azure_send_sampling", action="store_true",
                    help="Force-send temperature/top_p on Azure (only for "
                         "deployments that accept them, e.g. some GPT-4o). Off by "
                         "default because GPT-5-class/reasoning deployments reject "
                         "them.")
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

    azure_api_version = (args.azure_api_version or "").strip() or None
    if args.provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ERROR: set AZURE_OPENAI_API_KEY in the environment.")
        if not os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip():
            raise SystemExit(
                "ERROR: set AZURE_OPENAI_ENDPOINT (e.g. "
                "https://<resource>.openai.azure.com/openai/v1 for the v1 API, or "
                "https://<resource>.openai.azure.com/openai/deployments/<name> plus "
                "--azure_api_version for a classic deployment)."
            )
        if not args.azure_send_sampling and (args.temperature != 0.7 or args.top_p != 0.95):
            print(
                "[note] Azure without --azure_send_sampling ignores temperature/top_p; "
                "the deployment runs at its fixed default temperature. Samples are "
                "still drawn for pass@k. This is recorded in each record's meta.",
                file=sys.stderr,
            )
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ERROR: set OPENROUTER_API_KEY in the environment.")

    tasks = load_tasks(args.input)
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    existing = load_existing(args.output) if args.resume else {}
    todo = [r for r in tasks
            if str(r.get("task_id", r.get("id"))) not in existing]

    az_note = ""
    if args.provider == "azure" and not args.azure_send_sampling:
        samples_str = f"{args.samples} (deployment-default temp)"
        az_note = f" | api_version={azure_api_version or 'v1'}"
    else:
        samples_str = "1 (greedy)" if args.temperature == 0 else str(args.samples)
    print(
        f"provider={args.provider} model={args.model} mode={args.mode} "
        f"samples={samples_str}{az_note} "
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
            provider=args.provider,
            api_key=api_key, model=args.model, mode=args.mode,
            samples=args.samples, temperature=args.temperature,
            top_p=args.top_p, max_tokens=args.max_tokens,
            timeout=args.timeout, referer=args.referer, title=args.title,
            max_asm_tokens=args.max_asm_tokens, chars_per_token=args.chars_per_token,
            azure_api_version=azure_api_version,
            azure_send_sampling=args.azure_send_sampling,
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
