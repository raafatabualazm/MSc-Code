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

Three prompt modes answer distinct reviewer questions with one script:
  --mode asm        assembly only                -> "does a frontier LLM beat our 8B?"
  --mode asm_cfg    assembly + serialized CFG    -> textual CFG vs learned graph-prefix
  --mode graph_only CFG with block instructions  -> graph-structured textual conditioning

  IMPORTANT – old script vs this (V2) script:
  The original (V1) baseline always sent the *raw* GDB dump (including
  "info functions" / symbol-search noise). This V2 script *by default cleans*
  the assembly via _extract_assembler_dump() before anything else, because the
  CFG modes (asm_cfg / graph_only) need a clean instruction stream to align
  blocks and drop contaminated headers. Consequently:

    - Default --mode asm  = cleaned assembly (the proper control for the
      graph modes; the instructions the model sees are *exactly* those used
      to build the CFG).
    - --mode asm --raw_assembly = exact reproduction of the old V1 baseline
      (raw dump, no cleaning). Use this when you need an apples-to-apples
      number against previously published V1 runs.

  CRITICAL FOR CAUSALITY (V1 vs V2 / serializer ablations):
  Because graph modes always clean the assembly in order to construct a
  consistent CFG, a pure "old-asm vs new-asm_cfg" comparison confounds the
  cleaning step with the addition of the graph. Always run the *cleaned*
  control:

      python llm_baselinev2.py ... --mode asm        # cleaned-asm control
      python llm_baselinev2.py ... --mode asm_cfg    # cleaned-asm + CFG-v2

  The two arms share the identical cleaner, so any gain is caused by the
  CFG text, not by header removal. If you also want the raw-vs-cleaned
  delta itself, add the --raw_assembly flag on a separate --mode asm run.

Fairness ablations (address the "not architecture-controlled" reviewer objection):
  1. Same-budget frontier:  --max_asm_tokens 2048
     Runs a frontier model under the same ~2048-token prompt budget the specialized
     decoder sees, separating frontier reasoning from the full-assembly context
     advantage. (~46/154 tasks exceed this budget and get truncated.)
  2. Local base zero-shot:  --model qwen/qwen3-8b   (no code change; just pick the model)
     Separates local-base weakness from the training/graph effects.
  3. Raw vs cleaned:        --raw_assembly
     Quantifies how much of any gain comes from simply removing GDB junk.

Design choices that make this a *fair* baseline, not a strawman:
  * Full assembly is sent by default (long context is the frontier model's natural
    advantage); truncation is opt-in via --max_asm_tokens for the ablation above.
  * Cleaned-assembly-only control (--mode asm, default) is first-class so that any
    CFG / serializer change can be evaluated against a pure cleaned-asm arm that
    uses the *same* cleaner the graph is built from; without it the comparison
    is not causal.
  * --raw_assembly reproduces the old V1 baseline exactly when needed.
  * The system/user prompt is frozen here; tune it on a handful of tasks, then keep
    it fixed for the reported run. A lazy prompt invites a "you sandbagged the
    baseline" objection.
  * Provider, model, params, exact prompt hash, run fingerprint, cleanup/
    truncation metadata, token/reasoning-token counts, and all raw generations are
    written into the output for provenance (API models drift).

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
  * --resume reuses only tasks with a matching run fingerprint; stale rows
    from older prompt/serializer configurations are ignored.
  * Azure reasoning deployments spend part of --max_tokens on reasoning tokens; if
    you see empty completions, raise --max_tokens (the warning reports finish_reason
    and reasoning_tokens to confirm).
  * Always include a cleaned-assembly-only (--mode asm) run as the control when
    claiming that a CFG / graph change (e.g. V2 serializer) is responsible for
    a gain; otherwise the comparison is not causal.
  * --raw_assembly makes --mode asm (and the assembly half of --mode asm_cfg)
    emit the original raw GDB dump exactly as the old V1 script did. The CFG
    itself is still cleaned so that graph modes remain consistent.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
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
# Prompt construction / CFG cleanup
# --------------------------------------------------------------------------- #

CFG_SERIALIZER_VERSION = "cfg-text-v2"
ASSEMBLY_CLEANER_VERSION = "asm-dump-v1"  # isolates GDB header / symbol-search noise

SYSTEM_PROMPT = (
    "You are an expert reverse engineer specializing in Dart Ahead-of-Time (AOT) "
    "compiled binaries. You are given x86-64 Dart AOT disassembly and/or a "
    "control-flow graph reconstructed from that disassembly. Recover the original "
    "high-level Dart source for the target function.\n\n"
    "Rules:\n"
    "1. Output ONLY the Dart function, inside a single ```dart code block.\n"
    "2. Match the given function signature exactly (name, parameter types, return "
    "type).\n"
    "3. Do NOT include a main() function, tests, imports, or commentary.\n"
    "4. Produce idiomatic, compilable Dart that is functionally equivalent to the "
    "input -- the same inputs must produce the same outputs.\n"
    "5. Dart AOT code uses runtime-specific object layouts, boxed numbers, bounds "
    "checks, and allocation stubs; recover the program logic rather than those "
    "runtime artifacts."
)

_DUMP_LINE_RE = re.compile(
    r"^\s*(0x[0-9a-fA-F]+)\s+<[^>]+>:\s*(.*?)\s*$"
)


def _normalise_instruction(text: str) -> str:
    """Canonicalise spacing so CFG instructions can be matched to the GDB dump."""
    return " ".join(str(text).strip().split())


def _extract_assembler_dump(assembly: str) -> tuple[str, list[tuple[str, str]], dict]:
    """Remove GDB `info functions` output and retain only the assembler dump.

    The uploaded dataset's `assembly` field begins with GDB symbol-search output.
    For short function names (for example `f` or `tri`) that output can be very
    large and can leak into CFG construction. This function isolates lines between
    `Dump of assembler code for function ...` and `End of assembler dump.`.

    Returns:
      clean_text: address-bearing assembly lines suitable for the model prompt.
      entries:    [(address, normalised instruction), ...] in layout order.
      meta:       cleanup/provenance counters.
    """
    raw_lines = str(assembly or "").splitlines()
    in_dump = False
    found_marker = False
    dump_lines: list[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("Dump of assembler code for function "):
            in_dump = True
            found_marker = True
            continue
        if in_dump and stripped == "End of assembler dump.":
            break
        if in_dump:
            dump_lines.append(line.rstrip())

    # A conservative fallback keeps compatibility with already-clean datasets.
    if not found_marker:
        dump_lines = [line.rstrip() for line in raw_lines]

    entries: list[tuple[str, str]] = []
    clean_lines: list[str] = []
    for line in dump_lines:
        match = _DUMP_LINE_RE.match(line)
        if not match:
            continue
        address = match.group(1).lower()
        instruction = _normalise_instruction(match.group(2))
        if not instruction:
            continue
        entries.append((address, instruction))
        # Keep the original GDB line in the assembly prompt: addresses bind the
        # textual CFG blocks to the raw listing.
        clean_lines.append(line.strip())

    if not entries:
        raise ValueError("No address-bearing instructions found in assembler dump")

    return "\n".join(clean_lines), entries, {
        "assembly_cleaner_version": ASSEMBLY_CLEANER_VERSION,
        "assembly_raw_lines": len(raw_lines),
        "assembly_dump_lines": len(clean_lines),
        "assembly_header_removed": found_marker,
    }


def _find_subsequence(
    haystack: list[str], needle: list[str], start: int
) -> int | None:
    """Find a contiguous instruction sequence, preferring layout order."""
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for index in range(max(0, start), last):
        if haystack[index:index + len(needle)] == needle:
            return index
    # Defensive fallback for unusual block ordering.
    for index in range(0, min(max(0, start), last)):
        if haystack[index:index + len(needle)] == needle:
            return index
    return None


def _clean_cfg(
    cfg: list[dict], assembly_entries: list[tuple[str, str]]
) -> tuple[list[dict], dict]:
    """Remove GDB-header contamination and rebuild a self-consistent CFG.

    A line is retained as a block instruction only when it occurs in the actual
    address-bearing assembler dump. Blocks containing only symbol-search output are
    discarded. Remaining blocks are remapped to dense IDs, start addresses are
    recovered from the dump, and predecessor lists are recomputed from successors.
    """
    if not cfg:
        raise ValueError("CFG is missing")

    assembly_instructions = [instruction for _, instruction in assembly_entries]
    assembly_instruction_set = set(assembly_instructions)
    cursor = 0
    retained: list[tuple[dict, int, list[str]]] = []
    original_instruction_count = 0

    for block in cfg:
        raw_instructions = block.get("instructions", []) or []
        original_instruction_count += len(raw_instructions)
        matched_instructions = [
            normalised
            for instruction in raw_instructions
            if (normalised := _normalise_instruction(instruction))
            in assembly_instruction_set
        ]
        if not matched_instructions:
            continue

        start_index = _find_subsequence(
            assembly_instructions, matched_instructions, cursor
        )
        if start_index is None:
            raise ValueError(
                "CFG block instructions could not be aligned to the assembler "
                f"dump (block id={block.get('id')})"
            )
        cursor = max(cursor, start_index + len(matched_instructions))
        retained.append((block, start_index, matched_instructions))

    if not retained:
        raise ValueError("CFG contains no instructions from the assembler dump")

    old_to_new = {
        block.get("id"): new_id
        for new_id, (block, _, _) in enumerate(retained)
    }

    cleaned: list[dict] = []
    for new_id, (block, start_index, instructions) in enumerate(retained):
        successors: list[int] = []
        edge_types: list[str] = []
        raw_successors = block.get("successors", []) or []
        raw_edge_types = block.get("edge_types", []) or []

        for edge_index, old_target in enumerate(raw_successors):
            if old_target not in old_to_new:
                continue
            successors.append(old_to_new[old_target])
            edge_types.append(
                str(raw_edge_types[edge_index])
                if edge_index < len(raw_edge_types)
                else "unknown"
            )

        cleaned.append({
            "id": new_id,
            "original_id": block.get("id"),
            "label": f"block_{new_id}",
            "start_address": assembly_entries[start_index][0],
            "instructions": instructions,
            "predecessors": [],
            "successors": successors,
            "edge_types": edge_types,
            "instruction_count": len(instructions),
            "block_type": block.get("block_type", "unknown"),
        })

    # Recompute predecessors after dropping/remapping contaminated blocks.
    predecessors: dict[int, list[int]] = {block["id"]: [] for block in cleaned}
    for block in cleaned:
        for target in block["successors"]:
            predecessors[target].append(block["id"])
    for block in cleaned:
        block["predecessors"] = predecessors[block["id"]]

    cleaned_instruction_count = sum(
        block["instruction_count"] for block in cleaned
    )
    return cleaned, {
        "cfg_blocks_original": len(cfg),
        "cfg_blocks_clean": len(cleaned),
        "cfg_blocks_dropped": len(cfg) - len(cleaned),
        "cfg_instructions_original": original_instruction_count,
        "cfg_instructions_clean": cleaned_instruction_count,
        "cfg_instructions_dropped": (
            original_instruction_count - cleaned_instruction_count
        ),
        "cfg_cleaned": (
            len(cfg) != len(cleaned)
            or original_instruction_count != cleaned_instruction_count
        ),
    }


def _select_cfg_blocks(cfg: list[dict], max_blocks: int) -> list[dict]:
    """Select a connected entry-reachable subset, then emit it in layout order."""
    if max_blocks <= 0 or len(cfg) <= max_blocks:
        return list(cfg)

    by_id = {block["id"]: block for block in cfg}
    queue = [cfg[0]["id"]]
    queued = {cfg[0]["id"]}
    selected: set[int] = set()

    while queue and len(selected) < max_blocks:
        block_id = queue.pop(0)
        selected.add(block_id)
        for target in by_id[block_id].get("successors", []) or []:
            if target in by_id and target not in selected and target not in queued:
                queue.append(target)
                queued.add(target)

    # Should only matter for malformed/disconnected graphs; fill deterministically.
    if len(selected) < max_blocks:
        for block in cfg:
            selected.add(block["id"])
            if len(selected) >= max_blocks:
                break

    return [block for block in cfg if block["id"] in selected]


def _clip_block_instructions(
    instructions: list[str], max_instructions: int
) -> tuple[list[str | None], bool]:
    """Keep a block's head and tail when a graph-only instruction cap is used."""
    if max_instructions <= 0 or len(instructions) <= max_instructions:
        return list(instructions), False
    head_count = (max_instructions + 1) // 2
    tail_count = max_instructions // 2
    clipped: list[str | None] = list(instructions[:head_count])
    clipped.append(None)  # rendered as an omission marker
    if tail_count:
        clipped.extend(instructions[-tail_count:])
    return clipped, True


def _format_cfg(
    cfg: list[dict],
    *,
    include_all_instructions: bool,
    max_blocks: int = 0,
    tail_instructions: int = 2,
    max_instructions_per_block: int = 0,
) -> tuple[str, dict]:
    """Serialize a cleaned CFG for `asm_cfg` or `graph_only`.

    `asm_cfg` uses addresses, typed edges, and final instructions to bind topology
    to the accompanying raw assembly without duplicating every instruction.
    `graph_only` includes block instructions because topology alone cannot encode
    program semantics.
    """
    shown = _select_cfg_blocks(cfg, max_blocks)
    shown_ids = {block["id"] for block in shown}
    total_instructions = sum(block["instruction_count"] for block in cfg)
    serialized_instruction_count = 0
    truncated_instruction_blocks = 0

    lines = [
        f"Control-flow graph: {len(cfg)} basic blocks, "
        f"{total_instructions} instructions.",
        f"Entry block: B{cfg[0]['id']} @ {cfg[0]['start_address']}",
    ]

    if len(shown) < len(cfg):
        lines.append(
            f"Connected subset shown: {len(shown)}/{len(cfg)} blocks. "
            "Edges to omitted blocks are marked explicitly."
        )

    for block in shown:
        block_id = block["id"]
        predecessors = block.get("predecessors", []) or []
        successors = block.get("successors", []) or []
        edge_types = block.get("edge_types", []) or []

        def block_ref(target: int) -> str:
            suffix = "[omitted]" if target not in shown_ids else ""
            return f"B{target}{suffix}"

        predecessor_text = (
            ", ".join(block_ref(source) for source in predecessors) or "-"
        )
        successor_items = []
        for index, target in enumerate(successors):
            edge_type = (
                edge_types[index] if index < len(edge_types) else "unknown"
            )
            successor_items.append(f"{block_ref(target)}[{edge_type}]")
        successor_text = ", ".join(successor_items) or "-"

        lines.extend([
            "",
            f"B{block_id} @ {block.get('start_address', '?')} "
            f"type={block.get('block_type', 'unknown')} "
            f"instructions={block.get('instruction_count', 0)}",
            f"  predecessors: {predecessor_text}",
            f"  successors: {successor_text}",
        ])

        instructions = block.get("instructions", []) or []
        if include_all_instructions:
            clipped, was_clipped = _clip_block_instructions(
                instructions, max_instructions_per_block
            )
            if was_clipped:
                truncated_instruction_blocks += 1
            lines.append("  instructions:")
            rendered_index = 0
            for instruction in clipped:
                if instruction is None:
                    omitted = len(instructions) - max_instructions_per_block
                    lines.append(f"    ... {omitted} instructions omitted ...")
                    continue
                lines.append(f"    {rendered_index:02d}: {instruction}")
                rendered_index += 1
                serialized_instruction_count += 1
        else:
            tail_count = max(0, tail_instructions)
            tail = instructions[-tail_count:] if tail_count else []
            serialized_instruction_count += len(tail)
            if tail:
                lines.append("  final instructions:")
                for instruction in tail:
                    lines.append(f"    {instruction}")
            else:
                lines.append("  final instructions: -")

    return "\n".join(lines), {
        "cfg_serializer_version": CFG_SERIALIZER_VERSION,
        "cfg_blocks_total": len(cfg),
        "cfg_blocks_serialized": len(shown),
        "cfg_blocks_truncated": len(shown) < len(cfg),
        "cfg_instructions_total": total_instructions,
        "cfg_instructions_serialized": serialized_instruction_count,
        "cfg_instruction_truncated_blocks": truncated_instruction_blocks,
    }


def _truncate_asm(
    asm: str, max_asm_tokens: int | None, chars_per_token: float
) -> tuple[str, bool]:
    """Truncate cleaned assembly to an approximate token budget."""
    asm = asm.strip()
    if not max_asm_tokens or max_asm_tokens <= 0:
        return asm, False
    budget_chars = int(max_asm_tokens * chars_per_token)
    if len(asm) <= budget_chars:
        return asm, False
    head = asm[:budget_chars].rstrip()
    return (
        head + "\n; ... [assembly truncated to fit the prompt budget] ...",
        True,
    )


def build_user_prompt(
    row: dict,
    mode: str,
    *,
    max_asm_tokens: int | None = None,
    chars_per_token: float = 3.6,
    max_cfg_blocks: int = 0,
    cfg_tail_instructions: int = 2,
    graph_only_max_instructions_per_block: int = 0,
    raw_assembly: bool = False,
) -> tuple[str, dict]:
    """Assemble one prompt and return it with cleanup/serialization metadata.

    Modes:
      asm        : signature + assembly (cleaned by default, raw if raw_assembly).
      asm_cfg    : assembly + compact textual CFG tied to addresses/terminators.
      graph_only : semantic CFG with per-block instructions and no raw assembly.

    The cleaner is *always* run so that CFG modes receive a consistent instruction
    stream. When raw_assembly=True the *emitted* assembly text is the original
    GDB dump (exactly as the old V1 script did); the CFG itself stays cleaned.
    """
    signature = row.get("dart_function_signature", "").strip()
    raw_asm = str(row.get("assembly", "") or "")
    clean_asm, assembly_entries, assembly_meta = _extract_assembler_dump(raw_asm)
    # What the model actually sees for the assembly listing
    emit_asm = raw_asm if raw_assembly else clean_asm
    prompt_meta: dict = dict(assembly_meta)
    prompt_meta["assembly_cleaned"] = not raw_assembly
    prompt_meta["raw_assembly_flag"] = raw_assembly
    parts: list[str] = []

    if signature:
        parts.append(f"Target Dart function signature:\n{signature}\n")

    if mode == "asm":
        asm_text, asm_truncated = _truncate_asm(
            emit_asm, max_asm_tokens, chars_per_token
        )
        heading = "x86-64 Dart AOT disassembly"
        if raw_assembly:
            heading += " (raw GDB dump)"
        if asm_truncated:
            heading += " (truncated to the prompt budget)"
        parts.append(f"{heading}:\n```asm\n{asm_text}\n```")
        prompt_meta.update({
            "assembly_truncated": asm_truncated,
            "cfg_serializer_version": None,
        })

    elif mode in {"asm_cfg", "graph_only"}:
        cfg, cleanup_meta = _clean_cfg(row.get("cfg", []) or [], assembly_entries)
        prompt_meta.update(cleanup_meta)

        if mode == "asm_cfg":
            asm_text, asm_truncated = _truncate_asm(
                emit_asm, max_asm_tokens, chars_per_token
            )
            heading = "x86-64 Dart AOT disassembly"
            if raw_assembly:
                heading += " (raw GDB dump)"
            if asm_truncated:
                heading += " (truncated to the prompt budget)"
            parts.append(f"{heading}:\n```asm\n{asm_text}\n```")
            graph_text, graph_meta = _format_cfg(
                cfg,
                include_all_instructions=False,
                max_blocks=max_cfg_blocks,
                tail_instructions=cfg_tail_instructions,
            )
            parts.append(
                "Extracted control-flow graph. Block addresses and final "
                "instructions bind this topology to the disassembly:\n"
                + graph_text
            )
            prompt_meta["assembly_truncated"] = asm_truncated
        else:
            graph_text, graph_meta = _format_cfg(
                cfg,
                include_all_instructions=True,
                max_blocks=max_cfg_blocks,
                max_instructions_per_block=(
                    graph_only_max_instructions_per_block
                ),
            )
            parts.append(
                "Control-flow graph reconstructed from the Dart AOT function. "
                "No separate raw assembly listing is provided; each block includes "
                "its normalized assembly instructions:\n"
                + graph_text
            )
            prompt_meta["assembly_truncated"] = False

        prompt_meta.update(graph_meta)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    parts.append(
        "\nReconstruct the original Dart function. Output only a single "
        "```dart code block."
    )
    prompt = "\n\n".join(parts)
    prompt_meta["prompt_chars"] = len(prompt)
    return prompt, prompt_meta


def prompt_hash(system: str, user: str) -> str:
    """Hash the exact system+user prompt sent for one task."""
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(user.encode("utf-8"))
    return digest.hexdigest()[:20]


def configuration_hash(config: dict) -> str:
    """Stable fingerprint used to prevent unsafe resume across prompt changes."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(SYSTEM_PROMPT.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()[:20]


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
    run_fingerprint: str,
    max_asm_tokens: int | None = None,
    chars_per_token: float = 3.6,
    max_cfg_blocks: int = 0,
    cfg_tail_instructions: int = 2,
    graph_only_max_instructions_per_block: int = 0,
    raw_assembly: bool = False,
    azure_api_version: str | None = None,
    azure_send_sampling: bool = False,
) -> dict:
    task_id = str(row.get("task_id", row.get("id")))
    user_prompt, prompt_meta = build_user_prompt(
        row,
        mode,
        max_asm_tokens=max_asm_tokens,
        chars_per_token=chars_per_token,
        max_cfg_blocks=max_cfg_blocks,
        cfg_tail_instructions=cfg_tail_instructions,
        graph_only_max_instructions_per_block=(
            graph_only_max_instructions_per_block
        ),
        raw_assembly=raw_assembly,
    )
    exact_prompt_hash = prompt_hash(SYSTEM_PROMPT, user_prompt)

    predictions: list[str] = []
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    errors = 0
    empty = 0
    started = time.time()

    # Greedy single sample only when the API actually accepts temperature=0.
    controls_temperature = not (
        provider == "azure" and not azure_send_sampling
    )
    greedy = controls_temperature and temperature == 0
    draws = 1 if greedy else samples

    for _ in range(draws):
        try:
            text, usage = call_model(
                provider,
                api_key,
                model,
                SYSTEM_PROMPT,
                user_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                referer=referer,
                title=title,
                azure_api_version=azure_api_version,
                azure_send_sampling=azure_send_sampling,
            )
            predictions.append(text)
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            reasoning_tokens += int(usage.get("_reasoning_tokens", 0) or 0)
            if not text.strip():
                empty += 1
                finish_reason = usage.get("_finish_reason")
                print(
                    f"[warn] task {task_id}: empty content "
                    f"(finish_reason={finish_reason}, "
                    f"reasoning_tokens={usage.get('_reasoning_tokens', 0)}); "
                    "consider raising --max_tokens for reasoning deployments.",
                    file=sys.stderr,
                )
        except Exception as exc:  # Keep pass@k array length stable.
            predictions.append("")
            errors += 1
            print(f"[warn] task {task_id}: {exc}", file=sys.stderr)

    meta = {
        "provider": provider,
        "model": model,
        "run_fingerprint": run_fingerprint,
        "prompt_hash": exact_prompt_hash,
        "n_candidates": len(predictions),
        "n_errors": errors,
        "n_empty": empty,
        "temperature": (
            temperature if controls_temperature else "deployment_default"
        ),
        "top_p": top_p if controls_temperature else "deployment_default",
        "max_tokens": max_tokens,
        "max_asm_tokens": max_asm_tokens or 0,
        "max_cfg_blocks": max_cfg_blocks,
        "cfg_tail_instructions": cfg_tail_instructions,
        "graph_only_max_instructions_per_block": (
            graph_only_max_instructions_per_block
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "seconds": round(time.time() - started, 2),
    }
    meta.update(prompt_meta)

    return {
        "id": task_id,
        "task_id": task_id,
        "predictions": predictions,
        "tests": row.get("tests", ""),
        "function": row.get("function", ""),
        "dart_function_signature": row.get("dart_function_signature", ""),
        "prompt_mode": mode,
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an external Dart AOT decompilation baseline via "
            "OpenRouter or Azure OpenAI."
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to grpo_data_cfg.jsonl.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the predictions JSON array.",
    )
    parser.add_argument(
        "--provider", choices=["openrouter", "azure"], default="openrouter",
        help=(
            "openrouter uses OPENROUTER_API_KEY; azure uses "
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
        ),
    )
    parser.add_argument(
        "--model", required=True,
        help="OpenRouter model ID or Azure deployment name.",
    )
    parser.add_argument(
        "--mode", choices=["asm", "asm_cfg", "graph_only"], default="asm",
        help=(
            "asm: assembly only (cleaned by default); asm_cfg: assembly plus "
            "compact serialized CFG; graph_only: CFG with per-block instructions "
            "and no separate disassembly listing."
        ),
    )
    parser.add_argument(
        "--raw_assembly", action="store_true",
        help=(
            "Emit the original raw GDB dump instead of the cleaned assembler "
            "dump. Makes --mode asm (and the assembly half of --mode asm_cfg) "
            "identical to the old V1 baseline. The CFG is still cleaned so that "
            "graph modes remain consistent with the cleaned instruction stream. "
            "Inert for pure graph_only (which never shows a raw listing)."
        ),
    )
    parser.add_argument(
        "--max_asm_tokens", type=int, default=0,
        help=(
            "Approximate assembly prompt cap (0 = full). Set 2048 for the "
            "same-budget frontier ablation. Inert in graph_only mode."
        ),
    )
    parser.add_argument(
        "--chars_per_token", type=float, default=3.6,
        help="Character/token estimate for --max_asm_tokens.",
    )
    parser.add_argument(
        "--max_cfg_blocks", type=int, default=0,
        help=(
            "Maximum connected CFG blocks to serialize (0 = all). Truncation is "
            "explicitly marked and recorded in metadata."
        ),
    )
    parser.add_argument(
        "--cfg_tail_instructions", type=int, default=2,
        help=(
            "In asm_cfg mode, include this many final instructions per block to "
            "bind CFG nodes to the assembly (default 2)."
        ),
    )
    parser.add_argument(
        "--graph_only_max_instructions_per_block", type=int, default=0,
        help=(
            "In graph_only mode, cap instructions serialized per block; head and "
            "tail are retained (0 = all, recommended for the semantic ablation)."
        ),
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Stochastic samples per task for pass@k.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help=(
            "OpenRouter sampling temperature. Azure ignores this unless "
            "--azure_send_sampling is enabled."
        ),
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus p; ignored on Azure unless --azure_send_sampling.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help=(
            "Maximum completion tokens. Azure reasoning deployments often need "
            "4096-8192 because reasoning tokens share this budget."
        ),
    )
    parser.add_argument(
        "--azure_api_version",
        default=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        help=(
            "Leave empty for .../openai/v1; set an API version for a classic "
            ".../openai/deployments/<name> endpoint."
        ),
    )
    parser.add_argument(
        "--azure_send_sampling", action="store_true",
        help="Send temperature/top_p to Azure deployments that accept them.",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel tasks; keep modest to respect rate limits.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Run only the first N tasks (0 = all).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Reuse only records whose run fingerprint matches the current "
            "provider/model/prompt configuration."
        ),
    )
    parser.add_argument(
        "--referer", default="https://example.org/dart-decompilation",
        help="OpenRouter HTTP-Referer attribution header.",
    )
    parser.add_argument(
        "--title", default="Dart AOT Decompilation Baseline",
        help="OpenRouter X-Title attribution header.",
    )
    args = parser.parse_args()

    if args.max_cfg_blocks < 0:
        parser.error("--max_cfg_blocks must be >= 0")
    if args.cfg_tail_instructions < 0:
        parser.error("--cfg_tail_instructions must be >= 0")
    if args.graph_only_max_instructions_per_block < 0:
        parser.error("--graph_only_max_instructions_per_block must be >= 0")
    if args.chars_per_token <= 0:
        parser.error("--chars_per_token must be > 0")

    azure_api_version = (args.azure_api_version or "").strip() or None
    if args.provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ERROR: set AZURE_OPENAI_API_KEY in the environment.")
        if not os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip():
            raise SystemExit(
                "ERROR: set AZURE_OPENAI_ENDPOINT, for example "
                "https://<resource>.openai.azure.com/openai/v1."
            )
        if (
            not args.azure_send_sampling
            and (args.temperature != 0.7 or args.top_p != 0.95)
        ):
            print(
                "[note] Azure without --azure_send_sampling ignores "
                "temperature/top_p; deployment defaults will be recorded.",
                file=sys.stderr,
            )
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ERROR: set OPENROUTER_API_KEY in the environment.")

    tasks = load_tasks(args.input)
    if args.limit > 0:
        tasks = tasks[:args.limit]

    fingerprint_config = {
        "cfg_serializer_version": CFG_SERIALIZER_VERSION,
        "assembly_cleaner_version": ASSEMBLY_CLEANER_VERSION,
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "raw_assembly": args.raw_assembly,
        "samples": args.samples,
        "temperature": (
            args.temperature
            if not (args.provider == "azure" and not args.azure_send_sampling)
            else "deployment_default"
        ),
        "top_p": (
            args.top_p
            if not (args.provider == "azure" and not args.azure_send_sampling)
            else "deployment_default"
        ),
        "max_tokens": args.max_tokens,
        "max_asm_tokens": args.max_asm_tokens,
        "chars_per_token": args.chars_per_token,
        "max_cfg_blocks": args.max_cfg_blocks,
        "cfg_tail_instructions": args.cfg_tail_instructions,
        "graph_only_max_instructions_per_block": (
            args.graph_only_max_instructions_per_block
        ),
        "azure_api_version": azure_api_version or "v1",
        "azure_send_sampling": args.azure_send_sampling,
    }
    run_fingerprint = configuration_hash(fingerprint_config)

    all_existing = load_existing(args.output) if args.resume else {}
    existing = {
        task_id: record
        for task_id, record in all_existing.items()
        if (record.get("meta", {}) or {}).get("run_fingerprint")
        == run_fingerprint
    }
    stale_count = len(all_existing) - len(existing)
    todo = [
        row for row in tasks
        if str(row.get("task_id", row.get("id"))) not in existing
    ]

    if args.provider == "azure" and not args.azure_send_sampling:
        samples_text = f"{args.samples} (deployment-default temp)"
        azure_note = f" | api_version={azure_api_version or 'v1'}"
    else:
        samples_text = "1 (greedy)" if args.temperature == 0 else str(args.samples)
        azure_note = ""

    clean_note = " raw" if args.raw_assembly else " cleaned"
    print(
        f"provider={args.provider} model={args.model} mode={args.mode}{clean_note} "
        f"samples={samples_text}{azure_note} | {len(tasks)} tasks, "
        f"{len(existing)} compatible cached, {stale_count} stale ignored, "
        f"{len(todo)} to run | run_fingerprint={run_fingerprint}",
        file=sys.stderr,
    )

    results: dict[str, dict] = dict(existing)
    write_lock = threading.Lock()

    def flush() -> None:
        """Atomically write current compatible results in input order."""
        ordered = [
            results[str(row.get("task_id", row.get("id")))]
            for row in tasks
            if str(row.get("task_id", row.get("id"))) in results
        ]
        temporary_path = args.output + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            json.dump(ordered, output_file, indent=2)
        os.replace(temporary_path, args.output)

    def run_one(row: dict) -> dict:
        record = generate_for_task(
            row,
            provider=args.provider,
            api_key=api_key,
            model=args.model,
            mode=args.mode,
            samples=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            referer=args.referer,
            title=args.title,
            run_fingerprint=run_fingerprint,
            max_asm_tokens=args.max_asm_tokens,
            chars_per_token=args.chars_per_token,
            max_cfg_blocks=args.max_cfg_blocks,
            cfg_tail_instructions=args.cfg_tail_instructions,
            graph_only_max_instructions_per_block=(
                args.graph_only_max_instructions_per_block
            ),
            raw_assembly=args.raw_assembly,
            azure_api_version=azure_api_version,
            azure_send_sampling=args.azure_send_sampling,
        )
        with write_lock:
            results[record["task_id"]] = record
            flush()
        return record

    if todo:
        if args.workers > 1:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_one, row) for row in todo]
                iterator = cf.as_completed(futures)
                if tqdm is not None:
                    iterator = tqdm(
                        iterator, total=len(futures), desc="generate", unit="task"
                    )
                for future in iterator:
                    # Surface prompt/data failures immediately rather than silently
                    # producing a partially incompatible experiment.
                    future.result()
        else:
            iterator = (
                tqdm(todo, desc="generate", unit="task") if tqdm else todo
            )
            for row in iterator:
                run_one(row)

    flush()

    total_prompt = sum(
        record["meta"]["prompt_tokens"] for record in results.values()
    )
    total_completion = sum(
        record["meta"]["completion_tokens"] for record in results.values()
    )
    total_errors = sum(
        record["meta"]["n_errors"] for record in results.values()
    )
    print(
        f"\nDone. {len(results)} tasks written to {args.output}\n"
        f"tokens: prompt={total_prompt:,} completion={total_completion:,} "
        f"| candidate errors={total_errors}\n"
        "Next: python graph_pass_at_k_antigravity.py "
        f"--predictions {args.output} --k_values 1,5,10",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
