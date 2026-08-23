# X86 re-validation: graph-architecture value + assembly-channel ablation (Qwen3-8B)

> **Operational step-by-step commands: see `STUDY_RUNBOOK.md`** (model-selectable via a
> single `MODEL` toggle: `qwen3-8b-base` or `qwen-9b-base`). This doc = design rationale.
> Step-0 split-rebuild of the x86 `_cfg` files is DONE.

> **UPDATED 2026-07-04 for the fixed pipeline. Two things changed:**
> 1. **This ablation is only valid on the updated code.** Before the 2026-07-04 fix,
>    `--qwen_prefix_tokens 0` did NOT disable the graph — it prepended the raw
>    variable-length block states as the prefix (and for the untrained R arm, a
>    *random* projection of them). All three "no-graph" arms (R, G1, G0) would have
>    been graph-conditioned and R would not have been a clean zero-shot baseline.
>    With the fix, prefix=0 is true text-only: zero-width prefix, graph compute
>    skipped (init log prints `Qwen graph prefix DISABLED`). Sync the updated
>    `scripts/`, `models/`, `configs/` to the pod before running.
> 2. **Graph arms now measure the post-fix architecture**: cross-block DFG edges
>    (`--dfg_mode edges`), pretraining-faithful encoder positions
>    (`--position_scheme roberta`), and SFT RoPE positions matching generation
>    (`cumsum`, the new default — no flag needed). These go in COMMON so dataset
>    prep is identical across arms; the no-graph arms simply ignore the graph.

**Why:** supervisor wants the *fixed* (CFG-corrected, de-leaked) old datasets re-run
on **Qwen3-8B** — the TOSEM comparison model — to measure **how good the
encoder+graph architecture actually is** now that the graph is live (it was dead in
every prior run). This plan folds in the **with/without-assembly** question (does the
decoder need raw assembly text, or can the encoder carry it?) as the same 2x2.

**One ablation answers both:**

| Arm | `--qwen_prefix_tokens` | `--prompt_assembly_mode` | Isolates |
|---|---|---|---|
| **R** reference (no FT) | 0 | full | plain Qwen3-8B-Base zero-shot ≈ TOSEM base |
| **G1** text-only | 0 | full | FT decoder on assembly text, no graph |
| **G2** graph+text | 16 | full | the encoder+graph architecture (CFG+DFG) |
| **G3** graph-only | 16 | none | encoder carries assembly, no text |
| **G0** null (control) | 0 | none | prior-only floor (signature+tests, no assembly) |
| *G2c (optional)* | *16* | *full* | *G2 with `--dfg_mode off`: splits DFG value from CFG value* |

- **Supervisor's question = G2 − G1** (does the live graph add value over text alone?).
- **Your with/without-assembly = G2 vs G3 vs G1** (graph+text vs graph-only vs text-only).
- **R** grounds it against TOSEM's published base Qwen3-8B (pass@1 6.36%, pass@5 14.89%).
- **G0** bounds the floor any real channel must beat.
- **G2 − G2c** (run only if G2 − G1 > 0): how much of the graph's value is the new
  cross-block dataflow edges vs the CFG alone.

The comparison is **internally rigorous**: all FT arms train on the *same* data
(`dart_all_cfg_clean`), so only the architecture (graph on/off, text on/off) varies.
TOSEM's numbers are looser external context (different training mix). Leakage
verified zero (train vs the 154-task eval: 0 by source and by name).

## Datasets (fixed / de-leaked)
- SFT train: `data/datasets/dart_all_cfg_clean.jsonl` (1195)
- pass@k eval: `data/testing/grpo_data_cfg.jsonl` (154 HumanEval-Dart)
- compile@k eval: `data/testing/compile-test2_cfg.jsonl` (126)

## Step 0 — prep (recommended, local/free)
The x86 `_cfg` files predate block-splitting, so the **graph-only arm G3** would be
unfairly handicapped by encoder truncation (~39% of compile rows have a >512-tok
block). Rebuild them split so the encoder is lossless (same rows, finer blocks,
semantics unchanged):
```bash
for f in data/datasets/dart_all_cfg_clean.jsonl data/testing/grpo_data_cfg.jsonl data/testing/compile-test2_cfg.jsonl; do
  tmp="${f}.split24.tmp"
  GRAPH_MAX_BLOCK_INSTRS=24 python scripts/data/build_cfg_jsonl.py --input "$f" --output "$tmp" --overwrite
  test -s "$tmp"
  mv "$tmp" "$f"
done
wc -l data/datasets/dart_all_cfg_clean.jsonl data/testing/grpo_data_cfg.jsonl data/testing/compile-test2_cfg.jsonl
```
(G1/G2 are unaffected; only G3 needs this. Skip if you want byte-identical inputs.)

The cross-block **dataflow edges do NOT need a dataset rebuild**: with
`--dfg_mode edges` they are computed at load time inside `ensure_cfg_blocks`
from the stored blocks+CFG, identically for SFT and inference.

## The runs (SFT only; Qwen3-8B; one g7e.12xlarge, 2 arms per wave)
```bash
# Requires the patched runner where checkpointless --skip_training evaluates
# the raw base model instead of trying to load a missing pytorch_model.bin.
python -m py_compile configs/run_sweeps_antigravity.py

NEW=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128
METRIC_WORKERS=${METRIC_WORKERS:-64}
GEN_BS=${GEN_BS:-4}

# Sanity check that the overridden sweep name exists after --lora_r/--lora_alpha.
python configs/run_sweeps_antigravity.py \
  --encoder gcb \
  --max_risk high \
  --epochs 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  | grep qwen3-8b-base | grep 5e6

COMMON=(
  --experiment "$NEW"
  --encoder gcb
  --max_risk high
  --hardware_profile auto
  --force_rerun
  --train_file data/datasets/dart_all_cfg_clean.jsonl
  --eval_file data/testing/grpo_data_cfg.jsonl
  --compile_dataset data/testing/compile-test2_cfg.jsonl
  --pass_dataset data/testing/grpo_data_cfg.jsonl
  --prompt_fit_assembly 1
  --auto_cfg 1
  --max_block_instrs 24
  --dfg_mode edges
  --position_scheme roberta
  --epochs 4
  --sft_lr 5e-6
  --lora_r 64
  --lora_alpha 128
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 2
  --grad_accum 16
  --decoder_prompt_max_length 2048
  --eval_max_new_tokens 768
  --generation_batch_size "$GEN_BS"
  --num_samples 5
  --pass_num_samples 10
  --metric_workers "$METRIC_WORKERS"
)

# R  reference: plain Qwen3-8B-Base zero-shot (no training), text-only
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --skip_training \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _x86_ref_base

# G1  text-only (FT, no graph)
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _x86_g1_textonly

# G2  graph + text  (the encoder+graph architecture)
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _x86_g2_graphtext

# G3  graph only, NO assembly text  (encoder-carries)
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode none \
  --name_suffix _x86_g3_graphonly

# G0  null control: no graph, no assembly (prior-only floor)
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode none \
  --name_suffix _x86_g0_null

# G2c (optional; run only if G2 beats G1): CFG-only graph, no DFG edges
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --dfg_mode off \
  --name_suffix _x86_g2c_cfgonly
```

Sanity per arm: the startup banner prints
`Graph channel: dfg_mode=... | position_scheme=... | causal_position_ids=cumsum ...`,
and the no-graph arms must log `Qwen graph prefix DISABLED (GRAPH_QWEN_PREFIX_TOKENS=0)`.
If an arm with prefix 0 does NOT print that line, the pod is running pre-fix code and
the arm is invalid — stop and re-sync.

If training/inference finished but the pod died during metrics, rerun metrics only:
```bash
for SUFFIX in _x86_ref_base _x86_g1_textonly _x86_g2_graphtext _x86_g3_graphonly _x86_g0_null _x86_g2c_cfgonly; do
  EXTRA=()
  case "$SUFFIX" in
    _x86_ref_base) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _x86_g1_textonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _x86_g2_graphtext) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _x86_g3_graphonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode none) ;;
    _x86_g0_null) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode none) ;;
    _x86_g2c_cfgonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1 --dfg_mode off) ;;
  esac
  # skip arms that were never run
  [ -f "results/sweeps_antigravity/${NEW}${SUFFIX}.json" ] || [ -f "results/${NEW}${SUFFIX}_pass_predictions.json" ] || continue
  python configs/run_sweeps_antigravity.py "${COMMON[@]}" "${EXTRA[@]}" --name_suffix "$SUFFIX"
done
```

Pull the final comparison table from the runner summaries:
```bash
python - <<'PY'
import json
from pathlib import Path

base = "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128"
suffixes = [
    "_x86_ref_base",
    "_x86_g1_textonly",
    "_x86_g2_graphtext",
    "_x86_g3_graphonly",
    "_x86_g0_null",
    "_x86_g2c_cfgonly",  # optional arm; skipped below if never run
]

print("arm,codebleu,codebleu_compiled,compile@1,compile@5,pass@1,pass@5,pass@10")
for suffix in suffixes:
    path = Path("results/sweeps_antigravity") / f"{base}{suffix}.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    print(",".join([
        suffix.lstrip("_"),
        str(data["codebleu"]["mean_codebleu"]),
        str(data["codebleu_compiled_only"]["mean_codebleu"]),
        str(data["compile_at_k"].get("compile_at_1")),
        str(data["compile_at_k"].get("compile_at_5")),
        str(data["pass_at_k"].get("pass_at_1")),
        str(data["pass_at_k"].get("pass_at_5")),
        str(data["pass_at_k"].get("pass_at_10")),
    ]))
PY
```

The runner writes, for each arm:
- predictions: `results/${NEW}${SUFFIX}_compile_predictions.json` and `results/${NEW}${SUFFIX}_pass_predictions.json`
- summary: `results/sweeps_antigravity/${NEW}${SUFFIX}.json`
- candidate stats: `results/sweeps_antigravity/${NEW}${SUFFIX}_compile_stats.csv` and `results/sweeps_antigravity/${NEW}${SUFFIX}_pass_stats.csv`

## Reading the result
- **G2 > G1** → the live encoder+graph helps; report the gain and vs TOSEM.
  Then run **G2c** to attribute it: G2 − G2c = DFG edges' share, G2c − G1 = CFG's share.
- **G2 ≈ G1** → even live, the 16-token graph bottleneck doesn't earn its place at
  x86 scale (your earlier hypothesis); text is doing the work.
- **G3 ≈ G2** → the encoder *carries* the assembly (text removable).
- **G3 ≈ G0 (floor)** → the 16-token graph carries almost nothing; it's decorative.
- **G1 ≫ G3** → text is the dominant channel.

## Notes
- **Optional wide arms** (`--qwen_prefix_tokens 128`) for G2/G3 if the 16-token
  results are ambiguous — separates "graph too small" from "graph unhelpful".
  Capacity analysis says the prefix width is the likelier binding constraint than
  GNN size, so run the wide arms before considering a bigger GNN.
- **Per-arm graph flags:** `--dfg_mode edges --position_scheme roberta` sit in
  COMMON so all arms share identical dataset prep; they are inert in the
  prefix-0 arms (the graph is never computed there). `--causal_position_ids`
  defaults to `cumsum` (SFT/generation position parity) — do not set `arange`
  here; that exists only to reproduce pre-fix checkpoints.
- **Gate diagnostic (free):** for G2/G3, the runner logs
  `Qwen graph prefix gate: <value>` at checkpoint load. Gate drifting up from
  0.2 = the decoder leans on the graph; gate collapsing toward 0 = the graph
  channel is being ignored — corroborates the G2−G1 reading.
- **Same matrix on the 9B**: swap `--decoder qwen-9b-base` (your main architecture)
  — easy add once the 8B picture is clear.
- **SFT-only** answers the architecture question; GRPO is a separate follow-up.
- **Scope:** this is x86 TOSEM re-validation — *separate* from the ARM64 Flagship-B
  pilot in `PROPOSAL_AWS.md`. Cheap (8B + short x86 asm): ~8-10 instance-hours total.
