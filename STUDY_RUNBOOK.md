# Study runbook — graph-architecture value + assembly-channel ablation (x86, fixed datasets)

Step-by-step commands for every stage. Design rationale: `RUN_PLAN_X86_ABLATION.md`.

> **UPDATED 2026-07-04 — requires the fixed pipeline code on the pod.** Before the
> fix, `--qwen_prefix_tokens 0` still prepended the (raw, and for R random) block
> states, so the R/G1/G0 "no-graph" arms were invalid. On the fixed code, prefix-0
> arms must log `Qwen graph prefix DISABLED (GRAPH_QWEN_PREFIX_TOKENS=0)` at model
> init — if that line is missing, STOP and re-sync the repo. Graph arms now also
> use the post-fix architecture (`--dfg_mode edges --position_scheme roberta`,
> cumsum SFT positions by default).

**What it answers (one 2x2):**

| Arm | `--qwen_prefix_tokens` | `--prompt_assembly_mode` | Isolates |
|---|---|---|---|
| R  | 0  | full | plain base model, zero-shot (≈ TOSEM base) |
| G1 | 0  | full | FT text decoder, no graph |
| G2 | 16 | full | the encoder+graph architecture |
| G3 | 16 | none | encoder carries assembly, no text |
| G0 | 0  | none | prior-only floor |

Supervisor's "is encoder+graph good" = **G2 − G1**.  Your with/without-assembly = **G2 vs G3 vs G1**.

---

## Stage 0 — prerequisites (state: x86 datasets DONE)
Already completed locally (no action needed unless rebuilding fresh):
- Fixed/de-leaked + block-split datasets: `dart_all_cfg_clean.jsonl` (1195),
  `grpo_data_cfg.jsonl` (154), `compile-test2_cfg.jsonl` (126). Leak verified 0.
- If you ever need to regenerate the split CFG:
  ```bash
  for f in data/datasets/dart_all_cfg_clean.jsonl data/testing/grpo_data_cfg.jsonl data/testing/compile-test2_cfg.jsonl; do
    GRAPH_MAX_BLOCK_INSTRS=24 python scripts/data/build_cfg_jsonl.py --input "$f" --output "$f.tmp" --overwrite && mv -f "$f.tmp" "$f"
  done
  ```
- The cross-block dataflow edges (`--dfg_mode edges`) need NO dataset rebuild —
  they are computed at load time from the stored blocks+CFG.

Sync the repo (code + `data/`) to the GPU instance before Stage 2 — must include
the 2026-07-04 pipeline fixes (`scripts/data/`, `scripts/training/`, `models/`,
`scripts/evaluation/`, `configs/`).

## Stage 1 — choose the model (the toggle)
Set ONE variable; everything downstream follows. This is the "either model" option.
```bash
MODEL=qwen3-8b-base          # supervisor's TOSEM base model  ← default
# MODEL=qwen-9b-base         # your main graph architecture (swap to run the same study on the 9B)

EXP=${MODEL}_lora_enc_dec_r64_5e6_gcb_a128   # verified generated name for both models
```
(Selection works because `--experiment $EXP --name_suffix _x86_<arm>` matches the
runner's modified config name exactly. `--decoder $MODEL` also filters, but the
`$EXP` form pins a single config — use it.)

## Stage 2 — shared settings + smoke test
```bash
# Dataflow-edge cap: the 512 default truncates dense functions (observed up to
# 3312 edges/function on dart_all) and the truncation is index-biased (late
# blocks lose their dataflow). Set
  --train_file data/datasets/dart_all_cfg_clean.jsonl \
  --eval_file  data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset    data/testing/grpo_data_cfg.jsonl \
  --prompt_fit_assembly 1 --auto_cfg 1 --max_block_instrs 24 \
  --dfg_mode edges --position_scheme roberta \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 1 \
  --train_batch_size 2 --grad_accum 16 \
  --decoder_prompt_max_length 2048 --eval_max_new_tokens 768 \
  --generation_batch_size 4 --num_samples 5 --pass_num_samples 10"
```
Smoke-test each arm first (2 train steps + 2 eval samples — catches config errors
for free). Append `--dry_run` to any Stage-3 command and confirm before the real run:
- the banner shows the right prefix-token count and assembly mode;
- the `Graph channel:` line shows `dfg_mode=edges | position_scheme=roberta |
  causal_position_ids=cumsum`;
- prefix-0 arms log `Qwen graph prefix DISABLED (GRAPH_QWEN_PREFIX_TOKENS=0)`
  (missing line = pre-fix code on the pod = invalid arm).

## Stage 3 — run the five arms
Two GPUs on g7e.12xlarge ⇒ run two at a time. Order: G1+G2, then G3+G0, then R (eval-only).
```bash
# G1  text-only (FT, no graph)
python configs/run_sweeps_antigravity.py $COMMON \
  --qwen_prefix_tokens 0  --prompt_assembly_mode full --prompt_clean_asm 1 \
  --name_suffix _x86_g1_textonly

# G2  graph + text  (the encoder+graph architecture)
python configs/run_sweeps_antigravity.py $COMMON \
  --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1 \
  --name_suffix _x86_g2_graphtext

# G3  graph only, NO assembly text (encoder-carries)
python configs/run_sweeps_antigravity.py $COMMON \
  --qwen_prefix_tokens 16 --prompt_assembly_mode none \
  --name_suffix _x86_g3_graphonly

# G0  null floor: no graph, no assembly
python configs/run_sweeps_antigravity.py $COMMON \
  --qwen_prefix_tokens 0  --prompt_assembly_mode none \
  --name_suffix _x86_g0_null

# R  reference: base model zero-shot (no training), text-only — fast, eval only
python configs/run_sweeps_antigravity.py $COMMON --skip_training \
  --qwen_prefix_tokens 0  --prompt_assembly_mode full --prompt_clean_asm 1 \
  --name_suffix _x86_ref_base

# G2c (optional; run only if G2 > G1): CFG-only graph — attributes the G2 gain
# between CFG (G2c − G1) and DFG edges (G2 − G2c)
python configs/run_sweeps_antigravity.py $COMMON \
  --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1 \
  --dfg_mode off \
  --name_suffix _x86_g2c_cfgonly
```

## Stage 4 — collect metrics
Each run writes, under `results/` on the instance:
- `results/sweeps_antigravity/${EXP}_x86_<arm>.json`   — summary (pass@k, compile@k, CodeBLEU)
- `results/${EXP}_x86_<arm>_pass_stats.csv`             — per-task pass flags (for stratified pass@k)
- `results/${EXP}_x86_<arm>_pass_predictions.json`      — raw candidates

Cost-saver (optional): Ctrl-C the pod once predictions are written and compute
metrics locally from the CSVs (saves paid minutes). Pull `results/` back, then
tabulate the five arms into the comparison table (pass@1 / pass@5 / pass@10 /
compile@1 / compile@5 / CodeBLEU per arm). Length-stratified pass@k can be read
off the `_pass_stats.csv` files.

## Stage 5 — interpret
| Comparison | Reads as |
|---|---|
| R vs TOSEM base (p@1 6.36%, p@5 14.89%) | sanity/grounding |
| G1 − R | did corrected-pipeline FT help? |
| **G2 − G1** | **does the live encoder+graph help over text?** (supervisor) |
| **G2 vs G3 vs G1** | **graph+text vs graph-only vs text-only** (with/without assembly) |
| G3 vs G0 | does the 16-token graph carry anything above the prior floor? |
| G2 − G2c (optional arm) | DFG edges' share of the graph gain (G2c − G1 = CFG's share) |

- G2 > G1 → graph earns its place; report the gain + vs TOSEM — then run G2c to attribute it.
- G2 ≈ G1 → 16-token bottleneck doesn't help even when live; text does the work.
- G3 ≈ G2 → encoder carries the assembly (text removable).
- G3 ≈ G0 → graph is decorative.  G1 ≫ G3 → text is the dominant channel.

## Stage 6 — optional extensions
- **Run the other model:** change `MODEL` in Stage 1, rerun Stages 2–4. Nothing else changes.
- **Wide-prefix arms** (separate "graph too small" from "graph unhelpful"):
  MEASURED 2026-07-05: 64/128 tokens DIVERGE under the plain recipe (eval_loss
  rises monotonically after epoch 1; p128 final eval_loss 0.68 vs G3's 0.40,
  compile@5 collapsed to 0.04). Wide arms REQUIRE the stabilized recipe:
  ```bash
  python configs/run_sweeps_antigravity.py $COMMON \
    --qwen_prefix_tokens 128 --prompt_assembly_mode none \
    --qwen_prefix_rms_match 1 \
    --qwen_prefix_gate_init 0.05 \
    --save_strategy epoch --save_total_limit 4 \
    --name_suffix _x86_g3_p128r
  ```
  `--qwen_prefix_rms_match 1` rescales prefix vectors to the decoder's
  token-embedding RMS (root-cause fix; train and inference must both set it -
  the shared runner env handles that). Quiet gate + per-epoch saves are the
  safety margin: if eval_loss still rises after any epoch, evaluate the best
  epoch's checkpoint instead of the final one. Tripwire while training:
  epoch-N eval_loss must be <= epoch-(N-1); grad_norm spikes >50 are the
  early symptom.
- **GRPO follow-up** (only on the winning arm, if the architecture question warrants
  it): add `--use_grpo --grpo_checkpoint artifacts/${EXP}_x86_<winner>/pytorch_model.bin`
  with the pass@k-GRPO recipe (`--grpo_passk_k 5 --grpo_group_size 8 --grpo_score_chunk_size 2`).

## Notes
- Same recipe across arms (only `--qwen_prefix_tokens`, `--prompt_assembly_mode`,
  and — for G2c — `--dfg_mode` vary) → the comparison is internally rigorous;
  TOSEM numbers are external context. The graph flags in COMMON are inert in
  prefix-0 arms (graph compute is skipped entirely).
- G2/G3 checkpoint loads log `Qwen graph prefix gate: <value>` — gate rising from
  0.2 means the decoder leans on the graph; collapsing toward 0 corroborates a
  null G2−G1.
- SFT-only answers the architecture question; GRPO is Stage 6.
- Compute: 8B + short x86 asm is light (~8–10 instance-hours for all 5 arms); the 9B
  pass costs a bit more. Both fit one g7e.12xlarge (96 GB/GPU).
