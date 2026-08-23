# ARM64 Flutter run plan (Flagship B engine)

> **SUPERSEDED FOR CONFIRMATORY RUNS (2026-07-12).** This plan uses legacy
> graph-v1 data and the older 9B lineage. Do not execute it as written. The
> corrected, audited graph-v2.1 files and protocol are documented in
> `ARM64_GRAPHV21_PREP.md`; a Qwen3-8B confirmatory command matrix must replace
> the commands below before GPU training.

Resolves, **before any GRPO or big-budget spend**, the architecture question:
*which channel actually carries the assembly into the decoder* — the raw text,
the graph prefix, or both? The answer decides the truncation strategy and
whether the GNN-encoder earns its complexity.

Data (frozen, local, built 2026-06-19/20): `flutter_train_cfg.jsonl` (1371) /
`flutter_eval_cfg.jsonl` (343), length-stratified, leak-free, CFG live (100%
branch resolution), basic blocks split to <=24 instr (encoder truncation ~0).

Base model selector (same as prior runs): `qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128`.
Eval = held-out `flutter_eval_cfg` (in-distribution). **Do NOT eval against the
x86 HumanEval set** (ISA + style mismatch = the old synthetic-transfer failure).

```bash
python -m py_compile configs/run_sweeps_antigravity.py

NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
METRIC_WORKERS=${METRIC_WORKERS:-64}
GEN_BS=${GEN_BS:-4}

# Sanity check that the overridden sweep name exists after --lora_r/--lora_alpha.
python configs/run_sweeps_antigravity.py \
  --encoder gcb \
  --max_risk high \
  --epochs 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  | grep qwen-9b-base | grep 5e6

# Shared SFT settings. decoder budget 8192 = generous so the monolithic/text arms
# are a FAIR baseline (cliff = capability, not truncation). gradient_checkpointing
# on + small batch for the long prompt; raise batch on a 192/272GB card if memory allows.
COMMON=(
  --experiment "$NEW"
  --encoder gcb
  --max_risk high
  --hardware_profile auto
  --force_rerun
  --train_file data/datasets/flutter_train_cfg.jsonl
  --eval_file data/datasets/flutter_eval_cfg.jsonl
  --compile_dataset data/datasets/flutter_eval_cfg.jsonl
  --pass_dataset data/datasets/flutter_eval_cfg.jsonl
  --prompt_fit_assembly 1
  --auto_cfg 1
  --max_block_instrs 24
  --epochs 4
  --sft_lr 5e-6
  --lora_r 64
  --lora_alpha 128
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 1
  --grad_accum 64
  --decoder_prompt_max_length 8192
  --eval_max_new_tokens 1024
  --generation_batch_size "$GEN_BS"
  --num_samples 3
  --pass_num_samples 8
  --metric_workers "$METRIC_WORKERS"
)
```

## Phase -1 — free smoke test (do this first)
Append `--dry_run` to every arm below (2 train steps + 2 eval samples). Confirms
the flutter data + new flags run end-to-end on the pod before any real spend.
Watch the banner: SFT mode, prefix-token count, and assembly-mode must match the arm.

## Phase 0 — architecture ablation (SFT only, cheap eval)

```bash
# A0  text-only (no graph): does the GNN earn its place?
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a0_textonly

# A2  wide hybrid (graph + text): does a real graph bridge help over text?
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a2_wide

# A3  encoder-only (graph, NO text): can the encoder CARRY the assembly?
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode none \
  --name_suffix _arm_a3_enconly

# A1  current 16-token hybrid (OPTIONAL: 16->128 dose-response for the paper)
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a1_hybrid16
```

For a smoke test, append `--dry_run` to each arm command. For example:
```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --dry_run \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a2_wide_dry
```

If training/inference finished but the pod died during metrics, rerun metrics only:
```bash
for SUFFIX in _arm_a0_textonly _arm_a2_wide _arm_a3_enconly _arm_a1_hybrid16; do
  EXTRA=()
  case "$SUFFIX" in
    _arm_a0_textonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _arm_a2_wide) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 128 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _arm_a3_enconly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 128 --prompt_assembly_mode none) ;;
    _arm_a1_hybrid16) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
  esac
  python configs/run_sweeps_antigravity.py "${COMMON[@]}" "${EXTRA[@]}" --name_suffix "$SUFFIX"
done
```

Pull the comparison table from the runner summaries:
```bash
python - <<'PY'
import json
from pathlib import Path

base = "qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128"
suffixes = [
    "_arm_a0_textonly",
    "_arm_a2_wide",
    "_arm_a3_enconly",
    "_arm_a1_hybrid16",
]

print("arm,codebleu,codebleu_compiled,compile@1,compile@5,pass@1,pass@5")
for suffix in suffixes:
    path = Path("results/sweeps_antigravity") / f"{base}{suffix}.json"
    data = json.loads(path.read_text())
    print(",".join([
        suffix.lstrip("_"),
        str(data["codebleu"]["mean_codebleu"]),
        str(data["codebleu_compiled_only"]["mean_codebleu"]),
        str(data["compile_at_k"].get("compile_at_1")),
        str(data["compile_at_k"].get("compile_at_5")),
        str(data["pass_at_k"].get("pass_at_1")),
        str(data["pass_at_k"].get("pass_at_5")),
    ]))
PY
```

The runner writes, for each arm:
- predictions: `results/${NEW}${SUFFIX}_compile_predictions.json` and `results/${NEW}${SUFFIX}_pass_predictions.json`
- summary: `results/sweeps_antigravity/${NEW}${SUFFIX}.json`
- candidate stats: `results/sweeps_antigravity/${NEW}${SUFFIX}_compile_stats.csv` and `results/sweeps_antigravity/${NEW}${SUFFIX}_pass_stats.csv`

You can still Ctrl-C after predictions are written and do metrics locally; use the
same `--skip_training --skip_inference` recovery loop on the local box. Do the
length-stratified readout from the stats CSVs for the <50/50-100/100-200/200-500/>500 bins.

## Decision rule (this picks the architecture + truncation strategy)
Compare length-stratified pass@1 / pass@5 on `flutter_eval`:

- **A2 ≈ A0** → the GNN adds nothing → drop the encoder, go **text-only**;
  truncation is then handled by B's decomposition. Simplest, cheapest.
- **A2 > A0 and A3 ≈ A2** → the encoder **carries** the assembly →
  adopt **encoder-only**, raw text removable, decoder truncation ≈ free
  (encoder-only block problem, ~0 after the <=24 split). Your instinct, vindicated.
- **A2 > A0 and A3 << A2** → graph helps but the decoder **needs the tokens** →
  keep **hybrid**, and truncation is real → **B's decompose-and-recompose is the
  right fix** (this is positive evidence FOR Flagship B, not against it).
- **A0 > A2** → the graph **interferes** → text-only + B.

## Phase 1 — winner only: pass@k-GRPO + full eval
Carry the winning arch's flags forward. Use the gate-passing pass@k-GRPO recipe
(Stage D). Budgets get one `--dry_run` probe on the actual card first (GRPO is the
OOM-prone stage; chunked scoring bounds peak memory by chunk, not group).

```bash
WIN=${NEW}_arm_a2_wide   # <-- replace with the winning arm's suffix

# Carry these from the winning arm:
WIN_PREFIX_TOKENS=128
WIN_ASM_MODE=full
WIN_CLEAN_ASM=1

GRPO_COMMON=(
  --experiment "$NEW"
  --name_suffix _arm_grpo
  --encoder gcb
  --max_risk high
  --hardware_profile auto
  --force_rerun
  --use_grpo
  --grpo_checkpoint "artifacts/${WIN}/pytorch_model.bin"
  --grpo_train_file data/datasets/flutter_train_cfg.jsonl
  --compile_dataset data/datasets/flutter_eval_cfg.jsonl
  --pass_dataset data/datasets/flutter_eval_cfg.jsonl
  --prompt_fit_assembly 1
  --auto_cfg 1
  --max_block_instrs 24
  --qwen_prefix_tokens "$WIN_PREFIX_TOKENS"
  --prompt_assembly_mode "$WIN_ASM_MODE"
  --prompt_clean_asm "$WIN_CLEAN_ASM"
  --grpo_passk_k 5
  --grpo_group_size 8
  --grpo_score_chunk_size 2
  --grpo_epochs 1
  --grpo_lr 5e-7
  --grpo_perfect_bonus 1.5
  --grpo_max_new_tokens 1024
  --grpo_test_timeout 6
  --grpo_reward_workers 32
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 1
  --grad_accum 64
  --lora_r 64
  --lora_alpha 128
  --decoder_prompt_max_length 8192
  --use_reasoning 0
  --eval_max_new_tokens 1024
  --generation_batch_size 2
  --num_samples 5
  --pass_num_samples 16
  --metric_workers "$METRIC_WORKERS"
  --save_strategy epoch
  --save_total_limit 2
)

python configs/run_sweeps_antigravity.py "${GRPO_COMMON[@]}"
```

If only the GRPO metrics stage fails after predictions exist:
```bash
python configs/run_sweeps_antigravity.py "${GRPO_COMMON[@]}" \
  --skip_training \
  --skip_inference
```

## Phase 2 — Flagship B technique (deferred, local build first)
Compositional decompilation: use the CFG/slice boundaries
(`flutter_function_symbol_ranges`) to decompile blocks/loop-bodies/slices
independently and recompose; eval monolithic vs compositional, length-stratified.
Build + validate locally before any pod time.

## Cost discipline
- Phase 0 eval uses n=8 (rank the arms); only the winner gets n=16 in Phase 1.
- Ctrl-C after predictions are written; do metrics locally (saves paid minutes).
- Archive every checkpoint + predictions CSV locally after each arm.
