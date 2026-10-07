# Neural decompiler — results snapshot 2026-07-23

Snapshot of the compact-decoder (encoder-free) experiments and the frontier ceiling.
All local student numbers are **leakage-clean**, same 175 held-out sigless tasks, pass@10.

## Student arms (local Qwen3-8B compact decoder, 175 held-out, pass@10)

| arm | pass@10 | compile@10 | note |
|---|---|---|---|
| baseline sigless (no enrichment) | 0.57% | — | confabulation; info-starved |
| soft-KD (47 Qwen repairs, top-5) | 1.14% | — | improper KD (see below); null |
| GPT hard-target RS-SFT (48 repairs) | 1.71% | 60% | objective-side null |
| v3 (ChatGPT corpus, pool+CFG-types+edges) | 1.71% | — | scale/graph-richness null |
| real-enriched (my object-pool extractor) | 2.29% | — | real constants preamble |
| **proxy-ceiling enrichment** | **5.71%** | 69% | **only lever that moved pass@k (10×)** |

Enrichment (recovered constant pool prepended) is the **only** intervention that ever
moved pass@k (0.57 -> 5.71). Every objective change (soft-KD, GPT-hard, graph RS-SFT +0.0pp,
GRPO) is null. => **information bottleneck, not a capability/objective bottleneck.**

## Frontier ceiling — sigless from raw gdb disasm + recovered constants, K=10

| teacher | dataset | pass@10 | compile@10 | tokens |
|---|---|---|---|---|
| DeepSeek-V4-Pro | my dataset (60-task subset) | **8.33%** (5/60) | 26.7% | 7.48M |
| DeepSeek-V4-Pro | v3 dataset (60-task subset) | **16.67%** (10/60) | 36.7% | 6.60M |
| Qwen3.7-Max | my + v3 | pending (quota reset ~13:36 UTC 07-23) | | |

Key reads:
- Even a frontier reasoning model barely clears single digits on **my** sigless set (8.33%),
  and compiles LESS than our compact student (27% vs 69%) — the frontier's edge is semantic
  correctness, not syntax. Teacher–student gap only ~2.6pp.
- **v3 doubles the ceiling (16.67%)**: richer enrichment (call-edges + CFG-types) raises the
  frontier ceiling — direct evidence that **the lever is more recovered semantics**, not the
  objective.

## "KD fully" — not viable as imagined (codex true_kd_patch_v1 audit)

Dense full-distribution KD (`KL(teacher||student)` over full vocab) requires a **stronger
LOCAL compact-conditioned teacher** with byte-identical tokenizer. We have none — our best
local checkpoint IS the student. API teachers (Qwen/DeepSeek) categorically cannot do dense
KD. Machine-readable audit (`true_kd_patch_v1/reports/verified_repairs_lp.audit.json`):
- `usable_for_dense_full_kl: false`
- `usable_for_sparse_topk_tail_kl: false`
- `usable_verified_code_for_rs_sft: true` (48 verified Qwen repairs = RS-SFT hard targets only)
- rejection: only top-5 logprobs, rounded (negative inferred tail), no token IDs, tokenizer
  identity unsealed, no EOS distribution, 48/1580 coverage.

The soft_kd_trainer.py path was fail-closed by codex for exactly this reason (its 1.14% is junk).

## Harness bugs caught + fixed this session (frontier_passk.py)

1. **Reasoning-model empty output** faked pass@k = 0.0: `max_tokens=1600` fully consumed by
   hidden reasoning (`reasoning_tokens=6400`), `content` length 0. Fix: `max_tokens>=8000`,
   probe `finish_reason`/`content_len` first, round-trip GOLD through the evaluator (gold
   compiles+passes -> harness OK).
2. **Disasm-cache miss on v3**: guard `len(asm_cache) < len(dev)` skipped building disasm for
   v3's distinct task_ids (cache already had 175) -> tokens=0, false 0%. Fix: per-task missing
   check.
3. `ex.map` head-of-line block -> `as_completed` + per-request `timeout`; total-token BUDGET cap
   (DeepSeek earmarked for VeRPO judging).

## Files in this snapshot

- `logs/` — all run logs incl `frontier_deepseek.log`, `frontier_v3_deepseek.log`, `gpt_chain.log`
- `artifacts/compact_fn0_rebuild/` — datasets (train/dev fn0 real/enr/v3), verified repairs
  (Qwen `verified_repairs_lp.jsonl`, GPT `verified_repairs_gpt.jsonl`), constants, contract, audit
- `artifacts/v3_clean/` — v3 leakage-clean corpus
- `true_kd_patch_v1/` — codex dense-KD infra + audit + tests
- `codex_staging_20260723/fixed_training_launchers/` — RS-SFT / VeRPO / true-KD launchers (RS/VeRPO are OLD graph arch)

Model checkpoints (8.2G each: `direct_compact_*_sft_v1`) are uploaded to HF
`raafatabualazm/antigravity-qwen3-8b-artifacts`, not included here.
