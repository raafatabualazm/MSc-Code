# Session Handoff — Neural Decompilation Thesis Eval

_Last updated: 2026-06-01_

## 0. TL;DR

- Thesis: **Assembly → Swift/Dart neural decompilation** via a graph-conditioned
  encoder-decoder (NOT a causal LLM).
- Two eval bugs caused an **impossible result** (pass@k > compile@k). Both fixed
  locally; **must re-upload to H100 and re-run**.
- **Bigger open question:** the `qwen-2b` predictions are **degenerate** (decoding
  collapse). Must confirm whether the `codet5p-2b` CLAP run is also degenerate. If
  so, the metric fix is moot for that run and the real bug is the decoder /
  projection bridge — and the CodeBLEU 0.403 must have come from a *different*
  working checkpoint.

---

## 1. Project / Architecture

- **Task:** Assembly → Swift/Dart decompilation.
- **Stack:** PyTorch 2.x + CUDA, transformers 5.x, accelerate, PEFT LoRA, optional
  BitsAndBytes NF4 (bf16 remote), torch_geometric.
- **Model pipeline:** graph encoder → hierarchical encoder → projection bridge
  (`enc_to_dec_proj` patched to **Identity**) → decoder `codet5p-2b`.
  - Encoder options: **CLAP** (`hustcw/clap-asm`) or **graphcodebert-base**.
  - It is a **graph-conditioned encoder-decoder**, not a causal LLM.

### Baselines v1–v6 are FRONTIER causal LLMs (for the thesis table)
| Versions | Model | Size |
|----------|-------|------|
| v1, v2 | Qwen3-4B-2507 | 4B |
| v3, v4 | DeepSeek-R1-Distill-Qwen3-8B | 8B |
| v5, v6 | Qwen3-8B | 8B |

- They use multi-attempt compile-retry (attempts ≤ 5); caches have many
  `codebleu = null`.
- **Thesis framing:** parameter efficiency (2B graph model) + functional metrics +
  graph-encoder ablation. The CodeBLEU 0.403 gap vs these frontier LLMs is a
  capacity/methodology artifact (CodeBLEU measures surface similarity, not
  correctness; baselines are 4–8B frontier models).

---

## 2. Environments

**Remote (H100):**
- H100 80GB Lightning, `/teamspace/studios/this_studio/experiment_workspace`
- conda env `cloudspace`, python3.12
- Dart SDK at `/home/zeus/dart-sdk/bin/dart`

**Local (this machine):**
- RTX 3060 6GB, Windows PowerShell, conda `base`
- cwd `C:\Users\Raafat Abualazm\Desktop\Train Data\experiment_workspace`

**Dart CLI:** `dart analyze` (static, strict), `dart run` (compile+execute),
`dart compile aot-snapshot` / `exe`. Unbiased @k via comb-style estimator.

---

## 3. The Eval Bug (root cause)

`pass@k > compile@k` is logically impossible. Causes:

1. **Fake pass@k** — `pseudo_pass()` in
   `scripts/evaluation/graph_pass_at_k_antigravity.py` was a token-overlap
   heuristic (≥0.3 overlap) that **never ran Dart**. The reported
   `pass@1=0.0133, pass@5=0.0606` were noise.
2. **Too-strict compile@k** — `scripts/evaluation/graph_compile_at_k_antigravity.py`
   used `dart analyze` (flags lint / unused-import as errors) and did not strip
   markdown ```dart fences or leading prose. Produced all-zero compile@k.
3. **Degenerate model output** — the `qwen-2b` predictions are `/`, single lines
   of `/ / / /`, pure spaces, walls of `"`. Decoding collapse.

**Also found:** `data/datasets/test-set.jsonl` (165 problems; keys
`filename, function, source, assembly, language`) targets are full
`@pragma('vm:entry-point') void main(){...}` programs with **NO per-problem
unit-test harness** → assertion-based pass@k is impossible on this set; only
**compile-and-run (run@k)** is feasible.

---

## 4. FIXES APPLIED (local only — error-free — NOT yet on H100)

### `scripts/evaluation/graph_compile_at_k_antigravity.py`
- Added `_extract_code()` — strips ```dart fences + leading prose.
- `compile_dart` now runs
  `dart analyze --no-fatal-warnings --no-fatal-infos` (lint/style/unused-import
  no longer count as compile failure; only real errors do), with a timeout and a
  binary-not-found guard.
- Kept: `_resolve_dart_binary()`/`DART_BIN`, comb estimator
  `compile_at_k_estimator(n,c,k)`, n=5 candidate loop, CLI `--predictions` /
  `--k_values` (default `'1,5'`).

### `scripts/evaluation/graph_pass_at_k_antigravity.py`
- **Removed** `pseudo_pass`.
- Added `_extract_code()` + `run_dart()` — executes `dart --disable-dart-dev run`;
  **pass == program runs to completion (exit 0)**.
- Loop now calls `run_dart(cand)`; `reference` no longer used.
- Kept comb estimator + CLI.

**Result:** pass@k ≤ compile@k by construction (running requires compiling). The
contradiction can no longer occur.

---

## 5. Other relevant code

- `scripts/training/graph_grpo_decompiler_antigravity.py` (GRPO reward — **DIFFERENT
  task format**): `compute_reward(solution_code, test_code)` expects a **function
  body** (no `main()`); `_has_main_function` → −5 penalty; empty → −3.
  `_run_single_expect_test` wraps body + `expect` test in a generated `main()`, runs
  `['dart','--disable-dart-dev','run',p]`, returncode==0. Parallel via
  `GRPO_REWARD_WORKERS`, `GRPO_TEST_TIMEOUT`. **Edited prior session (error-free) —
  MUST RE-UPLOAD to H100 before GRPO.**
- `configs/run_sweeps_antigravity.py` (**edited prior session — MUST RE-UPLOAD**):
  GRPO speed knobs (`GRPO_GROUP_SIZE=4`, `GRPO_EPOCHS=1`, `GRPO_TEST_TIMEOUT=3`,
  `GRPO_REWARD_WORKERS`), 2B OOM fix (`batch_size=2`/`grad_accum=16`,
  `expandable_segments`). SFT→`artifacts/{name}`,
  `results/{name}_predictions.json`; GRPO adds `_grpo` suffix; inference limit 165,
  num_samples 5, k_values `'1,5'`.
- `results/qwen-2b_lora_enc_dec_1e5_predictions.json`: schema = list of
  `{id, predictions:[str,...], reference, language}`. Output **degenerate**.
- `scripts/evaluation/compile-test.py`: `compile_dart_code` uses
  `dart compile aot-snapshot` (~L129); `compute_compile_at_k` (~L99). Reference for
  real compile logic.
- Older unused harness copies (single-prediction, NOT the log producers):
  `evaluation/graph_compile_at_k.py`, `evaluation/graph_pass_at_k.py`,
  `scripts/evaluation/graph_compile_at_k.py`, `scripts/evaluation/graph_pass_at_k.py`.

---

## 6. Progress

- ✅ CodeBLEU gap explanation + thesis reframing.
- ✅ Located the actual evaluator scripts (`_antigravity` versions).
- ✅ Diagnosed both bugs (fake pass@k, too-strict compile@k).
- ✅ Confirmed test set lacks unit-test harness.
- ✅ Confirmed qwen-2b output degenerate.
- ✅ Rewrote `graph_compile_at_k_antigravity.py` + `graph_pass_at_k_antigravity.py`
  (local, error-free).
- ⏳ Re-run both metrics on H100 with the fixed scripts.
- ⏳ **OPEN:** confirm codet5p-2b CLAP predictions degenerate vs real.

---

## 7. Next actions

1. **Re-upload to H100:** the two fixed eval scripts, plus (still pending from prior
   session) `configs/run_sweeps_antigravity.py` and
   `scripts/training/graph_grpo_decompiler_antigravity.py`.
2. **Re-run** compile@k and pass@k on the predictions JSON (old numbers were from
   broken scripts).
3. **Decision blocker — confirm decoding collapse.** Run on H100:
   ```bash
   python -c "import json,glob
   fs=glob.glob('results/*clap*predictions.json')
   print(fs)
   for f in fs:
       d=json.load(open(f)); print('===',f)
       [print(r.get('id'),'->',repr((r.get('predictions') or [r.get('prediction','')])[0][:140])) for r in d[:6]]"
   ```
   - If degenerate → real bug is **decoding collapse** (projection bridge feeding
     noise / greedy repetition / sampling params). Metric fix is moot for that run,
     and the CodeBLEU 0.403 came from a **different** working checkpoint.
   - Do NOT conclude "model doesn't compile" until decoding collapse is ruled out.
4. Full GRPO chain command (after re-uploads):
   ```bash
   python configs/run_sweeps_antigravity.py --run_all --decoder codet5p-2b \
     --mode lora_enc_dec --max_risk high --encoder clap --epochs 5 \
     --lora_r 32 --lora_alpha 64 --chain
   ```

## 8. Reminders
- Do NOT create markdown docs (this file was explicitly requested).
- pass@k now aliases to run@k on this test set (no unit-test harness).
