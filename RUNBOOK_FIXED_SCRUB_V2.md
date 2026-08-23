# Fixed Signature-Scrubbed Experiment (v2) — State & Runbook

2026-07-16. Successor to the drifted P7 run reviewed in
`results-20260713/FABLE_SIGNATURE_SCRUBBED_REVIEW_FINDINGS.md`.

## What was built (all gates pass)

Two dataset variants over the frozen 154-task benchmark (`grpo_data_graphv2.jsonl`,
input sha `8453876a…`), built on the pinned box `167.172.150.125`
(`/root/fixed_scrub_build`, Dart 3.11.5, GDB 17.1, benchmark extractor `3b522afc…`):

- `data/testing/grpo_data_graphv2_sigscrub_v2_nameonly_{public,private}.jsonl`
  — prompt gets only the neutral name (harsh arm; replaces the drifted P7 set).
- `data/testing/grpo_data_graphv2_sigscrub_v2_neutralexact_{public,private}.jsonl`
  — prompt gets the exact typed signature with neutral function AND parameter
  names (`bool candidate(List<double> a, double b)`), in-distribution `exact`
  phrasing. Decomposes name-recognition from type/arity scaffolding.

Key design decision: `--assembly_mode rename_frozen`. Recompiling the neutralized
source can NEVER match the frozen comparator (Dart AOT register allocation
drifted on 87/154 tasks even fully pinned), so the scrubbed assembly is derived
by **context-targeted symbol renaming of the frozen benchmark dump** (annotations,
dump headers, quoted regex headers, synopsis lines incl. `name.<anonymous
closure>`; helper functions renamed to `helperN`; file paths neutralized;
mnemonic collisions like task `add` untouched). Result: G1-strict — instructions
byte-exact modulo `<symbol>` masking, edges exact — passes 154/154 for both
variants.

Gates (in `logs/fixed_scrub_v2/` on the pod, `/root/fixed_scrub_build/logs/` on
the box): G1 strict parity 0 mismatches; G2 leak scan clean (benign whitelist:
dot-qualified SDK members like `JsonCodec.encode` and `dart:` paths, which are
byte-identical in the comparator dumps); G3 neutrality/hygiene/shuffle
(fingerprint hashes stripped from public rows, public order shuffled seeds
4242/4243); G4 stub-compile 150/154 (4 known-broken legacy tasks: 121 tests
alias a `solution` wrapper, 127 record-args, 153 `Strongest_Extension` alias,
161 `$a` interpolation — all comparator zero-compile ties); G5 references
compile 151/154, pass 142/154 on Dart 3.11.5.

Manifest: `fixed_scrub_v2_sha256.txt` (verified 8/8 on pod + local).

## Queued GPU stage (autonomous)

`/workspace/fixed_scrub_v2_pod_chain.sh` on the vast pod (98.218.15.126:24424),
launched 09:11Z, status `/workspace/results/fixed_scrub_v2_queue.status`:

1. Waits for `arm64_regions16_s42.status` to leave RUNNING, then ≥60 GB free VRAM.
2. Verifies the frozen checkpoint sha `e8e872…` before use.
3. Per arm (`nameonly` then `neutralexact`): inference (frozen regions16 env,
   seed 42, 10×154, 768 new tokens) → mode-aware rehydrate → codebleu ×2 →
   compile@k `jit_tests` (patched crash classifier) → pass@k → stats CSV →
   sha manifest. Stems: `…_regions16_sigscrub_v2_{nameonly,neutralexact}`.

Monitor: `ssh -p 24424 root@98.218.15.126 'cat /workspace/results/fixed_scrub_v2_queue.status /workspace/results/*sigscrub_v2*.status 2>/dev/null'`

## When it completes (task 6)

Sync `results/*sigscrub_v2*` + `results/sweeps_antigravity/*sigscrub_v2*` +
`logs/*sigscrub_v2*` to `results-20260713/`, re-run the review-style
verification (hashes, digest, provenance, paired tables vs the exact-signature
comparator — 42-solved/147-compiling baseline), and update
`FABLE_SIGNATURE_SCRUBBED_REVIEW_FINDINGS.md` + memory with both arms' numbers.

Predictions to check first: candidate-naming rate and arity-match rate per arm
(the neutralexact arm should emit `candidate` with correct typed params if the
in-distribution signature phrasing restores decompile mode).

## Code changes (local workspace, shipped to box+pod)

- `scripts/data/build_signature_scrubbed_eval.py`: `--public_signature_mode`,
  `--assembly_mode rename_frozen`, `--shuffle_public_seed`, helper/closure
  symbol neutralization, toolchain stamps, public fingerprint-hash stripping,
  schema v2.
- `scripts/evaluation/rehydrate_signature_scrubbed_predictions.py`:
  `--expected_signature_mode`, sealed-signature equality check, v2 hygiene
  validation.
- `scripts/evaluation/graph_compile_at_k_antigravity.py`: `Crash when
  compiling` now classified as compile failure (F4).
- Tests: 20 local (incl. renamer collision/closure/helper cases), 7 on pod/box.
- Box gate suite: `/root/fixed_scrub_build/gates.py`.
