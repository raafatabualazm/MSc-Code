# Phase-0 split manifest (seed 44) — canonical split for the compact rebuild

Extracted 2026-07-20 from the pod's live Phase-0 artifacts
(`artifacts/hybrid_v2_3_s44/00_phase0/approved_all_length_{train,dev}.jsonl`).

- `phase0_split_manifest_s44_20260720.jsonl` — one row per approved task:
  `{task_id, split: train|dev, family, in_long_dev_ge200}`.
  3,305 rows = 2,975 train / 330 dev. The 140 rows flagged
  `in_long_dev_ge200` are the ≥200-instruction eval slice used by the
  02a2/02a3 gates (all 140 are inside `dev`).
- Input corpus was `data/training/combined_fresh_s44_train_input.clean.jsonl`
  (3,306 rows); exactly 1 row was dropped at stage 00a
  (`--drop_invalid_references`: non-deterministic differential-oracle
  reference).
- `family` is best-effort from local archives: `master` (sigless_*),
  `topup_chatgpt_s46`, `topup_deepseek_s45`, `topup_batch_unknown`
  (fresh-eval-* rows whose generation batch archive is not local — the
  generator's records are authoritative for these; the split assignment is
  the contract, the family label is metadata).
- SHA-256 in `SHA256SUMS.txt`:
  `d69d7110a63d768207ec4eaf5bf03ce1afc0cb431326358c102fef8c40093258`.

Rule for the compact rebuild: codebook/materialization must fit on
`split=train` rows ONLY; `split=dev` is measure-only; the sealed evaluation
families (fresh_graphv2_holdout_s44, rebuilt 490-row functional eval, all
HumanEval scrub variants) remain forbidden and are NOT in this manifest at
all — anything not listed here is not trainable.
