# Arm C typed folded-union plus gold replay preregistration

Status: fixed before the Arm B seed-42 evaluation completed. At fixation, Arm B
generation was still in progress and no Arm B score file or promotion decision
existed.

## Question and claim

Arm C tests the practical replay-assisted recipe against Arm B. It does not
isolate gold content alone: matching optimizer-update count requires doubling
gradient accumulation, which also doubles effective batch size and halves the
direct-target share of each epoch's examples. A duplicated-direct GA16 control
would be required for a causal claim about gold content alone.

## Fixed training contract

- Fresh branch from the two-epoch typed SFT checkpoint `optstep348`; never warm
  start from Arm B or a stacked RS-SFT checkpoint.
- Reuse Arm B's exact sealed union of 458 verified direct targets.
- Add exactly 458 gold-replay targets from the clean typed TRAIN universe.
- One epoch, batch size 1, gradient accumulation 16, learning rate `5e-6`, zero
  warmup, zero weight decay, seed 42, and the same LoRA/base/tokenizer lineage.
- Total rows: 916. Planned optimizer updates: `ceil(916 / 16) = 58`, equal to
  Arm B.
- Model-visible input remains only the opaque typed contract plus F2 text.
  Acceptance tests, compiler diagnostics, held-out content, semantic names, and
  semantic parameter names remain hidden.
- Checkpoint interval 20 (updates 20, 40, and 58) is an operational storage
  choice and does not alter optimization.

## Replay selection

Selection is independent of Arm B outputs and of all labels/evaluation evidence.

1. Load the sealed clean typed TRAIN universe (2,775 rows after the known
   contaminant exclusion).
2. Exclude all 458 Arm B direct task identities.
3. Exclude every candidate whose byte-identical typed source hash occurs in the
   Arm B direct union. Arm B has 457 distinct source hashes; ID-only exclusion
   would leave two aliases of direct model-visible inputs.
4. Deterministically order remaining candidates using only the sealed Arm C
   dataset schema, seed, `gold_replay` kind, task ID, and source SHA-256. Do not
   use target text/hash, private tests, B predictions, or B scores in selection.
5. Admit at most one row per typed source SHA-256 and take the first 458 rows.
6. Assert exactly 458 unique replay task IDs, 458 unique replay source hashes,
   zero replay/direct task overlap, zero replay/direct source-hash overlap, and
   zero held-out/known-contaminant overlap.

On the sealed corpus, the pool contains 2,315 rows after direct-ID and
direct-source exclusions; one remaining duplicate-source pair must be reduced to
one eligible source identity before taking 458 rows.

## Evaluation and decisions

- Evaluate the frozen Arm C checkpoint on the same ordered 175 held-out tasks,
  seed 42, temperature 0.8, top-p 0.95, and K=10.
- Primary metric: pass@10. Report pass@1, compile@10, and mean distinct programs
  per task as diagnostics.
- Seed 42 may veto or trigger replication but can never promote a checkpoint.
- Mean distinct programs below 9.50 is collapsed. At least 9.90 is required for
  replication eligibility.
- Seed-42 pass@10 of at least 18/175 together with diversity of at least 9.90
  triggers matched seeds 43 and 44.
- Promotion requires at least three matched seeds, with Arm B evaluated at the
  same seeds. Seeds are repeated measurements, not independent tasks, and must
  not be reported as pass@30.
- VeRPO remains held until the matched-seed decision is sealed.

## Operational gate

Arm C must remain stopped until Arm B's supervisor has exited and Arm B's stable
single-seed audit exists with status `pass`, promotion on hold, and VeRPO on
hold. Cleanup of superseded checkpoints is a separate explicit operation; the
Arm C launcher must fail closed on inadequate free storage rather than deleting
artifacts automatically.
