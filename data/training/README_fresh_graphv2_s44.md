# Fresh Graph V2 S44 training inputs

These files preserve the sealed 500-row evaluation split while making the
leakage-clean remainder available to Hybrid Training Patch v2.3.

## Training inputs

- `scrubbed_master_v2_fresh_s44_train_input.jsonl`: holdout-bound copy of the
  private scrubbed master, with one alpha-structural duplicate removed.
- `fresh_graphv2_extra_s44_train_input.jsonl`: unused fresh candidates that
  passed exact, alpha-structural, name, and fuzzy source-overlap controls.

Both files are raw Phase-0 inputs. They are not approved training outputs until
`prepare_hybrid_training_data_antigravity.py` replays their references and
writes a passing Phase-0 report.

## Frozen evaluation input

Use `../testing/fresh_graphv2_holdout_s44.jsonl` only as `--forbidden_eval`.
Never include it in `--input`.

Once the fresh extras are used for training, describe the 500 rows as a
same-corpus held-out evaluation split. Do not describe them as an independently
fresh holdout for that trained model.

## Phase-0 command

Run from `hybrid_training_patch_v2_3`:

```powershell
python -m scripts.training.prepare_hybrid_training_data_antigravity `
  --input "../data/training/scrubbed_master_v2_fresh_s44_train_input.jsonl,../data/training/fresh_graphv2_extra_s44_train_input.jsonl" `
  --forbidden_eval "../data/testing/fresh_graphv2_holdout_s44.jsonl" `
  --output "../data/prepared/fresh_s44/approved_all_length_train.jsonl" `
  --dev_output "../data/prepared/fresh_s44/approved_all_length_dev.jsonl" `
  --short_output "../data/prepared/fresh_s44/approved_short.jsonl" `
  --bridge_output "../data/prepared/fresh_s44/approved_bridge.jsonl" `
  --long_output "../data/prepared/fresh_s44/approved_long.jsonl" `
  --rejected_output "../data/prepared/fresh_s44/phase0_rejections.jsonl" `
  --report "../data/prepared/fresh_s44/phase0_report.json" `
  --seed 44
```

The split provenance and output hashes are recorded in
`fresh_graphv2_train_inputs_s44.manifest.json`. Rejected source rows remain in
their immutable source pools; the rejection ledger contains only provenance and
match evidence.
