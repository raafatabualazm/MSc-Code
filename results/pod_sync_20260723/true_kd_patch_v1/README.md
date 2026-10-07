# True distribution KD patch

This patch separates two mathematically different objectives and records which
one was used in checkpoint provenance.

- `dense_full_kl` is true full-distribution KD. A frozen local teacher and the
  student must have byte-identical tokenizer JSON, identical vocabulary IDs and
  output-vocabulary size, and the same sealed direct-compact input contract. It
  computes `KL(teacher || student)` over every vocabulary item at every target
  position, including EOS, with temperature-squared (`T^2`) scaling.
- `sparse_topk_tail_kl` is a lower-bound/coarsened objective for future API
  collections. It uses each observed top-k probability plus one aggregate tail
  event. It never renormalizes top-k as if the tail were zero and cannot use a
  temperature other than 1.

API top-5 responses can never enter `dense_full_kl`. The teacher for dense mode
must be a **local compact-conditioned checkpoint**: it needs a decoder adapter
and source-token embedding overlay trained against the same compact contract.
A generic Qwen base model or an API teacher cannot understand the custom compact
source IDs.

## Current Qwen artifact decision

The remote artifact
`/workspace/artifacts/compact_fn0_rebuild/verified_repairs_lp.jsonl` has 48 rows
for a 1,580-row train set. Its converted `softkd_repairs.jsonl` has 47 rows. The
48-row file has 13,997 teacher positions, but stores only rounded top-5 decoded
strings. It omits token bytes/IDs, immutable model and tokenizer provenance,
explicit tail mass, and an EOS distribution. Retokenizing the strings happens
to produce one student ID for 13,996/13,997 observed positions, but that does not
prove the API used the same IDs. Rounding even makes the inferred tail negative
at many positions.

Therefore:

- do not use either file for dense or sparse KD;
- do not run `/workspace/soft_kd_trainer.py` again;
- retain `verified_repairs_lp.jsonl`, because all 48 verified `code` values remain
  useful hard targets for an RS-SFT join by `task_id`.

The captured decision is machine-readable in
`reports/verified_repairs_lp.audit.json`.

## Audit before training

After copying `true_kd_patch_v1` to `/workspace/true_kd_patch_v1`, Opus should run
the audit-only launcher (it never starts training):

```bash
bash /workspace/true_kd_patch_v1/run_true_kd.sh audit-legacy
```

The report must say both KD compatibility flags are `false`. That is expected;
the code targets are being retained, not deleted.

For a future sparse artifact, run the fail-closed validator:

```bash
/venv/main/bin/python \
  /workspace/true_kd_patch_v1/scripts/data/validate_qwen_kd_artifacts.py \
  sparse-validate \
  --input /workspace/artifacts/kd/sparse_topk_tail.jsonl \
  --manifest /workspace/artifacts/kd/sparse_topk_tail.manifest.json \
  --student_tokenizer_json \
    /workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json \
  --contract /workspace/artifacts/compact_fn0_rebuild/fn0_contract.json
```

The exact manifest and row layouts are in `examples/`. The manifest seals the
data, source dataset, contract, collector, immutable teacher identity, identical
teacher/student tokenizer hashes, vocabulary size, EOS, probability temperature,
and absence of logprob rounding.

## Dense full-KL launch

Pick a genuinely stronger local compact-conditioned checkpoint as
`TEACHER_CHECKPOINT` and a different checkpoint as
`STUDENT_WARMSTART_CHECKPOINT`. The exact teacher model and immutable revision
are required rather than guessed. Using the same adapter and overlay for teacher
and student is rejected because it starts with exactly zero KL.

```bash
cd /workspace
TEACHER_CHECKPOINT=/workspace/artifacts/LOCAL_STRONG_COMPACT_TEACHER \
TEACHER_MODEL=Qwen/Qwen3-8B \
TEACHER_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
STUDENT_WARMSTART_CHECKPOINT=/workspace/artifacts/STUDENT_WARMSTART \
OUTPUT_DIR=/workspace/artifacts/direct_compact_dense_full_kd_v1 \
bash /workspace/true_kd_patch_v1/run_true_kd.sh dense
```

For a one-optimizer-step smoke run, add `MAX_STEPS=1` while keeping a distinct
output directory. Both launchers use `set -Eeuo pipefail`, and the Python trainer
exits before training if any seal, hash, tokenizer, vocabulary, EOS, warm-start,
or teacher contract check fails.

Before loading either large model, validate the real checkpoint pair:

```bash
TEACHER_CHECKPOINT=/workspace/artifacts/LOCAL_STRONG_COMPACT_TEACHER \
STUDENT_WARMSTART_CHECKPOINT=/workspace/artifacts/STUDENT_WARMSTART \
bash /workspace/run_true_kd.sh checkpoint-preflight
```

Direct-compact checkpoints reserialize `compact_contract.json`, and
`save_pretrained` can rewrite non-vocabulary tokenizer decoder defaults.
Checkpoint binding therefore compares canonical contract semantics and the
contract's embedded original `tokenizer_json_sha256`; it does not incorrectly
require those saved copies to have identical file bytes. The teacher and
student tokenizers actually loaded for KD still must have identical
vocabulary/special-token mappings, and the supplied source tokenizer JSON files
must be byte-identical.

The launcher uses:

- `project_root=/workspace/direct_compact_stage/hybrid_training_patch_v2_3`;
- all 1,580 sealed `train_fn0_real.jsonl` rows;
- the exact Qwen3-8B tokenizer revision;
- full-vocabulary target-position logits in chunks to cap peak LM-head memory;
- a frozen 4-bit teacher and trainable 4-bit LoRA student by default;
- EOS supervision, warm-start loading, strict checkpoint contract/tokenizer
  checks, correct final partial gradient accumulation, periodic checkpoints, and
  SHA-256 provenance.

Quantizing the teacher means the student matches the complete distribution of
that explicitly recorded quantized teacher. Remove `--teacher_load_4bit` in a
custom invocation when memory permits and an unquantized teacher distribution is
desired.

Run tests before spending GPU time:

```bash
/venv/main/bin/python -m unittest discover \
  -s /workspace/true_kd_patch_v1/tests -v
```
