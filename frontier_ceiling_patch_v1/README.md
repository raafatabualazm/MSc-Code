# Audited frontier pass@k runner

This patch replaces the unauditable raw-disassembly script with a fail-closed
runner. Its default `compact` arm evaluates the teacher on the same information
available to the student:

- the pinned `compact_input_ids`;
- the frozen instruction codebook expanded back to normalized instructions;
- the compact CFG rendered explicitly;
- the exact binary-constant prefix that `build_real_enriched.py` prepended.

The expansion is necessary for a black-box API: provider models do not know the
student's private custom-token IDs. F2 uses tokenizer-proved, visible one-token
task-local symbols for block, instruction, and nonnested sequence references;
compact x86 aliases; explicit typed CFG edges; and only unambiguous
CFG-reconstructed terminal branch targets. Constants are UTF-8 byte-length
framed. The encoder decodes every F2 prompt back to the exact constant prefix,
instruction order, block membership, and ordered CFG before returning it. No
row is truncated or silently excluded.

For the matched multi-function binary representation, blocks from F0 and all
producer-attested user helpers/closures are concatenated with global block IDs.
The first instruction of each function entry is a boundary marker:
`fn @SELF` identifies F0 and `fn @U<n>` identifies helper `@U<n>`. Calls use
the same aliases globally. SDK/runtime calls remain a separate `@X<n>` domain;
the exact source-blind X dictionary (including explicit redaction class when a
label is not trusted) is carried in the byte-framed binary-enrichment prefix.
F2 serializes and decodes these markers, embedded aliases, the whole ordered
instruction stream, and every offset CFG edge before a prompt is accepted.

## What is fixed

- The common 175-task dev JSONL and `real_constants.jsonl` are SHA-256 pinned.
  Passing the v3 training cohort through the default command fails before an
  API call.
- Contract, codebook, codec, tokenizer, per-row hashes, compact text, compact
  IDs, and the reconstructed student constant prefix are all verified.
- Prompts are token-counted with the sealed Qwen tokenizer plus an explicit chat
  reserve. An over-limit prompt aborts the entire preflight; no input is
  truncated.
- Every task must obtain exactly K valid API responses. API errors, refusals,
  empty content, unsafe harness-termination code, zero/missing usage, and any
  `finish_reason` other than `stop` do not count toward K.
- There is no early stopping after a passing sample.
- Every prompt, raw response, reasoning field, candidate, token-usage record,
  harness outcome, model identity, source/test hash, and artifact hash is
  persisted.
- The only accepted evaluator entrypoint is
  `evaluate_dart_jit_tests_detail` from the package-owned hardened evaluator.
  Its source hash and pinned Dart-binary hash are persisted. The runner requires
  completion-attestation contract
  `per-run-256-bit-marker-exactly-once-v1`; a return code alone can never pass.
- Every acceptance-test `main` is checked for attestation compatibility during
  preflight. A missing Dart executable or evaluator exception aborts the run
  instead of becoming a false model failure.
- A completed result is emitted only if every selected task has exactly K valid
  completions. Otherwise the run writes `failure.json` and exits non-zero.
- Candidates are evaluated twice by default. A compile/pass counts only when
  both executions agree.
- `raw` and `raw_constants` are optional controls over the same pinned cohort.
  They disassemble every `fn0`-family symbol, use a source/compiler/GDB-bound
  cache, and never character-truncate. They are not described as
  compression-only comparisons.

The previous `5/60` result cannot be upgraded into a frontier ceiling: its
candidate responses were discarded, valid-sample counts are unknowable, and
its raw input was truncated. This patch produces a new auditable measurement.

## Files

- `frontier_core.py` — importable verified serializer, response validation,
  budget accounting, raw-control cache, and persistence primitives.
- `frontier_f2.py` — lossless semantic F2 encoder/decoder and the single
  exported cross-pipeline grammar prompt.
- `frontier_passk.py` — drop-in runner accepting the old environment variables.
- `serialize_compact_inputs.py` — materializes verified readable inputs for a
  Qwen black-box sequence-KL sampler.
- `tests/test_frontier_core.py` — compact round-trip, hash, safety, response,
  token, and budget tests.
- `install_remote_fixes.sh` — install helper. It does not start, stop, or kill
  any job.
- `REAL175_PREFLIGHT_REPORT.json` — hash-bound full-cohort preflight evidence.

## Install on the pod

Run only after deciding how to handle an already-armed legacy queue:

```bash
cd /workspace/frontier_ceiling_patch_v1
bash install_remote_fixes.sh
```

The installer backs up an existing `/workspace/frontier_passk.py`, installs both
Python modules and the serializer, then runs `py_compile`. It does not launch an
evaluation.

## Preflight first

```bash
cd /workspace
EVAL_PATH=hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py
EVAL_SHA="$(sha256sum "$EVAL_PATH" | awk '{print $1}')"
DART_PATH=dart-3.12.2/usr/bin/dart
DART_SHA="$(sha256sum "$DART_PATH" | awk '{print $1}')"
PROVIDER=qwen MODEL=qwen3.8-max-preview K=10 MAXTOK=8192 \
  EXPECTED_EVALUATOR_SHA256="$EVAL_SHA" \
  EXPECTED_DART_SHA256="$DART_SHA" \
  python frontier_passk.py --preflight-only
```

The default primary inputs are pinned to:

- dataset SHA-256
  `a4ed1cf185d52c3d212e2d7348fdb2a1dffd0035f4c395e2e897fd072fa70001`;
- constants SHA-256
  `ec9b7086f03f1099cee31903cb4933c326df4f39160cd6820ebc47cd94860b13`;
- expected dataset size: 175.

The real 175-row release preflight passes with 0 truncations and a 256-token
chat reserve: min 676, p50 2,566, p90 5,107, p95 6,987, p99 11,095, max
11,988. One constants-extractor error is surfaced for
`sigless_d544567b9189`; its exact student prefix is still reproduced and
verified. No paid API request was made during preflight.

For any intentionally different cohort, both
`--expected-dev-sha256 <reviewed digest>` and `--expected-task-count <n>` must be
provided. A different cohort is a separate experiment.

## Primary runs

Qwen:

```bash
cd /workspace
EVAL_PATH=hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py
EVAL_SHA="$(sha256sum "$EVAL_PATH" | awk '{print $1}')"
DART_PATH=dart-3.12.2/usr/bin/dart
DART_SHA="$(sha256sum "$DART_PATH" | awk '{print $1}')"
PROVIDER=qwen MODEL=qwen3.8-max-preview K=10 WORKERS=10 MAXTOK=8192 \
  EXPECTED_EVALUATOR_SHA256="$EVAL_SHA" \
  EXPECTED_DART_SHA256="$DART_SHA" \
  python frontier_passk.py
```

DeepSeek:

```bash
cd /workspace
EVAL_PATH=hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py
EVAL_SHA="$(sha256sum "$EVAL_PATH" | awk '{print $1}')"
DART_PATH=dart-3.12.2/usr/bin/dart
DART_SHA="$(sha256sum "$DART_PATH" | awk '{print $1}')"
PROVIDER=deepseek MODEL=deepseek-v4-pro K=10 WORKERS=10 MAXTOK=12000 \
  EXPECTED_EVALUATOR_SHA256="$EVAL_SHA" \
  EXPECTED_DART_SHA256="$DART_SHA" \
  python frontier_passk.py
```

`MAXTOK=8192` and `MAXTOK=12000` are both accepted. A response that reaches its
cap with `finish_reason=length` is saved as an invalid attempt and retried; it
can never count toward K.

## Output contract

Each unique run directory contains:

- `provenance.json`: exact config, file hashes, Python/API/evaluator/Dart
  identity, and completion-attestation contract;
- `tasks.jsonl`: source/test/input hashes and raw-control provenance;
- `prompts.jsonl`: exact messages and preflight token counts;
- `attempts.jsonl`: all valid and invalid API attempts, including full response,
  content, reasoning content, candidate, finish reason, usage, and resolved
  model;
- `outcomes.jsonl`: each candidate's repeated completion-attested harness
  results, evaluator hash, and attestation identity;
- `summary.json`: pass@K/compile@K, Wilson intervals, task results, and usage;
- `manifest.json`: final hashes of all run artifacts;
- `failure.json`: fail-closed diagnostic when no valid summary can be emitted.

Anthropic runs add `anthropic_native_stop_report` and
`capability_metric_assessment` to progress and metric summaries. The former
counts provider-native `stop_reason` values and explicit `stop_details`
categories (for example, `refusal:cyber`; missing provider detail is reported
as `refusal:unspecified`). The latter reports non-refusal slot/task coverage.
The sealed pass/compile denominator is never changed, but any provider refusal
makes that execution result a lower bound rather than a capability ceiling;
refusal-dominated runs are explicitly marked `invalid_refusal_dominated` with
`ceiling_claim_allowed=false`. The compatibility `finish_reason` remains
`content_filter` for native Anthropic refusals.

## Reuse from a sequence-KL sampler

`prepare_api_readable_compact` is the stable helper requested for a black-box
Qwen sampler:

```python
from pathlib import Path
from frontier_core import (
    COMPACT_F2_SYSTEM_PROMPT,
    CompactArtifactBundle,
    prepare_api_readable_compact,
)

bundle = CompactArtifactBundle(
    contract_path=Path("/workspace/artifacts/compact_fn0_rebuild/fn0_contract.json"),
    codebook_path=Path("/workspace/direct_compact_stage/scrubbed_master_v2_release/"
                       "direct_compact_split_v1/compact_qwen_confirmatory_v1/codebook.json"),
    tokenizer_path=Path("/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/"
                        "b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"),
    codec_path=Path("/workspace/direct_compact_stage/scripts/data/"
                    "build_compact_qwen_v1.py"),
    constants_path=Path("/workspace/artifacts/compact_fn0_rebuild/real_constants.jsonl"),
    expected_constants_sha256=(
        "ec9b7086f03f1099cee31903cb4933c326df4f39160cd6820ebc47cd94860b13"
    ),
)

verified = prepare_api_readable_compact(bundle, dataset_row)
messages = [
    {"role": "system", "content": COMPACT_F2_SYSTEM_PROMPT},
    {"role": "user", "content": verified["text"]},
]
```

For bulk materialization:

```bash
python serialize_compact_inputs.py \
  --dataset /workspace/artifacts/compact_fn0_rebuild/train_fn0_real.jsonl \
  --expected-dataset-sha256 REVIEWED_TRAIN_SHA256 \
  --expected-rows 1580 \
  --expected-constants-sha256 \
    ec9b7086f03f1099cee31903cb4933c326df4f39160cd6820ebc47cd94860b13 \
  --max-prompt-tokens 12000 \
  --chat-overhead-reserve 256 \
  --out /workspace/artifacts/compact_fn0_rebuild/train_api_readable_f2.jsonl
```

The bulk command writes a v2 sidecar manifest containing the exact F2 system
prompt, its SHA-256, representation schema, pinned tokenizer hash, and maximum
complete-prompt count. It refuses any row whose sealed representation or
constant prefix cannot be reproduced exactly, or whose complete prompt plus
reserve exceeds the configured limit. Every consumer must use the manifest
prompt verbatim and reject a hash/schema mismatch.
