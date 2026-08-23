# Fixed Signature-Scrubbed Experiment (v3)

Status: queued on 2026-07-16 behind the unrelated ARM64 Regions16 job. The
frozen v1 result and the rejected v2 artifacts remain intact; the v2 GPU queue
is paused.

## Why v3 supersedes v2

The v1 all-zero result is authentic, but it combined removal of the semantic
name, return type, parameter types, and arity with a weaker name-only prompt
contract and the semantically loaded word `candidate`. The first v2 repair was
not a clean decomposition: it still used `candidate`, gave the two arms
different public permutations, did not preserve all graph metadata, exposed
linkable build fingerprints, and did not rescore the comparator under the
patched classifier and pinned SDK.

V3 makes the two new policy inputs a matched pair:

- target name: opaque `fn0` in source, tests, signatures, assembly, and prompt;
- name-only arm: no signature, type, parameter count, arity, or parameter names;
- neutral-exact arm: typed `fn0` signature with neutral parameter names;
- both arms: the same top-level/anti-`main` structural contract and the same
  assembly-via-graph marker;
- both arms: the same seed-4343 underlying task permutation;
- frozen binary evidence: `rename_frozen`, with exact instruction/edge parity
  modulo declared target/helper/path symbol rewrites;
- public IDs: high-entropy private salt, only its SHA-256 recorded;
- public rows: source, tests, evaluator signature, source/name fingerprints,
  and frozen pre-rename assembly hash removed.

The old report's statement that the 770-row SFT corpus used exact signatures
was incorrect. The corpus has 0 signature fields and 0 explicit signature-mode
fields; `name == main` on 741 rows, 29 other semantic names, and no `candidate`
or `fn0`. V3 is therefore an inference-contract ablation on a frozen
checkpoint, not an in-distribution signature-robustness estimate.

## Canonical v3 inputs

- `data/testing/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_{public,private}.jsonl`
- `data/testing/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_{public,private}.jsonl`
- `data/testing/fixed_scrub_v3_known_broken_tasks.json`
- `fixed_scrub_v3_build_sha256.txt`
- `fixed_scrub_v3_gpu_inputs.sha256`

Dataset SHA-256:

| Artifact | SHA-256 |
|---|---|
| opaque name-only private | `0fa377019ed558fc2fb2ebbb723690ca90ab2391b9e8bdaed48f184856fcae87` |
| opaque name-only public | `75c9a6c663c87c9f95ad3a98197abffba811debbb366eaa7966058ec3866a2b4` |
| opaque neutral-exact private | `b242c4897af7df335110e67ff6a7ed4e26ee7a301075cb5a449f7fad9dbff2e0` |
| opaque neutral-exact public | `4aff74b026c2ac015f69f7e79923dbf8557decb0e7b0dd5cd9b8e80eb1a8171c` |
| 56-file GPU input manifest | `918e1cae955b0cffec762dfb59fe2990308110b78d50127cc8277e2bd8bd3428` |
| frozen Regions16 checkpoint | `e8e872608f22ae8e1c5607d6179feeb5f133401fb2a7d1fc40fe8894d8c347fc` |

## Prompt-control proof

Prompt schema: `antigravity-v3-matched-function-contract`.

Expected policy prompt-stream SHA-256 values, independently rendered over the
154 public rows and enforced after inference:

- opaque name-only: `e0c13c8169598851eb3363728bf792adb2389e420e3b9af6dd98834288ead622`
- opaque neutral-exact: `5c4ae01d07e3ff962fc1c555ed1463365fca614bf4316cfb096fe96f99b64971`

The current exact-signature renderer over the frozen comparator dataset
recomputes to `55adb80e8a24df956c82c2eed260523a2f6c1b6e00a566bfb7b269c7eab75d0d`,
exactly matching the frozen comparator's recorded prompt stream. The comparator
candidate pool is therefore inference-prompt valid; it is rescored, not
regenerated.

## Acceptance gates

Pinned build box: `167.172.150.125`, Dart 3.11.5, GDB 17.1, extractor digest
`3b522afc7ea9d24440c4ed0e1bafd2c4a047bb76f0f592560451b61e10c2613d`.

`scripts/evaluation/fixed_scrub_v3_gates.py` passes all static and executable
gates:

- 154 rows per side, 0 rejects, one shared underlying public permutation;
- exact public redaction and no target/helper semantic-name residue;
- instructions equal modulo declared symbols and edges exactly equal for
  154/154 tasks;
- matched rendered prompt structure and pinned prompt schema;
- complete Dart/GDB/extractor assertions;
- patched `Crash when compiling` front-end classifier;
- contract stubs compile 150/154;
- neutral references compile 151/154 and pass 142/154.

The faithful staging-only Linux overlay verifies all 56 manifest entries,
passes shell syntax validation, and passes 106/106 unit and protocol tests.
The frozen comparator is also checked by a dedicated verifier that binds its
154x10 candidate pool, benchmark dataset, checkpoint, historical renderer,
inference source, model revisions, and recorded prompt-stream digest.

The four exact inherited contract defects are benchmark task IDs 121, 127,
153, and 161. Reports must include all 154 tasks and the valid-150 sensitivity.

## GPU queue and scoring

Pod: `ssh -p 24424 root@98.218.15.126`.

Queue script:
`/workspace/fixed_scrub_v3_staging/scripts/evaluation/run_fixed_scrub_v3_gpu.sh`.
It was relaunched after the final staging preflight as PID 2286577 with status
`/workspace/results/fixed_scrub_v3_queue.status`. While ARM64 is running it
does not deploy or overwrite canonical sources. After ARM64 exits it requires
at least 60 GiB free VRAM, verifies the 56-file staging manifest, deploys those
exact files, verifies them again, and runs:

1. unit/static preflight and frozen-comparator input verification;
2. opaque name-only inference, prompt/load/gate verification, private join,
   primary metrics, candidate-level stats, and provenance;
3. opaque neutral-exact through the identical chain;
4. frozen comparator rescore with the same patched classifier and the same
   pinned Dart 3.11.5 SDK;
5. a secondary candidate-only legacy wrapped standalone-AOT acceptance@1/5/10
   diagnostic on all three frozen candidate pools; each projection strips
   tests, references, signatures, graph metadata, and semantic task IDs while
   proving the ordered candidate stream is unchanged;
6. fail-closed paired analysis (all 154 primary plus valid-150 sensitivity)
   and final SHA-256 manifest.

The pod's system Dart 3.12.2 was not replaced. The official 3.11.5 SDK is
installed alongside it at `/workspace/toolchains/dart-sdk-3.11.5`, exposed to
the evaluators through `/home/zeus/dart-sdk`, and its release ZIP verifies as
`57f3ab5ac24883060b1ff12bcdac472ed76563ec7364e88f8a6d41e4f0db075f`.

Monitor without mutating the run:

```bash
ssh -p 24424 root@98.218.15.126 \
  'cat /workspace/results/fixed_scrub_v3_queue.status; \
   cat /workspace/results/*sigscrub_v3*.status 2>/dev/null'
```

Because `/workspace` is not a host volume on this Vast instance, all result,
log, stats, provenance, and manifest files must be synced locally before the
instance is recycled.

## Interpretation boundaries

- The established “aligned JIT compile@k” primary metric is test-linked JIT
  front-end acceptance@k: it compiles the candidate together with hidden
  callers. It measures top-level symbol/caller compatibility as well as
  classifier-defined front-end acceptance; it is neither exact ABI recovery
  nor standalone syntax compilation.
- The separately reported legacy wrapped standalone-AOT acceptance@k receives
  only an opaque candidate-only projection. It is diagnostic because its AOT
  wrapper and normalization differ from the primary harness; subtracting the
  two metrics is not a pure estimate of interface cost.
- The two v3 arms use the same row order and seed, but the generator seeds only
  once. Autoregressive draw consumption can diverge with output length, so do
  not claim candidate-level common random numbers. Aggregate pass@k and
  source/test-digest-paired task outcomes remain valid, with Monte Carlo noise.
- The public IDs and order no longer provide a trivial lookup, but frozen
  binary content remains linkable to a party holding the original benchmark.
  V3 is safe for this closed no-tools policy run; it is not cryptographic
  anonymization for retrieval-capable external baselines.
