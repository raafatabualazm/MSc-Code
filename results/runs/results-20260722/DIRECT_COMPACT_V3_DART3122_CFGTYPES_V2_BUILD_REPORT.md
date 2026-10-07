# Direct compact-Qwen Phase-0 s44: Dart 3.12.2 build report

Date: 2026-07-22

## Verdict

The encoder-free compact-Qwen Phase-0 s44 corpus has been rebuilt from actual
Dart 3.12.2 x86-64 AOT binaries, compacted losslessly, privately joined to the
training labels, audited, sealed, copied back to the local workspace, and
mirrored to the GPU pod. The original Dart 3.11.5 release was not modified.

All 3,277 model rows pass canonical graph-plus-pool and compact-ID round-trip
checks. All 515,077 DFG edges and all 225,100 CFG edges are accounted for.
There is no truncation, no unknown compact token, and no row above the 9,000
source-token gate. A two-step Qwen3-8B no-encoder smoke then trained, saved,
reloaded, and generated through the cached inference path successfully.

This is a release/integration result, not a decompilation-quality result. A
full SFT run and compile/pass evaluation were deliberately not started as part
of this validation.

## Release identity

| Item | Value |
|---|---|
| Local release | `scrubbed_master_v2_release/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release` |
| Pod mirror | `/workspace/releases/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release` |
| Release manifest SHA-256 | `5d527dbe733df1a98fe55bd3d2d7e6403d465627674b27ba20dc596129cab8f1` |
| Root `SHA256SUMS.txt` SHA-256 | `00a5ad10cb343ce2489d1ab5e6d069b4507921720bcba9111e658d90efc2db94` |
| Verified root entries | 43/43; all six nested seals also verify (35/35 entries) |
| Sealed archive SHA-256 | `d82017851b6a1bc52a65be5443db77cc6dd9416596eeae80505701c1e46be59b` |
| Compact contract SHA-256 | `bdfc16373fd55b708a6edb8210067238faeb975bd921c86016cbe9a74ab02dda` |
| Codec SHA-256 | `b30f531761fbc0497e20eea8873f787bbfe3fa8e9fbdbc5ad97c8ad78e55325e` |
| Train-only codebook SHA-256 | `5197f9007c001686572d1efbcc672a455d29c4917538f30ba2255ea8ef7b591f` |

The release is path-independent. Its multi-gigabyte AOT payload is deliberately
not duplicated into the release, but all 3,277 external binaries are bound by
the shipped AOT manifest, byte lengths, and SHA-256 values.

## Corpus and split parity

The inherited reconciliation accounts for every one of the 3,306 canonical
input rows:

| Canonical status | Rows |
|---|---:|
| Included train | 2,951 |
| Included dev | 326 |
| Quarantined by the inherited canonical preparation | 14 |
| Excluded by the inherited canonical preparation | 15 |
| Total | 3,306 |

This reconciles the 3,305 rows present in the supplied Phase-0 manifest plus
the one explicitly excluded non-manifest input row. The supplied 140-row
long-dev flag resolves to 138 retained development rows plus two inherited
quarantines.

The 3.12.2 binary and compact stages introduced zero additional quarantines or
failures. The 3,277 included rows retain their original task IDs, order, split,
and family metadata:

| Family | Rows |
|---|---:|
| `master` | 2,172 |
| `topup_s45` | 994 |
| `topup_s46` | 111 |

The private supervised join is bijective and hash-bound:

| Split | Rows | Joined file SHA-256 | Join-seal SHA-256 |
|---|---:|---|---|
| Train | 2,951 | `8341723ae0a74e3ea70ffc2650b4d9d2bc8a39fb9753f6d91c2381cdcd35fba9` | `678ee933cb384375d2e7a5ad566023e9e47f136159ff55e463817371d3ca2e33` |
| Dev | 326 | `3daea1653debd6b748913a7b0f37e23e0b9670c6af3d038374f981f9b4bb82ba` | `5a757707c924e120a2750ae2bad5efb8883c56660969b942a98cce17d6f9b018` |

Only the strict four-field compact inputs are public to the model. Labels,
task IDs, family/source-pool metadata, graphs, assembly, binary receipts, and
alignment information stay in private files or seals. All targets are named
`candidate` under `runtime-symbol-policy-v1`.

## Dart 3.12.2 toolchain and AOT validation

The build used the official stable Linux x64 Dart SDK:

```text
Dart SDK version: 3.12.2 (stable) (Tue Jun 9 01:11:39 2026 -0700) on "linux_x64"
```

The SDK archive SHA-256 is
`28e47b44cf075f36771046c068bb0d174201cf9c7608744aed1cc23204299c2d`.
The shipped toolchain manifest has SHA-256
`7a4e2d16214d4934c7d97c81239b35b66d6b0df64a5fb8bbdf2fbd380888c677`
and pins Dart, `dartaotruntime`, `gen_snapshot`, `gen_kernel`, the platform
DILL, GDB 15.1, and GNU binutils 2.42 by resolved path, version, and hash.

Four AOT/runtime layout canaries spanning composites, nested records, strings,
large integers, doubles, and null values passed. Static and runtime pool-pointer
offset sets agreed exactly, every target xref was reconciled, and every receipt
was bound to the 3.12.2 runtime and toolchain manifests. The corpus contains no
top-level Boolean pool entry, so that exact layout case was not exercised by a
canary.

Full binary-stage totals:

| Measure | Count |
|---|---:|
| AOT payload bytes | 2,699,635,168 |
| Exact target xrefs | 30,310 |
| Supported/represented literal xrefs | 16,943 |
| Pool projection records | 11,952 |
| Excluded non-graph xrefs | 162 |

The binary-build seal SHA-256 is
`f44f4d574aec0dc6faa465fd3514d594cc1b7b72536b6300278c151c62879c15`;
the complete AOT manifest SHA-256 is
`e5bdb05eaf08281113298b9f37f26960a6f630b258de51eee92e672af722e633`.
For clarity, the compact contract's historically named
`aot_manifest_sha256` field contains the finalized binary-build-seal hash
(`f44f4d...`), while the literal row-level AOT manifest is `e5bdb05e...`.

## Lossless compression and the 9,000-token gate

The first 3.12.2 materialization exposed one honest failure in the old explicit
CFG-triple representation: `sigless_f17b4d628f1b` required 10,490 tokens. Its
cost was 3,569 instruction atoms, 816 block atoms, 3,642 explicit CFG tokens,
2,453 pool tokens, and 10 control markers. This was not an OOV problem.

The row was not dropped or truncated, and the gate was not raised. The codec
was versioned with
`cfg_encoding=inline-source-implicit-next-fallthrough-targets-v2`:

- CFG atoms are emitted under their source block, making the repeated source
  atom implicit.
- Only next-block `conditional_false` and `linear_fallthrough` destinations are
  implicit, after proving this invariant over the complete corpus.
- Calls keep an explicit call-edge atom and destination.
- Conditional-true, jump, and loop-back destinations remain explicit.
- Decoding reconstructs the canonical ordered edge list and fails closed on
  any mismatch.

This preserves modeling-visible branch structure while removing redundant
serialization. The existing `<G2C3>` stream marker and v3 trainer schema remain
for compatibility; exact semantics are identified by the codec SHA and the
`cfg_encoding` contract field, not by the marker alone.

Preflight results:

| Invariant | Result |
|---|---:|
| Exact graph-plus-pool round trips | 3,277/3,277 |
| Exact compact-ID stream round trips | 3,277/3,277 |
| DFG regeneration and match | 3,277 rows / 515,077 edges |
| CFG edges preserved | 225,100 |
| Explicit call edges preserved | 135 |
| Pool records / use sites | 11,952 / 16,943 |
| Unknown tokens | 0 |
| Truncated rows | 0 |
| Compact failures / quarantines | 0 / 0 |

Token lengths are min 29, p50 335, p95 1,195, p99 2,267, and max 8,684.
All 3,277 rows fit the 9,000-token limit.

The compression is therefore lossless over the contract's stated scrubbed
canonical graph-plus-source-blind-pool domain. The privacy scrub remains the
only intentional irreversibility; the codec does not claim recovery of names
or other information removed before encoding.

## Generalization audits

The instruction codebook was fit on the 2,951 train rows only. Dev and all
forbidden/evaluation families were excluded from fitting.

| Audit | Fallback occurrences | Rate | Rows with fallback |
|---|---:|---:|---:|
| Dev | 410 / 70,135 | 0.5846% | 110 / 326 |
| Dev `master` | 291 / 49,575 | 0.5870% | 79 / 219 |
| Dev `topup_s45` | 115 / 19,738 | 0.5826% | 29 / 100 |
| Dev `topup_s46` | 4 / 822 | 0.4866% | 2 / 7 |
| Top-up families across train+dev | 119 / 200,437 | 0.0594% | 31 / 1,105 |
| Scrubbed HumanEval | 517 / 18,629 | 2.7752% | 131 / 154 |

Fallback instructions use exact native-Qwen token sequences between reserved
raw markers, so fallback remains reversible rather than lossy. For HumanEval,
all 154 canonical/DFG round trips and all 13,616 regenerated DFG edges matched;
all 517 fallback occurrences round-tripped exactly.

The HumanEval result is an instruction-codebook coverage audit only. A complete
3.12.2 HumanEval binary-pool stream was not built here, so this report does not
claim a full-v3 HumanEval token count or a HumanEval 9,000-token-gate result.

## Tests and encoder-free GPU smoke

Local validation passed:

- Build/release/toolchain/codec suites: 83 passed, 1 skipped.
- Direct compact trainer-path suites: 41 passed.
- Final targeted inline-CFG codec suite after assertion cleanup: 8 passed.
- Final local umbrella checksum verification: 43 passed, 0 failed.
- Pod mirror checksum verification: 43 passed, 0 failed.

The bounded GPU smoke used Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, BF16, FlashAttention 2, LoRA
rank 16/alpha 32, batch size 1, and two optimizer steps. Its two training
losses were 1.1789911 and 0.8378679.

Provenance explicitly records:

```text
architecture = qwen-causal-compact-tokens-no-encoder
graph_encoder = null
soft_prefix = null
lm_head_rows = 151936
source_embedding_overlay_rows = 16512
```

The source overlay shape was checked as `[16512, 4096]` in BF16. Training,
adapter save, overlay save, reload, cache use, and a one-row/one-token measure
generation all completed with matching contract, codebook, codec, overlay, and
adapter hashes. This directly checks that the new compact stream is the sole
source-conditioning path; there is no graph encoder or injected soft prefix to
ignore.

The local smoke folder contains the small provenance/state/checksum metadata.
The adapter and overlay weight bytes remain on the pod at
`/workspace/artifacts/direct_compact_v3_dart3122_cfgtypes_v2_sft_smoke2` and
are represented locally by their recorded hashes, not by copied weight files.

## Scope and next scientific gate

- This release is x86-64 only. It does not yet provide the requested paired
  ARM64 representation of the same functions.
- The smoke proves the data/model integration path, not compile@k, pass@k, or
  semantic decompilation quality.
- A representation-only comparison must rebuild every comparison arm from the
  exact same Dart 3.12.2 AOT/graph artifacts. Historical 3.11.5 metrics are not
  a clean comparator for this release.
- The next meaningful action is the full sealed Phase-0 encoder-free SFT,
  followed by frozen scrubbed-HumanEval and fresh-holdout evaluation with the
  same 3.12.2 binary/codec contract.
