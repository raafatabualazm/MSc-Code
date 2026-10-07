# Direct compact-Qwen Phase-0 s44 v3 build report

Date: 2026-07-22

## Verdict

The encoder-free x86-64 release is built, sealed, installed on the GPU pod,
and accepted by a real Qwen3-8B train/save/reload/generate smoke.  No full SFT
was launched.

Local release:

`scrubbed_master_v2_release/direct_compact_phase0_s44_pool_v3_release`

Pod release:

`/workspace/releases/direct_compact_phase0_s44_v3`

The release manifest SHA-256 is
`ce8fbabeaf6a2d748c1bd6ef7b571ecace5610a0e773a580ace32144798d6c66`.
The root `SHA256SUMS.txt` SHA-256 is
`128ff020d452e46315b709b4a0ad2725cfafadfa4504e50192f562ee7d8a5270`.
All 43 shipped files verified locally and again after transfer to the pod.

## Corpus and sealing

- Canonical input ledger: 3,306 rows.
- Included train: 2,951 rows.
- Included dev: 326 rows.
- Inherited quarantine: 14 rows.
- Inherited exclusion: 15 rows.
- Included model/supervised rows: 3,277.
- Original task IDs and Phase-0 train/dev membership are preserved.
- Public model rows contain exactly the four contract fields.
- Targets are joined privately, bijectively, and uniformly relabeled to
  `candidate`.
- The instruction codebook was fitted on train only.

## Representation and AOT enrichment

The release uses actual Dart 3.11.5 AOT binaries.  Their 3,277-row external
payload is 2,621,066,728 bytes and is not duplicated in the release; every
binary remains bound by its manifest SHA-256 and size.

The compact representation is lossless over the explicit scrubbed contract
domain: canonical graph v2 plus source-blind literal-pool values at retained
graph uses.  It does not claim losslessness over unreachable raw disassembly,
and the privacy scrub is intentional irreversibility.

Verified invariants:

- 3,277/3,277 graph, pool, and compact-ID round-trips.
- 511,678/511,678 regenerated DFG edges matched edge-for-edge.
- Explicit `call` CFG atom: 135 edges.
- 11,795 pool records and 16,690 supported graph use-sites preserved.
- 29,601 raw target xrefs reconciled.
- Zero truncation and zero unknown compact tokens.

The positional pool encoding is exact: fixed schema fields are contract
constants; finite enums use contract tags; offsets and indices use reversible
deltas; UTF-16 strings round-trip exactly, including unpaired surrogates.

## Token gate and generalization

Full graph-plus-AOT-pool source tokens:

- min 33, p50 407, p95 1,455, p99 2,651, max 8,791.
- 0/3,277 rows exceed the 9,000-token gate.

Instruction-codebook fallback:

- Train: 0/696,219 instructions.
- Dev: 401/69,878 = 0.5739%; 105/326 rows contain a fallback.
- Top-up aggregate: 117/200,482 = 0.05836%.
- Scrubbed HumanEval: 190/18,629 = 1.0199%, all reversible; 154/154
  canonical/DFG round-trips passed.

The HumanEval audit is deliberately labeled instruction-codebook-only because
its full v3 AOT-pool stream was not rebuilt in this release; it makes no 9,000
token claim for HumanEval.

## GPU acceptance smoke

Pod output:

`/workspace/artifacts/direct_compact_v3_sft_smoke2`

The smoke loaded pinned Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218` in BF16 with FlashAttention 2.
It trained LoRA r=16 plus the input-only compact source overlay for exactly two
steps:

- Loss: 1.123446 -> 0.821279; mean 0.9724.
- Gradient norm: 2.2952 -> 1.3886.
- Overlay: BF16 `[16289, 4096]`; 292 rows changed from initialization.
- LoRA: 43,646,976 parameters; all 252 B tensors became nonzero.
- Base LM head remained 151,936 rows.
- `graph_encoder=null`; `soft_prefix=null`.

Saved artifact bindings:

- Source overlay SHA-256:
  `b67a346b6751f03f0eec509703d37d39fa2c0d0441f5a65e223cc77cecfb90c3`.
- Decoder adapter tree SHA-256:
  `452d908b587bda22e6cebc4b1861c3c4862d2a5568bc08a10de988827fe320b1`.
- Sealed contract SHA-256:
  `4a310b98d6c1915bfb027c8da443ef2b9eb6a27eba5bd2044316413cd1c7d5e8`.

A separate reload gate then composed base Qwen -> PEFT adapter -> source
overlay and generated one cached token from a sealed dev row.  Its provenance
reconfirmed the dataset, alignment, contract, model revision, adapter, overlay,
LM-head size, and absence of an encoder/prefix.

## Bugs found and fixed during acceptance

1. A real nested Dart `Record` descriptor was missing from the finite
   fail-closed runtime-object allowlist.  The reviewed pair was added without
   weakening descriptor validation.
2. Verbose literal-pool JSON made eight rows exceed 9,000 tokens.  It was
   replaced with the exact positional encoding described above; max is now
   8,791.
3. The umbrella packager compared normalized supervised targets to raw labels
   with trailing newlines.  Its comparison now mirrors the private join's
   normalization.
4. Reload inference installed the source overlay after moving PEFT to CUDA,
   leaving its lookup buffer on CPU.  The buffer is now allocated beside the
   base embedding.  A device-placement regression test was added; cached
   generation passed after the fix.
5. Checkpoint saving reserialized only the trainer-consumed contract fields,
   omitting release-only provenance.  It now copies the validated sealed
   contract byte-for-byte and verifies the copied hash.

The direct compact path now has 41 passing focused tests.  The build/release
suite passed 79 tests (plus one Linux-only skip) before the final two trainer
fixes; those fixes are isolated to overlay device placement and exact contract
copying and are covered by the 41-test direct-path suite.

## Scope boundary

This release is x86-64 only.  It does not yet provide the requested exact
x86-64/ARM64 paired corpus.  The build removes the encoder-ignore failure mode,
but a full SFT and sealed functional evaluation are still needed to establish
decompilation capability; the two-step smoke proves integration, not quality.
