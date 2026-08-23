# Exact neutral-source x86-64/AArch64 corpus

## Status

The four-pair pilot is complete. The same byte-exact neutral Dart program was
compiled once for Linux x64 and once for Linux arm64 under Dart 3.11.5. The
pilot produced eight valid Graph-v2.1 rows (four per ISA) with the recorded
`7a89b10f...` CFG/DFG family.

Pilot artifacts:

- Public paired rows: `scrubbed_master_v2_release/exact_cross_arch_pilot_v1/paired_public.jsonl`
- Strict model rows: `scrubbed_master_v2_release/exact_cross_arch_pilot_v1/model_public.jsonl`
- Public manifest: `scrubbed_master_v2_release/exact_cross_arch_pilot_v1/public_manifest.json`
- Private alignment/build evidence: `scrubbed_master_v2_release/exact_cross_arch_pilot_v1/private/`

The public leakage gate passed, all 18 transferred artifact hashes verify, all
eight stored DFGs regenerate exactly, all four programs pass Dart JIT and x64
AOT execution, and all four arm64 AOT snapshots compile and contain an exact
`candidate` symbol. ARM64 runtime execution is recorded as not run because the
CPU builder currently has neither a native ARM host nor QEMU user-mode runtime.

## Why the historical Flutter ARM64 pool is not merged

The old Flutter pool has 1,712 byte-identical *original* sources in the x86
synthetic pool, and 1,690 of those sources survive in the scrubbed x86 release.
That is useful private linkage, but it is not an exact scrubbed architecture
pair: the Flutter binaries were built from named sources, whereas the new x86
release was rebuilt from `candidate`-neutral sources. The original Flutter/APK
builder, source manifest, APK/libapp artifacts and complete Flutter/Dart/NDK
toolchain stamps are also absent.

Do not merge the historical ARM train/eval files into the direct-Qwen corpus or
claim them as same-neutral-source pairs.

## Pair contract

One private neutral source and test harness form an immutable `program.dart`.
The release stores the source/harness without its fixed compile envelope, so
the builder deterministically restores `dart:async` and `dart:convert` exactly
as `build_scrubbed_dataset.py` did. This is fixed infrastructure, not inferred
task content. Missing this envelope is fail-closed: an initial full-run attempt
published zero pairs when async harness symbols did not resolve.
The program includes both retention pragmas on `candidate`:

```dart
@pragma('vm:never-inline')
@pragma('vm:entry-point')
```

The exact same program SHA-256 must appear for both architectures. A salted
HMAC of the neutral source gives an opaque `semantic_pair_id`; the salt remains
private. All pair members must be assigned to the same dataset split.

`paired_public.jsonl` contains the opaque pair ID, architecture, neutral target,
assembly, CFG, DFG and graph integrity. `model_public.jsonl` is the stricter
four-field allowlist used before compact encoding:

```text
architecture, assembly, cfg, edges
```

Source, tests, typed signature, release row identity, build commands, true
addresses, AOT files, hashes and pair salt are confined to `private/`.

## Reproduce the pilot on the Linux CPU builder

The builder VM is `root@167.172.150.125`. It has Dart 3.11.5 and the x64/arm64
snapshot compilers cached. Stage these files under one workspace root:

```text
scripts/data/build_exact_cross_arch_pairs.py
scripts/data/build_graph_v2_jsonl.py
scripts/data/cfg_extractor.py
scripts/data/dfg_extractor.py
scrubbed_master_v2_release/master_dart_graphv2_signature_scrubbed_private.jsonl
```

Then run:

```bash
/workspace/.venv/bin/python scripts/data/build_exact_cross_arch_pairs.py \
  --private-input scrubbed_master_v2_release/master_dart_graphv2_signature_scrubbed_private.jsonl \
  --indices 508,509,510,511 \
  --output-dir exact_cross_arch_pilot_v1 \
  --dart /opt/dart-3.11.5/dart-sdk/bin/dart \
  --dartaotruntime /opt/dart-3.11.5/dart-sdk/bin/dartaotruntime \
  --readelf readelf \
  --objdump-x64 objdump \
  --objdump-arm64 aarch64-linux-gnu-objdump \
  --expected-dart-version 3.11.5
```

The builder fails closed on the wrong Dart version, missing retention pragmas,
missing tests, missing candidate symbols, graph failures, unrecognized public
symbol annotations, or public source/signature/task residues.

The pilot selector remains intentionally limited to 2-8 inline indices. The
same builder now also supports full-corpus operation through a frozen indices
file, deterministic shard slicing, per-pair dual-ISA receipts, `--resume`, and
private quarantine records. A pair receipt is committed only after JIT, x64
AOT execution, both AOT compilations, both symbol extractions, and both graphs
succeed. Therefore an ARM failure can never publish an x64-only pair. Final
JSONLs and manifests are atomically replaced, their checksums are re-read, and
`COMPLETE` is written last. A shard with any quarantined row has no `COMPLETE`
marker and exits nonzero, while successful pair receipts remain resumable.

## Full frozen split launch (do not exceed two workers)

The frozen direct split contains 1,975 train rows and 219 dev rows. Its
alignment files use one-based `original_line`; select them only with the
explicit `alignment-jsonl` mode, which converts to zero-based release indices
and verifies unique contiguous `split_line` values. Static preflight of all
2,194 selected rows found:

- all sources/tests present;
- both retention pragmas on every neutral `candidate`;
- no `main` left in a source and one test-driver `main` on every row;
- 2,194 unique neutral sources and byte-exact programs (no pair-ID collision).

There are 31 train shards and 4 dev shards at 64 pairs per shard. Generate one
private shared salt once, then launch train and dev sequentially so that no
more than two Dart compilers run concurrently:

```bash
set -euo pipefail

ROOT=/root/exact_cross_arch_pilot
OUT=/root/exact_cross_arch_full_v1
PYTHON="$ROOT/.venv/bin/python"
BUILDER="$ROOT/scripts/data/build_exact_cross_arch_pairs.py"
PRIVATE="$ROOT/input/master_dart_graphv2_signature_scrubbed_private.jsonl"
SPLITS="$ROOT/input/direct_compact_split_v1"
SALT="$OUT/private/shared_semantic_pair_salt"

if [ ! -x "$PYTHON" ]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install --no-input networkx==3.6.1
fi
test "$("$PYTHON" -c 'import networkx; print(networkx.__version__)')" = 3.6.1
mkdir -p "$OUT/private"
if [ ! -s "$SALT" ]; then
  umask 077
  head -c 32 /dev/urandom > "$SALT"
fi

run_pair_shard() {
  split="$1"
  shard="$2"
  shard_name=$(printf '%03d' "$shard")
  "$PYTHON" "$BUILDER" \
    --private-input "$PRIVATE" \
    --indices-file "$SPLITS/${split}_private_alignment.jsonl" \
    --indices-file-format alignment-jsonl \
    --shard-size 64 \
    --shard-index "$shard" \
    --output-dir "$OUT/$split/shard-$shard_name" \
    --pair-salt "$SALT" \
    --dart /opt/dart-3.11.5/dart-sdk/bin/dart \
    --dartaotruntime /opt/dart-3.11.5/dart-sdk/bin/dartaotruntime \
    --readelf /usr/bin/readelf \
    --objdump-x64 /usr/bin/x86_64-linux-gnu-objdump \
    --objdump-arm64 /usr/bin/aarch64-linux-gnu-objdump \
    --expected-dart-version 3.11.5 \
    --resume
}
export -f run_pair_shard
export ROOT OUT PYTHON BUILDER PRIVATE SPLITS SALT

printf '%s\n' {0..30} | xargs -P2 -I{} bash -c 'run_pair_shard train "$1"' _ {}
printf '%s\n' {0..3}  | xargs -P2 -I{} bash -c 'run_pair_shard dev "$1"' _ {}
```

The split alignment files and current builder/extractor files must first be
staged on the VM with their frozen hashes. `--resume` is safe on a new output
directory and skips a pair only after verifying its receipt contract, source
and program hashes, both graph records, and stored AOT hashes. A corrupt or
stale receipt is moved to private quarantine and rebuilt.

Do not merge shard JSONLs merely because the processes exited. Require exactly
35 `COMPLETE` files, zero non-empty `private/quarantine.jsonl` files, and a
successful `sha256sum -c private/SHA256SUMS.txt` from every shard. Merge in
global `pair_slot` order with x86-64 before AArch64, then rerun the public
allowlist/leakage gate and verify exactly two architecture rows per pair.

The four-pair pilot used 21.88 seconds in the eight AOT compiler calls and
about 29 seconds end to end, or roughly 7.3 seconds per semantic pair. That
projects to about 4.4 hours sequential. With at most two workers, allow roughly
2.5-4 hours for the 2,194-row corpus, plus retries for unusually large rows.
The paired AOT artifacts are expected to use about 3.8-4.2 GB; the VM currently
has 149 GB free. ARM64 remains compile- and extraction-verified, not
behavior-verified, until a pinned native/QEMU runtime is added.

## Scale-up gates

1. Freeze the private semantic split first. Both ISAs for a pair, plus all exact
   and near-source variants, remain in one split. HumanEval scrubbed and fresh
   s44 hashes remain forbidden from training.
2. Compile every retained neutral program for `x64` and `arm64` with the pinned
   Dart SDK. Do not mix Dart 3.11.5 and 3.12.2 artifacts.
3. Require JIT and x64 AOT behavior to pass. Before calling the full ARM corpus
   behavior-verified, add a native ARM64 worker or pinned QEMU/sysroot and run
   every ARM AOT harness too.
4. Extract only the exact top-level `candidate` body by the recorded
   largest-size/lowest-address rule. Keep raw AOT and true addresses private;
   rebase the public candidate entry to `0x100000` and zero external absolute
   destinations.
5. Rebuild both graphs with the same `7a89b10f...` CFG/DFG family and require
   exact DFG regeneration. The older release DFG is invalid for ARM64 (only
   4/1,714 historical rows reproduce).
6. Extend the compact codec with explicit `<AX64>` and `<AA64>` markers and a
   fail-closed 67-opcode AArch64 allowlist. Fit ISA-scoped instruction entries
   on training rows only. The held-out ARM audit already shows that 1,024 ARM
   codebook entries give p50 784, p95 2,806 and max 5,210 source tokens, with
   zero rows over 9,000.
7. Run correct-source versus matched-permuted/null NLL and free-generation
   gates. Direct source tokens remove the ignored encoder prefix, but they do
   not by themselves prove the decoder is behaviorally source-conditioned.

## Flutter-specific continuation

If the final corpus must be actual Flutter/Android rather than paired Linux
Dart AOT, recreate the missing build pipeline instead of reusing the old pool.
Compile each *same neutral source artifact* for Android x64 and Android arm64
under one pinned Flutter/Dart/Android SDK/NDK revision, retain the exact build
commands and `libapp.so` hashes, extract `candidate` from both binaries, and use
the same private pair/split contract above.
