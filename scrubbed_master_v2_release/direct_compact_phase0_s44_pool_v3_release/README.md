# Encoder-free compact-Qwen Phase-0 s44 v3

This is the fail-closed, path-independent release of the direct compact-Qwen
v3 representation.  It preserves the canonical Phase-0 assignment exactly:
2,951 supervised training rows and 326 supervised development rows.  The
inherited reconciliation accounts for all 3,306 canonical input rows.

The model sees only the strict four-field compact input plus the supervised
Dart target after the private join.  Task IDs, family/source-pool metadata,
assembly, graphs, binary-pool receipts, tests, and join mappings remain in
private sidecars or seals.  The target function is uniformly `candidate`.

The compact stream is lossless over the contract's scrubbed graph-plus-pool
domain: canonical graph/pool and token-ID round-trips passed for all 3,277
model rows, DFG edges were regenerated and matched, call edges are explicit,
and no row exceeds the 9,000-source-token gate.

## AOT payload policy

The multi-gigabyte AOT/ELF payload is intentionally **not shipped**.  The
release includes the finalized binary-build seal and its complete 3,277-row
AOT manifest, which bind every external AOT by SHA-256 and byte size.  The
external payload totals 2621066728 bytes.

## Verification

Verify `SHA256SUMS.txt` recursively.  `release_manifest.json` records every
shipped payload with a release-relative path, SHA-256, size, and JSONL count.
Nested producer checksums remain included.  No absolute host path or timestamp
is used by the umbrella manifest.
