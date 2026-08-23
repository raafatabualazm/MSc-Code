# Attested multi-function Dart AOT extraction

`extract_dart_aot_user_function_bundle.py` fixes the exact-symbol projection
bug while keeping the public/model representation source-name-free.

There are two distinct privacy domains:

1. `build_dart_user_symbol_attestation.py` is a **private preparation step**.
   It reads the sealed `source_symbols` projection in the hash-pinned build
   inputs and emits only per-task, ordered, domain-separated HMAC-SHA256
   digests (`AF#` for declared functions and `T#` for declared types).
2. `extract_dart_aot_user_function_bundle.py` reads AOT/GDB output, that
   name-free attestation, and its private HMAC key. It never reads Dart source
   text. It uses keyed equality tests in memory and emits no raw attested
   source name, source path, or key.

Accordingly, public rows truthfully say `source_text_read: false` and
`source_symbol_attestation_used: true`. The full pipeline is not described as
source-blind: its private attestation preparation is source-derived.

## Selection and recursive recovery contract

For every AOT ELF, the extractor runs unfiltered GDB `info functions`, finds
the single `File ...:` section containing exact root `candidate`, and requires
every declaration in that section to parse. It disassembles every listed
function. Missing, duplicate, or unparseable declarations fail the row.

That GDB `File` section is authoritative for explicit top-level functions,
methods, local functions, and closures. Attestation does not recursively claim
an SDK call merely because its leaf collides with a source function name:
`_List.map` and bare `map` are not recovered from an attested source `map`.

Dart AOT omits some implicit symbols from the `File` table. The extractor
handles the two forms established by the sealed-corpus audit:

- `new T`: it is a user constructor only when the **immediate owner after
  `new`** exactly matches a type in that task's keyed attestation. A generic
  argument is not ownership evidence, so `new List<Dog>` is not claimed merely
  because source type `Dog` exists.
- `assert type is T`: exact base identifiers in the complete type expression
  are intersected with that task's attested types and replaced by `@T#`.
  Assertion stubs remain external transfers; the private source type name is
  never published.

User constructors are recursively disassembled by the direct call/jump
**operand address**, never by symbol name. The returned function entry must
equal that operand exactly. This is mandatory because GDB can contain duplicate
same-name functions (observed for `new StateError`), for which name lookup can
select the wrong body. Recursive discovery repeats to convergence and fails on
missing bodies, entry mismatch, identity collision, or a fixed safety limit.

Trusted SDK/runtime evidence is applied before attestation interpretation.
Other labels proven non-user by the complete per-task attestation receive a
distinct `X#` identity but a public `null` label. Indirect calls are expected
dynamic dispatch: their normalized operands are retained and they are
accounted as unresolved, not treated as missing functions.

Public IDs are:

- `F0`: exact root;
- `F1...`: every other explicit or recovered user function in entry-address
  order;
- `X0...`: external annotations in encounter order;
- `T0...`: name-free attested source-type aliases;
- `AF0...`: name-free ordered declared-function attestation aliases.

Only these external `symbol_class` values are emitted:

- `trusted_runtime`;
- `neutralized_untrusted_runtime`.

## Producer scaffold

The sealed AOT manifest must use
`phase0-s44-source-only-aot-row-v1` and bind valid analysis-program,
function-source, and producer-script hashes. The producer removes every gold
top-level `main` and appends exact `void main() {}`. If GDB exposes exact
`main`, the extractor disassembles and separately accounts for it, then excludes
that producer-owned scaffold from the model projection. It never excludes
`main.*`, candidate closures, helpers, constructors, instructions, or call
sites.

Every public function contains all machine instructions and bytes,
function-relative offsets, a non-pruning CFG, and an explicit function kind.
Every call instruction has an `interfunction_transfers` record, including
indirect and unlabelled direct calls.

## Sealed inputs

The intended 1,755-row scope is 1,580 training plus 175 held-out tasks:

```text
/workspace/multifunction_v1/input/aot_manifest_1755.jsonl
```

It must be an order-preserving, task-ID-exact projection of:

```text
/workspace/releases/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release/binary_build/aot_manifest.jsonl
SHA-256 e5bdb05eaf08281113298b9f37f26960a6f630b258de51eee92e672af722e633
```

External AOT payload root:

```text
/workspace/compact_qwen_pool_v3_full_dart3122_v1
```

Private source-only build inputs:

```text
/workspace/releases/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release/source_preparation/private_build_inputs/train.jsonl
SHA-256 a56b8fb9d8a26c872b4e26f28823cecef76e00cdb9dec08c87c72678c012ad1a

/workspace/releases/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release/source_preparation/private_build_inputs/dev.jsonl
SHA-256 1464004d3970b5b4eb5aecfe5a74a9f40f74de35e9a246af3521e33cb1ad17a8
```

Every manifest digest and AOT size is checked. Never synthesize AOT-row fields
from the compact corpus.

## Build the private keyed attestation

Set `SUBSET_SHA256` to the independently recorded SHA-256 of the exact
1,755-row subset manifest. Keep the binary key private and stable across smoke
and full extraction; changing it intentionally changes the attestation file.

```bash
cd /workspace/experiment_workspace

export SUBSET_MANIFEST=/workspace/multifunction_v1/input/aot_manifest_1755.jsonl
export SUBSET_SHA256='<recorded-64-lowercase-hex>'
export RELEASE=/workspace/releases/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release
export PRIVATE_ATTEST=/workspace/multifunction_v1/private_attestation_v1

test "$(sha256sum "$SUBSET_MANIFEST" | cut -d' ' -f1)" = "$SUBSET_SHA256"
test "$(wc -l < "$SUBSET_MANIFEST")" -eq 1755

install -d -m 700 "$PRIVATE_ATTEST"
if [ ! -e "$PRIVATE_ATTEST/hmac.key" ]; then
  umask 077
  head -c 32 /dev/urandom > "$PRIVATE_ATTEST/hmac.key"
fi
test "$(wc -c < "$PRIVATE_ATTEST/hmac.key")" -ge 32

python scripts/data/build_dart_user_symbol_attestation.py \
  --aot-manifest "$SUBSET_MANIFEST" \
  --aot-manifest-sha256 "$SUBSET_SHA256" \
  --train-build-input "$RELEASE/source_preparation/private_build_inputs/train.jsonl" \
  --train-build-input-sha256 a56b8fb9d8a26c872b4e26f28823cecef76e00cdb9dec08c87c72678c012ad1a \
  --dev-build-input "$RELEASE/source_preparation/private_build_inputs/dev.jsonl" \
  --dev-build-input-sha256 1464004d3970b5b4eb5aecfe5a74a9f40f74de35e9a246af3521e33cb1ad17a8 \
  --key-file "$PRIVATE_ATTEST/hmac.key" \
  --expected-rows 1755 \
  --output-jsonl "$PRIVATE_ATTEST/symbol_attestation_1755.jsonl" \
  --report "$PRIVATE_ATTEST/build_report.json"

export ATTESTATION_SHA256="$(
  sha256sum "$PRIVATE_ATTEST/symbol_attestation_1755.jsonl" | cut -d' ' -f1
)"
test "$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_jsonl_sha256"])' \
    "$PRIVATE_ATTEST/build_report.json"
)" = "$ATTESTATION_SHA256"
```

The builder fails unless the AOT/source rows align, content hashes match,
`source_symbols` exactly equals `transform_metadata.source_symbols`,
`candidate` appears exactly once, symbols are unique, every linked library URI
uses the `dart:` scheme (with no `part` or `export` directive), and the
producer/hash contract is valid. The output contains no raw name, source, path,
or key.

## One-task smoke

Use a fresh receipt directory. Old receipts lack the attestation binding and
must not be resumed.

```bash
export TASK_ID=sigless_eb2688fbe445
export SMOKE=/workspace/multifunction_v1/smoke_attested_v1
mkdir -p "$SMOKE/receipts"

python scripts/data/extract_dart_aot_user_function_bundle.py \
  --aot-manifest "$SUBSET_MANIFEST" \
  --aot-manifest-sha256 "$SUBSET_SHA256" \
  --aot-root /workspace/compact_qwen_pool_v3_full_dart3122_v1 \
  --symbol-attestation "$PRIVATE_ATTEST/symbol_attestation_1755.jsonl" \
  --symbol-attestation-sha256 "$ATTESTATION_SHA256" \
  --symbol-attestation-key-file "$PRIVATE_ATTEST/hmac.key" \
  --task-id "$TASK_ID" \
  --expected-rows 1 \
  --output-jsonl "$SMOKE/user_function_bundles.jsonl" \
  --receipt-dir "$SMOKE/receipts" \
  --report "$SMOKE/preflight.json" \
  --failures-jsonl "$SMOKE/failures.jsonl" \
  --gdb /usr/bin/gdb \
  --root-symbol candidate \
  --workers 1
```

The smoke report must show one extracted row, zero failures, and the exact
attestation file/key IDs.

## Full 1,755-row extraction

Do not reuse the failed pre-attestation receipts.

```bash
export EXTRACTION=/workspace/multifunction_v1/extraction_attested_v1
mkdir -p "$EXTRACTION/receipts"

python scripts/data/extract_dart_aot_user_function_bundle.py \
  --aot-manifest "$SUBSET_MANIFEST" \
  --aot-manifest-sha256 "$SUBSET_SHA256" \
  --aot-root /workspace/compact_qwen_pool_v3_full_dart3122_v1 \
  --symbol-attestation "$PRIVATE_ATTEST/symbol_attestation_1755.jsonl" \
  --symbol-attestation-sha256 "$ATTESTATION_SHA256" \
  --symbol-attestation-key-file "$PRIVATE_ATTEST/hmac.key" \
  --expected-rows 1755 \
  --output-jsonl "$EXTRACTION/user_function_bundles_1755.jsonl" \
  --receipt-dir "$EXTRACTION/receipts" \
  --report "$EXTRACTION/preflight_1755.json" \
  --failures-jsonl "$EXTRACTION/failures_1755.jsonl" \
  --gdb /usr/bin/gdb \
  --root-symbol candidate \
  --workers 8
```

The preflight must report:

- 1,755 extracted rows and zero failed rows;
- zero excluded user functions, instructions, and call sites;
- `selected_function_count == successfully_disassembled_function_count`;
- `gdb_file_function_count + attested_recursive_function_count ==
  successfully_disassembled_function_count`;
- exact-address recursive recovery and returned-entry verification enabled;
- no truncation.

This extraction phase does not itself claim training readiness. It emits the
canonical objects needed to fit the shared multi-function compact codebook.

## Mandatory encoded gate

The compact builder must emit one measurement per task:

```json
{
  "schema": "dart-aot-multifunction-encoded-measurement-v1",
  "task_id": "sigless_...",
  "model_projection_sha256": "<64 lowercase hex>",
  "student_tokens": 7123,
  "api_tokens": 10117
}
```

Finalize using the same manifest and attestation arguments plus:

```bash
--encoded-measurements /workspace/multifunction_v1/encoded/measurements_1755.jsonl \
--require-budget-measurements
```

Production readiness requires every row/hash measurement to match the exact
canonical projection, `student_tokens <= 9000`, and `api_tokens <= 12000`.
No partial final corpus is published when extraction or the encoded gate fails.

## Tests

```bash
python -m py_compile \
  scripts/data/build_dart_user_symbol_attestation.py \
  scripts/data/extract_dart_aot_user_function_bundle.py

python -m pytest -q \
  scripts/data/test_build_dart_user_symbol_attestation.py \
  scripts/data/test_extract_dart_aot_user_function_bundle.py

python scripts/data/extract_dart_aot_user_function_bundle.py --help
```

The focused tests cover keyed/order-bound and name-free attestations, incomplete
or wrong-key rejection, exact same-file selection, missing-body rejection,
non-pruning CFG/call accounting, producer-scaffold accounting, exact-address
constructor recovery, duplicate same-name functions, returned-entry mismatch,
aliased generic type assertions, `_List.map`/bare-`map` collisions,
`new List<Dog>` ownership rejection, source-path rejection, untrusted-runtime
neutralization, privacy residue checks, and standalone CLI import bootstrap.
