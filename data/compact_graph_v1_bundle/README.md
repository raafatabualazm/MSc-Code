# Compact CFG+DFG conversion

This bundle replaces verbose textual `cfg` and `edges` fields with `compact_graph_v1`.
It is intended for a structured graph/instruction encoder, **not** for direct JSON prompting.

## What is preserved

- Every normalized assembly instruction.
- Opcode and typed operands.
- Register operands.
- Immediate values as integer side features.
- Memory width, base, index, scale and displacement.
- Direct call/runtime symbols.
- Basic-block boundaries and block types.
- Every control-flow edge and edge type.
- Every original block-level data-flow edge.
- A SHA-256 digest of the source CFG+DFG.

Redundant text was removed: block labels, repeated predecessor/successor fields, addresses,
instruction counts, JSON property names per node, and repeated edge descriptions.
Branch addresses are replaced with block targets.

## Model-position accounting

Two supported layouts are reported per sample:

- `positions_flat = instructions + operands + blocks + 1`
- `positions_pooled = instructions + blocks + 1`

In pooled mode, the operand fields are embedded locally and pooled into the opcode/instruction
vector. CFG and DFG edges are sparse tensors and consume no sequence positions.

Across the 2,211 rows:

- 682 original graph serializations exceed 9,000 tokens using a CL100K proxy.
- Only one row exceeds 9,000 positions in flat opcode+operand mode.
- No row exceeds 9,000 positions in pooled-instruction mode.
- Maximum pooled length is 4,276 positions.

The 2,089-instruction row `sigless_452c11877e17` changes from 77,596 proxy text tokens to:

- 6,054 flat structured positions.
- 2,476 pooled structured positions.

The largest 3,570-instruction row changes from 194,632 proxy text tokens to:

- 10,297 flat positions.
- 4,276 pooled positions.

## Schema

Compact keys are intentionally short because the JSON is storage/interchange only:

- `n`: instruction count.
- `b`: basic-block count.
- `bp`: block pointer array into `i`.
- `bt`: block-type IDs.
- `i`: instruction records: `[opcode_id, operand, ...]`.
- `c`: control edges: `[source_block, target_block, edge_type_id]`.
- `d`: data-flow edges: `[source_block, target_block]`.

Operand records:

- `[0, register_id]`
- `[1, immediate_value]`
- `[2, width_bytes, base_register_id, index_register_id, scale, displacement]`
- `[3, symbol_id]`
- `[4, target_block_id]`
- `[5, raw_operand_id]`

Use `compact_graph_v1_tools.py` to validate rows and create columnar arrays or PyTorch tensors.

## Recommended encoder use

1. Embed opcode, registers, operand kind, memory width, symbol and block type.
2. Encode numeric immediates/displacements with a signed logarithmic or byte-level numeric encoder.
3. Pool each instruction's operand embeddings into its opcode vector.
4. Run sparse relational message passing using `c` and `d`.
5. Feed instruction/block memory to the decoder through persistent cross-attention rather than input-only prefix concatenation.

The token report uses CL100K only as a reproducible proxy for the old text serialization. The
new structured position counts are tokenizer-independent.
