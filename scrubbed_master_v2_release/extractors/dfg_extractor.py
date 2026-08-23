
"""
Lightweight DFG extraction.

Two layers:

1. `LightweightDFGExtractor` — the ORIGINAL intra-block def-use chain builder
   (x86 mov/arith/cmp only, no register aliasing). Kept byte-identical because
   its node COUNT feeds the legacy GraphCodeBERT token appendage
   (GRAPH_DFG_MODE=legacy); changing it would silently change the block
   tensors that old checkpoints were trained on.

2. Module-level functions (`instruction_def_use`, `block_def_use`,
   `build_cross_block_dfg`) — the WIRED dataflow channel
   (GRAPH_DFG_MODE=edges). Dual-ISA (x86-64 Intel-syntax + AArch64
   llvm-objdump), canonical register aliasing (eax/ax/al -> rax, wN -> xN),
   and a block-granularity reaching-definitions pass that emits typed
   'dataflow' edges (def block -> use block) for the CFG GNN.

Known limitations (both layers): float/SIMD registers (xmm*, v*/d*/s*),
flags, and memory locations are not tracked.
"""

from __future__ import annotations

import os
import re
import sys


class LightweightDFGExtractor:
    def __init__(self):
        self.register_regex = re.compile(
            r'\b([re]?(ax|bx|cx|dx|si|di|bp|sp)|r[0-9]{1,2})\b',
            re.IGNORECASE,
        )

    def extract_block_dfg_structured(self, instructions):
        dfg_edges = []
        dfg_nodes = []
        instruction_to_nodes = {}
        last_defined_by = {}

        for instruction_index, instruction in enumerate(instructions):
            instruction = instruction.replace(',', ' ')
            parts = instruction.split()

            if not parts:
                continue

            mnemonic = parts[0].lower()
            operands = parts[1:]

            reads = []
            writes = []

            if mnemonic in ['mov', 'lea', 'movzx', 'movsx']:
                if len(operands) >= 1:
                    writes.extend(self.register_regex.findall(operands[0]))

                if len(operands) >= 2:
                    reads.extend(self.register_regex.findall(' '.join(operands[1:])))

            elif mnemonic in ['add', 'sub', 'xor', 'and', 'or', 'imul']:
                if len(operands) >= 1:
                    destination = self.register_regex.findall(operands[0])
                    writes.extend(destination)
                    reads.extend(destination)

                if len(operands) >= 2:
                    reads.extend(self.register_regex.findall(' '.join(operands[1:])))

            elif mnemonic in ['cmp', 'test']:
                reads.extend(self.register_regex.findall(' '.join(operands)))

            normalized_reads = [r[0].lower() if isinstance(r, tuple) else r.lower() for r in reads]
            normalized_writes = [r[0].lower() if isinstance(r, tuple) else r.lower() for r in writes]

            for register in normalized_reads:
                if register in last_defined_by:
                    source_node = f"{register}_{last_defined_by[register]}"
                    target_node = f"{register}_{instruction_index}"

                    if source_node not in dfg_nodes:
                        dfg_nodes.append(source_node)

                    if target_node not in dfg_nodes:
                        dfg_nodes.append(target_node)

                    dfg_edges.append((source_node, target_node))

            touched_nodes = []

            for register in normalized_writes:
                node_name = f"{register}_{instruction_index}"

                if node_name not in dfg_nodes:
                    dfg_nodes.append(node_name)

                touched_nodes.append(node_name)
                last_defined_by[register] = instruction_index

            for register in normalized_reads:
                node_name = f"{register}_{last_defined_by.get(register, instruction_index)}"

                if node_name not in dfg_nodes:
                    dfg_nodes.append(node_name)

                touched_nodes.append(node_name)

            instruction_to_nodes[instruction_index] = touched_nodes

        return {
            'nodes': dfg_nodes,
            'edges': dfg_edges,
            'instruction_to_nodes': instruction_to_nodes,
        }


# ---------------------------------------------------------------------------
# Wired dataflow channel (GRAPH_DFG_MODE=edges): dual-ISA def-use analysis.
# ---------------------------------------------------------------------------

def _build_x86_alias_map():
    alias = {}
    for canon, subs in {
        'rax': ('eax', 'ax', 'al', 'ah'),
        'rbx': ('ebx', 'bx', 'bl', 'bh'),
        'rcx': ('ecx', 'cx', 'cl', 'ch'),
        'rdx': ('edx', 'dx', 'dl', 'dh'),
        'rsi': ('esi', 'si', 'sil'),
        'rdi': ('edi', 'di', 'dil'),
        'rbp': ('ebp', 'bp', 'bpl'),
        'rip': ('eip',),
    }.items():
        alias[canon] = canon
        for sub in subs:
            alias[sub] = canon
    # rsp/esp/spl fold to rsp; bare 'sp' stays its own name because it is also
    # the AArch64 stack pointer (16-bit x86 sp never appears in these dumps).
    alias['rsp'] = 'rsp'
    alias['esp'] = 'rsp'
    alias['spl'] = 'rsp'
    for n in range(8, 16):
        base = f'r{n}'
        alias[base] = base
        for suffix in ('d', 'w', 'b'):
            alias[f'{base}{suffix}'] = base
    return alias


_X86_ALIAS = _build_x86_alias_map()

# One scan pattern for both ISAs. AArch64 zero registers (xzr/wzr) are matched
# so they can be EXCLUDED (reads are constant zero, writes are discarded).
_REGISTER_SCAN = re.compile(
    r'\b(?:'
    r'r(?:[89]|1[0-5])[dwb]?'                      # x86 r8-r15 (+32/16/8-bit views)
    r'|[er]?(?:ax|bx|cx|dx|si|di|bp|sp|ip)'        # x86 legacy family
    r'|sil|dil|bpl|spl'
    r'|[xw](?:30|[12][0-9]|[0-9])'                 # aarch64 x0-x30 / w0-w30
    r'|xzr|wzr|sp|fp|lr'
    r')\b',
    re.IGNORECASE,
)


def _canonical_register(name):
    name = name.lower()
    if name in ('xzr', 'wzr'):
        return None
    if name in _X86_ALIAS:
        return _X86_ALIAS[name]
    if name == 'fp':
        return 'x29'
    if name == 'lr':
        return 'x30'
    if name and name[0] in ('x', 'w') and name[1:].isdigit():
        return f'x{int(name[1:])}'
    return name  # 'sp' and anything else already canonical


def _scan_registers(text):
    return [
        canon
        for canon in (_canonical_register(m) for m in _REGISTER_SCAN.findall(text))
        if canon is not None
    ]


def _split_operands(text):
    """Split an operand string on top-level commas ('[sp, #-16]!' stays whole)."""
    operands, depth, current = [], 0, []
    for ch in text:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == ',' and depth == 0:
            operands.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        operands.append(''.join(current).strip())
    return [op for op in operands if op]


# dst-first: destination written, remaining operands read.
_X86_MOV_LIKE = {'mov', 'movzx', 'movsx', 'movsxd', 'movabs', 'lea', 'movbe'}
# destination is read AND written (read-modify-write).
_X86_RMW = {'add', 'sub', 'xor', 'or', 'and', 'imul', 'adc', 'sbb',
            'shl', 'shr', 'sar', 'rol', 'ror', 'not', 'neg', 'inc', 'dec'}
_X86_READ_ONLY = {'cmp', 'test', 'push'}
_X86_ZERO_IDIOMS = {'xor', 'sub'}

_ARM_DST_WRITE = {'mov', 'movz', 'movn', 'mvn', 'neg', 'ngc',
                  'sxtw', 'sxth', 'sxtb', 'uxtw', 'uxth', 'uxtb',
                  'rev', 'rev16', 'rev32', 'clz', 'rbit',
                  'add', 'adds', 'adc', 'sub', 'subs', 'sbc',
                  'and', 'ands', 'orr', 'orn', 'eor', 'eon', 'bic', 'bics',
                  'lsl', 'lsr', 'asr', 'ror', 'lslv', 'lsrv', 'asrv', 'rorv',
                  'mul', 'mneg', 'smull', 'umull', 'smulh', 'umulh',
                  'sdiv', 'udiv', 'adr', 'adrp',
                  'csel', 'csinc', 'csinv', 'csneg', 'cset', 'csetm',
                  'cinc', 'cinv', 'cneg', 'ubfx', 'sbfx', 'ubfiz', 'sbfiz', 'bfi', 'bfxil'}
_ARM_DST_RMW = {'movk'}
_ARM_MADD = {'madd', 'msub', 'smaddl', 'smsubl', 'umaddl', 'umsubl'}
_ARM_LOAD = {'ldr', 'ldur', 'ldrb', 'ldrh', 'ldrsb', 'ldrsh', 'ldrsw', 'ldursb', 'ldursh', 'ldursw', 'ldurb', 'ldurh', 'ldar', 'ldaxr', 'ldxr'}
_ARM_LOAD_PAIR = {'ldp', 'ldpsw', 'ldnp'}
_ARM_STORE = {'str', 'stur', 'strb', 'strh', 'sturb', 'sturh', 'stlr', 'stxr', 'stlxr'}
_ARM_STORE_PAIR = {'stp', 'stnp'}
_ARM_READ_ONLY = {'cmp', 'cmn', 'tst', 'ccmp', 'ccmn', 'cbz', 'cbnz', 'tbz', 'tbnz', 'prfm'}
_ARM_ZERO_IDIOMS = {'eor', 'sub'}


def _memory_writeback_regs(operands):
    """Base registers written back by AArch64 pre-index ('[sp, #-16]!') or
    post-index ('[x1], #8' -> split keeps '[x1]' and '#8' separate) addressing."""
    written = []
    for index, op in enumerate(operands):
        if op.endswith('!') and '[' in op:
            base = _scan_registers(op)
            if base:
                written.append(base[0])
        elif op.startswith('[') and index + 1 < len(operands) and operands[index + 1].startswith('#'):
            base = _scan_registers(op)
            if base:
                written.append(base[0])
    return written


# Mnemonics that exist only on AArch64 (used to disambiguate shared names
# like add/sub/mov, whose operand semantics differ between the two ISAs).
_ARM_ONLY = (
    (_ARM_DST_WRITE - _X86_RMW - _X86_MOV_LIKE)
    | _ARM_DST_RMW | _ARM_MADD | _ARM_LOAD | _ARM_LOAD_PAIR
    | _ARM_STORE | _ARM_STORE_PAIR
    | (_ARM_READ_ONLY - _X86_READ_ONLY)
)

_ARM_REG_HINT = re.compile(r'\b(?:[xw]\d{1,2}|xzr|wzr)\b', re.IGNORECASE)


def instruction_def_use(instruction):
    """(reads, writes) canonical register lists for one x86-64/AArch64 instruction.

    Memory is not tracked: registers inside a memory operand are address READS,
    and a store 'writes' nothing (its data operand is a read).
    """
    text = instruction.strip()
    if not text:
        return [], []
    # Drop symbol annotations and trailing comments before scanning: the
    # '<foo+0x12>' text and '// =42' notes contain no live registers.
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'//.*$', '', text)
    parts = text.split(None, 1)
    mnemonic = parts[0].lower()
    operand_text = parts[1] if len(parts) > 1 else ''
    operands = _split_operands(operand_text)

    # ISA dispatch: AArch64 register names, ARM-only mnemonics, or the
    # 3-operand arithmetic form ('add sp, sp, #0x10') mark an ARM instruction.
    is_arm = bool(
        _ARM_REG_HINT.search(operand_text)
        or mnemonic in _ARM_ONLY
        or (len(operands) >= 3 and mnemonic in _ARM_DST_WRITE)
    )

    reads, writes = [], []

    def dst_regs():
        if not operands:
            return []
        if '[' in operands[0]:
            reads.extend(_scan_registers(operands[0]))  # store-to-memory: address regs are reads
            return []
        return _scan_registers(operands[0])

    def src_regs(from_index=1):
        for op in operands[from_index:]:
            reads.extend(_scan_registers(op))

    if is_arm:
        if mnemonic in _ARM_DST_RMW:
            destination = dst_regs()
            writes.extend(destination)
            reads.extend(destination)
            src_regs()
        elif mnemonic in _ARM_DST_WRITE:
            if (
                mnemonic in _ARM_ZERO_IDIOMS
                and len(operands) == 3
                and operands[1] == operands[2]
                and '[' not in operands[1]
            ):
                writes.extend(dst_regs())  # eor x0, x1, x1: dst def, no live read
            else:
                writes.extend(dst_regs())
                src_regs()
        elif mnemonic in _ARM_MADD:
            writes.extend(dst_regs())
            src_regs()
        elif mnemonic in _ARM_LOAD:
            writes.extend(dst_regs())
            src_regs()
            writes.extend(_memory_writeback_regs(operands))
        elif mnemonic in _ARM_LOAD_PAIR:
            if operands and '[' not in operands[0]:
                writes.extend(_scan_registers(operands[0]))
            if len(operands) > 1 and '[' not in operands[1]:
                writes.extend(_scan_registers(operands[1]))
            src_regs(from_index=2)
            writes.extend(_memory_writeback_regs(operands))
        elif mnemonic in _ARM_STORE or mnemonic in _ARM_STORE_PAIR:
            for op in operands:
                reads.extend(_scan_registers(op))
            writes.extend(_memory_writeback_regs(operands))
        elif mnemonic in _ARM_READ_ONLY:
            for op in operands:
                reads.extend(_scan_registers(op))
        else:
            return [], []
    elif mnemonic in _X86_MOV_LIKE or mnemonic.startswith('cmov'):
        writes.extend(dst_regs())
        src_regs()
    elif mnemonic in _X86_RMW:
        if (
            mnemonic in _X86_ZERO_IDIOMS
            and len(operands) == 2
            and operands[0] == operands[1]
            and '[' not in operands[0]
        ):
            writes.extend(_scan_registers(operands[0]))  # xor rax, rax: pure def
        else:
            destination = dst_regs()
            writes.extend(destination)
            reads.extend(destination)
            src_regs()
    elif mnemonic in _X86_READ_ONLY:
        for op in operands:
            reads.extend(_scan_registers(op))
    elif mnemonic == 'pop':
        writes.extend(dst_regs())
    else:
        # Branches, calls, ret, nop, unrecognized: no tracked dataflow.
        return [], []

    return reads, writes


def block_def_use(instructions):
    """(defs, upward_exposed_uses) register sets for one basic block.

    upward-exposed = read before any in-block definition, i.e. the reads this
    block exposes to reaching definitions from predecessor blocks.
    """
    defs, upward_exposed = set(), set()
    for instruction in instructions or []:
        reads, writes = instruction_def_use(instruction)
        for register in reads:
            if register not in defs:
                upward_exposed.add(register)
        defs.update(writes)
    return defs, upward_exposed


def build_cross_block_dfg(blocks, cfg_edges, max_edges=None):
    """Block-level cross-block dataflow edges via register reaching definitions.

    blocks: list of {'instructions': [...]} (or objects with .instructions),
    positionally indexed like the CFG. cfg_edges: dicts/objects with
    source/target/edge_type; existing 'dataflow' edges are ignored as inputs.
    Returns [{'source': d, 'target': u, 'edge_type': 'dataflow'}, ...] where a
    register defined in block d reaches an upward-exposed use in block u along
    CFG paths with no intervening redefinition. Self-loops = loop-carried flow.
    """
    block_count = len(blocks)
    if block_count <= 1:
        return []
    if max_edges is None:
        max_edges = int(os.environ.get('GRAPH_MAX_DATAFLOW_EDGES', '512'))

    gen, upward = [], []
    for block in blocks:
        if isinstance(block, dict):
            instructions = block.get('instructions') or []
        else:
            instructions = getattr(block, 'instructions', None) or []
        defs, uses = block_def_use(instructions)
        gen.append(defs)
        upward.append(uses)

    predecessors = [[] for _ in range(block_count)]
    for edge in cfg_edges or []:
        if isinstance(edge, dict):
            source, target = edge.get('source'), edge.get('target')
            edge_type = edge.get('edge_type', '')
        else:
            source, target = getattr(edge, 'source', None), getattr(edge, 'target', None)
            edge_type = getattr(edge, 'edge_type', '')
        if edge_type == 'dataflow' or source is None or target is None:
            continue
        if 0 <= source < block_count and 0 <= target < block_count:
            predecessors[target].append(source)

    def compute_in(block_id):
        in_map = {}
        for pred in predecessors[block_id]:
            for register, sources in reaching_out[pred].items():
                bucket = in_map.get(register)
                if bucket is None:
                    in_map[register] = set(sources)
                else:
                    bucket.update(sources)
        return in_map

    # reaching_out[b][reg] = set of blocks whose def of reg reaches b's exit.
    # Blocks are stream-ordered (approximately reverse-postorder), so forward
    # flow converges in ~1 pass; the cap only bounds pathological backedge nests.
    reaching_out = [dict() for _ in range(block_count)]
    max_passes = min(2 * block_count + 4, 64)
    for _ in range(max_passes):
        changed = False
        for block_id in range(block_count):
            in_map = compute_in(block_id)
            out_map = {}
            for register in set(in_map) | gen[block_id]:
                if register in gen[block_id]:
                    out_map[register] = {block_id}
                else:
                    out_map[register] = in_map[register]
            if out_map != reaching_out[block_id]:
                reaching_out[block_id] = out_map
                changed = True
        if not changed:
            break

    seen, dataflow_edges = set(), []
    for block_id in range(block_count):
        in_map = compute_in(block_id)
        for register in upward[block_id]:
            for def_block in in_map.get(register, ()):
                key = (def_block, block_id)
                if key not in seen:
                    seen.add(key)
                    dataflow_edges.append(
                        {'source': def_block, 'target': block_id, 'edge_type': 'dataflow'}
                    )

    dataflow_edges.sort(key=lambda edge: (edge['source'], edge['target']))
    if len(dataflow_edges) > max_edges:
        print(
            f"[build_cross_block_dfg] NOTICE: capping dataflow edges "
            f"{len(dataflow_edges)} -> {max_edges} (GRAPH_MAX_DATAFLOW_EDGES)",
            file=sys.stderr,
        )
        dataflow_edges = dataflow_edges[:max_edges]
    return dataflow_edges
