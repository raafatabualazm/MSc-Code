
"""
CFG extraction and basic-block recovery from assembly text.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

try:
    import networkx as nx
except Exception:  # networkx is optional: block/edge construction does not need it,
    nx = None      # only the integrity report and dominator-based backedge refinement do.


# x86-64 (gdb / objdump) branch mnemonics.
_X86_CONDITIONAL = {
    'je', 'jne', 'jg', 'jge', 'jl', 'jle',
    'ja', 'jae', 'jb', 'jbe', 'jz', 'jnz',
    'jo', 'jno', 'js', 'jns', 'jp', 'jpe', 'jnp', 'jpo',
    'jcxz', 'jecxz', 'jrcxz',
    'loop', 'loope', 'loopz', 'loopne', 'loopnz',
}
_X86_UNCONDITIONAL = {'jmp'}

# AArch64 (llvm-objdump) branch mnemonics. Conditional = the b.<cond> family
# plus the compare/test-and-branch ops (cbz/cbnz/tbz/tbnz). The plain `b` is
# unconditional; `br` is an indirect branch (tail call / jump table) whose
# register target is not statically resolvable, so it is treated as an
# unconditional terminator with no edge. `bl`/`blr` are branch-with-LINK CALLS:
# they return and must NOT split a block, so they are deliberately kept out of
# every jump/terminator set (mirroring how x86 `call` is handled).
_ARM64_CONDITIONAL = {
    'b.eq', 'b.ne', 'b.cs', 'b.hs', 'b.cc', 'b.lo', 'b.mi', 'b.pl',
    'b.vs', 'b.vc', 'b.hi', 'b.ls', 'b.ge', 'b.lt', 'b.gt', 'b.le',
    'cbz', 'cbnz', 'tbz', 'tbnz',
}
_ARM64_UNCONDITIONAL = {'b', 'b.al', 'b.nv', 'br'}

CONDITIONAL_JUMPS = _X86_CONDITIONAL | _ARM64_CONDITIONAL
UNCONDITIONAL_JUMPS = _X86_UNCONDITIONAL | _ARM64_UNCONDITIONAL

# Branch-with-link (call) mnemonics: kept inline as a fall-through, never a leader.
CALL_MNEMONICS = {'bl', 'blr'}

TRAP_TERMINATORS = {'ud2', 'int3', 'hlt', 'brk', 'hlt', 'eret'}
TERMINATORS = CONDITIONAL_JUMPS | UNCONDITIONAL_JUMPS | TRAP_TERMINATORS

_SOURCE_DECLARATION_RE = re.compile(
    r'^(?:static\b|void\b|class\b|int\s|double\s|bool\s|String\b|List<|Map<|Set<)'
)
_GDB_LINE_RE = re.compile(
    r'^(?P<address>0x[0-9a-fA-F]+)(?:\s+<[^>]*>)?:\s*(?P<instruction>.*)$'
)
_OBJDUMP_LINE_RE = re.compile(
    r'^(?P<address>[0-9a-fA-F]{3,}):\s*(?P<instruction>.*)$'
)
_OBJDUMP_BYTE_RE = re.compile(r'^(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{8})$')


@dataclass
class CFGEdge:
    source: int
    target: int
    edge_type: str


@dataclass
class BasicBlock:
    id: int
    label: str
    start_address: str
    instructions: List[str]
    predecessors: List[int]
    successors: List[int]
    edge_types: List[str] = field(default_factory=list)
    instruction_count: int = 0
    block_type: str = 'linear'


class AssemblyCFGExtractor:
    ADDRESS_PATTERN = re.compile(r'0x[0-9a-fA-F]+')

    CONDITIONAL_JUMPS = CONDITIONAL_JUMPS
    UNCONDITIONAL_JUMPS = UNCONDITIONAL_JUMPS

    def __init__(self, assembly_text: str):
        self.raw_lines = assembly_text.splitlines()
        self.instructions = []
        self.entry_address = None
        self.function_name = None
        self.duplicate_address_count = 0
        self.rejected_symbol_line_count = 0
        self.rejected_non_instruction_count = 0
        self.candidate_line_count = 0

    def clean_lines(self):
        cleaned = []

        for line in self.raw_lines:
            line = line.split(';')[0].strip()

            if not line or 'End of assembler dump' in line:
                continue

            cleaned.append(line)

        return cleaned

    @staticmethod
    def _strip_objdump_bytes(instruction: str) -> str:
        tokens = instruction.split()
        while tokens and _OBJDUMP_BYTE_RE.fullmatch(tokens[0]):
            tokens.pop(0)
        return ' '.join(tokens)

    def parse(self):
        self.instructions = []
        self.entry_address = None
        self.function_name = None
        self.duplicate_address_count = 0
        self.rejected_symbol_line_count = 0
        self.rejected_non_instruction_count = 0
        self.candidate_line_count = 0
        has_gdb_dump_marker = any(
            'Dump of assembler code for function' in line for line in self.raw_lines
        )
        in_dump = not has_gdb_dump_marker

        for raw_line in self.raw_lines:
            if 'Dump of assembler code for function' in raw_line:
                function_match = re.search(
                    r'Dump of assembler code for function\s+(.+?):\s*$',
                    raw_line.strip(),
                )
                if function_match:
                    self.function_name = function_match.group(1).strip()
                in_dump = True
                continue
            if in_dump and 'End of assembler dump' in raw_line:
                break
            if not in_dump:
                if re.match(r'^\s*\d+:\s*static\b', raw_line):
                    self.rejected_symbol_line_count += 1
                continue

            stripped = raw_line.strip()
            if stripped.startswith('=>'):
                stripped = stripped[2:].lstrip()
            if not stripped:
                continue

            match = _GDB_LINE_RE.match(stripped)
            is_objdump = False
            if match is None:
                match = _OBJDUMP_LINE_RE.match(stripped)
                is_objdump = match is not None
            if match is None:
                continue

            self.candidate_line_count += 1
            instruction = match.group('instruction').split(';', 1)[0].strip()
            if is_objdump:
                instruction = self._strip_objdump_bytes(instruction)
            if not instruction or _SOURCE_DECLARATION_RE.match(instruction):
                if _SOURCE_DECLARATION_RE.match(instruction):
                    self.rejected_symbol_line_count += 1
                else:
                    self.rejected_non_instruction_count += 1
                continue

            instruction_address = int(match.group('address'), 16)
            offset_match = re.search(r'<\+(\d+)>:', stripped)
            offset = int(offset_match.group(1)) if offset_match else None

            self.instructions.append({
                'address': instruction_address,
                'instruction': instruction,
                'offset': offset,
            })

        zero_positions = [
            index for index, item in enumerate(self.instructions)
            if item.get('offset') == 0
        ]
        if zero_positions:
            if len(zero_positions) == 1:
                # GDB sometimes prints the tail before <+0>. Offset order is
                # the executable order; no-offset int3 padding is outside it.
                selected = [item for item in self.instructions if item.get('offset') is not None]
            else:
                # Concatenated symbol regions occur in the old SFT corpus. The
                # requested function is the final <+0> region in those dumps;
                # earlier fragments belong to adjacent symbols/stubs.
                selected = [
                    item for item in self.instructions[zero_positions[-1]:]
                    if item.get('offset') is not None
                ]
            self.instructions = sorted(selected, key=lambda item: item['offset'])
            self.entry_address = self.instructions[0]['address'] if self.instructions else None
        elif self.instructions:
            # llvm-objdump output has no GDB <+offset> annotation.
            self.instructions.sort(key=lambda item: item['address'])

        # Deduplicate only inside the selected function region. An adjacent
        # symbol fragment that was discarded above must not suppress an
        # instruction from the retained function merely because addresses
        # were reused in a concatenated dump.
        deduplicated = []
        seen_addresses = set()
        for item in self.instructions:
            if item['address'] in seen_addresses:
                self.duplicate_address_count += 1
                continue
            seen_addresses.add(item['address'])
            deduplicated.append(item)
        self.instructions = deduplicated
        if self.instructions:
            self.entry_address = self.instructions[0]['address']


    def canonicalize_address(self, value):
        if value is None:
            return None

        text = str(value).strip()

        # Prefer an explicit 0x-prefixed token: covers x86 line addresses,
        # branch operands on both ISAs, and the hex() round-trip of a block's
        # start_address.
        match = self.ADDRESS_PATTERN.search(text)
        if match:
            return int(match.group(0), 16)

        # AArch64 line addresses are bare hex in the address column
        # ("226304:\tstp ..."); accept a leading bare-hex run (the trailing ':'
        # is not a hex digit, so re.match stops before it).
        bare = re.match(r'[0-9a-fA-F]+', text)
        if bare:
            return int(bare.group(0), 16)

        return None

    def extract_jump_target(self, instruction: str):
        # Strip "<symbol+0xoffset>" annotations first: on AArch64 the offset
        # inside the annotation is itself a 0x token, so a naive last-token scan
        # would return the annotation's offset instead of the real destination.
        cleaned = re.sub(r'<[^>]*>', '', instruction)
        parts = cleaned.split(None, 1)
        operand_text = parts[1] if len(parts) > 1 else ''
        # Intel indirect jumps/calls contain a memory operand; its displacement
        # is not a control-flow destination. AArch64 br/blr use a register.
        if '[' in operand_text or (parts and parts[0].lower() in {'br', 'blr'}):
            return None
        addresses = self.ADDRESS_PATTERN.findall(cleaned)

        if addresses:
            # The destination is the final address operand, after any
            # bit-immediate (e.g. tbz's "#0x0").
            return self.canonicalize_address(addresses[-1])

        # llvm-objdump prints direct AArch64 destinations as bare hexadecimal
        # addresses (e.g. `b.eq 226310`). Select the final address-like operand,
        # after bit immediates used by tbz/tbnz.
        bare_addresses = re.findall(r'(?<![\w#])([0-9a-fA-F]{3,})(?!\w)', operand_text)
        if bare_addresses:
            return int(bare_addresses[-1], 16)

        return None

    def is_external_direct_target(self, instruction: str) -> bool:
        """Whether a direct target annotation names another function/stub.

        A missing `<main+offset>` target is evidence of a truncated dump and
        remains fatal. A jump to `<stub ...>` or another named symbol is a
        legitimate external exit and cannot be represented as an intra-function
        edge without inventing a node.
        """
        annotations = re.findall(r'<([^>]*)>', instruction)
        if not annotations:
            return False
        symbol = annotations[-1].strip()
        if not symbol:
            return False
        if self.function_name:
            return not (
                symbol == self.function_name
                or symbol.startswith(self.function_name + "+")
            )
        return symbol.lower().startswith(("stub ", "new ", "runtime "))

    @staticmethod
    def _is_return(opcode: str) -> bool:
        return opcode.startswith('ret')

    @staticmethod
    def _looks_like_unknown_branch(opcode: str) -> bool:
        return (
            opcode.startswith('j')
            or opcode.startswith('loop')
            or opcode.startswith('b.')
            or opcode in {'braa', 'brab', 'braaz', 'brabz'}
        )

    def discover_leaders(self):
        leaders = {self.instructions[0]['address'], self.entry_address}

        for index, item in enumerate(self.instructions):
            instruction = item['instruction']
            opcode = instruction.split()[0].lower()

            is_unknown_branch = (
                self._looks_like_unknown_branch(opcode)
                and opcode not in self.CONDITIONAL_JUMPS
                and opcode not in self.UNCONDITIONAL_JUMPS
            )
            if opcode in TERMINATORS or self._is_return(opcode) or is_unknown_branch:
                if index + 1 < len(self.instructions):
                    leaders.add(self.instructions[index + 1]['address'])

            if opcode in self.CONDITIONAL_JUMPS | self.UNCONDITIONAL_JUMPS:
                target = self.extract_jump_target(instruction)

                if target is not None:
                    leaders.add(target)

        return sorted(leaders)

    def infer_block_type(self, instruction: str):
        opcode = instruction.split()[0].lower()

        if opcode in self.CONDITIONAL_JUMPS:
            return 'conditional'

        if opcode in self.UNCONDITIONAL_JUMPS:
            return 'jump'

        if opcode.startswith('call') or opcode in CALL_MNEMONICS:
            return 'call'

        if self._is_return(opcode):
            return 'return'

        if opcode in TRAP_TERMINATORS:
            return 'trap'

        return 'linear'

    def build_blocks(self):
        self.parse()

        if not self.instructions:
            return [], [], {
                'isolated_nodes': [],
                'unreachable_nodes': [],
                'valid': False,
                'error': 'no parsed instructions',
            }

        leaders = self.discover_leaders()
        address_to_index = {
            item['address']: index
            for index, item in enumerate(self.instructions)
        }
        # Some assembly dumps contain jumps to addresses outside the dumped
        # instruction range (runtime stubs, landing pads, truncated contexts).
        # They are valid edge targets only if they map to a parsed instruction;
        # otherwise treating them as leaders raises a KeyError and loses the
        # whole row's CFG.
        leaders = [leader for leader in leaders if leader in address_to_index]

        # Optionally cap basic-block length so every block fits the encoder's
        # per-block token window (GraphCodeBERT, 512 tokens). A long straight-line
        # block is split into a fall-through chain of <=max_run-instruction
        # sub-blocks by inserting synthetic leaders; control flow is unchanged
        # (the new boundaries are linear edges, the terminator stays on the last
        # sub-block). Off by default (GRAPH_MAX_BLOCK_INSTRS=0) so x86/legacy
        # extraction is byte-identical; ARM64 builds set e.g. 24.
        max_run = int(os.environ.get("GRAPH_MAX_BLOCK_INSTRS", "0") or "0")
        if max_run > 0:
            ordered = sorted(address_to_index[leader] for leader in leaders)
            extra = []
            for pos, start in enumerate(ordered):
                end = ordered[pos + 1] if pos + 1 < len(ordered) else len(self.instructions)
                cut = start + max_run
                while cut < end:
                    extra.append(self.instructions[cut]['address'])
                    cut += max_run
            if extra:
                leaders = sorted(set(leaders) | set(extra))

        # Cut blocks in INSTRUCTION-STREAM order, not address order. Dart AOT
        # dumps are not strictly ascending by address (the entry region is often
        # listed after the body, and jumps point backwards), so ordering leaders
        # by address produced start>end empty-slice blocks and crashed CFG
        # extraction (~39% of dart_all rows). Ordering leaders by their position
        # in the parsed stream guarantees non-empty, contiguous blocks and makes
        # the index+1 fall-through edge the true next block.
        leader_indices = sorted({address_to_index[leader] for leader in leaders})

        blocks = []

        for block_id, start in enumerate(leader_indices):
            if block_id + 1 < len(leader_indices):
                end = leader_indices[block_id + 1]
            else:
                end = len(self.instructions)

            block_items = self.instructions[start:end]
            block_instructions = [item['instruction'] for item in block_items]

            leader_address = self.instructions[start]['address']

            block = BasicBlock(
                id=block_id,
                label=f'block_{block_id}',
                start_address=hex(leader_address),
                instructions=block_instructions,
                predecessors=[],
                successors=[],
                instruction_count=len(block_instructions),
                block_type=self.infer_block_type(block_instructions[-1]),
            )

            blocks.append(block)

        address_to_block = {}

        for block in blocks:
            start_index = address_to_index[self.canonicalize_address(block.start_address)]
            end_index = start_index + len(block.instructions)

            for item in self.instructions[start_index:end_index]:
                address_to_block[item['address']] = block.id

        typed_edges = []
        unresolved_direct_branches = 0
        external_direct_branches = 0
        indirect_branches = 0
        unknown_branch_mnemonics = set()

        for index, block in enumerate(blocks):
            last_instruction = block.instructions[-1]
            opcode = last_instruction.split()[0].lower()

            if opcode in self.CONDITIONAL_JUMPS:
                target = self.extract_jump_target(last_instruction)

                if target in address_to_block:
                    successor = address_to_block[target]
                    block.successors.append(successor)
                    block.edge_types.append('conditional_true')
                    typed_edges.append(CFGEdge(block.id, successor, 'conditional_true'))
                elif target is not None and self.is_external_direct_target(last_instruction):
                    external_direct_branches += 1
                elif target is not None:
                    unresolved_direct_branches += 1

                if index + 1 < len(blocks):
                    block.successors.append(index + 1)
                    block.edge_types.append('conditional_false')
                    typed_edges.append(CFGEdge(block.id, index + 1, 'conditional_false'))

            elif opcode in self.UNCONDITIONAL_JUMPS:
                target = self.extract_jump_target(last_instruction)

                if target in address_to_block:
                    successor = address_to_block[target]
                    block.successors.append(successor)
                    block.edge_types.append('unconditional_jump')
                    typed_edges.append(CFGEdge(block.id, successor, 'unconditional_jump'))
                elif target is None:
                    indirect_branches += 1
                elif self.is_external_direct_target(last_instruction):
                    external_direct_branches += 1
                else:
                    unresolved_direct_branches += 1

            elif (
                self._looks_like_unknown_branch(opcode)
                and opcode not in self.CONDITIONAL_JUMPS
                and opcode not in self.UNCONDITIONAL_JUMPS
            ):
                # Conservatively terminate an unknown branch instead of
                # fabricating a linear fall-through edge.
                unknown_branch_mnemonics.add(opcode)
            elif not self._is_return(opcode) and opcode not in TRAP_TERMINATORS:
                if index + 1 < len(blocks):
                    block.successors.append(index + 1)
                    block.edge_types.append('linear_fallthrough')
                    typed_edges.append(CFGEdge(block.id, index + 1, 'linear_fallthrough'))

        if nx is not None:
            graph = nx.DiGraph()

            for block in blocks:
                graph.add_node(block.id)

            for edge in typed_edges:
                graph.add_edge(edge.source, edge.target)

            entry_block = address_to_block.get(self.entry_address)
            integrity = verify_graph_integrity(graph, entry_block)

            # Compiler dumps can contain closed, unreachable epilogues or
            # landing-pad fragments after the reachable function body. Drop
            # those blocks only when every direct branch resolved: unresolved
            # control flow indicates a truncated dump and must remain invalid.
            pruned_unreachable_starts = []
            unreachable = sorted(integrity.get('unreachable_nodes') or [])
            if (
                unreachable
                and unresolved_direct_branches == 0
                and not unknown_branch_mnemonics
                and entry_block is not None
            ):
                unreachable_set = set(unreachable)
                pruned_unreachable_starts = [
                    blocks[block_id].start_address for block_id in unreachable
                ]
                kept_ids = [
                    block.id for block in blocks if block.id not in unreachable_set
                ]
                remap = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
                blocks = [blocks[old_id] for old_id in kept_ids]
                for new_id, block in enumerate(blocks):
                    block.id = new_id
                    block.label = f'block_{new_id}'
                typed_edges = [
                    CFGEdge(remap[edge.source], remap[edge.target], edge.edge_type)
                    for edge in typed_edges
                    if edge.source in remap and edge.target in remap
                ]
                address_to_block = {
                    address: remap[block_id]
                    for address, block_id in address_to_block.items()
                    if block_id in remap
                }
                entry_block = address_to_block.get(self.entry_address)
                graph = nx.DiGraph()
                graph.add_nodes_from(block.id for block in blocks)
                graph.add_edges_from((edge.source, edge.target) for edge in typed_edges)
                integrity = verify_graph_integrity(graph, entry_block)

            integrity['pruned_unreachable_block_count'] = len(pruned_unreachable_starts)
            integrity['pruned_unreachable_block_starts'] = pruned_unreachable_starts

            try:
                dominators = nx.immediate_dominators(graph, entry_block)

                def _dominates(candidate, node):
                    # candidate dominates node iff candidate lies on node's
                    # idom chain (every node dominates itself). Walking the
                    # chain catches backedges to NON-immediate dominators,
                    # which the old idom-equality check missed.
                    steps = 0
                    while node in dominators and steps <= len(dominators):
                        if node == candidate:
                            return True
                        parent = dominators[node]
                        if parent == node:  # reached the entry node
                            return False
                        node = parent
                        steps += 1
                    return False

                for edge in typed_edges:
                    if (
                        edge.source in dominators
                        and edge.target in dominators
                        and _dominates(edge.target, edge.source)
                    ):
                        edge.edge_type = 'loop_backedge'
            except Exception:
                pass
        else:
            # Without networkx, block/edge construction still works, but
            # reachability pruning, integrity checks, and dominator-based
            # backedge refinement are unavailable.
            integrity = {
                'isolated_nodes': [],
                'unreachable_nodes': [],
                'valid': None,
                'note': 'networkx unavailable; integrity not computed',
                'pruned_unreachable_block_count': 0,
                'pruned_unreachable_block_starts': [],
            }

        # Dominator refinement changes the canonical edge labels. Rebuild the
        # duplicated per-block predecessor/successor metadata afterwards so the
        # stored views cannot disagree.
        for block in blocks:
            block.predecessors.clear()
            block.successors.clear()
            block.edge_types.clear()
        for edge in typed_edges:
            blocks[edge.source].successors.append(edge.target)
            blocks[edge.source].edge_types.append(edge.edge_type)
            blocks[edge.target].predecessors.append(edge.source)

        entry_block = address_to_block.get(self.entry_address)
        all_edges_in_range = all(
            0 <= edge.source < len(blocks) and 0 <= edge.target < len(blocks)
            for edge in typed_edges
        )
        all_blocks_nonempty = all(bool(block.instructions) for block in blocks)
        integrity.update({
            'entry_address': hex(self.entry_address) if self.entry_address is not None else None,
            'entry_block': entry_block,
            'has_entry': entry_block is not None,
            'all_edges_in_range': all_edges_in_range,
            'all_blocks_nonempty': all_blocks_nonempty,
            'parsed_instruction_count': len(self.instructions),
            'candidate_line_count': self.candidate_line_count,
            'rejected_symbol_line_count': self.rejected_symbol_line_count,
            'rejected_non_instruction_count': self.rejected_non_instruction_count,
            'duplicate_address_count': self.duplicate_address_count,
            'unresolved_direct_branch_count': unresolved_direct_branches,
            'external_direct_branch_count': external_direct_branches,
            'indirect_branch_count': indirect_branches,
            'unknown_branch_mnemonics': sorted(unknown_branch_mnemonics),
        })
        integrity['valid'] = bool(
            entry_block is not None
            and all_edges_in_range
            and all_blocks_nonempty
            and self.duplicate_address_count == 0
            and unresolved_direct_branches == 0
            and not integrity.get('unreachable_nodes')
            and not integrity.get('isolated_nonentry_nodes')
            and not unknown_branch_mnemonics
        )

        for block in blocks:
            runtime_hits = [
                ins for ins in block.instructions
                if 'stub' in ins.lower() or 'runtime' in ins.lower()
            ]

            if runtime_hits:
                block.block_type = 'runtime_stub'

        return blocks, typed_edges, integrity




def verify_graph_integrity(graph, entry_node=0):
    isolated = list(nx.isolates(graph))
    has_entry = entry_node in graph
    reachable = (nx.descendants(graph, entry_node) | {entry_node}) if has_entry else set()
    unreachable = set(graph.nodes()) - reachable
    isolated_nonentry = [node for node in isolated if node != entry_node]

    integrity = {
        'isolated_nodes': isolated,
        'isolated_nonentry_nodes': isolated_nonentry,
        'unreachable_nodes': list(unreachable),
        # A one-block function is an isolated entry and is perfectly valid.
        'valid': has_entry and not isolated_nonentry and not unreachable,
    }

    return integrity


class CFGDatasetBuilder:
    def convert_record(self, record: dict):
        converted = dict(record)
        extractor = AssemblyCFGExtractor(record['assembly'])
        blocks, edges, integrity = extractor.build_blocks()

        # Preserve all original SFT/GRPO fields (dart_source, tests,
        # dart_function_signature, assembly, task metadata, etc.).  Earlier
        # versions returned a tiny replacement record and could silently break
        # downstream training/evaluation datasets.
        converted['language'] = converted.get('language', converted.get('lang', 'unknown'))
        converted['cfg'] = [asdict(block) for block in blocks]
        converted['edges'] = [asdict(edge) for edge in edges]
        converted['integrity'] = integrity
        return converted

    def convert_jsonl(self, input_path: str, output_path: str):
        output_rows = []

        with open(input_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                output_rows.append(self.convert_record(record))

        with open(output_path, 'w', encoding='utf-8') as handle:
            for row in output_rows:
                handle.write(json.dumps(row) + '\n')


def _normalize_blocks(cfg_blocks):
    """Project blocks (dicts or BasicBlock) to a uniform {'instructions': [...]}.

    The model only consumes per-block instructions; node identity is positional
    (block i == graph node i == edge endpoints i), so dropping the other fields
    keeps a single Arrow schema across rows regardless of where the CFG came
    from (precomputed file, inline extraction, or single-block fallback).
    """
    normalized = []
    for block in cfg_blocks:
        if isinstance(block, dict):
            instructions = block.get('instructions') or []
        else:
            instructions = getattr(block, 'instructions', []) or []
        normalized.append({'instructions': list(instructions)})
    return normalized


def _normalize_edges(edges):
    normalized = []
    for edge in edges:
        if isinstance(edge, dict):
            source = edge.get('source')
            target = edge.get('target')
            edge_type = edge.get('edge_type', 'linear_fallthrough')
        else:
            source = getattr(edge, 'source', None)
            target = getattr(edge, 'target', None)
            edge_type = getattr(edge, 'edge_type', 'linear_fallthrough')
        if source is None or target is None:
            continue
        normalized.append({'source': int(source), 'target': int(target), 'edge_type': edge_type})
    return normalized


def _maybe_add_dataflow_edges(blocks, edges):
    """Append cross-block 'dataflow' edges when GRAPH_DFG_MODE=edges.

    Computed at load time from the normalized blocks + CFG edges (works for
    precomputed *_cfg.jsonl rows and inline extraction alike), so the data
    files never need regenerating. Idempotent: rows that already carry
    dataflow edges are left untouched.
    """
    dfg_mode = os.environ.get('GRAPH_DFG_MODE', 'legacy').lower()
    if dfg_mode != 'edges':
        # Graph-v2 files pin their DFG edges for reproducibility. An ablation
        # that requests DFG off must remove those stored edges explicitly.
        return [edge for edge in edges if edge.get('edge_type') != 'dataflow']
    if not edges or any(edge.get('edge_type') == 'dataflow' for edge in edges):
        return edges
    try:
        try:
            from scripts.data.dfg_extractor import build_cross_block_dfg
        except ModuleNotFoundError:  # direct execution from scripts/data
            from dfg_extractor import build_cross_block_dfg
        return edges + build_cross_block_dfg(blocks, edges)
    except Exception as exc:
        if (
            os.environ.get('GRAPH_STRICT_GRAPH', '0') == '1'
            or os.environ.get('GRPO_STRICT_GRAPH', '0') == '1'
        ):
            raise RuntimeError(
                'Strict graph mode could not construct cross-block DFG edges'
            ) from exc
        print(
            f"[ensure_cfg_blocks] WARNING: cross-block DFG failed ({exc!r}); "
            f"continuing with CFG edges only",
            file=sys.stderr,
        )
        return edges


def ensure_cfg_blocks(record, auto_extract=None):
    """Return (cfg_blocks, edges) for a record in a uniform minimal schema.

    Resolution order:
      1. record['cfg']/['edges'] when present (precomputed by build_cfg_jsonl).
      2. else, when auto_extract (env GRAPH_AUTO_CFG=1), parse the raw assembly
         into a real basic-block CFG so the GNN gets actual control-flow and
         each block fits GraphCodeBERT's 512-token window.
      3. else, degrade to a single 'entry' block with no edges (legacy behavior).

    The default for (2) is OFF so that re-evaluating older single-block-trained
    checkpoints on the original files stays reproducible. New runs should either
    point at the *_cfg.jsonl files (case 1) or pass --auto_cfg (case 2).

    With GRAPH_DFG_MODE=edges, block-level cross-block dataflow edges are
    appended to the CFG edges here, so SFT, GRPO, and inference all agree.
    """
    if auto_extract is None:
        auto_extract = os.environ.get('GRAPH_AUTO_CFG', '0') == '1'
    strict_graph = (
        os.environ.get('GRAPH_STRICT_GRAPH', '0') == '1'
        or os.environ.get('GRPO_STRICT_GRAPH', '0') == '1'
    )

    cfg = record.get('cfg') or []
    edges = record.get('edges') or []
    if cfg:
        integrity = record.get('integrity')
        if strict_graph and (
            not isinstance(integrity, dict) or integrity.get('valid') is not True
        ):
            raise ValueError(
                'Strict graph mode requires integrity.valid=true on every '
                'precomputed graph row'
            )
        blocks = _normalize_blocks(cfg)
        normalized_edges = _normalize_edges(edges)
        return blocks, _maybe_add_dataflow_edges(blocks, normalized_edges)

    assembly = record.get('assembly') or ''
    if auto_extract and assembly.strip():
        try:
            blocks, typed_edges, integrity = AssemblyCFGExtractor(assembly).build_blocks()
            if strict_graph and integrity.get('valid') is not True:
                raise ValueError(
                    f"inline CFG failed strict integrity: {integrity}"
                )
            if blocks:
                normalized_blocks = _normalize_blocks(blocks)
                normalized_edges = _normalize_edges(typed_edges)
                return normalized_blocks, _maybe_add_dataflow_edges(normalized_blocks, normalized_edges)
        except Exception as exc:
            if strict_graph:
                raise RuntimeError(
                    'Strict graph mode refused an invalid inline CFG'
                ) from exc
            print(
                f"[ensure_cfg_blocks] CFG extraction failed ({exc}); "
                f"falling back to single-block",
                file=sys.stderr,
            )

    if strict_graph:
        raise ValueError(
            'Strict graph mode requires a valid precomputed CFG or '
            'GRAPH_AUTO_CFG=1 with successful extraction'
        )
    return [{'instructions': assembly.splitlines()}], []


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    builder = CFGDatasetBuilder()
    builder.convert_jsonl(args.input, args.output)
