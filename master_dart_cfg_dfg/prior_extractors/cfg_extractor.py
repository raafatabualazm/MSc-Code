
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

TERMINATORS = CONDITIONAL_JUMPS | UNCONDITIONAL_JUMPS | {
    'ret', 'retn', 'retq',
}


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

    def clean_lines(self):
        cleaned = []

        for line in self.raw_lines:
            line = line.split(';')[0].strip()

            if not line or 'End of assembler dump' in line:
                continue

            cleaned.append(line)

        return cleaned

    def parse(self):
        for line in self.clean_lines():
            stripped = line.strip()

            # A real disassembly line starts with its address (e.g.
            # "0x...bfc <+0>:\tpush rbp" or "0x...179:\tint3"). Requiring the
            # address at the START rejects interleaved source code that merely
            # contains a hex literal (some corpora, e.g. dart_all.jsonl, append
            # the numbered Dart source after the dump) instead of mis-parsing it
            # as an instruction.
            # x86 gdb dumps prefix the line address with 0x ("0x...bfc <+0>:").
            # AArch64 llvm-objdump uses a bare-hex address column ("226304:").
            # Accept either form at the START; the bare-hex form additionally
            # requires the trailing ':' so interleaved numbered Dart source
            # ("  42  int foo() {") is still rejected, not mis-parsed.
            match = re.match(r'0x[0-9a-fA-F]+|[0-9a-fA-F]+:', stripped)
            if not match:
                continue

            instruction_address = self.canonicalize_address(match.group(0))

            if ':' in stripped:
                instruction = stripped.split(':', 1)[1].strip()
            else:
                instruction = stripped[match.end():].strip()

            # Skip address-only lines: an empty instruction would crash the
            # downstream `instruction.split()[0]` opcode lookups.
            if not instruction:
                continue

            self.instructions.append({
                'address': instruction_address,
                'instruction': instruction,
            })


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
        addresses = self.ADDRESS_PATTERN.findall(cleaned)

        if addresses:
            # The destination is the final address operand, after any
            # bit-immediate (e.g. tbz's "#0x0").
            return self.canonicalize_address(addresses[-1])

        return None

    def discover_leaders(self):
        leaders = {self.instructions[0]['address']}

        for index, item in enumerate(self.instructions):
            instruction = item['instruction']
            opcode = instruction.split()[0].lower()

            if opcode in TERMINATORS:
                if index + 1 < len(self.instructions):
                    leaders.add(self.instructions[index + 1]['address'])

            if opcode in self.CONDITIONAL_JUMPS | self.UNCONDITIONAL_JUMPS:
                target = self.extract_jump_target(instruction)

                if target:
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

        if opcode.startswith('ret'):
            return 'return'

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

        for index, block in enumerate(blocks):
            last_instruction = block.instructions[-1]
            opcode = last_instruction.split()[0].lower()

            if opcode in self.CONDITIONAL_JUMPS:
                target = self.extract_jump_target(last_instruction)

                if target in address_to_block:
                    successor = address_to_block[target]

                    edge_type = 'conditional_true'

                    if target < self.canonicalize_address(block.start_address):
                        edge_type = 'loop_backedge'

                    block.successors.append(successor)
                    block.edge_types.append(edge_type)
                    typed_edges.append(CFGEdge(block.id, successor, edge_type))

                if index + 1 < len(blocks):
                    block.successors.append(index + 1)
                    block.edge_types.append('conditional_false')
                    typed_edges.append(CFGEdge(block.id, index + 1, 'conditional_false'))

            elif opcode in self.UNCONDITIONAL_JUMPS:
                target = self.extract_jump_target(last_instruction)

                if target in address_to_block:
                    successor = address_to_block[target]

                    edge_type = 'unconditional_jump'

                    if target < self.canonicalize_address(block.start_address):
                        edge_type = 'loop_backedge'

                    block.successors.append(successor)
                    block.edge_types.append(edge_type)
                    typed_edges.append(CFGEdge(block.id, successor, edge_type))

            elif not opcode.startswith('ret'):
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

            integrity = verify_graph_integrity(graph, 0)

            try:
                dominators = nx.immediate_dominators(graph, 0)

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
            # No networkx: blocks/edges are still correct (loop backedges are also
            # detected by the target<start_address heuristic above). Only the
            # integrity report and dominator refinement are skipped.
            integrity = {
                'isolated_nodes': [],
                'unreachable_nodes': [],
                'valid': None,
                'note': 'networkx unavailable; integrity not computed',
            }

        for block in blocks:
            for successor in block.successors:
                blocks[successor].predecessors.append(block.id)

            runtime_hits = [
                ins for ins in block.instructions
                if 'stub' in ins.lower() or 'runtime' in ins.lower()
            ]

            if runtime_hits:
                block.block_type = 'runtime_stub'

        return blocks, typed_edges, integrity




def verify_graph_integrity(graph, entry_node=0):
    isolated = list(nx.isolates(graph))

    reachable = nx.descendants(graph, entry_node) | {entry_node}
    unreachable = set(graph.nodes()) - reachable

    integrity = {
        'isolated_nodes': isolated,
        'unreachable_nodes': list(unreachable),
        'valid': len(isolated) == 0,
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
    if os.environ.get('GRAPH_DFG_MODE', 'legacy').lower() != 'edges':
        return edges
    if not edges or any(edge.get('edge_type') == 'dataflow' for edge in edges):
        return edges
    try:
        try:
            from scripts.data.dfg_extractor import build_cross_block_dfg
        except ModuleNotFoundError:  # direct execution from scripts/data
            from dfg_extractor import build_cross_block_dfg
        return edges + build_cross_block_dfg(blocks, edges)
    except Exception as exc:
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

    cfg = record.get('cfg') or []
    edges = record.get('edges') or []
    if cfg:
        blocks = _normalize_blocks(cfg)
        normalized_edges = _normalize_edges(edges)
        return blocks, _maybe_add_dataflow_edges(blocks, normalized_edges)

    assembly = record.get('assembly') or ''
    if auto_extract and assembly.strip():
        try:
            blocks, typed_edges, _ = AssemblyCFGExtractor(assembly).build_blocks()
            if blocks:
                normalized_blocks = _normalize_blocks(blocks)
                normalized_edges = _normalize_edges(typed_edges)
                return normalized_blocks, _maybe_add_dataflow_edges(normalized_blocks, normalized_edges)
        except Exception as exc:  # never let CFG extraction crash training/inference
            print(
                f"[ensure_cfg_blocks] CFG extraction failed ({exc}); "
                f"falling back to single-block",
                file=sys.stderr,
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
