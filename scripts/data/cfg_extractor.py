
"""
CFG extraction and basic-block recovery from assembly text.
"""

from __future__ import annotations

import json
import re
import networkx as nx
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List


CONDITIONAL_JUMPS = {
    'je', 'jne', 'jg', 'jge', 'jl', 'jle',
    'ja', 'jae', 'jb', 'jbe', 'jz', 'jnz'
}

UNCONDITIONAL_JUMPS = {
    'jmp'
}

TERMINATORS = CONDITIONAL_JUMPS | UNCONDITIONAL_JUMPS | {
    'ret', 'retn', 'retq'
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
            addresses = self.ADDRESS_PATTERN.findall(line)

            if not addresses:
                continue

            instruction_address = self.canonicalize_address(addresses[0])

            if ':' in line:
                instruction = line.split(':', 1)[1].strip()
            else:
                instruction = line

            self.instructions.append({
                'address': instruction_address,
                'instruction': instruction,
            })


    def canonicalize_address(self, value):
        if value is None:
            return None

        match = self.ADDRESS_PATTERN.search(str(value))

        if not match:
            return None

        return int(match.group(0), 16)

    def extract_jump_target(self, instruction: str):
        addresses = self.ADDRESS_PATTERN.findall(instruction)

        if len(addresses) >= 1:
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

        if opcode.startswith('call'):
            return 'call'

        if opcode.startswith('ret'):
            return 'return'

        return 'linear'

    def build_blocks(self):
        self.parse()

        leaders = self.discover_leaders()
        address_to_index = {
            item['address']: index
            for index, item in enumerate(self.instructions)
        }

        blocks = []

        for block_id, leader in enumerate(leaders):
            start = address_to_index[leader]

            if block_id + 1 < len(leaders):
                end = address_to_index[leaders[block_id + 1]]
            else:
                end = len(self.instructions)

            block_items = self.instructions[start:end]
            block_instructions = [item['instruction'] for item in block_items]

            block = BasicBlock(
                id=block_id,
                label=f'block_{block_id}',
                start_address=hex(leader),
                instructions=block_instructions,
                predecessors=[],
                successors=[],
                instruction_count=len(block_instructions),
                block_type=self.infer_block_type(block_instructions[-1]),
            )

            blocks.append(block)

        leader_to_block = {
            leaders[index]: index
            for index in range(len(leaders))
        }

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

        graph = nx.DiGraph()

        for block in blocks:
            graph.add_node(block.id)

        for edge in typed_edges:
            graph.add_edge(edge.source, edge.target)

        integrity = verify_graph_integrity(graph, 0)

        try:
            dominators = nx.immediate_dominators(graph, 0)

            for edge in typed_edges:
                if edge.target in dominators and dominators.get(edge.source) == edge.target:
                    edge.edge_type = 'loop_backedge'
        except Exception:
            pass

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
        extractor = AssemblyCFGExtractor(record['assembly'])
        blocks, edges, integrity = extractor.build_blocks()

        return {
            'language': record.get('language', 'unknown'),
            'source': record.get('source', ''),
            'reasoning': record.get('reasoning', ''),
            'cfg': [asdict(block) for block in blocks],
            'edges': [asdict(edge) for edge in edges],
            'integrity': integrity,
        }

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    builder = CFGDatasetBuilder()
    builder.convert_jsonl(args.input, args.output)
