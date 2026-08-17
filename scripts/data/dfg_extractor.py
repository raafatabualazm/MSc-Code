
"""
Lightweight intra-block DFG extraction.
"""

from __future__ import annotations

import re


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
