
"""
Hierarchical basic-block chunking utilities.
"""

from __future__ import annotations

from typing import List


def chunk_cfg_blocks(cfg_blocks: List[dict], max_instructions: int = 32):
    chunks = []

    for block in cfg_blocks:
        instructions = block['instructions']

        for start in range(0, len(instructions), max_instructions):
            sub_chunk = instructions[start:start + max_instructions]

            chunks.append({
                'block_id': block['id'],
                'block_type': block['block_type'],
                'predecessors': block['predecessors'],
                'successors': block['successors'],
                'instructions': sub_chunk,
            })

    return chunks
