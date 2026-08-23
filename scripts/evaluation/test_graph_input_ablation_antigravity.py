from __future__ import annotations

import unittest

from scripts.evaluation.graph_inference_antigravity import apply_graph_input_ablation


def sample_row(task_id: int, block_count: int = 3) -> dict:
    cfg = []
    for index in range(block_count):
        cfg.append({
            "id": index,
            "label": f"block_{index}",
            "instructions": [f"mov rax,{task_id * 10 + index}"],
            "predecessors": [] if index == 0 else [index - 1],
            "successors": [] if index + 1 == block_count else [index + 1],
        })
    edges = [
        {"source": index, "target": index + 1, "edge_type": "linear_fallthrough"}
        for index in range(block_count - 1)
    ]
    return {
        "task_id": task_id,
        "dart_function_signature": f"int task{task_id}(int x)",
        "tests": f"expect(task{task_id}(1), {task_id});",
        "cfg": cfg,
        "edges": edges,
        "integrity": {"entry_block": 0, "isolated_nodes": []},
        "graph_v2": {"schema": "antigravity-graph-v2"},
    }


class GraphInputAblationTests(unittest.TestCase):
    def test_cyclic_shift_is_a_derangement_and_preserves_target_fields(self) -> None:
        rows = [sample_row(index) for index in range(4)]
        graph_rows, records, mapping_sha = apply_graph_input_ablation(
            rows, "cyclic_shift", 42
        )

        self.assertEqual(len(mapping_sha), 64)
        self.assertTrue(all(r["target_id"] != r["donor_id"] for r in records))
        for index, graph_row in enumerate(graph_rows):
            donor = records[index]["donor_index"]
            self.assertEqual(graph_row["cfg"], rows[donor]["cfg"])
            self.assertEqual(graph_row["edges"], rows[donor]["edges"])
            self.assertEqual(graph_row["tests"], rows[index]["tests"])
            self.assertEqual(
                graph_row["dart_function_signature"],
                rows[index]["dart_function_signature"],
            )

    def test_cyclic_shift_is_deterministic(self) -> None:
        rows = [sample_row(index) for index in range(5)]
        first = apply_graph_input_ablation(rows, "cyclic_shift", 43)
        second = apply_graph_input_ablation(rows, "cyclic_shift", 43)
        self.assertEqual(first, second)

    def test_shuffle_blocks_preserves_graph_topology_under_reindexing(self) -> None:
        rows = [sample_row(7, block_count=6)]
        graph_rows, records, _ = apply_graph_input_ablation(
            rows, "shuffle_blocks", 42
        )
        shuffled = graph_rows[0]

        self.assertTrue(records[0]["changed"])
        self.assertEqual(
            sorted(block["instructions"][0] for block in shuffled["cfg"]),
            sorted(block["instructions"][0] for block in rows[0]["cfg"]),
        )
        self.assertEqual(
            [block["id"] for block in shuffled["cfg"]],
            list(range(len(shuffled["cfg"]))),
        )
        self.assertEqual(len(shuffled["edges"]), len(rows[0]["edges"]))
        self.assertTrue(
            all(
                0 <= edge["source"] < len(shuffled["cfg"])
                and 0 <= edge["target"] < len(shuffled["cfg"])
                for edge in shuffled["edges"]
            )
        )

    def test_cyclic_shift_rejects_single_row(self) -> None:
        with self.assertRaises(ValueError):
            apply_graph_input_ablation([sample_row(0)], "cyclic_shift", 42)


if __name__ == "__main__":
    unittest.main()
