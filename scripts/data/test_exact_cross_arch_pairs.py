from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.build_exact_cross_arch_pairs import (
    FIXED_PUBLIC_BASE,
    PAIR_RECEIPT_SCHEMA,
    atomic_write_text,
    input_row_count,
    load_indices_file,
    load_selected,
    neutral_program,
    public_graph,
    scrub_and_rebase_instruction,
    select_shard,
    validate_pair_receipt,
)


class ExactCrossArchPairBuilderTests(unittest.TestCase):
    def test_alignment_indices_are_explicitly_converted_to_zero_based(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "alignment.jsonl"
            path.write_text(
                '{"original_line":5,"semantic_group":9,"split_line":1}\n'
                '{"original_line":2,"semantic_group":3,"split_line":2}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_indices_file(path, "alignment-jsonl"), [1, 4])

            path.write_text(
                '{"original_line":5,"split_line":1}\n'
                '{"original_line":2,"split_line":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid_split_line"):
                load_indices_file(path, "alignment-jsonl")

    def test_contiguous_shards_are_deterministic_disjoint_and_complete(self) -> None:
        source = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
        shards = [select_shard(source, 4, index)[0] for index in range(3)]
        self.assertEqual(shards, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        self.assertEqual(
            sorted(value for shard in shards for value in shard), list(range(10))
        )
        self.assertEqual(sum(len(shard) for shard in shards), 10)
        with self.assertRaisesRegex(ValueError, "out_of_range"):
            select_shard(source, 4, 3)

    def test_jsonl_row_count_rejects_blank_physical_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "private.jsonl"
            path.write_text('{"row":0}\n\n{"row":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank_jsonl_line"):
                input_row_count(path)

    def test_atomic_write_replaces_file_and_leaves_no_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.jsonl"
            atomic_write_text(path, "old\n")
            atomic_write_text(path, "new\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
            self.assertEqual([item.name for item in path.parent.iterdir()], [path.name])

    def test_resume_receipt_requires_complete_dual_isa_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            receipt_path = output / "pair_receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema": PAIR_RECEIPT_SCHEMA,
                        "source_release_index": 4,
                        "pair_slot": 0,
                        "source_sha256": "source",
                        "program_sha256": "program",
                        "semantic_pair_id": "pair",
                        "build_contract_sha256": "contract",
                        "public_rows": [{}],
                        "model_rows": [{}],
                        "private_rows": [{}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "two_architecture_rows"):
                validate_pair_receipt(
                    receipt_path,
                    source_index=4,
                    pair_slot=0,
                    source_sha256="source",
                    program_sha256="program",
                    pair_id="pair",
                    contract_sha256="contract",
                    output_dir=output,
                )

    def test_load_selected_preserves_release_order_not_request_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "private.jsonl"
            path.write_text(
                "".join(json.dumps({"row": index}) + "\n" for index in range(5)),
                encoding="utf-8",
            )

            selected = load_selected(path, {4, 1, 3})

        self.assertEqual([index for index, _ in selected], [1, 3, 4])
        self.assertEqual([row["row"] for _, row in selected], [1, 3, 4])

    def test_load_selected_fails_closed_when_an_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "private.jsonl"
            path.write_text('{"row":0}\n{"row":1}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, r"missing requested indices: \[7\]"
            ):
                load_selected(path, {0, 7})

    def test_neutral_program_requires_both_retention_pragmas(self) -> None:
        row = {
            "dart_source": (
                "@pragma('vm:never-inline')\n"
                "@pragma('vm:entry-point')\n"
                "int candidate(int x) => x + 1;\n"
            ),
            "tests": "void main() { if (candidate(1) != 2) throw 'bad'; }\n",
        }
        source, tests, program = neutral_program(row)
        self.assertEqual(
            program,
            "import 'dart:async';\n"
            "import 'dart:convert';\n" + source.rstrip() + "\n\n" + tests,
        )
        with self.assertRaisesRegex(ValueError, "entry_point"):
            neutral_program({**row, "dart_source": "int candidate(int x) => x + 1;"})

    def test_neutral_program_deduplicates_standard_harness_imports(self) -> None:
        row = {
            "dart_source": (
                "import 'dart:async';\n"
                "@pragma('vm:never-inline')\n"
                "@pragma('vm:entry-point')\n"
                "void candidate() {}\n"
            ),
            "tests": "Future<void> main() async { await Future.sync(candidate); }\n",
        }
        _, _, program = neutral_program(row)
        self.assertEqual(program.count("import 'dart:async';"), 1)
        self.assertEqual(program.count("import 'dart:convert';"), 1)

    def test_symbol_scrub_is_contextual_and_addresses_are_rebased(self) -> None:
        aliases: dict[str, str] = {}
        nested: dict[str, str] = {}
        internal = scrub_and_rebase_instruction(
            "b.eq 1010 <candidate+0x10>",
            true_base=0x1000,
            true_stop=0x1100,
            aliases=aliases,
            nested=nested,
        )
        external = scrub_and_rebase_instruction(
            "bl abc <SecretType.helper>",
            true_base=0x1000,
            true_stop=0x1100,
            aliases=aliases,
            nested=nested,
        )
        self.assertEqual(
            internal, f"b.eq 0x{FIXED_PUBLIC_BASE + 0x10:x} <candidate+0x10>"
        )
        self.assertEqual(external, "bl 0x0 <symbol_0>")

    def test_current_dual_isa_graph_builder_accepts_rebased_function(self) -> None:
        assembly = (
            'All functions matching regular expression "candidate":\n\n'
            "Dump of assembler code for function candidate:\n"
            "   0x0000000000100000 <+0>:\tmov rax,QWORD PTR [rsp+0x8]\n"
            "   0x0000000000100005 <+5>:\tret\n"
            "End of assembler dump.\n"
        )
        graph = public_graph(assembly, "x86_64")
        self.assertTrue(graph["integrity"]["valid"])
        self.assertEqual(graph["graph_v2"]["schema"], "antigravity-graph-v2.1")
        self.assertEqual(sum(len(block["instructions"]) for block in graph["cfg"]), 2)


if __name__ == "__main__":
    unittest.main()
