from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from scripts.data.extract_dart_aot_pool_runtime_receipt import (
    RuntimePoolReceiptError,
    SDK_LAYOUT_CONTRACT,
    assemble_runtime_receipt,
    build_runtime_receipt,
    parse_objdump_pool_xrefs,
    parse_target_symbols,
)


class RuntimePoolReceiptTests(unittest.TestCase):
    def test_symbol_and_xref_parsers_are_target_scoped(self) -> None:
        nm_output = """000000000009031c 0000000000000096 t candidate
00000000000903dc 000000000000002d t candidate
0000000000092ce0 0000000000000160 t candidate.<anonymous closure>
0000000000094000 0000000000000020 t candidateExtra
0000000000095000 0000000000000020 t unrelated
"""
        symbols = parse_target_symbols(nm_output, "candidate")
        self.assertEqual(len(symbols), 3)
        self.assertEqual(
            [symbol["function_id"] for symbol in symbols],
            ["candidate", "candidate_exact_1", "candidate_descendant_0"],
        )

        objdump = """000000000009031c <candidate>:
   90357: mov    rax,QWORD PTR [r15+0x787]
   9035e: ret
   90367: mov    rax,QWORD PTR [r15+0x797]
"""
        xrefs = parse_objdump_pool_xrefs(objdump, function_id="candidate")
        self.assertEqual(
            xrefs,
            [
                {"function_id": "candidate", "pc": 0x90357, "pool_offset": 0x787},
                {"function_id": "candidate", "pc": 0x90367, "pool_offset": 0x797},
            ],
        )

    def test_assembled_receipt_is_deterministic_typed_and_address_free(self) -> None:
        functions = [
            {
                "function_id": "candidate",
                "scope_role": "exact",
                "aot_address": "0x9031c",
                "size_bytes": 150,
            }
        ]
        xrefs = [
            {"function_id": "candidate", "pc": 0x90367, "pool_offset": 0x797},
            {"function_id": "candidate", "pc": 0x90357, "pool_offset": 0x787},
            {"function_id": "candidate", "pc": 0x90370, "pool_offset": 0x787},
        ]
        resolved = [
            {
                "pool_offset": "0x797",
                "class_id": 62,
                "runtime_type": "Double",
                "category": "literal",
                "literal": {"type": "double", "bits_hex": "400c000000000000"},
            },
            {
                "pool_offset": "0x787",
                "class_id": 94,
                "runtime_type": "OneByteString",
                "category": "literal",
                "literal": {"type": "string", "code_units": [97, 108, 112, 104, 97]},
            },
        ]
        receipt = assemble_runtime_receipt(
            target="candidate",
            aot_sha256="a" * 64,
            runtime_sha256="b" * 64,
            functions=functions,
            xrefs=xrefs,
            resolved_entries=resolved,
        )
        self.assertEqual([entry["pool_offset"] for entry in receipt["entries"]], ["0x787", "0x797"])
        self.assertEqual([entry["pp_offset"] for entry in receipt["entries"]], [0x787, 0x797])
        self.assertEqual(len(receipt["entries"][0]["uses"]), 2)
        self.assertEqual(receipt["summary"]["literal_entries"], 2)
        self.assertEqual(
            receipt["pool_literal_presence"], {"bool": False, "null": False}
        )
        self.assertTrue(receipt["source_blind"])
        self.assertEqual(
            receipt["layout_contract"],
            "dart-3.12.2-linux-x64-object-layout-v1",
        )
        self.assertEqual(receipt["layout_contract"], SDK_LAYOUT_CONTRACT)

        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("tagged_value", serialized)
        self.assertNotIn("slot_address", serialized)
        self.assertNotIn("raw_address", serialized)

    def test_resolved_and_disassembled_offset_sets_must_match(self) -> None:
        with self.assertRaisesRegex(RuntimePoolReceiptError, "offset_set_mismatch"):
            assemble_runtime_receipt(
                target="candidate",
                aot_sha256="a",
                runtime_sha256="b",
                functions=[],
                xrefs=[{"function_id": "candidate", "pc": 1, "pool_offset": 0x787}],
                resolved_entries=[],
            )

    def test_bool_and_null_pool_presence_is_explicit(self) -> None:
        receipt = assemble_runtime_receipt(
            target="candidate",
            aot_sha256="a",
            runtime_sha256="b",
            functions=[],
            xrefs=[
                {"function_id": "candidate", "pc": 1, "pool_offset": 0x10},
                {"function_id": "candidate", "pc": 2, "pool_offset": 0x18},
            ],
            resolved_entries=[
                {
                    "pool_offset": "0x10",
                    "category": "literal",
                    "literal": {"type": "bool", "value": True},
                },
                {
                    "pool_offset": "0x18",
                    "category": "literal",
                    "literal": {"type": "null", "value": None},
                },
            ],
        )
        self.assertEqual(receipt["pool_literal_presence"], {"bool": True, "null": True})
        self.assertEqual(receipt["summary"]["bool_pool_entries"], 1)
        self.assertEqual(receipt["summary"]["null_pool_entries"], 1)

    def test_missing_target_symbols_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimePoolReceiptError, "target_symbols_not_found"):
            parse_target_symbols("0000000000010000 00000010 t unrelated\n", "candidate")

    @unittest.skipUnless(
        os.environ.get("DART_AOT_EMPTY_MAIN_FIXTURE")
        and os.environ.get("DART_AOTRUNTIME_3_12_2"),
        "set DART_AOT_EMPTY_MAIN_FIXTURE and DART_AOTRUNTIME_3_12_2 for Linux integration",
    )
    def test_empty_main_aot_is_resolved_before_candidate_execution(self) -> None:
        receipt = build_runtime_receipt(
            Path(os.environ["DART_AOT_EMPTY_MAIN_FIXTURE"]),
            Path(os.environ["DART_AOTRUNTIME_3_12_2"]),
        )
        literals = [entry["literal"] for entry in receipt["entries"] if entry["category"] == "literal"]
        self.assertIn(
            {"type": "string", "code_units": [114, 101, 116, 97, 105, 110, 101, 100]},
            literals,
        )
        self.assertIn({"type": "double", "bits_hex": "3ff8000000000000"}, literals)
        self.assertIn({"type": "int", "decimal": "4611686018427387904"}, literals)


if __name__ == "__main__":
    unittest.main()
