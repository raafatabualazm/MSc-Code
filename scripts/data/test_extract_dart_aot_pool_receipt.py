from __future__ import annotations

import json
import unittest

from scripts.data.extract_dart_aot_pool_receipt import (
    PoolReceiptError,
    build_pool_receipt,
    pool_index_from_offset,
    pool_offset_from_index,
)


def fixture_profile(*, string_value: str = "dead") -> dict:
    node_fields = ["type", "name", "id", "self_size", "edge_count"]
    edge_fields = ["type", "name_or_index", "to_node"]
    node_types = [
        "ArtificialRoot",
        "ObjectPool",
        "CanonicalString",
        "double",
        "TypeArguments",
        "Array",
    ]
    edge_types = ["context", "element", "property", "internal"]
    strings = [
        "root",
        "pool",
        string_value,
        "Unnamed [double] (nil)",
        "type arguments",
        "array",
    ]

    # Flat node offsets are 0, 5, 10, 15, 20 and 25.
    nodes = [
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        2,
        64,
        4,
        2,
        2,
        3,
        16,
        0,
        3,
        3,
        4,
        16,
        0,
        4,
        4,
        5,
        16,
        0,
        5,
        5,
        6,
        32,
        2,
    ]
    edges = [
        # Global pool indices map to offsets 0x787, 0x797, 0x7bf, 0x7c7.
        1,
        239,
        10,
        1,
        241,
        15,
        1,
        246,
        20,
        1,
        247,
        25,
        # The composite array recursively refers to the string and double.
        1,
        0,
        10,
        1,
        1,
        15,
    ]
    return {
        "snapshot": {
            "meta": {
                "node_fields": node_fields,
                "node_types": [node_types],
                "edge_fields": edge_fields,
                "edge_types": [edge_types],
            },
            "node_count": 6,
            "edge_count": 6,
        },
        "nodes": nodes,
        "edges": edges,
        "strings": strings,
    }


def fixture_disassembly(*, malformed_offset: bool = False) -> str:
    first_offset = "0x788" if malformed_offset else "0x787"
    return f"""Code for optimized function 'file:///private/original.dart_::_candidate' (RegularFunction) {{
0x2a    498b8787070000         movq rax,[pp+{first_offset}]   \"dead\"
0x68    4d8b9f97070000         movq tmp,[pp+0x797]   2.0
0xc9    4d8b9fbf070000         movq tmp,[pp+0x7bf]   TypeArguments: [String]
0xeb    498b87c7070000         movq rax,[pp+0x7c7]   _ImmutableList len:2
}}
Code for optimized function 'file:///private/original.dart_::_candidate_helper' (ClosureFunction) {{
0x4    498b8787070000          movq rax,[pp+0x787]   \"dead\"
}}
Code for optimized function 'file:///private/original.dart_::_other' (RegularFunction) {{
0x9    498b8787070000          movq rax,[pp+0x787]   \"dead\"
}}
Global object pool:
ObjectPool len:248 {{
""" + "".join(
        f"  [pp+0x{pool_offset_from_index(index):x}] placeholder (raw)\n"
        for index in range(239)
    ) + f"""  [pp+0x787] dead (obj)
  [pp+0x78f] placeholder (raw)
  [pp+0x797] 2.0 (obj)
  [pp+0x79f] placeholder (raw)
  [pp+0x7a7] placeholder (raw)
  [pp+0x7af] placeholder (raw)
  [pp+0x7b7] placeholder (raw)
  [pp+0x7bf] TypeArguments: [String] (obj)
  [pp+0x7c7] _ImmutableList len:2 (obj)
}}
"""


class PoolReceiptTests(unittest.TestCase):
    def test_pp_mapping_is_exact_and_fail_closed(self) -> None:
        self.assertEqual(pool_index_from_offset(0xF), 0)
        self.assertEqual(pool_index_from_offset(0x787), 239)
        self.assertEqual(pool_offset_from_index(239), 0x787)
        with self.assertRaisesRegex(PoolReceiptError, "invalid_pp_offset"):
            pool_index_from_offset(0x788)

    def test_receipt_is_target_scoped_typed_recursive_and_source_blind(self) -> None:
        receipt = build_pool_receipt(fixture_disassembly(), fixture_profile())
        entries = {entry["pool_offset"]: entry for entry in receipt["entries"]}

        self.assertEqual(entries["0x787"]["pp_offset"], 0x787)
        self.assertEqual(
            entries["0x787"]["literal"],
            {"type": "string", "code_units": [100, 101, 97, 100], "value": "dead"},
        )
        self.assertEqual(
            entries["0x787"]["uses"],
            [
                {"function_id": "candidate", "pc": "0x2a"},
                {"function_id": "candidate_descendant_0", "pc": "0x4"},
            ],
        )
        self.assertEqual(entries["0x797"]["literal"]["value"], 2.0)
        self.assertEqual(entries["0x7bf"]["nonliteral_kind"], "type_metadata")
        composite = entries["0x7c7"]
        self.assertEqual(composite["category"], "composite")
        self.assertEqual(composite["composite_type"], "array_storage")
        self.assertEqual(composite["elements"][0]["value"]["literal"]["value"], "dead")
        self.assertEqual(composite["elements"][1]["value"]["literal"]["value"], 2.0)
        self.assertTrue(composite["complete"])

        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("original.dart", serialized)
        self.assertNotIn("::_candidate", serialized)
        self.assertNotIn("::_other", serialized)
        self.assertTrue(receipt["source_blind"])
        self.assertEqual(receipt["summary"]["pool_uses"], 5)

    def test_misaligned_target_pool_offset_is_rejected(self) -> None:
        with self.assertRaisesRegex(PoolReceiptError, "invalid_pp_offset"):
            build_pool_receipt(fixture_disassembly(malformed_offset=True), fixture_profile())

    def test_global_pool_count_is_sealed(self) -> None:
        malformed = fixture_disassembly().replace("ObjectPool len:248", "ObjectPool len:249")
        with self.assertRaisesRegex(PoolReceiptError, "global_pool_entry_count"):
            build_pool_receipt(malformed, fixture_profile())

    def test_receipt_is_deterministic(self) -> None:
        first = build_pool_receipt(fixture_disassembly(), fixture_profile())
        second = build_pool_receipt(fixture_disassembly(), fixture_profile())
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
