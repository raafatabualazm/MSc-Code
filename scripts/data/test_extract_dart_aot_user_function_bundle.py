from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.data.build_dart_user_symbol_attestation import (
    key_id_sha256,
    ordered_commitment,
    row_salt,
    symbol_digest,
)
from scripts.data.extract_dart_aot_user_function_bundle import (
    AttestedSymbols,
    UserFunctionExtractionError,
    build_user_function_bundle,
    canonical_sha256,
    discover_attested_direct_callees,
    external_symbol_evidence,
    model_projection,
    parse_combined_gdb_disassemblies,
    parse_combined_gdb_disassemblies_by_address,
    select_same_file_symbols,
)


KEY = bytes(range(32))
ATTESTATION_FILE_SHA256 = "f" * 64
SOURCE_ONLY_CONTRACT = {
    "contract": "source-only-program-empty-main-v1",
    "aot_row_schema": "phase0-s44-source-only-aot-row-v1",
    "analysis_program_sha256": "c" * 64,
    "function_source_sha256": "d" * 64,
    "producer_script_sha256": "e" * 64,
}


def make_attestation(
    task_id: str,
    *,
    functions: tuple[str, ...] = ("candidate", "helper"),
    types: tuple[str, ...] = (),
) -> AttestedSymbols:
    salt = row_salt(
        KEY,
        task_id=task_id,
        analysis_program_sha256=SOURCE_ONLY_CONTRACT[
            "analysis_program_sha256"
        ],
    )
    function_entries = [
        {
            "alias": f"AF{index}",
            "digest": symbol_digest(
                KEY,
                task_id=task_id,
                salt_hex=salt,
                kind="function",
                index=index,
                symbol=symbol,
            ),
        }
        for index, symbol in enumerate(functions)
    ]
    type_entries = [
        {
            "alias": f"T{index}",
            "digest": symbol_digest(
                KEY,
                task_id=task_id,
                salt_hex=salt,
                kind="type",
                index=index,
                symbol=symbol,
            ),
        }
        for index, symbol in enumerate(types)
    ]
    row = {
        "schema": "dart-user-symbol-attestation-v1",
        "task_id": task_id,
        "split": "train",
        "split_row": 0,
        "analysis_program_sha256": SOURCE_ONLY_CONTRACT[
            "analysis_program_sha256"
        ],
        "function_source_sha256": SOURCE_ONLY_CONTRACT[
            "function_source_sha256"
        ],
        "producer_script_sha256": SOURCE_ONLY_CONTRACT[
            "producer_script_sha256"
        ],
        "key_id_sha256": key_id_sha256(KEY),
        "salt_hex": salt,
        "function_symbols": function_entries,
        "type_symbols": type_entries,
        "completeness": {
            "complete_source_symbols_projection": True,
            "source_symbols_bound_to_transform_metadata": True,
            "only_dart_scheme_imports": True,
            "ordered_function_count": len(function_entries),
            "ordered_type_count": len(type_entries),
            "ordered_commitment": ordered_commitment(
                KEY,
                task_id=task_id,
                salt_hex=salt,
                function_digests=[
                    entry["digest"] for entry in function_entries
                ],
                type_digests=[
                    entry["digest"] for entry in type_entries
                ],
            ),
        },
    }
    return AttestedSymbols(row, KEY)


INFO_FUNCTIONS = """All defined functions:

File dart:core/string.dart:
10:\tstatic void _StringBase._interpolate(void);

File file:///private/build/program.dart:
8:\tstatic void helper(void);
3:\tstatic void candidate(void);
4:\tstatic void candidate.<anonymous closure>(void);
20:\tstatic void main(void);

Non-debugging symbols:
0x0000000000001000  stub AllocateArray
"""


DISASSEMBLIES = """Dump of assembler code for function helper:
   0x0000000000001200 <+0>:\t55\tpush rbp
   0x0000000000001201 <+1>:\t48 89 e5\tmov rbp,rsp
   0x0000000000001204 <+4>:\tc3\tret
End of assembler dump.
Dump of assembler code for function candidate:
   0x0000000000001000 <+0>:\t55\tpush rbp
   0x0000000000001001 <+1>:\te8 fa 01 00 00\tcall 0x1200 <helper>
   0x0000000000001006 <+6>:\t75 05\tjne 0x100d <candidate+13>
   0x0000000000001008 <+8>:\te8 03 03 00 00\tcall 0x1310 <dart:core__StringBase._interpolate>
   0x000000000000100d <+13>:\t41 ff 56 20\tcall QWORD PTR [r14+0x20]
   0x0000000000001011 <+17>:\tc3\tret
End of assembler dump.
Dump of assembler code for function candidate.<anonymous closure>:
   0x0000000000001100 <+0>:\tb8 01 00 00 00\tmov eax,0x1
   0x0000000000001105 <+5>:\tc3\tret
End of assembler dump.
Dump of assembler code for function main:
   0x0000000000001300 <+0>:\t31 c0\txor eax,eax
   0x0000000000001302 <+2>:\tc3\tret
End of assembler dump.
"""


def build_fixture_bundle(
    *,
    task_id: str,
    parsed: dict,
    selected_symbols: list[str],
    attestation: AttestedSymbols,
    gdb_file_symbols: list[str] | None = None,
    trusted_runtime_symbols: set[str] | None = None,
    known_nonruntime_symbols: set[str] | None = None,
) -> dict:
    return build_user_function_bundle(
        task_id=task_id,
        root_symbol="candidate",
        private_file_identity="file:///private/build/program.dart",
        selected_symbols=selected_symbols,
        parsed_by_symbol=parsed,
        info_output_sha256="a" * 64,
        aot_sha256="b" * 64,
        aot_size_bytes=1234,
        source_only_contract=SOURCE_ONLY_CONTRACT,
        symbol_attestation=attestation,
        symbol_attestation_file_sha256=ATTESTATION_FILE_SHA256,
        gdb_file_symbols=gdb_file_symbols,
        trusted_runtime_symbols=trusted_runtime_symbols,
        known_nonruntime_symbols=known_nonruntime_symbols,
        split="train",
        split_row=0,
    )


class SameFileUserFunctionBundleTests(unittest.TestCase):
    def test_standalone_cli_help_bootstraps_repository_imports(self) -> None:
        script = Path(__file__).with_name(
            "extract_dart_aot_user_function_bundle.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--symbol-attestation", result.stdout)
        self.assertIn("--symbol-attestation-key-file", result.stdout)

    def test_file_selection_is_exact_complete_and_private(self) -> None:
        private_file, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        self.assertEqual(private_file, "file:///private/build/program.dart")
        self.assertEqual(
            symbols,
            ["helper", "candidate", "candidate.<anonymous closure>", "main"],
        )

    def test_duplicate_root_sections_fail_closed(self) -> None:
        duplicate = INFO_FUNCTIONS.replace(
            "Non-debugging symbols:",
            "File file:///other/program.dart:\n"
            "1:\tstatic void candidate(void);\n\n"
            "Non-debugging symbols:",
        )
        with self.assertRaisesRegex(
            UserFunctionExtractionError, "root_file_section_count:2"
        ):
            select_same_file_symbols(duplicate)

    def test_unparsed_selected_declaration_fails_closed(self) -> None:
        broken = INFO_FUNCTIONS.replace(
            "20:\tstatic void main(void);",
            "20:\tstatic int unsupported_signature(int);",
        )
        with self.assertRaisesRegex(
            UserFunctionExtractionError,
            "unparsed_declarations_in_root_file",
        ):
            select_same_file_symbols(broken)

    def test_complete_bundle_preserves_cfg_calls_runtime_and_truth_fields(
        self,
    ) -> None:
        private_file, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        trusted_runtime, known_nonruntime = external_symbol_evidence(
            INFO_FUNCTIONS, private_file
        )
        parsed = parse_combined_gdb_disassemblies(
            DISASSEMBLIES, symbols
        )
        task_id = "sigless_test"
        bundle = build_fixture_bundle(
            task_id=task_id,
            parsed=parsed,
            selected_symbols=symbols,
            attestation=make_attestation(task_id),
            trusted_runtime_symbols=trusted_runtime,
            known_nonruntime_symbols=known_nonruntime,
        )

        self.assertEqual(
            [function["function_id"] for function in bundle["functions"]],
            ["F0", "F1", "F2"],
        )
        self.assertEqual(
            [function["function_kind"] for function in bundle["functions"]],
            ["RegularFunction", "ClosureFunction", "RegularFunction"],
        )
        root = bundle["functions"][0]
        texts = [
            instruction["text"] for instruction in root["instructions"]
        ]
        self.assertIn("call @F2", texts)
        self.assertIn("jne @L+0xd", texts)
        self.assertIn("call @X0", texts)
        self.assertIn("call QWORD PTR [r14+0x20]", texts)
        self.assertEqual(
            root["integrity"]["represented_instruction_count"],
            len(root["instructions"]),
        )
        self.assertEqual(
            root["integrity"]["pruned_unreachable_block_count"], 0
        )
        self.assertEqual(
            [
                transfer["transfer_kind"]
                for transfer in bundle["interfunction_transfers"]
            ],
            [
                "direct_internal_call",
                "direct_external_call",
                "indirect_call",
            ],
        )
        self.assertEqual(
            bundle["external_symbols"],
            [
                {
                    "external_id": "X0",
                    "symbol": "dart:core__StringBase._interpolate",
                    "symbol_class": "trusted_runtime",
                }
            ],
        )
        accounting = bundle["accounting"]
        self.assertEqual(accounting["selected_function_count"], 4)
        self.assertEqual(accounting["gdb_file_function_count"], 4)
        self.assertEqual(accounting["attested_recursive_function_count"], 0)
        self.assertEqual(accounting["producer_scaffold_function_count"], 1)
        self.assertEqual(accounting["emitted_function_count"], 3)
        self.assertEqual(accounting["call_site_count"], 3)
        self.assertEqual(accounting["represented_call_site_count"], 3)
        self.assertFalse(bundle["source_text_read"])
        self.assertFalse(bundle["raw_source_names_serialized"])
        self.assertFalse(bundle["raw_source_paths_serialized"])
        self.assertTrue(bundle["source_symbol_attestation_used"])
        self.assertTrue(bundle["source_symbol_attestation_is_keyed"])
        self.assertEqual(
            bundle["symbol_attestation_binding"][
                "attestation_file_sha256"
            ],
            ATTESTATION_FILE_SHA256,
        )

        serialized = json.dumps(bundle, sort_keys=True)
        self.assertNotIn("program.dart", serialized)
        self.assertNotIn("file://", serialized)
        self.assertNotIn('"raw_symbol"', serialized)
        self.assertNotIn("<helper>", serialized)
        self.assertNotIn("<candidate", serialized)

        projection = model_projection(bundle)
        self.assertEqual(
            bundle["model_projection_sha256"],
            canonical_sha256(projection),
        )
        self.assertEqual(
            json.loads(
                json.dumps(
                    projection,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            projection,
        )

    def test_user_constructor_is_recursively_recovered_by_exact_address(
        self,
    ) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        initial_text = DISASSEMBLIES.replace(
            "dart:core__StringBase._interpolate", "new SecretType"
        )
        parsed = parse_combined_gdb_disassemblies(initial_text, symbols)
        task_id = "sigless_constructor"
        attestation = make_attestation(
            task_id, types=("SecretType",)
        )
        targets = discover_attested_direct_callees(
            parsed, attestation
        )
        self.assertEqual(
            targets,
            [
                {
                    "address": 0x1310,
                    "raw_annotation": "new SecretType",
                    "recovery_kind": "constructor",
                }
            ],
        )
        recovered_text = """Dump of assembler code for function new SecretType:
   0x0000000000001310 <+0>:\te9 eb 00 00 00\tjmp 0x1400 <stub AllocateObject>
End of assembler dump.
"""
        recovered = parse_combined_gdb_disassemblies_by_address(
            recovered_text, targets
        )
        parsed.update(recovered)
        selected = [*symbols, *recovered]
        bundle = build_fixture_bundle(
            task_id=task_id,
            parsed=parsed,
            selected_symbols=selected,
            gdb_file_symbols=symbols,
            attestation=attestation,
        )
        self.assertEqual(
            bundle["accounting"]["attested_recursive_function_count"], 1
        )
        self.assertEqual(
            bundle["accounting"]["selected_function_count"], 5
        )
        constructor = next(
            function
            for function in bundle["functions"]
            if function["function_kind"] == "Constructor"
        )
        self.assertEqual(
            constructor["function_kind_evidence"],
            "keyed_attestation_and_exact_call_target_address",
        )
        root_texts = [
            instruction["text"]
            for instruction in bundle["functions"][0]["instructions"]
        ]
        self.assertIn("call @F3", root_texts)
        self.assertEqual(bundle["type_aliases"], [{"type_alias": "T0"}])
        serialized = json.dumps(bundle, sort_keys=True)
        self.assertNotIn("SecretType", serialized)
        self.assertNotIn("new SecretType", serialized)

    def test_duplicate_function_names_are_disambiguated_by_address(
        self,
    ) -> None:
        output = """Dump of assembler code for function new StateError:
   0x0000000000002000 <+0>:\tc3\tret
End of assembler dump.
Dump of assembler code for function new StateError:
   0x0000000000003000 <+0>:\t90\tnop
   0x0000000000003001 <+1>:\tc3\tret
End of assembler dump.
"""
        parsed = parse_combined_gdb_disassemblies_by_address(
            output,
            [
                {"address": 0x2000, "recovery_kind": "constructor"},
                {"address": 0x3000, "recovery_kind": "constructor"},
            ],
        )
        self.assertEqual(
            sorted(
                function["entry_address"]
                for function in parsed.values()
            ),
            [0x2000, 0x3000],
        )
        with self.assertRaisesRegex(
            UserFunctionExtractionError,
            "address_disassembly_entry_mismatch",
        ):
            parse_combined_gdb_disassemblies_by_address(
                output.split(
                    "Dump of assembler code for function new StateError:",
                    2,
                )[0]
                + "Dump of assembler code for function new StateError:"
                + output.split(
                    "Dump of assembler code for function new StateError:",
                    2,
                )[1],
                [{"address": 0x2001, "recovery_kind": "constructor"}],
            )

    def test_type_assertion_uses_alias_without_recovering_stub(self) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        hostile = DISASSEMBLIES.replace(
            "dart:core__StringBase._interpolate",
            "assert type is Result<int>",
        )
        parsed = parse_combined_gdb_disassemblies(hostile, symbols)
        task_id = "sigless_type_assert"
        attestation = make_attestation(task_id, types=("Result",))
        self.assertEqual(
            discover_attested_direct_callees(parsed, attestation), []
        )
        bundle = build_fixture_bundle(
            task_id=task_id,
            parsed=parsed,
            selected_symbols=symbols,
            attestation=attestation,
        )
        self.assertIn(
            {
                "external_id": "X0",
                "symbol": "assert type is @T0<int>",
                "symbol_class": "trusted_runtime",
            },
            bundle["external_symbols"],
        )
        self.assertEqual(
            bundle["accounting"]["attested_type_assertion_count"], 1
        )
        self.assertNotIn("Result", json.dumps(bundle, sort_keys=True))

    def test_runtime_owner_and_generic_argument_do_not_claim_user_type(
        self,
    ) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        task_id = "sigless_collision"
        attestation = make_attestation(
            task_id,
            functions=("candidate", "map"),
            types=("Dog",),
        )
        for label, trusted, expected_symbol in (
            ("_List.map", {"_List.map"}, "_List.map"),
            ("map", set(), None),
            ("new List<Dog>", set(), None),
        ):
            with self.subTest(label=label):
                hostile = DISASSEMBLIES.replace(
                    "dart:core__StringBase._interpolate", label
                )
                parsed = parse_combined_gdb_disassemblies(
                    hostile, symbols
                )
                self.assertEqual(
                    discover_attested_direct_callees(
                        parsed, attestation
                    ),
                    [],
                )
                bundle = build_fixture_bundle(
                    task_id=task_id,
                    parsed=parsed,
                    selected_symbols=symbols,
                    attestation=attestation,
                    trusted_runtime_symbols=trusted,
                )
                self.assertEqual(
                    bundle["external_symbols"][0]["symbol"],
                    expected_symbol,
                )
                serialized = json.dumps(bundle, sort_keys=True)
                if label != "_List.map":
                    self.assertNotIn(label, serialized)
                self.assertNotIn("Dog", serialized)

    def test_source_path_annotation_is_rejected_before_serialization(
        self,
    ) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        hostile = DISASSEMBLIES.replace(
            "dart:core__StringBase._interpolate",
            "file:///private/gold.dart_::_secretHelper",
        )
        parsed = parse_combined_gdb_disassemblies(hostile, symbols)
        task_id = "sigless_path_leak"
        with self.assertRaisesRegex(
            UserFunctionExtractionError,
            "source_identity_in_external_annotation",
        ):
            build_fixture_bundle(
                task_id=task_id,
                parsed=parsed,
                selected_symbols=symbols,
                attestation=make_attestation(task_id),
            )

    def test_untrusted_runtime_stub_is_neutralized_without_raw_text(
        self,
    ) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        hostile = DISASSEMBLIES.replace(
            "dart:core__StringBase._interpolate",
            "stub SecretRuntimeThunk",
        )
        parsed = parse_combined_gdb_disassemblies(hostile, symbols)
        task_id = "sigless_runtime_stub"
        bundle = build_fixture_bundle(
            task_id=task_id,
            parsed=parsed,
            selected_symbols=symbols,
            attestation=make_attestation(task_id),
        )
        serialized = json.dumps(bundle, sort_keys=True)
        self.assertNotIn("SecretRuntimeThunk", serialized)
        self.assertIn("call @X0", serialized)
        self.assertEqual(
            bundle["external_symbols"],
            [
                {
                    "external_id": "X0",
                    "symbol": None,
                    "symbol_class": "neutralized_untrusted_runtime",
                }
            ],
        )

    def test_missing_or_incomplete_attestation_fails_closed(self) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        parsed = parse_combined_gdb_disassemblies(
            DISASSEMBLIES, symbols
        )
        with self.assertRaisesRegex(
            UserFunctionExtractionError, "symbol_attestation_required"
        ):
            build_user_function_bundle(
                task_id="sigless_missing_attestation",
                root_symbol="candidate",
                private_file_identity="private",
                selected_symbols=symbols,
                parsed_by_symbol=parsed,
                info_output_sha256="a" * 64,
                aot_sha256="b" * 64,
                aot_size_bytes=1,
                source_only_contract=SOURCE_ONLY_CONTRACT,
                symbol_attestation=None,  # type: ignore[arg-type]
                symbol_attestation_file_sha256=ATTESTATION_FILE_SHA256,
            )

        valid = make_attestation("sigless_incomplete")
        incomplete = copy.deepcopy(valid.row)
        incomplete["completeness"][
            "complete_source_symbols_projection"
        ] = False
        with self.assertRaisesRegex(
            UserFunctionExtractionError, "symbol_attestation_incomplete"
        ):
            AttestedSymbols(incomplete, KEY)

        with self.assertRaisesRegex(
            UserFunctionExtractionError,
            "symbol_attestation_key_binding_mismatch",
        ):
            AttestedSymbols(valid.row, b"x" * 32)

    def test_missing_disassembly_is_never_treated_as_exclusion(self) -> None:
        _, symbols = select_same_file_symbols(INFO_FUNCTIONS)
        with self.assertRaisesRegex(
            UserFunctionExtractionError, "disassembly_set_mismatch"
        ):
            parse_combined_gdb_disassemblies(
                DISASSEMBLIES.replace(
                    "Dump of assembler code for function helper:",
                    "Dump of assembler code for function omitted_helper:",
                ),
                symbols,
            )


if __name__ == "__main__":
    unittest.main()
