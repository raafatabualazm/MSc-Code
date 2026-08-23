from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.fixed_scrub_v3_gates import (
    DEFAULT_REFERENCE_COMPILE_FAILURES,
    DEFAULT_REFERENCE_PASS_FAILURES,
    DEFAULT_STUB_COMPILE_FAILURES,
    GateError,
    _allowed_semantic_collision,
    _semantic_leaks,
    build_bundle,
    canonical_cfg,
    extract_user_function_identifiers,
    main,
    original_source_sha256,
    run_dart_gates,
    validate_contracts_and_hygiene,
    validate_frozen_cfg_parity,
    validate_harness_patch,
    validate_rendered_prompts,
    validate_toolchain_and_summaries,
)


DART_VERSION = "Dart SDK version: 3.11.5 (stable)"
GDB_VERSION = "GNU gdb (GDB) 17.1"


def _assembly(name: str) -> str:
    return (
        f'All functions matching regular expression "{name}":\n'
        f"Dump of assembler code for function {name}:\n"
        f"0x10 <{name}+0>:\tmov eax,edi\n"
        f"0x12 <{name}+2>:\tadd eax,0x1\n"
        f"0x15 <{name}+5>:\tret\n"
    )


def _cfg(name: str) -> list[dict]:
    return [
        {
            "id": 0,
            "label": "block_0",
            "start_address": "0x10",
            "instructions": [
                "mov eax,edi",
                "add eax,0x1",
                f"jmp 0x15 <{name}+5>",
            ],
            "predecessors": [],
            "successors": [1],
            "edge_types": ["unconditional_jump"],
            "instruction_count": 3,
            "block_type": "jump",
        },
        {
            "id": 1,
            "label": "block_1",
            "start_address": "0x15",
            "instructions": ["ret"],
            "predecessors": [0],
            "successors": [],
            "edge_types": [],
            "instruction_count": 1,
            "block_type": "return",
        },
    ]


def _edges() -> list[dict]:
    return [{"source": 0, "target": 1, "edge_type": "unconditional_jump"}]


def _build_stamp() -> dict:
    return {
        "mode": "rename_frozen",
        "assembly_derivation": "text_rename_of_frozen_benchmark_assembly",
        "asserted_frozen_dart_version": DART_VERSION,
        "asserted_frozen_gdb_version": GDB_VERSION,
        "extractor_sha256": "e" * 64,
    }


def _benchmark_row(task_id: str, name: str) -> dict:
    return {
        "task_id": task_id,
        "filename": f"{task_id}.dart",
        "function": name,
        "camel_case_function_name": name,
        "dart_function_signature": f"int {name}(int value)",
        "dart_source": f"int {name}(int value) {{ return value + 1; }}",
        "tests": f"void main() {{ final candidate = {name}; expect(candidate(1), 2); }}",
        "assembly": _assembly(name),
        "cfg": _cfg(name),
        "edges": _edges(),
    }


def _private_row(
    benchmark: dict, mode: str, opaque_id: str, target: str = "fn0"
) -> dict:
    source_hash = original_source_sha256(benchmark)
    exact = mode == "neutral_exact"
    public_signature = f"int {target}(int a)" if exact else ""
    return {
        "task_id": opaque_id,
        "filename": f"{opaque_id}.dart",
        "function": target,
        "camel_case_function_name": target,
        "dart_function_signature": public_signature,
        "public_prompt_signature": public_signature,
        "evaluation_only_dart_function_signature": f"int {target}(int value)",
        "prompt_signature_mode": "exact" if exact else "name_only",
        "dart_source": f"int {target}(int value) {{ return value + 1; }}",
        "tests": f"void main() {{ final implementation = {target}; expect(implementation(1), 2); }}",
        "assembly": _assembly(target),
        "cfg": _cfg(target),
        "edges": _edges(),
        "benchmark_protocol": {
            "schema": "dart-signature-scrubbed-v3",
            "public_signature_mode": mode,
            "neutral_target_name": target,
            "original_source_sha256": source_hash,
            "semantic_function_name_sha256": hashlib.sha256(
                benchmark["function"].lower().encode()
            ).hexdigest(),
            "prompt_exposes": (
                ["assembly_or_graph", "neutral_target_name", "typed_signature_neutral_name_neutral_params"]
                if exact
                else ["assembly_or_graph", "neutral_target_name"]
            ),
            "prompt_withholds": (
                ["parameter_names", "reference_source", "tests", "semantic_function_name"]
                if exact
                else [
                    "return_type",
                    "arity",
                    "parameter_count",
                    "parameter_types",
                    "parameter_names",
                    "reference_source",
                    "tests",
                    "semantic_function_name",
                ]
            ),
            "assembly_build": {
                **_build_stamp(),
                "frozen_assembly_sha256": "f" * 64,
            },
        },
    }


def _public_row(private: dict) -> dict:
    row = copy.deepcopy(private)
    for field in ("dart_source", "evaluation_only_dart_function_signature", "tests"):
        row.pop(field)
    protocol = row["benchmark_protocol"]
    for field in (
        "original_source_sha256",
        "semantic_function_name_sha256",
    ):
        protocol.pop(field)
    protocol["assembly_build"].pop("frozen_assembly_sha256")
    return row


def _fixture():
    benchmark = [_benchmark_row("10", "alphaTask"), _benchmark_row("11", "betaTask")]
    name_private = [
        _private_row(benchmark[0], "name_only", "n_alpha"),
        _private_row(benchmark[1], "name_only", "n_beta"),
    ]
    exact_private = [
        _private_row(benchmark[0], "neutral_exact", "e_alpha"),
        _private_row(benchmark[1], "neutral_exact", "e_beta"),
    ]
    # Both public arms use the same underlying shuffled source order.
    name_public = [_public_row(name_private[1]), _public_row(name_private[0])]
    exact_public = [_public_row(exact_private[1]), _public_row(exact_private[0])]
    return benchmark, name_public, name_private, exact_public, exact_private


def _bundle():
    benchmark, name_public, name_private, exact_public, exact_private = _fixture()
    return build_bundle(
        nameonly_public=name_public,
        nameonly_private=name_private,
        neutralexact_public=exact_public,
        neutralexact_private=exact_private,
        benchmark=benchmark,
        expected_rows=2,
        target_name="fn0",
    )


def _summary(mode: str) -> dict:
    return {
        "schema": "dart-signature-scrubbed-v3",
        "public_signature_mode": mode,
        "effective_id_salt_sha256": "a" * 64,
        "target_name": "fn0",
        "shuffle_public_seed": 4343,
        "accepted_rows": 2,
        "rejected_rows": 0,
        "toolchain": {
            "asserted_frozen_dart_version": DART_VERSION,
            "asserted_frozen_gdb_version": GDB_VERSION,
        },
    }


class StaticGateTests(unittest.TestCase):
    def test_complete_static_bundle_passes(self) -> None:
        bundle = _bundle()
        self.assertIn("fingerprint clean", validate_contracts_and_hygiene(bundle))
        self.assertIn("exact edges", validate_frozen_cfg_parity(bundle))
        self.assertIn("prompt schema", validate_rendered_prompts(bundle))
        evidence = validate_toolchain_and_summaries(
            bundle,
            expected_dart_version=DART_VERSION,
            expected_gdb_version=GDB_VERSION,
            nameonly_summary=_summary("name_only"),
            neutralexact_summary=_summary("neutral_exact"),
        )
        self.assertIn("Dart SDK version: 3.11.5", evidence)

    def test_public_order_must_map_to_same_source_sequence(self) -> None:
        benchmark, name_public, name_private, exact_public, exact_private = _fixture()
        exact_public.reverse()
        with self.assertRaisesRegex(GateError, "row order"):
            build_bundle(
                nameonly_public=name_public,
                nameonly_private=name_private,
                neutralexact_public=exact_public,
                neutralexact_private=exact_private,
                benchmark=benchmark,
                expected_rows=2,
            )

    def test_public_fingerprint_is_rejected_recursively(self) -> None:
        bundle = _bundle()
        bundle.nameonly.public[0]["benchmark_protocol"]["assembly_build"][
            "frozen_assembly_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(GateError, "frozen_assembly_sha256"):
            validate_contracts_and_hygiene(bundle)

    def test_semantic_name_leak_is_rejected(self) -> None:
        bundle = _bundle()
        bundle.nameonly.public[0]["assembly"] += "\n; betaTask"
        with self.assertRaisesRegex(GateError, "semantic-name leak"):
            validate_contracts_and_hygiene(bundle)

    def test_cfg_instruction_drift_is_rejected(self) -> None:
        bundle = _bundle()
        bundle.neutralexact.public[0]["cfg"][0]["instructions"][0] = "xor eax,eax"
        with self.assertRaisesRegex(GateError, "CFG drift"):
            validate_frozen_cfg_parity(bundle)

    def test_summary_must_not_expose_raw_salt(self) -> None:
        summary = _summary("name_only")
        summary["effective_id_salt"] = "secret"
        with self.assertRaisesRegex(GateError, "exposes effective salt"):
            validate_toolchain_and_summaries(
                _bundle(), nameonly_summary=summary
            )

    def test_harness_crash_classifier_is_patched(self) -> None:
        self.assertIn("front-end crash", validate_harness_patch())


class AddCollisionRegressionTests(unittest.TestCase):
    def test_real_gdb_opcode_and_cfg_opcode_are_benign(self) -> None:
        gdb_line = "   0x0000000000090ace <+30>:\tadd    rcx,rdx"
        self.assertTrue(
            _allowed_semantic_collision(
                "add", gdb_line, gdb_line.index("add"), ("assembly",)
            )
        )
        cfg_instruction = "add    rax,rax"
        self.assertTrue(
            _allowed_semantic_collision(
                "add",
                cfg_instruction,
                cfg_instruction.index("add"),
                ("cfg", "4", "instructions", "2"),
            )
        )

    def test_comparator_identical_qualified_sdk_members_are_benign(self) -> None:
        lines = (
            "77:\tstatic void _AsyncStarStreamController.add(void);",
            "283:\tstatic void List.add(void);",
            "35:\tstatic void _TimerHeap.add(void);",
        )
        comparator = "\n".join(lines)
        for line in lines:
            with self.subTest(line=line):
                occurrence = comparator.index(line) + line.index("add")
                self.assertTrue(
                    _allowed_semantic_collision(
                        "add",
                        comparator,
                        occurrence,
                        ("assembly",),
                        comparator_assembly=comparator,
                    )
                )
        encode_line = "236:\tstatic void JsonCodec.encode(void);"
        self.assertTrue(
            _allowed_semantic_collision(
                "encode",
                encode_line,
                encode_line.index("encode"),
                ("assembly",),
                comparator_assembly=encode_line,
            )
        )

    def test_bare_target_contexts_and_introduced_member_are_leaks(self) -> None:
        comparator_lines = (
            'All functions matching regular expression "add":',
            "1:\tstatic void add(void);",
            "   0x0000000000000010 <add+0>:\tret",
            "   0x0000000000000011 <+1>:\tcall add",
            "283:\tstatic void List.add(void);",
        )
        comparator = "\n".join(comparator_lines)
        rejected = (*comparator_lines[:4], "999:\tstatic void Evil.add(void);")
        for line in rejected:
            with self.subTest(line=line):
                occurrence = line.index("add")
                self.assertFalse(
                    _allowed_semantic_collision(
                        "add",
                        line,
                        occurrence,
                        ("assembly",),
                        comparator_assembly=comparator,
                    )
                )

    def test_new_duplicate_of_valid_sdk_line_fails_positional_context(self) -> None:
        sdk_line = "283:\tstatic void List.add(void);"
        comparator = {"assembly": sdk_line}
        public = {"assembly": sdk_line + "\n" + sdk_line}
        leaks = _semantic_leaks(
            public, re.compile(r"\badd\b"), comparator=comparator
        )
        self.assertEqual(leaks, ["assembly:add"])


class HelperLeakRegressionTests(unittest.TestCase):
    def test_target_qualified_symbol_backstops_source_extraction(self) -> None:
        comparator = {
            "function": "outerTask",
            "dart_source": "int outerTask(int value) => value;",
            "assembly": (
                "1:\tstatic void outerTask(void);\n"
                "9:\tstatic void outerTask.hiddenLocal(void);"
            ),
        }
        self.assertEqual(
            extract_user_function_identifiers(comparator),
            {"outerTask", "hiddenLocal"},
        )

    def test_parse_paren_group_target_qualified_symbol_is_a_leak(self) -> None:
        comparator = {
            "function": "parseNestedParens",
            "dart_source": (
                "List<int> parseNestedParens(String value) {\n"
                "  int parseParenGroup(String s) { return s.length; }\n"
                "  return [parseParenGroup(value)];\n"
                "}"
            ),
            "assembly": (
                "1:\tstatic void parseNestedParens(void);\n"
                "3:\tstatic void parseNestedParens.parseParenGroup(void);"
            ),
        }
        self.assertIn("parseParenGroup", extract_user_function_identifiers(comparator))
        public = {
            "assembly": (
                "1:\tstatic void fn0(void);\n"
                "3:\tstatic void fn0.parseParenGroup(void);"
            )
        }
        leaks = _semantic_leaks(
            public,
            re.compile(r"\bparseParenGroup\b"),
            comparator=comparator,
        )
        self.assertEqual(leaks, ["assembly:parseParenGroup"])

    def test_digits_sum_and_qualified_closure_symbols_are_leaks(self) -> None:
        comparator = {
            "function": "countNums",
            "dart_source": (
                "int countNums(List<int> values) {\n"
                "  int digitsSum(int n) { return n; }\n"
                "  return values.map((x) => digitsSum(x)).length;\n"
                "}"
            ),
            "assembly": (
                "1:\tstatic void countNums(void);\n"
                "3:\tstatic void countNums.digitsSum(void);\n"
                "12:\tstatic void countNums.digitsSum.<anonymous closure>(void);"
            ),
        }
        self.assertIn("digitsSum", extract_user_function_identifiers(comparator))
        public = {
            "assembly": (
                "1:\tstatic void fn0(void);\n"
                "3:\tstatic void fn0.digitsSum(void);\n"
                "12:\tstatic void fn0.digitsSum.<anonymous closure>(void);"
            )
        }
        leaks = _semantic_leaks(
            public, re.compile(r"\bdigitsSum\b"), comparator=comparator
        )
        self.assertEqual(
            leaks, ["assembly:digitsSum", "assembly:digitsSum"]
        )

    def test_opcode_named_helper_is_allowed_only_at_opcode_position(self) -> None:
        instruction = "   0x0000000000000010 <+0>:\txor    eax,eax"
        self.assertTrue(
            _allowed_semantic_collision(
                "xor", instruction, instruction.index("xor"), ("assembly",)
            )
        )
        symbol = "3:\tstatic void fn0.xor(void);"
        self.assertFalse(
            _allowed_semantic_collision(
                "xor", symbol, symbol.index("xor"), ("assembly",)
            )
        )


class CfgNormalizationRegressionTests(unittest.TestCase):
    def test_scalar_cfg_fields_are_single_values_not_character_sequences(self) -> None:
        row = {
            "task_id": "scalar_cfg",
            "cfg": [
                {
                    "id": 0,
                    "start_address": "0x10",
                    "instructions": "add rax,rax",
                    "predecessors": "entry",
                    "successors": "exit",
                    "edge_types": "linear_fallthrough",
                    "instruction_count": 1,
                    "block_type": "linear",
                }
            ],
        }
        block = canonical_cfg(row)[0]
        self.assertEqual(block[3], ("add rax,rax",))
        self.assertEqual(block[4], ("entry",))
        self.assertEqual(block[5], ("exit",))
        self.assertEqual(block[6], ("linear_fallthrough",))


class ExecutableGateTests(unittest.TestCase):
    @staticmethod
    def evaluator(code: str, tests: str, task_id: str, timeout: int, stability: int):
        del tests, task_id, timeout, stability
        if "throw UnimplementedError" in code:
            return True, False, "runtime stub", code
        return True, True, "", code

    def test_dart_gate_accepts_exact_mocked_sets(self) -> None:
        evidence = run_dart_gates(
            _bundle(),
            evaluator=self.evaluator,
            workers=2,
            expected_stub_compile_failures=set(),
            expected_reference_compile_failures=set(),
            expected_reference_pass_failures=set(),
        )
        self.assertIn("references compile 2/2, pass 2/2", evidence)

    def test_dart_gate_rejects_unexpected_failure_set(self) -> None:
        with self.assertRaisesRegex(GateError, "stub compile-failure set"):
            run_dart_gates(
                _bundle(),
                evaluator=self.evaluator,
                workers=1,
                expected_stub_compile_failures={"10"},
                expected_reference_compile_failures=set(),
                expected_reference_pass_failures=set(),
            )

    def test_default_known_defect_sets_are_frozen(self) -> None:
        self.assertEqual(DEFAULT_STUB_COMPILE_FAILURES, {"121", "127", "153", "161"})
        self.assertEqual(DEFAULT_REFERENCE_COMPILE_FAILURES, {"127", "153", "161"})
        self.assertEqual(
            DEFAULT_REFERENCE_PASS_FAILURES,
            {"8", "20", "54", "87", "90", "107", "112", "127", "136", "153", "155", "161"},
        )


class CliTests(unittest.TestCase):
    def test_skip_dart_cli_runs_all_static_gates(self) -> None:
        benchmark, name_public, name_private, exact_public, exact_private = _fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {
                "benchmark": benchmark,
                "name_public": name_public,
                "name_private": name_private,
                "exact_public": exact_public,
                "exact_private": exact_private,
            }
            paths = {}
            for label, rows in values.items():
                path = root / f"{label}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                paths[label] = path
            code = main(
                [
                    "--nameonly_public", str(paths["name_public"]),
                    "--nameonly_private", str(paths["name_private"]),
                    "--neutralexact_public", str(paths["exact_public"]),
                    "--neutralexact_private", str(paths["exact_private"]),
                    "--benchmark", str(paths["benchmark"]),
                    "--expected_rows", "2",
                    "--skip_dart",
                ]
            )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
