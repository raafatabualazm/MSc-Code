from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_synthetic_tasks_parallel as generator
from scripts.data.build_signature_scrubbed_eval import (
    build_one,
    effective_id_salt,
    extract_source_function_names,
    id_salt_summary_fields,
    neutralize_param_names,
    prepare_scrubbed_row,
    protocol_summary_fields,
    public_row,
    read_jsonl,
    rename_assembly_symbols,
    rename_frozen_toolchain_stamp,
    rewrite_tests,
)
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
    build_decoder_prompt,
)


class SignatureScrubbedEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = """@pragma('vm:entry-point')
int semanticSum(List<int> values, int bias) {
  return values.fold(bias, (total, value) => total + value);
}

void main() {
  assert(semanticSum([1, 2], 3) == 6);
}
"""
        self.tests = """void main() {
  final candidate = semanticSum;
  expect(candidate([1, 2], 3), 6);
  expect(candidate([], 4), 4);
}

void expect(dynamic actual, dynamic expected) {
  if (actual != expected) throw '$actual != $expected';
}
"""
        self.row = {
            "task_id": "old_0",
            "function": "semanticSum",
            "dart_function_signature": "int semanticSum(List<int> values, int bias)",
            "dart_source": self.source,
            "tests": self.tests,
            "assembly": "semanticSum assembly",
            "lang": "Dart",
        }

    def test_existing_harness_is_retargeted_without_self_shadowing(self) -> None:
        rewritten = rewrite_tests(self.tests, "semanticSum")
        self.assertIn("final implementation = candidate;", rewritten)
        self.assertIn("expect(implementation([1, 2], 3), 6);", rewritten)
        self.assertNotIn("final candidate = candidate;", rewritten)
        self.assertNotIn("semanticSum", rewritten)

    def test_prepared_row_hides_semantic_name_and_public_signature(self) -> None:
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt="unit-test",
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
        )
        self.assertEqual(prepared["function"], "candidate")
        self.assertEqual(prepared["dart_function_signature"], "")
        self.assertEqual(prepared["prompt_signature_mode"], "name_only")
        self.assertEqual(
            prepared["evaluation_only_dart_function_signature"],
            "int candidate(List<int> values, int bias)",
        )
        self.assertNotIn("semanticSum", prepared["dart_source"])
        self.assertNotIn("semanticSum", prepared["tests"])
        self.assertEqual(
            prepared["benchmark_protocol"]["semantic_function_name_sha256"],
            hashlib.sha256(b"semanticsum").hexdigest(),
        )
        self.assertIn("arity", prepared["benchmark_protocol"]["prompt_withholds"])
        self.assertIn(
            "parameter_count", prepared["benchmark_protocol"]["prompt_withholds"]
        )

    def test_custom_opaque_target_threads_through_the_protocol(self) -> None:
        arm_salt = effective_id_salt("unit-test", "name_only", "fn0")
        self.assertEqual(arm_salt, "unit-test|name_only|target_name=fn0")
        self.assertEqual(
            effective_id_salt("unit-test", "name_only"),
            "unit-test|name_only",
        )
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt=arm_salt,
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
            target_name="fn0",
        )
        default_prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt=effective_id_salt("unit-test", "name_only"),
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
        )
        self.assertEqual(prepared["function"], "fn0")
        self.assertEqual(prepared["camel_case_function_name"], "fn0")
        self.assertEqual(
            prepared["evaluation_only_dart_function_signature"],
            "int fn0(List<int> values, int bias)",
        )
        self.assertIn("int fn0(List<int> values, int bias)", prepared["dart_source"])
        self.assertIn("final candidate = fn0;", prepared["tests"])
        self.assertNotIn("semanticSum", prepared["dart_source"] + prepared["tests"])
        self.assertNotEqual(prepared["task_id"], default_prepared["task_id"])
        self.assertEqual(prepared["benchmark_protocol"]["neutral_target_name"], "fn0")

        prepared["assembly"] = "Dump of assembler code for function fn0:\nret"
        prompt = build_decoder_prompt(prepared)
        self.assertIn("function named exactly fn0.", prompt)

        dump = (
            "Dump of assembler code for function semanticSum:\n"
            "   0x1000 <semanticSum+0>:\tret\n"
        )
        renamed = rename_assembly_symbols(dump, "semanticSum", target_name="fn0")
        self.assertIn("function fn0:", renamed)
        self.assertIn("<fn0+0>", renamed)
        self.assertNotIn("candidate", renamed)

    def test_custom_opaque_target_supports_neutral_exact_signature(self) -> None:
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt=effective_id_salt("unit-test", "neutral_exact", "fn0"),
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
            public_signature_mode="neutral_exact",
            target_name="fn0",
        )
        self.assertEqual(
            prepared["dart_function_signature"],
            "int fn0(List<int> a, int b)",
        )

    def test_summary_can_redact_effective_id_salt(self) -> None:
        secret = "high-entropy-secret|name_only|target_name=fn0"
        self.assertEqual(id_salt_summary_fields(secret), {"effective_id_salt": secret})
        redacted = id_salt_summary_fields(secret, redact=True)
        self.assertEqual(
            redacted,
            {
                "effective_id_salt_sha256": hashlib.sha256(
                    secret.encode("utf-8")
                ).hexdigest()
            },
        )
        self.assertNotIn(secret, json.dumps(redacted))

    def test_v3_row_and_summary_redact_effective_id_salt(self) -> None:
        schema = "dart-signature-scrubbed-v3"
        secret = "another-high-entropy-secret|name_only|target_name=fn0"
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt=secret,
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
            target_name="fn0",
            protocol_schema=schema,
        )
        self.assertEqual(prepared["benchmark_protocol"]["schema"], schema)
        self.assertEqual(public_row(prepared)["benchmark_protocol"]["schema"], schema)

        summary_fields = protocol_summary_fields(schema, secret, redact_id_salt=True)
        self.assertEqual(summary_fields["schema"], schema)
        self.assertNotIn("effective_id_salt", summary_fields)
        self.assertEqual(
            summary_fields["effective_id_salt_sha256"],
            hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn(secret, json.dumps(summary_fields))

    def test_neutral_target_is_kept_as_an_aot_entrypoint(self) -> None:
        row = dict(self.row)
        row["dart_source"] = self.source.replace("@pragma('vm:entry-point')\n", "", 1)
        prepared = prepare_scrubbed_row(
            row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt="unit-test",
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
        )
        self.assertIn(
            "@pragma('vm:entry-point')\nint candidate(", prepared["dart_source"]
        )

    def test_nullable_and_nested_generic_targets_keep_aot_entrypoints(self) -> None:
        cases = (
            ("String? semanticSum(List<String> values) { return null; }", "String?"),
            (
                "List<List<int>> semanticSum(List<List<int>> values) { return values; }",
                "List<List<int>>",
            ),
        )
        for source, return_type in cases:
            with self.subTest(return_type=return_type):
                row = dict(self.row)
                row["dart_source"] = source
                row["dart_function_signature"] = source.split(" {")[0]
                prepared = prepare_scrubbed_row(
                    row,
                    index=0,
                    benchmark_kind="existing_ablation",
                    id_salt="unit-test",
                    input_sha256="a" * 64,
                    freeze_manifest_sha256=None,
                )
                self.assertIn(
                    f"@pragma('vm:entry-point')\n{return_type} candidate(",
                    prepared["dart_source"],
                )

    def test_prompt_exposes_only_neutral_name(self) -> None:
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt="unit-test",
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
        )
        prepared["assembly"] = "Dump of assembler code for function candidate:\nret"
        prompt = build_decoder_prompt(prepared)
        self.assertIn("function named exactly candidate.", prompt)
        self.assertNotIn("Implement this exact top-level Dart signature", prompt)
        self.assertNotIn("semanticSum", prompt)
        self.assertNotIn("List<int>", prompt)
        self.assertNotIn("expect(", prompt)

    def test_name_only_mode_overrides_accidentally_populated_signature(self) -> None:
        prompt = build_decoder_prompt(
            {
                "lang": "Dart",
                "assembly": "ret",
                "function": "candidate",
                "dart_function_signature": "int candidate(int leaked)",
                "prompt_signature_mode": "name_only",
            }
        )
        self.assertIn("function named exactly candidate.", prompt)
        self.assertNotIn("int candidate(int leaked)", prompt)

    def test_public_row_removes_evaluator_secrets(self) -> None:
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt="unit-test",
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
        )
        visible = public_row(prepared)
        self.assertNotIn("dart_source", visible)
        self.assertNotIn("tests", visible)
        self.assertNotIn("evaluation_only_dart_function_signature", visible)

    def test_public_row_strips_fingerprint_hashes(self) -> None:
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt="unit-test",
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
        )
        self.assertIn("semantic_function_name_sha256", prepared["benchmark_protocol"])
        self.assertIn("original_source_sha256", prepared["benchmark_protocol"])
        prepared["benchmark_protocol"]["assembly_build"] = {
            "mode": "rename_frozen",
            "frozen_assembly_sha256": "c" * 64,
            "extractor_sha256": "d" * 64,
        }
        visible = public_row(prepared)
        self.assertNotIn("semantic_function_name_sha256", visible["benchmark_protocol"])
        self.assertNotIn("original_source_sha256", visible["benchmark_protocol"])
        self.assertNotIn(
            "frozen_assembly_sha256",
            visible["benchmark_protocol"]["assembly_build"],
        )
        self.assertEqual(
            visible["benchmark_protocol"]["assembly_build"]["extractor_sha256"],
            "d" * 64,
        )
        self.assertEqual(
            prepared["benchmark_protocol"]["assembly_build"]["frozen_assembly_sha256"],
            "c" * 64,
        )

    def test_rename_assembly_symbols_is_context_targeted(self) -> None:
        dump = (
            'All functions matching regular expression "add":\n\n'
            "File file:///tmp/ag_rebuild_00000_add__xyz/program.dart:\n"
            "9:\tstatic void add(void);\n"
            "Dump of assembler code for function add:\n"
            "   0x1000 <add+0>:\tpush   rbp\n"
            "   0x1004 <add+4>:\tadd    rax,rax\n"
            "   0x1008 <add+8>:\tjne    0x1010 <add+16>\n"
        )
        renamed = rename_assembly_symbols(dump, "add")
        self.assertIn('"candidate"', renamed)
        self.assertIn("function candidate:", renamed)
        self.assertIn("static void candidate(void);", renamed)
        self.assertIn("<candidate+4>:\tadd    rax,rax", renamed)
        self.assertIn("jne    0x1010 <candidate+16>", renamed)
        self.assertNotIn("<add", renamed)
        self.assertNotIn("ag_rebuild", renamed)
        # the mnemonic must survive untouched
        self.assertIn("add    rax,rax", renamed)

    def test_rename_assembly_residue_check_allows_target_with_original_prefix(
        self,
    ) -> None:
        dump = (
            'All functions matching regular expression "f":\n\n'
            "1:\tstatic int f(void);\n"
            "2:\tstatic int f.<anonymous closure>(void);\n"
            "Dump of assembler code for function f:\n"
            "   0x1000 <f+0>:\tpush   rbp\n"
            "   0x1004 <f.<anonymous closure>+4>:\tret\n"
        )
        renamed = rename_assembly_symbols(dump, "f", target_name="fn0")
        self.assertIn('regular expression "fn0"', renamed)
        self.assertIn("static int fn0(void);", renamed)
        self.assertIn("static int fn0.<anonymous closure>(void);", renamed)
        self.assertIn("function fn0:", renamed)
        self.assertIn("<fn0+0>", renamed)
        self.assertIn("<fn0.<anonymous closure>+4>", renamed)

    def test_rename_assembly_symbols_neutralizes_closures_and_helpers(self) -> None:
        dump = (
            "1:\tstatic void histogram(void);\n"
            "5:\tstatic void histogram.<anonymous closure>(void);\n"
            "8:\tstatic List<int> sortArray(void);\n"
            "Dump of assembler code for function histogram:\n"
            "   0x1000 <histogram+0>:\tpush   rbp\n"
            "   0x1004 <histogram.<anonymous closure>+4>:\tret\n"
            "   0x1008 <sortArray+0>:\tret\n"
        )
        renamed = rename_assembly_symbols(dump, "histogram", ("sortArray",))
        self.assertIn("static void candidate.<anonymous closure>(void);", renamed)
        self.assertIn("<candidate.<anonymous closure>+4>", renamed)
        self.assertIn("static List<int> helper1(void);", renamed)
        self.assertIn("<helper1+0>", renamed)
        self.assertNotIn("histogram", renamed)
        self.assertNotIn("sortArray", renamed)

    def test_rename_assembly_symbols_neutralizes_target_qualified_helpers(
        self,
    ) -> None:
        cases = (
            (
                "parseNestedParens",
                "parseParenGroup",
                (
                    "3:\tstatic void parseNestedParens.parseParenGroup(void);\n"
                    "Dump of assembler code for function parseNestedParens:\n"
                    "   0x1000 <parseNestedParens+0>:\tret\n"
                ),
                ("static void fn0.helper1(void);",),
            ),
            (
                "countNums",
                "digitsSum",
                (
                    'All functions matching regular expression "countNums.digitsSum":\n'
                    "3:\tstatic void countNums.digitsSum(void);\n"
                    "12:\tstatic void countNums.digitsSum.<anonymous closure>(void);\n"
                    "Dump of assembler code for function countNums.digitsSum:\n"
                    "   0x1000 <countNums.digitsSum+0>:\tpush   rbp\n"
                    "   0x1004 <countNums.digitsSum.<anonymous closure>+4>:\tret\n"
                ),
                (
                    'regular expression "fn0.helper1"',
                    "static void fn0.helper1(void);",
                    "static void fn0.helper1.<anonymous closure>(void);",
                    "function fn0.helper1:",
                    "<fn0.helper1+0>",
                    "<fn0.helper1.<anonymous closure>+4>",
                ),
            ),
        )
        for original_name, helper_name, dump, expected_fragments in cases:
            with self.subTest(original_name=original_name, helper_name=helper_name):
                renamed = rename_assembly_symbols(
                    dump,
                    original_name,
                    (helper_name,),
                    target_name="fn0",
                )
                for fragment in expected_fragments:
                    self.assertIn(fragment, renamed)
                self.assertNotIn(original_name, renamed)
                self.assertNotIn(helper_name, renamed)

    def test_real_rows_neutralize_target_qualified_helpers(self) -> None:
        dataset = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "testing"
            / "grpo_data_graphv2.jsonl"
        )
        rows = read_jsonl(dataset)
        cases = ((6, "parseParenGroup"), (102, "digitsSum"))
        for index, leaked_helper in cases:
            with self.subTest(index=index, leaked_helper=leaked_helper):
                row = rows[index]
                original_name = str(row["function"])
                source = str(row.get("dart_source") or row.get("source") or "")
                helper_names = tuple(
                    name
                    for name in extract_source_function_names(source)
                    if name != original_name
                )
                self.assertIn(leaked_helper, helper_names)
                renamed = rename_assembly_symbols(
                    str(row["assembly"]),
                    original_name,
                    helper_names,
                    target_name="fn0",
                )
                self.assertIn("fn0.helper1", renamed)
                self.assertNotIn(original_name, renamed)
                self.assertNotIn(leaked_helper, renamed)

    def test_rename_assembly_symbols_raises_on_residue(self) -> None:
        with self.assertRaisesRegex(ValueError, "residue"):
            # malformed annotation the targeted patterns cannot reach
            rename_assembly_symbols(
                "<semanticSum!oops> function semanticSum :", "semanticSum"
            )
        with self.assertRaisesRegex(ValueError, "residue"):
            rename_assembly_symbols(
                "<parseNestedParens.parseParenGroup!oops>",
                "parseNestedParens",
                ("parseParenGroup",),
                target_name="fn0",
            )

    def test_rename_frozen_records_asserted_toolchain_versions(self) -> None:
        stamp = rename_frozen_toolchain_stamp(
            frozen_dart_version="Dart SDK version: 3.11.5",
            frozen_gdb_version="GNU gdb 17.1",
            frozen_toolchain_version="humaneval-dart-linux-x64-v1",
        )
        row = dict(self.row)
        row["assembly"] = (
            "Dump of assembler code for function semanticSum:\n"
            "   0x1000 <semanticSum+0>:\tret\n"
        )
        with patch(
            "scripts.data.build_signature_scrubbed_eval.build_record",
            side_effect=lambda built, **_: (built, {}),
        ):
            _, built, reject = build_one(
                (0, row),
                benchmark_kind="existing_ablation",
                id_salt=effective_id_salt("unit-test", "name_only", "fn0"),
                input_sha256="a" * 64,
                freeze_manifest_sha256=None,
                dart_bin="dart",
                gdb_bin="gdb",
                timeout=1,
                max_block_instrs=20,
                max_dataflow_edges=0,
                graph_extractor_sha256="b" * 64,
                prepare_only=False,
                public_signature_mode="name_only",
                toolchain_stamp=stamp,
                assembly_mode="rename_frozen",
                target_name="fn0",
                protocol_schema="dart-signature-scrubbed-v3",
            )
        self.assertIsNone(reject)
        self.assertIsNotNone(built)
        assembly_build = built["benchmark_protocol"]["assembly_build"]
        self.assertEqual(
            built["benchmark_protocol"]["schema"], "dart-signature-scrubbed-v3"
        )
        self.assertEqual(assembly_build["mode"], "rename_frozen")
        self.assertEqual(
            assembly_build["assembly_derivation"],
            "text_rename_of_frozen_benchmark_assembly",
        )
        self.assertEqual(
            assembly_build["asserted_frozen_dart_version"],
            "Dart SDK version: 3.11.5",
        )
        self.assertEqual(assembly_build["asserted_frozen_gdb_version"], "GNU gdb 17.1")
        self.assertEqual(
            assembly_build["asserted_frozen_toolchain_version"],
            "humaneval-dart-linux-x64-v1",
        )
        self.assertEqual(assembly_build["extractor_sha256"], "b" * 64)
        self.assertIn("function fn0:", built["assembly"])

    def test_neutralize_param_names(self) -> None:
        cases = (
            (
                "bool candidate(List<double> numbers, double threshold)",
                "bool candidate(List<double> a, double b)",
                True,
            ),
            ("double candidate(double number)", "double candidate(double a)", True),
            ("int candidate()", "int candidate()", True),
            (
                "String? candidate(Map<String, List<int>> lookup)",
                "String? candidate(Map<String, List<int>> a)",
                True,
            ),
        )
        for original, expected, expected_flag in cases:
            with self.subTest(original=original):
                got, flag = neutralize_param_names(original)
                self.assertEqual(got, expected)
                self.assertEqual(flag, expected_flag)
        unchanged, flag = neutralize_param_names("int candidate([int fallback = 0])")
        self.assertEqual(unchanged, "int candidate([int fallback = 0])")
        self.assertFalse(flag)

    def test_neutral_exact_mode_exposes_typed_neutral_signature(self) -> None:
        prepared = prepare_scrubbed_row(
            self.row,
            index=0,
            benchmark_kind="existing_ablation",
            id_salt="unit-test",
            input_sha256="a" * 64,
            freeze_manifest_sha256=None,
            public_signature_mode="neutral_exact",
        )
        self.assertEqual(prepared["prompt_signature_mode"], "exact")
        self.assertEqual(
            prepared["dart_function_signature"],
            "int candidate(List<int> a, int b)",
        )
        self.assertEqual(
            prepared["public_prompt_signature"],
            prepared["dart_function_signature"],
        )
        self.assertTrue(prepared["task_id"].startswith("sigtyped_"))
        self.assertTrue(
            prepared["benchmark_protocol"]["public_signature_params_neutralized"]
        )
        visible = public_row(prepared)
        prompt = build_decoder_prompt(
            {
                **visible,
                "assembly": "Dump of assembler code for function candidate:\nret",
            }
        )
        self.assertIn(
            "Implement this exact top-level Dart signature: "
            "int candidate(List<int> a, int b).",
            prompt,
        )
        self.assertNotIn("semanticSum", prompt)
        self.assertNotIn("values", prompt)
        self.assertNotIn("bias", prompt)

    def test_fresh_generator_neutralizes_before_compilation(self) -> None:
        generated = {
            "function_name": "countSignals",
            "signature": "int countSignals(List<int> values)",
            "dart_function": (
                "int countSignals(List<int> values) { "
                "return values.isEmpty ? 0 : countSignals(values.sublist(1)) + 1; }"
            ),
            "main_asserts": ["assert(countSignals([1, 2]) == 2);"],
            "test_expects": ["expect(candidate([]), 0);"],
        }
        scrubbed, hidden_signature = generator.scrub_generated_candidate(generated)
        self.assertEqual(scrubbed["function_name"], "candidate")
        self.assertEqual(hidden_signature, "int candidate(List<int> values)")
        self.assertNotIn("countSignals", scrubbed["dart_function"])
        harness = generator.build_signature_scrubbed_tests(scrubbed["test_expects"])
        self.assertIn("final implementation = candidate;", harness)
        self.assertIn("expect(implementation([]), 0);", harness)

        visible = generator.public_signature_scrubbed_record(
            {
                "assembly": "ret",
                "dart_source": scrubbed["dart_function"],
                "tests": harness,
                "evaluation_only_dart_function_signature": hidden_signature,
            }
        )
        self.assertEqual(visible, {"assembly": "ret"})


if __name__ == "__main__":
    unittest.main()
