from __future__ import annotations

import ast
import importlib.util
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch


PATCH_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    PATCH_ROOT / "scripts" / "evaluation" / "graph_compile_at_k_antigravity.py"
)
GRPO = PATCH_ROOT / "scripts" / "training" / "graph_grpo_decompiler_antigravity.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "hybrid_dart_completion_evaluator", EVALUATOR
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DartCompletionAttestationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = load_evaluator()
        cls.tests = """void main() {
  final candidate = foo;
  expect(candidate(1), 2);
}
void expect(dynamic a, dynamic b) { if (a != b) throw '$a != $b'; }
"""

    def test_marker_is_per_run_exact_and_duplicate_fails_closed(self) -> None:
        nonce = "c" * 64
        with patch.object(
            self.evaluator.secrets, "token_hex", return_value=nonce
        ):
            ok, diagnostic, source, marker = (
                self.evaluator.prepare_dart_test_completion_attestation(
                    "void main() { return; }\n"
                )
            )
        self.assertTrue(ok, diagnostic)
        self.assertIn("Future<void> main(List<String>", source)
        self.assertNotIn("void main() { return; }", source)
        self.assertTrue(
            self.evaluator.dart_test_completion_observed(marker + "\n", marker)
        )
        self.assertFalse(
            self.evaluator.dart_test_completion_observed(
                marker + "\n" + marker + "\n", marker
            )
        )

    def test_escaped_and_conditional_process_imports_are_rejected(self) -> None:
        check = self.evaluator.disallowed_dart_test_runtime_library
        self.assertEqual(check(r"import 'dart:\x69o';"), "dart:io")
        self.assertEqual(
            check("import 'safe.dart' if (dart.library.io) 'dart:io';"),
            "dart:io",
        )
        self.assertEqual(check("import r'dart:isolate';"), "")
        self.assertEqual(check("import 'dart:math';"), "")

    def test_extract_code_preserves_top_level_dart_declarations(self) -> None:
        declarations = (
            "class Solver { static int f() => 1; }",
            "abstract class Solver { int f(); }",
            "abstract base class Solver { int f(); }",
            "final class Solver { int f() => 1; }",
            "sealed class Solver {}",
            "base class Solver { int f() => 1; }",
            "interface class Solver { int f() => 1; }",
            "enum Choice { yes, no }",
            "mixin Helpers { int f() => 1; }",
            "extension Values on int { int f() => this + 1; }",
            "typedef Mapper = int Function(int);",
            "@immutable\nclass Solver { const Solver(); }",
        )
        for declaration in declarations:
            with self.subTest(declaration=declaration.splitlines()[0]):
                raw = "Here is the Dart implementation:\n" + declaration
                self.assertEqual(self.evaluator._extract_code(raw), declaration)

    def test_verpo_full_and_per_assertion_runners_require_attestation(self) -> None:
        tree = ast.parse(GRPO.read_text(encoding="utf-8"))
        reward_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TruePerTestReward"
        )
        methods = {
            node.name: ast.unparse(node)
            for node in reward_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in ("_run_full_test_harness", "_run_single_expect_test"):
            source = methods[name]
            self.assertIn("prepare_dart_test_completion_attestation", source)
            self.assertIn("dart_test_completion_observed", source)

    @unittest.skipUnless(shutil.which("dart"), "Dart is required")
    def test_sync_and_async_harnesses_still_pass(self) -> None:
        evaluate = self.evaluator.evaluate_dart_jit_tests_detail
        compiled, passed, diagnostic, _ = evaluate(
            "int foo(int x) => x + 1;",
            self.tests,
            "sync",
            timeout=30,
        )
        self.assertEqual((compiled, passed), (True, True), diagnostic)

        async_tests = """Future<void> main() async {
  await Future<void>.delayed(Duration.zero);
  final candidate = foo;
  expect(await Future.value(candidate(1)), 2);
}
void expect(dynamic a, dynamic b) { if (a != b) throw '$a != $b'; }
"""
        compiled, passed, diagnostic, _ = evaluate(
            "int foo(int x) => x + 1;",
            async_tests,
            "async",
            timeout=30,
        )
        self.assertEqual((compiled, passed), (True, True), diagnostic)

    @unittest.skipUnless(shutil.which("dart"), "Dart is required")
    def test_trusted_harness_imports_do_not_depend_on_candidate_output(self) -> None:
        evaluate = self.evaluator.evaluate_dart_jit_tests_detail
        stdout_tests = r"""
Future<void> main() async {
  final captured = <String>[];
  await runZoned(
    () async { await Future.sync(() => fn0()); },
    zoneSpecification: ZoneSpecification(
      print: (self, parent, zone, line) { captured.add(line); },
    ),
  );
  if (captured.join('\n') != 'ok') throw StateError('$captured');
}
"""
        compiled, passed, diagnostic, source = evaluate(
            "void fn0() { print('ok'); }",
            stdout_tests,
            "harness_async_import",
            timeout=30,
        )
        self.assertEqual((compiled, passed), (True, True), diagnostic)
        self.assertEqual(source.count("import 'dart:async';"), 1)
        self.assertEqual(source.count("import 'dart:convert';"), 1)

        differential_tests = r"""
Future<void> main() async {
  final value = await Future.sync(() => fn0());
  if (jsonEncode(value) != '[1,2]') throw StateError('$value');
}
"""
        compiled, passed, diagnostic, _ = evaluate(
            "List<int> fn0() => <int>[1, 2];",
            differential_tests,
            "harness_convert_import",
            timeout=30,
        )
        self.assertEqual((compiled, passed), (True, True), diagnostic)

    @unittest.skipUnless(shutil.which("dart"), "Dart is required")
    def test_exit_zero_isolate_exit_and_marker_spoof_never_pass(self) -> None:
        evaluate = self.evaluator.evaluate_dart_jit_tests_detail
        attacks = {
            "exit_zero": (
                "import 'dart:io';\nint foo(int x) { exit(0); }",
                "completion_attestation_disallowed_library",
            ),
            "isolate_exit": (
                "import 'dart:isolate';\n"
                "int foo(int x) { Isolate.exit(); }",
                "",
            ),
            "marker_spoof": (
                r"""import 'dart:io';
int foo(int x) {
  final source = File.fromUri(Platform.script).readAsStringSync();
  final marker = RegExp(
    r'__ANTIGRAVITY_DART_TEST_COMPLETED_[0-9a-f]+__'
  ).firstMatch(source)!;
  print(marker.group(0));
  exit(0);
}""",
                "completion_attestation_disallowed_library",
            ),
        }
        for task_id, (candidate, required_diagnostic) in attacks.items():
            with self.subTest(task_id=task_id):
                compiled, passed, diagnostic, _ = evaluate(
                    candidate, self.tests, task_id, timeout=30
                )
                self.assertFalse(passed)
                if required_diagnostic:
                    self.assertIn(required_diagnostic, diagnostic)


if __name__ == "__main__":
    unittest.main()
