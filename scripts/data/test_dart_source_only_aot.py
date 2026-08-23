from scripts.data.dart_source_only_aot import source_only_program


def test_removes_annotated_main_and_preserves_helpers() -> None:
    source = """
import 'dart:math';
@pragma('vm:entry-point')
int candidate(int x) {
  String brace = '}'; // source-only comment
  return helper(x) + brace.length;
}
int helper(int x) => x * 2;
@pragma('vm:entry-point')
void main() {
  if ('${candidate(2)}' != '5') throw StateError('bad } brace');
}
"""
    function_source, program, metadata = source_only_program(source)
    assert metadata["removed_top_level_main"] == 1
    assert "void main" not in function_source
    assert "StateError" not in function_source
    assert "source-only comment" not in function_source
    assert "int helper" in function_source
    assert function_source.count("vm:entry-point") == 1
    assert function_source.count("vm:never-inline") == 1
    assert program.endswith("void main() {}\n")
    assert metadata["source_symbols"] == {
        "functions": ["candidate", "helper"],
        "types": [],
    }


def test_rejects_multiple_candidates() -> None:
    source = "int candidate(int x) => x;\nint candidate(String x) => x.length;\n"
    try:
        source_only_program(source)
    except ValueError as error:
        assert "expected_one_candidate_function" in str(error)
    else:
        raise AssertionError("duplicate candidate was accepted")
