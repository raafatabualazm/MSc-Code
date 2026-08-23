"""Compile/test/repair diagnostics for the Dart AOT neural decompiler.

This script answers two separate questions without conflating them:

* ``analyze`` describes the observed compile/pass failure topology of an
  archived candidate pool. It does not infer a task-level repair ceiling from
  that static pool.
* ``repair`` runs sampled model-in-the-loop repair chains. ``compile`` feedback
  is deployable because only compiler diagnostics reach the policy.
  ``visible_tests`` uses a separate, held-out development-test file while the
  untouched scoring tests remain hidden for final evaluation.
  ``oracle_tests`` is a contaminated hidden-test diagnostic and is never a
  reportable evaluation result.

The repair path uses the same ``GraphInferenceModel`` constructor, checkpoint
loading, prompt builder, graph tensors, and ``generate`` method as
``graph_inference_antigravity.py``. A source inference provenance sidecar is
required by default so the checkpoint is reconstructed under the original
``GRAPH_*`` environment and its checkpoint/dataset hashes can be verified.

Examples
--------
Describe an archived seed-42 pool without a GPU::

  python scripts/evaluation/repair_loop_antigravity.py analyze \
    --stats results-20260713/sweeps_antigravity/<run>_pass_stats.csv

Run deployable compile-feedback chains on the pod::

  python scripts/evaluation/repair_loop_antigravity.py repair \
    --checkpoint artifacts/<run>/pytorch_model.bin \
    --pass_dataset data/testing/grpo_data_graphv2.jsonl \
    --source_provenance results/<run>_compile_predictions.json.provenance.json \
    --feedback compile --rounds 3 --num_seeds 10 \
    --out results/repair/<run>_compile_r3

For a model-call-matched resampling comparison, replace ``--num_seeds`` with
the same ``--generation_budget_per_task`` in every arm. For example, use 40
with ``--rounds 0`` for independent resampling and 40 with ``--rounds 3`` for
repair. Early-stopping repair chains are replaced until exactly 40 model calls
have been spent on that task. This matches calls, not token-level FLOPs, because
repair prompts can be longer than base prompts.

Run the explicitly contaminated hidden-test diagnostic by changing
``--feedback`` to ``oracle_tests``. Its output filename and provenance are
marked ``ORACLE`` and ``deployable=false``.

For a clean semantic-feedback experiment, use ``--feedback visible_tests`` and
provide ``--visible_tests_dataset``. When the hidden scoring JSONL was derived
from the original inference dataset, also pass ``--test_split_manifest`` so
the changed dataset hash is verified back to that original. Every sidecar row
must use ``visible_tests``; scoring-test fields are intentionally not accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lightweight harness imports. Model and tokenizer imports remain lazy so
# analyze and the unit tests do not initialize torch or Transformers.
from scripts.evaluation.rerank_predictions_antigravity import (  # noqa: E402
    _extract_code,
    _resolve_dart_binary,
    run_dart_compile,
    run_dart_tests,
    validate_dart_binary,
)


# ---------------------------------------------------------------------------
# pass@k (unbiased Chen et al. estimator)
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def aggregate(flags_per_task: list[list[int]], ks=(1, 5, 10)) -> dict[str, Any]:
    n_tasks = len(flags_per_task)
    out: dict[str, Any] = {}
    for k in ks:
        out[f"pass_at_{k}"] = (
            sum(pass_at_k(len(flags), sum(flags), min(k, len(flags)))
                for flags in flags_per_task)
            / max(n_tasks, 1)
        )
    out["solved_any"] = sum(1 for flags in flags_per_task if sum(flags) > 0)
    out["n_tasks"] = n_tasks
    return out


# ---------------------------------------------------------------------------
# Feedback construction
# ---------------------------------------------------------------------------
def compile_feedback(
    candidate_raw: str,
    dart_bin: str,
    timeout: int,
) -> tuple[bool, str]:
    """Return compiler status and deployable, test-free feedback."""
    ok, diagnostic = run_dart_compile(candidate_raw, dart_bin, timeout)
    if ok:
        return True, ""
    lines = [line for line in diagnostic.splitlines() if line.strip()][:6]
    cleaned = "\n".join(
        re.sub(r"[A-Za-z]:?[\\/][^\s:]+[\\/]", "", line) for line in lines
    )
    return False, cleaned or "did not compile"


def test_feedback(
    candidate_raw: str,
    tests: str,
    task_id: str,
    dart_bin: str,
    timeout: int,
) -> tuple[bool, str]:
    """Run a candidate against a Dart test harness and return (passed, hint).

    Used for BOTH held-out development tests (clean) and the hidden scoring
    tests (contaminated). The caller decides which harness is passed in and how
    the result is labelled; this function is signal-agnostic.
    """
    ok, diagnostic = run_dart_tests(
        candidate_raw,
        tests,
        task_id,
        dart_bin,
        timeout,
    )
    if ok:
        return True, ""
    nonempty = [line for line in diagnostic.splitlines() if line.strip()]
    tail = "\n".join(nonempty)[-500:]
    match = re.search(r"(Expected|expected).*?(Actual|actual).*", diagnostic, re.S)
    hint = match.group(0)[:300] if match else ""
    return False, hint or tail or "tests failed"


# Back-compat alias: the hidden-scoring-test path is just test_feedback on the
# contaminated harness.
oracle_test_feedback = test_feedback


_FORBIDDEN_VISIBLE_TEST_FIELDS = frozenset({
    "tests",
    "scoring_tests",
    "hidden_tests",
})
_CANDIDATE_BINDING_RE = re.compile(
    r"\bfinal\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;"
)


def _test_keys(row: dict[str, Any], index: int) -> list[str]:
    keys = [
        str(row[name])
        for name in ("task_id", "id", "filename")
        if row.get(name) not in (None, "")
    ]
    if not keys:
        raise ValueError(
            f"test row {index + 1} has no stable task_id, id, or filename"
        )
    return list(dict.fromkeys(keys))


def _candidate_binding(harness: str) -> str | None:
    match = _CANDIDATE_BINDING_RE.search(harness)
    return match.group(1) if match else None


def _balanced_paren_close(text: str, open_index: int) -> int | None:
    """Return the inclusive close-paren index, respecting quoted strings."""
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _candidate_calls(harness: str) -> set[str]:
    """Extract whitespace-normalized candidate(...) calls with balanced parens."""
    calls: set[str] = set()
    for match in re.finditer(r"\bcandidate\s*\(", harness):
        start = match.start()
        close = _balanced_paren_close(harness, match.end() - 1)
        if close is not None:
            calls.add(re.sub(r"\s+", "", harness[start:close + 1]))
    return calls


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def load_visible_tests(path: Path) -> dict[str, str]:
    """Load a JSONL of held-out development tests keyed by task id/filename.

    Each row needs a task identifier (task_id / id / filename) and a
    ``visible_tests`` field containing a complete Dart ``void main()`` harness
    whose assertions are DISJOINT from the scoring tests. These tests drive the
    repair loop; the untouched scoring tests still decide the reported pass@k.
    """
    mapping: dict[str, str] = {}
    key_owners: dict[str, int] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"visible-test row {index + 1} is not a JSON object")
        forbidden = sorted(_FORBIDDEN_VISIBLE_TEST_FIELDS & row.keys())
        if forbidden:
            raise ValueError(
                f"visible-test row {index + 1} contains forbidden hidden/scoring "
                f"test fields: {forbidden}"
            )
        harness = row.get("visible_tests") or ""
        if not harness.strip():
            raise ValueError(
                f"visible-test row {index + 1} must contain a nonempty "
                "visible_tests field; the scoring-tests field is not accepted"
            )
        if not re.search(r"\bvoid\s+main\s*\(", harness):
            raise ValueError(
                f"visible-test row {index + 1} is not a complete void main harness"
            )
        if _candidate_binding(harness) is None:
            raise ValueError(
                f"visible-test row {index + 1} has no final candidate binding"
            )
        for key in _test_keys(row, index):
            if key in key_owners:
                raise ValueError(
                    f"duplicate visible-test task key {key!r} on rows "
                    f"{key_owners[key] + 1} and {index + 1}"
                )
            key_owners[key] = index
            mapping[key] = harness
    return mapping


def visible_tests_for_row(
    mapping: dict[str, str],
    row: dict[str, Any],
    index: int,
) -> str:
    matches = {
        mapping[key]
        for key in _test_keys(row, index)
        if key in mapping
    }
    if not matches:
        raise RuntimeError(
            f"no visible-test harness matched task keys {_test_keys(row, index)!r}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"task keys {_test_keys(row, index)!r} resolve to conflicting harnesses"
        )
    return matches.pop()


def _main_body(harness: str) -> str | None:
    """Return the balanced body of ``void main(...)`` without helper code."""
    match = re.search(r"\bvoid\s+main\s*\([^)]*\)\s*\{", harness)
    if match is None:
        return None
    open_brace = match.end() - 1
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = open_brace
    while index < len(harness):
        char = harness[index]
        next_char = harness[index + 1] if index + 1 < len(harness) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 1
        elif quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and next_char == "/":
            line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            block_comment = True
            index += 1
        elif char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return harness[open_brace + 1:index]
        index += 1
    return None


def _normalize_assertions(harness: str) -> set[str]:
    """Return complete expect/assert statements from the test-driver main()."""
    driver = _main_body(harness)
    if driver is None:
        return set()
    assertions: set[str] = set()
    for match in re.finditer(r"\b(?:expect|assert)\s*\(", driver):
        close = _balanced_paren_close(driver, match.end() - 1)
        if close is None:
            continue
        semicolon = close + 1
        while semicolon < len(driver) and driver[semicolon].isspace():
            semicolon += 1
        if semicolon < len(driver) and driver[semicolon] == ";":
            assertions.add(
                re.sub(r"\s+", "", driver[match.start():semicolon + 1])
            )
    return assertions


def validate_visible_test_boundary(
    visible_tests: str,
    scoring_tests: str,
    task_id: str,
) -> None:
    """Require complete, same-target development tests disjoint from scoring."""
    visible_binding = _candidate_binding(visible_tests)
    scoring_binding = _candidate_binding(scoring_tests)
    if visible_binding is None:
        raise RuntimeError(f"task {task_id} visible harness has no candidate binding")
    if scoring_binding is None:
        raise RuntimeError(f"task {task_id} scoring harness has no candidate binding")
    if visible_binding != scoring_binding:
        raise RuntimeError(
            f"task {task_id} visible/scoring candidate bindings differ: "
            f"{visible_binding!r} != {scoring_binding!r}"
        )
    visible_assertions = _normalize_assertions(visible_tests)
    if not visible_assertions:
        raise RuntimeError(f"task {task_id} visible harness has no expect/assert calls")
    overlap = visible_assertions & _normalize_assertions(scoring_tests)
    if overlap:
        raise RuntimeError(
            f"task {task_id} visible/scoring test assertions overlap; "
            "the semantic-feedback run would be contaminated"
        )
    visible_calls = _candidate_calls(visible_tests)
    if not visible_calls:
        raise RuntimeError(f"task {task_id} visible harness never calls candidate")
    call_overlap = visible_calls & _candidate_calls(scoring_tests)
    if call_overlap:
        raise RuntimeError(
            f"task {task_id} visible/scoring candidate inputs overlap; "
            "the semantic-feedback run would be contaminated"
        )


# ---------------------------------------------------------------------------
# Repair prompt and chain orchestration
# ---------------------------------------------------------------------------
# Feedback-mode metadata. The prompt tag is deliberately different per mode so a
# reviewer reading a trace can tell instantly which signal reached the policy.
_FEEDBACK_TAGS = {
    "compile": "Compiler diagnostic",
    "visible_tests": "Development-test diagnostic (held out from the scoring tests)",
    "oracle_tests": "ORACLE hidden-test diagnostic (contaminated, diagnostic use only)",
}
_FEEDBACK_TERMINAL = {
    "compile": "compiled",
    "visible_tests": "visible_tests_passed",
    "oracle_tests": "tests_passed",
}


def build_repair_prompt(
    base_prompt: str,
    previous_code: str,
    feedback: str,
    feedback_kind: str,
) -> str:
    """Append one revision turn while keeping the feedback mode unmistakable."""
    tag = _FEEDBACK_TAGS[feedback_kind]
    return (
        f"{base_prompt}\n\n"
        f"--- Previous attempt ---\n{_extract_code(previous_code).strip()}\n\n"
        f"--- {tag} ---\n{feedback.strip()}\n\n"
        "Correct the previous attempt. Return one complete Dart implementation "
        "and output only source code."
    )


def validate_base_prompt(base_prompt: str, scoring_tests: str) -> None:
    """Reject known labels or a verbatim hidden-test payload in a base prompt."""
    if "Unit-test harness excerpt" in base_prompt or "ORACLE DIAGNOSTIC" in base_prompt:
        raise RuntimeError("scoring-test content reached the base policy prompt")
    normalized_tests = scoring_tests.strip()
    if normalized_tests and normalized_tests in base_prompt:
        raise RuntimeError("verbatim scoring tests reached the base policy prompt")
    compact_prompt = re.sub(r"\s+", "", base_prompt)
    if any(assertion in compact_prompt for assertion in _normalize_assertions(scoring_tests)):
        raise RuntimeError("a scoring-test assertion reached the base policy prompt")


def run_repair_chain(
    generate: Callable[[str, int], str],
    evaluate: Callable[[str], tuple[bool, str]],
    base_prompt: str,
    max_repair_rounds: int,
    feedback_kind: str,
) -> tuple[str, list[dict[str, Any]], str]:
    """Run one base generation followed by at most N repair generations."""
    if max_repair_rounds < 0:
        raise ValueError("max_repair_rounds must be non-negative")

    prompt = base_prompt
    candidate = ""
    trace: list[dict[str, Any]] = []
    for generation_index in range(max_repair_rounds + 1):
        candidate = generate(prompt, generation_index)
        satisfied, feedback = evaluate(candidate)
        trace.append(
            {
                "generation_index": generation_index,
                "feedback_satisfied": bool(satisfied),
                "feedback_chars": len(feedback),
            }
        )
        if satisfied:
            return candidate, trace, _FEEDBACK_TERMINAL[feedback_kind]
        if generation_index < max_repair_rounds:
            prompt = build_repair_prompt(
                base_prompt,
                candidate,
                feedback,
                feedback_kind,
            )

    return candidate, trace, "round_budget_exhausted"


def _next_chain_round_limit(
    max_repair_rounds: int,
    requested_chains: int,
    chains_started: int,
    generations_used: int,
    generation_budget_per_task: int | None,
) -> int | None:
    """Return the next chain's repair-round ceiling, or None when finished."""
    if generation_budget_per_task is None:
        return max_repair_rounds if chains_started < requested_chains else None
    remaining = generation_budget_per_task - generations_used
    if remaining <= 0:
        return None
    return min(max_repair_rounds, remaining - 1)


def generate_model_candidate(
    model: Any,
    decoder_tokenizer: Any,
    block_tensors: Any,
    graph_data: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
    prompt_max_length: int,
    do_sample: bool,
    preserve_prompt_suffix: bool,
) -> str:
    """Tokenize a causal prompt and call the repository's real generate API."""
    old_side = getattr(decoder_tokenizer, "truncation_side", None)
    if preserve_prompt_suffix and old_side is not None:
        decoder_tokenizer.truncation_side = "left"
    try:
        prompt_tensors = decoder_tokenizer(
            prompt,
            max_length=prompt_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    finally:
        if old_side is not None:
            decoder_tokenizer.truncation_side = old_side

    predictions = model.generate(
        block_tensors,
        graph_data,
        decoder_tokenizer,
        device,
        max_new_tokens=max_new_tokens,
        num_samples=1,
        decoder_prompt_input_ids=prompt_tensors["input_ids"],
        decoder_prompt_attention_mask=prompt_tensors["attention_mask"],
        do_sample=do_sample,
    )
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise RuntimeError(
            "GraphInferenceModel.generate must return exactly one prediction "
            f"for a repair-chain turn; got {type(predictions).__name__} "
            f"with length {len(predictions) if hasattr(predictions, '__len__') else '?'}"
        )
    return str(predictions[0])


# ---------------------------------------------------------------------------
# analyze mode
# ---------------------------------------------------------------------------
def load_stats_csv(path: Path) -> list[dict[str, list[int]]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    tasks: list[dict[str, list[int]]] = []
    for row in rows:
        compile_flags: list[int] = []
        pass_flags: list[int] = []
        index = 1
        while f"cand_{index}_compile" in row:
            compile_flags.append(int(row.get(f"cand_{index}_compile") or 0))
            pass_flags.append(int(row.get(f"cand_{index}_pass") or 0))
            index += 1
        tasks.append({"compile": compile_flags, "pass": pass_flags})
    return tasks


def failure_topology(tasks: list[dict[str, list[int]]]) -> dict[str, Any]:
    """Describe observed unsolved-task and candidate failure categories.

    This is deliberately not called a repair ceiling. In a mixed task, every
    noncompiling chain can receive compiler feedback even though another chain
    already compiles. A static pool cannot tell how often those repairs will
    become semantically correct.
    """
    unsolved: list[dict[str, list[int]]] = []
    for index, task in enumerate(tasks):
        compile_flags = task["compile"]
        pass_flags = task["pass"]
        if len(compile_flags) != len(pass_flags):
            raise ValueError(
                f"task {index} has {len(compile_flags)} compile flags but "
                f"{len(pass_flags)} pass flags"
            )
        if not pass_flags:
            raise ValueError(f"task {index} has no candidate flags")
        if sum(pass_flags) == 0:
            unsolved.append(task)

    no_compile = 0
    mixed = 0
    all_compile_wrong = 0
    unsolved_noncompile_candidates = 0
    unsolved_compile_wrong_candidates = 0
    for task in unsolved:
        n_candidates = len(task["compile"])
        n_compile = sum(task["compile"])
        unsolved_noncompile_candidates += n_candidates - n_compile
        unsolved_compile_wrong_candidates += n_compile
        if n_compile == 0:
            no_compile += 1
        elif n_compile == n_candidates:
            all_compile_wrong += 1
        else:
            mixed += 1

    total_candidates = sum(len(task["pass"]) for task in tasks)
    compiling_candidates = sum(sum(task["compile"]) for task in tasks)
    passing_candidates = sum(sum(task["pass"]) for task in tasks)
    return {
        "unsolved_tasks": len(unsolved),
        "no_compiling_candidates_tasks": no_compile,
        "mixed_compile_outcomes_tasks": mixed,
        "all_candidates_compile_wrong_tasks": all_compile_wrong,
        "tasks_with_base_compile_feedback": no_compile + mixed,
        "noncompiling_candidates_on_unsolved_tasks": unsolved_noncompile_candidates,
        "compiling_wrong_candidates_on_unsolved_tasks": unsolved_compile_wrong_candidates,
        "candidate_compile_rate": compiling_candidates / max(total_candidates, 1),
        "candidate_pass_rate": passing_candidates / max(total_candidates, 1),
        "pass_rate_among_compiling": (
            passing_candidates / max(compiling_candidates, 1)
        ),
    }


def run_harness_on_predictions(
    pred_path: Path,
    dart_bin: str,
    timeout: int,
    workers: int,
    limit: int | None,
) -> list[dict[str, list[int]]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pool_rows = json.loads(pred_path.read_text(encoding="utf-8"))
    if limit:
        pool_rows = pool_rows[:limit]
    tasks: list[dict[str, list[int]] | None] = [None] * len(pool_rows)

    def one(index_row: tuple[int, dict[str, Any]]):
        index, row = index_row
        task_id = str(row.get("id", row.get("task_id", index)))
        tests = row.get("tests", "")
        compile_flags: list[int] = []
        pass_flags: list[int] = []
        for candidate in row.get("predictions", []):
            compiles, _ = run_dart_compile(candidate, dart_bin, timeout)
            passes, _ = (
                run_dart_tests(candidate, tests, task_id, dart_bin, timeout)
                if tests
                else (False, "")
            )
            compile_flags.append(int(compiles))
            pass_flags.append(int(passes))
        return index, {"compile": compile_flags, "pass": pass_flags}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(one, (index, row))
            for index, row in enumerate(pool_rows)
        ]
        done = 0
        for future in as_completed(futures):
            index, result = future.result()
            tasks[index] = result
            done += 1
            if done % 25 == 0:
                print(f"  harness {done}/{len(pool_rows)}")

    if any(task is None for task in tasks):
        raise RuntimeError("harness did not produce a result for every task")
    return [task for task in tasks if task is not None]


def cmd_analyze(args: argparse.Namespace) -> int:
    if args.stats:
        tasks = load_stats_csv(Path(args.stats))
        source = f"stats CSV {args.stats}"
    elif args.predictions:
        dart_bin = _resolve_dart_binary(args.dart)
        validate_dart_binary(dart_bin)
        tasks = run_harness_on_predictions(
            Path(args.predictions),
            dart_bin,
            args.timeout,
            args.workers,
            args.limit,
        )
        source = f"harness on {args.predictions}"
    else:
        raise SystemExit("analyze needs --stats or --predictions")

    metrics = aggregate([task["pass"] for task in tasks])
    topology = failure_topology(tasks)
    print(f"\nsource: {source}\ntasks: {metrics['n_tasks']}")
    print(
        f"pass@1={metrics['pass_at_1']:.3f}  "
        f"pass@5={metrics['pass_at_5']:.3f}  "
        f"pass@10={metrics['pass_at_10']:.3f}  "
        f"solved={metrics['solved_any']}"
    )
    print("\n--- observed failure topology (not a repair ceiling) ---")
    print(f"unsolved tasks:                         {topology['unsolved_tasks']}")
    print(
        "  no candidate compiles:                "
        f"{topology['no_compiling_candidates_tasks']}"
    )
    print(
        "  mixed compile/noncompile candidates:  "
        f"{topology['mixed_compile_outcomes_tasks']}"
    )
    print(
        "  all candidates compile but are wrong: "
        f"{topology['all_candidates_compile_wrong_tasks']}"
    )
    print(
        "noncompiling chains on unsolved tasks:  "
        f"{topology['noncompiling_candidates_on_unsolved_tasks']}"
    )
    print(f"candidate compile-rate:                 {topology['candidate_compile_rate']:.3f}")
    print(f"candidate pass-rate:                    {topology['candidate_pass_rate']:.3f}")
    print(
        "pass-rate among compiling candidates:   "
        f"{topology['pass_rate_among_compiling']:.3f}"
    )
    print("\ninterpretation:")
    print(
        "  Compile feedback is available to every noncompiling chain in the "
        "no-compile and mixed categories."
    )
    print(
        "  The archived pool cannot establish a task-level improvement ceiling; "
        "that requires controlled repair runs."
    )
    print(
        "  All-compile-wrong chains provide no compiler signal. Hidden-test "
        "feedback can probe responsiveness,"
    )
    print(
        "  but that contaminated diagnostic is neither deployable nor an upper "
        "bound on execution-reward training."
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "source": source,
                    "metrics": metrics,
                    "failure_topology": topology,
                    "repair_ceiling_inferred": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nreport: {args.report}")
    return 0


# ---------------------------------------------------------------------------
# Source-provenance validation and model loading
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file_record(
    path: str | Path,
    expected: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if expected:
        expected_size = expected.get("size_bytes")
        expected_sha = expected.get("sha256")
        if expected_size is not None and int(expected_size) != actual["size_bytes"]:
            raise RuntimeError(
                f"{label} size does not match source provenance: "
                f"{actual['size_bytes']} != {expected_size}"
            )
        if expected_sha and str(expected_sha).lower() != actual["sha256"]:
            raise RuntimeError(
                f"{label} sha256 does not match source provenance: "
                f"{actual['sha256']} != {expected_sha}"
            )
    return actual


def _require_matching_records(
    expected: dict[str, Any],
    actual: dict[str, Any],
    label: str,
) -> None:
    for key in ("size_bytes", "sha256"):
        expected_value = expected.get(key)
        actual_value = actual.get(key)
        if expected_value is None or actual_value is None:
            raise RuntimeError(f"{label} records must both contain {key}")
        if str(expected_value).lower() != str(actual_value).lower():
            raise RuntimeError(
                f"{label} {key} mismatch: {actual_value!r} != {expected_value!r}"
            )


def _choose_provenance_value(
    label: str,
    explicit: Any,
    recorded: Any,
    fallback: Any = None,
) -> Any:
    if explicit not in (None, "") and recorded not in (None, ""):
        if str(explicit) != str(recorded):
            raise RuntimeError(
                f"{label} conflicts with source provenance: "
                f"{explicit!r} != {recorded!r}"
            )
    if explicit not in (None, ""):
        return explicit
    if recorded not in (None, ""):
        return recorded
    return fallback


def resolve_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    source: dict[str, Any] | None = None
    source_record: dict[str, Any] | None = None
    if args.source_provenance:
        source_path = Path(args.source_provenance)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_record = _verified_file_record(source_path, None, "source provenance")
        graph_env = source.get("graph_environment")
        if not isinstance(graph_env, dict):
            raise RuntimeError("source provenance has no graph_environment object")
        for key, value in graph_env.items():
            if key.startswith("GRAPH_") and value is not None:
                os.environ[key] = str(value)
    elif not args.allow_unverified_environment:
        raise RuntimeError(
            "repair requires --source_provenance so checkpoint architecture and "
            "hashes are verified; use --allow_unverified_environment only for "
            "non-study debugging"
        )

    models = source.get("models", {}) if source else {}
    generation = source.get("generation", {}) if source else {}
    decoder_source = models.get("decoder", {}) if isinstance(models, dict) else {}
    encoder_source = models.get("encoder", {}) if isinstance(models, dict) else {}

    decoder_model = _choose_provenance_value(
        "decoder model",
        args.decoder_model,
        decoder_source.get("requested_id"),
        os.environ.get("GRAPH_DECODER_MODEL"),
    )
    encoder_model = _choose_provenance_value(
        "encoder model",
        args.encoder_model,
        encoder_source.get("requested_id"),
        os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base"),
    )
    if not decoder_model:
        raise RuntimeError(
            "decoder model is unknown; provide --decoder_model or source provenance"
        )

    decoder_revision = _choose_provenance_value(
        "decoder revision",
        args.decoder_revision,
        decoder_source.get("requested_revision"),
        os.environ.get("GRAPH_DECODER_REVISION", ""),
    )
    encoder_revision = _choose_provenance_value(
        "encoder revision",
        args.encoder_revision,
        encoder_source.get("requested_revision"),
        os.environ.get("GRAPH_ENCODER_REVISION", ""),
    )
    seed = int(
        _choose_provenance_value(
            "seed",
            args.seed,
            source.get("seed") if source else None,
            os.environ.get("GRAPH_SEED", "42"),
        )
    )
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else generation.get("max_new_tokens", 768)
    )
    prompt_max_length = int(
        args.decoder_prompt_max_length
        if args.decoder_prompt_max_length is not None
        else generation.get(
            "decoder_prompt_max_length",
            os.environ.get("GRAPH_DECODER_PROMPT_MAX_LENGTH", 768),
        )
    )

    checkpoint_record = _verified_file_record(
        args.checkpoint,
        source.get("checkpoint") if source else None,
        "checkpoint",
    )
    split_manifest = None
    split_manifest_record = None
    split_manifest_path = getattr(args, "test_split_manifest", None)
    if split_manifest_path:
        manifest_path = Path(split_manifest_path)
        split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_manifest_record = _verified_file_record(
            manifest_path,
            None,
            "test-split manifest",
        )
        if split_manifest.get("stage") != "repair_test_split":
            raise RuntimeError("test-split manifest has the wrong stage")
        for key in ("input", "visible_output", "hidden_output"):
            if not isinstance(split_manifest.get(key), dict):
                raise RuntimeError(f"test-split manifest has no {key} file record")
        if source and isinstance(source.get("dataset"), dict):
            _require_matching_records(
                source["dataset"],
                split_manifest["input"],
                "source dataset versus test-split input",
            )
        dataset_record = _verified_file_record(
            args.pass_dataset,
            split_manifest["hidden_output"],
            "hidden scoring dataset",
        )
    else:
        dataset_record = _verified_file_record(
            args.pass_dataset,
            source.get("dataset") if source else None,
            "pass dataset",
        )
    return {
        "source": source,
        "source_record": source_record,
        "checkpoint_record": checkpoint_record,
        "dataset_record": dataset_record,
        "test_split_manifest": split_manifest,
        "test_split_manifest_record": split_manifest_record,
        "decoder_model": str(decoder_model),
        "encoder_model": str(encoder_model),
        "decoder_revision": str(decoder_revision or ""),
        "encoder_revision": str(encoder_revision or ""),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "prompt_max_length": prompt_max_length,
    }


def load_repair_model(config: dict[str, Any], checkpoint: str, device: str):
    """Reproduce the canonical graph-inference model/tokenizer load path."""
    os.environ["GRAPH_DECODER_MODEL"] = config["decoder_model"]
    os.environ["GRAPH_ENCODER_MODEL"] = config["encoder_model"]
    os.environ["GRAPH_DECODER_REVISION"] = config["decoder_revision"]
    os.environ["GRAPH_ENCODER_REVISION"] = config["encoder_revision"]
    os.environ["GRAPH_SEED"] = str(config["seed"])

    import torch
    from transformers import AutoTokenizer, set_seed

    set_seed(config["seed"])
    checkpoint_lower = checkpoint.lower()
    if "lora" in checkpoint_lower:
        os.environ.setdefault("GRAPH_ENCODER_PEFT", "lora")
        os.environ.setdefault("GRAPH_DECODER_PEFT", "lora")
    else:
        os.environ.setdefault("GRAPH_ENCODER_PEFT", "none")
        os.environ.setdefault("GRAPH_DECODER_PEFT", "none")

    from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
    from scripts.data.dfg_extractor import LightweightDFGExtractor
    from scripts.evaluation.graph_inference_antigravity import GraphInferenceModel
    from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
        maybe_override_qwen_prefix_gate,
    )

    encoder_tokenizer = AutoTokenizer.from_pretrained(
        config["encoder_model"],
        revision=config["encoder_revision"] or None,
        trust_remote_code=True,
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        config["decoder_model"],
        revision=config["decoder_revision"] or None,
        trust_remote_code=True,
    )
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(
        encoder_tokenizer,
        max_seq_len=512,
    )
    dfg_extractor = LightweightDFGExtractor()

    model = GraphInferenceModel(config["decoder_model"]).to(device)
    state = torch.load(checkpoint, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint must contain a PyTorch state dictionary")
    if any(key.startswith("local_encoder") for key in state) and not hasattr(
        model,
        "local_encoder",
    ):
        missing, unexpected = model.decompiler.load_state_dict(state, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint: {checkpoint}")
    print(
        "loaded parameters with strict=False: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    maybe_override_qwen_prefix_gate(model.decompiler)
    model.eval()
    for decoder in (model.t5_model, model.decompiler.base_decoder_model):
        if hasattr(decoder, "gradient_checkpointing_disable"):
            decoder.gradient_checkpointing_disable()
        if hasattr(decoder, "config") and hasattr(decoder.config, "use_cache"):
            decoder.config.use_cache = True
    model.decompiler.gradient_checkpointing = False

    if not model.is_causal:
        raise RuntimeError(
            "repair prompts require a causal decoder; the current seq2seq "
            "GraphInferenceModel.generate path has no text-prompt channel"
        )
    return model, decoder_tokenizer, tensor_builder, dfg_extractor


# ---------------------------------------------------------------------------
# repair mode
# ---------------------------------------------------------------------------
def cmd_repair(args: argparse.Namespace) -> int:
    import torch

    config = resolve_run_configuration(args)
    feedback_kind = args.feedback
    oracle = feedback_kind == "oracle_tests"
    uses_visible = feedback_kind == "visible_tests"
    if uses_visible and not args.visible_tests_dataset:
        raise RuntimeError(
            "--feedback visible_tests requires --visible_tests_dataset"
        )
    if not uses_visible and args.visible_tests_dataset:
        raise RuntimeError(
            "--visible_tests_dataset is only valid with --feedback visible_tests"
        )
    dart_bin = _resolve_dart_binary(args.dart)
    validate_dart_binary(dart_bin)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    visible_tests_map: dict[str, str] = {}
    visible_test_record = None
    if uses_visible:
        visible_tests_map = load_visible_tests(Path(args.visible_tests_dataset))
        if not visible_tests_map:
            raise RuntimeError(
                f"no visible tests loaded from {args.visible_tests_dataset}"
            )
        visible_test_record = _verified_file_record(
            args.visible_tests_dataset,
            (
                config["test_split_manifest"]["visible_output"]
                if config["test_split_manifest"]
                else None
            ),
            "visible-test dataset",
        )

    # Validate the complete visible/hidden split before allocating or loading
    # the model. A protocol error should cost milliseconds, not a GPU startup.
    rows = _load_jsonl_records(Path(args.pass_dataset))
    if args.limit:
        rows = rows[: args.limit]
    if config["test_split_manifest"] and args.limit is None:
        expected_rows = config["test_split_manifest"].get("output_rows")
        if expected_rows is not None and len(rows) != int(expected_rows):
            raise RuntimeError(
                "hidden scoring dataset row count does not match test-split "
                f"manifest: {len(rows)} != {expected_rows}"
            )
    visible_tests_by_index: dict[int, str] = {}
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id", row.get("id", index)))
        scoring_tests = row.get("tests", "") or ""
        if (oracle or uses_visible) and not scoring_tests.strip():
            raise RuntimeError(
                f"task {task_id} has no hidden scoring tests for {feedback_kind} mode"
            )
        if uses_visible:
            repair_tests = visible_tests_for_row(visible_tests_map, row, index)
            validate_visible_test_boundary(repair_tests, scoring_tests, task_id)
            visible_tests_by_index[index] = repair_tests

    model, decoder_tokenizer, tensor_builder, dfg_extractor = load_repair_model(
        config,
        args.checkpoint,
        device,
    )
    from scripts.evaluation.graph_inference_antigravity import build_blocks
    from scripts.provenance_antigravity import (
        file_record,
        graph_environment,
        runtime_record,
    )
    from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
        PROMPT_SCHEMA_VERSION,
        build_decoder_prompt,
    )

    outputs: list[dict[str, Any]] = []
    prompt_digest = hashlib.sha256()
    terminal_counts: dict[str, int] = {}
    total_generations = 0
    total_final_candidates = 0
    task_generation_counts: list[int] = []
    task_chain_counts: list[int] = []
    do_sample = (args.generation_budget_per_task or args.num_seeds) > 1

    for index, row in enumerate(rows):
        block_tensors, graph_data = build_blocks(
            row,
            tensor_builder,
            dfg_extractor,
        )
        base_prompt = build_decoder_prompt(
            row,
            decoder_tokenizer,
            config["prompt_max_length"],
        )
        tests = row.get("tests", "")
        validate_base_prompt(base_prompt, tests)
        task_id = str(row.get("task_id", row.get("id", index)))

        repair_tests = visible_tests_by_index.get(index, "")
        if uses_visible:
            validate_base_prompt(base_prompt, repair_tests)

        final_candidates: list[str] = []
        chain_records: list[dict[str, Any]] = []
        task_generations = 0
        chain_index = 0
        while True:
            chain_round_limit = _next_chain_round_limit(
                args.rounds,
                args.num_seeds,
                chain_index,
                task_generations,
                args.generation_budget_per_task,
            )
            if chain_round_limit is None:
                break

            def generate(prompt: str, generation_index: int) -> str:
                prompt_digest.update(
                    json.dumps(
                        [task_id, chain_index, generation_index, prompt],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                return generate_model_candidate(
                    model,
                    decoder_tokenizer,
                    block_tensors,
                    graph_data,
                    prompt,
                    device,
                    config["max_new_tokens"],
                    config["prompt_max_length"],
                    do_sample,
                    preserve_prompt_suffix=generation_index > 0,
                )

            if oracle:
                def evaluate(candidate: str) -> tuple[bool, str]:
                    return test_feedback(
                        candidate, tests, task_id, dart_bin, args.timeout,
                    )
            elif uses_visible:
                def evaluate(candidate: str) -> tuple[bool, str]:
                    return test_feedback(
                        candidate, repair_tests, task_id, dart_bin, args.timeout,
                    )
            else:
                def evaluate(candidate: str) -> tuple[bool, str]:
                    return compile_feedback(candidate, dart_bin, args.timeout)

            candidate, trace, terminal = run_repair_chain(
                generate,
                evaluate,
                base_prompt,
                chain_round_limit,
                feedback_kind,
            )
            final_candidates.append(candidate)
            total_generations += len(trace)
            task_generations += len(trace)
            terminal_counts[terminal] = terminal_counts.get(terminal, 0) + 1
            chain_records.append(
                {
                    "chain_index": chain_index,
                    "max_repair_rounds": chain_round_limit,
                    "generations": len(trace),
                    "terminal_status": terminal,
                }
            )
            chain_index += 1

        task_generation_counts.append(task_generations)
        task_chain_counts.append(len(chain_records))
        total_final_candidates += len(final_candidates)
        if (
            args.generation_budget_per_task is not None
            and task_generations != args.generation_budget_per_task
        ):
            raise RuntimeError(
                f"task {task_id} used {task_generations} generations; expected "
                f"the exact budget {args.generation_budget_per_task}"
            )

        outputs.append(
            {
                "id": row.get("task_id", row.get("id", index)),
                "source_line": index + 1,
                "filename": row.get("filename", ""),
                "predictions": final_candidates,
                "reference": row.get("source", row.get("dart_source", "")),
                "language": row.get("language", row.get("lang", "dart")),
                "tests": tests,
                "repair_chains": chain_records,
                "model_generations": task_generations,
                "final_candidate_count": len(final_candidates),
            }
        )
        print(
            f"[{index + 1}/{len(rows)}] {task_id}: "
            f"{len(final_candidates)} final candidates"
        )

    file_tag = {"oracle_tests": "ORACLE_", "visible_tests": "VISIBLE_"}.get(
        feedback_kind, ""
    )
    out_path = Path(f"{args.out}_{file_tag}pass_predictions.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    interpretation = {
        "compile": "deployable_compile_feedback",
        # visible_tests repairs on held-out development tests; the SCORING tests
        # are never seen, so hidden-test pass@k from this pool is a clean number
        # (it models the execution-feedback-training regime, not zero-test deploy).
        "visible_tests": "clean_held_out_test_feedback",
        "oracle_tests": "contaminated_hidden_test_diagnostic",
    }[feedback_kind]
    decoder_config = model.decompiler.base_decoder_model.config
    encoder_config = model.decompiler.local_encoder.encoder.config
    provenance = {
        "schema_version": 1,
        "stage": "repair_inference",
        "feedback": args.feedback,
        "deployable": not oracle,
        "deployment_requires_visible_tests": uses_visible,
        "clean_scoring_eval": feedback_kind in ("compile", "visible_tests"),
        "oracle": oracle,
        "interpretation": interpretation,
        "visible_test_dataset": visible_test_record,
        "test_split_manifest": config["test_split_manifest_record"],
        "test_split": (
            {
                "seed": config["test_split_manifest"].get("seed"),
                "strategy": config["test_split_manifest"].get("strategy"),
                "input_rows": config["test_split_manifest"].get("input_rows"),
                "output_rows": config["test_split_manifest"].get("output_rows"),
                "dropped": config["test_split_manifest"].get("dropped", []),
            }
            if config["test_split_manifest"]
            else None
        ),
        "scoring_tests_visible_to_policy": oracle,
        "development_tests_visible_to_policy": uses_visible,
        "scoring_test_source_visible_to_policy": False,
        "scoring_test_feedback_visible_to_policy": oracle,
        "development_test_source_visible_to_policy": False,
        "development_test_feedback_visible_to_policy": uses_visible,
        "test_boundary": {
            "preflight_before_model_load": True,
            "stable_task_identity_required": uses_visible,
            "hidden_fields_forbidden_in_visible_sidecar": uses_visible,
            "candidate_binding_match_required": uses_visible,
            "candidate_input_overlap_rejected": uses_visible,
        },
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "prompt_stream_sha256": prompt_digest.hexdigest(),
        "seed": config["seed"],
        "repair": {
            "max_repair_rounds": args.rounds,
            "num_chains": (
                args.num_seeds
                if args.generation_budget_per_task is None
                else None
            ),
            "requested_num_chains": args.num_seeds,
            "generation_budget_per_task": args.generation_budget_per_task,
            "exact_model_call_budget_per_task": (
                args.generation_budget_per_task is not None
            ),
            "budget_matches_token_flops": False,
            "do_sample": do_sample,
            "repair_prompt_truncation_side": "left",
            "total_generations": total_generations,
            "total_final_candidates": total_final_candidates,
            "task_generation_count_min": min(task_generation_counts),
            "task_generation_count_max": max(task_generation_counts),
            "task_chain_count_min": min(task_chain_counts),
            "task_chain_count_max": max(task_chain_counts),
            "terminal_counts": terminal_counts,
        },
        "generation": {
            "max_new_tokens": config["max_new_tokens"],
            "decoder_prompt_max_length": config["prompt_max_length"],
            "decoder_gradient_checkpointing": False,
            "use_cache": True,
        },
        "models": {
            "decoder": {
                "requested_id": config["decoder_model"],
                "requested_revision": config["decoder_revision"] or None,
                "resolved_name_or_path": getattr(decoder_config, "_name_or_path", None),
                "resolved_commit": getattr(decoder_config, "_commit_hash", None),
            },
            "encoder": {
                "requested_id": config["encoder_model"],
                "requested_revision": config["encoder_revision"] or None,
                "resolved_name_or_path": getattr(encoder_config, "_name_or_path", None),
                "resolved_commit": getattr(encoder_config, "_commit_hash", None),
            },
        },
        "checkpoint": config["checkpoint_record"],
        "dataset": config["dataset_record"],
        "source_inference_provenance": config["source_record"],
        "graph_environment": graph_environment(),
        "runtime": runtime_record(),
        "row_count": len(outputs),
        "output": file_record(out_path),
        "source_files": [
            file_record(Path(__file__)),
            file_record(ROOT / "scripts/evaluation/graph_inference_antigravity.py"),
            file_record(
                ROOT
                / "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py"
            ),
        ],
    }
    provenance_path = Path(f"{out_path}.provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    print(f"provenance: {provenance_path}")
    if oracle:
        print(
            "ORACLE diagnostic: hidden scoring tests reached the policy. "
            "This pool is contaminated and nondeployable."
        )
    return 0


# ---------------------------------------------------------------------------
# Lightweight built-in smoke test
# ---------------------------------------------------------------------------
def self_test(_args: argparse.Namespace) -> int:
    assert abs(pass_at_k(10, 1, 10) - 1.0) < 1e-9
    assert pass_at_k(10, 0, 10) == 0.0
    assert abs(pass_at_k(10, 1, 1) - 0.1) < 1e-9

    tasks = [
        {"compile": [1, 1], "pass": [1, 0]},
        {"compile": [0, 0], "pass": [0, 0]},
        {"compile": [1, 0], "pass": [0, 0]},
        {"compile": [1, 1], "pass": [0, 0]},
    ]
    topology = failure_topology(tasks)
    assert topology["unsolved_tasks"] == 3, topology
    assert topology["no_compiling_candidates_tasks"] == 1, topology
    assert topology["mixed_compile_outcomes_tasks"] == 1, topology
    assert topology["all_candidates_compile_wrong_tasks"] == 1, topology

    generated_prompts: list[str] = []

    def generate(prompt: str, generation_index: int) -> str:
        generated_prompts.append(prompt)
        return "int f()=>0;" if generation_index == 0 else "int f()=>5;"

    def evaluate(candidate: str) -> tuple[bool, str]:
        passed = "=>5;" in candidate
        return passed, "" if passed else "Expected: 5 Actual: 0"

    candidate, trace, terminal = run_repair_chain(
        generate,
        evaluate,
        "Convert assembly to Dart.",
        max_repair_rounds=2,
        feedback_kind="oracle_tests",
    )
    assert candidate == "int f()=>5;"
    assert len(trace) == 2 and terminal == "tests_passed"
    assert "ORACLE hidden-test diagnostic" in generated_prompts[1]

    # visible-tests mode: distinct prompt tag + clean terminal label
    generated_prompts.clear()
    _, trace_v, terminal_v = run_repair_chain(
        generate,
        evaluate,
        "Convert assembly to Dart.",
        max_repair_rounds=2,
        feedback_kind="visible_tests",
    )
    assert terminal_v == "visible_tests_passed", terminal_v
    assert "Development-test diagnostic" in generated_prompts[1]
    assert "ORACLE" not in generated_prompts[1]

    # contamination guard: shared assertions between visible and scoring tests
    hidden = (
        "void main(){ final candidate = f; "
        "expect(candidate(1), 2); expect(candidate(3), 4); }"
    )
    clean = "void main(){ final candidate = f; expect(candidate(5), 6); }"
    dirty = "void main(){ final candidate = f; expect(candidate(1), 1 + 1); }"
    validate_visible_test_boundary(clean, hidden, "self-test")
    try:
        validate_visible_test_boundary(dirty, hidden, "self-test")
    except RuntimeError as exc:
        assert "candidate inputs overlap" in str(exc), exc
    else:
        raise AssertionError("overlapping visible/scoring candidate input was accepted")

    print(
        "self_test OK: pass@k, three-way failure topology, compile/visible/oracle "
        "feedback boundary, contamination guard, and repair-chain convergence"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        allow_abbrev=False,
    )
    parser.add_argument("--self_test", action="store_true")
    subparsers = parser.add_subparsers(dest="cmd")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="describe archived compile/pass failure topology",
    )
    analyze_parser.add_argument("--stats", help="archived *_pass_stats.csv")
    analyze_parser.add_argument(
        "--predictions",
        help="*_pass_predictions.json (re-runs Dart)",
    )
    analyze_parser.add_argument("--dart", default=None)
    analyze_parser.add_argument("--timeout", type=int, default=30)
    analyze_parser.add_argument("--workers", type=int, default=8)
    analyze_parser.add_argument("--limit", type=int, default=None)
    analyze_parser.add_argument("--report", default=None)

    repair_parser = subparsers.add_parser(
        "repair",
        help="model-in-the-loop compile/test/repair",
    )
    repair_parser.add_argument("--checkpoint", required=True)
    repair_parser.add_argument("--pass_dataset", required=True)
    repair_parser.add_argument(
        "--source_provenance",
        help="source inference *.provenance.json used to reconstruct and verify the run",
    )
    repair_parser.add_argument(
        "--allow_unverified_environment",
        action="store_true",
        help="allow a debugging run without source provenance",
    )
    repair_parser.add_argument("--decoder_model", default=None)
    repair_parser.add_argument("--encoder_model", default=None)
    repair_parser.add_argument("--decoder_revision", default=None)
    repair_parser.add_argument("--encoder_revision", default=None)
    repair_parser.add_argument("--seed", type=int, default=None)
    repair_parser.add_argument(
        "--feedback",
        choices=["compile", "visible_tests", "oracle_tests"],
        default="compile",
    )
    repair_parser.add_argument(
        "--visible_tests_dataset",
        "--visible_tests",
        dest="visible_tests_dataset",
        default=None,
        help=(
            "separate JSONL with task ids and visible_tests harnesses; required "
            "for --feedback visible_tests"
        ),
    )
    repair_parser.add_argument(
        "--test_split_manifest",
        default=None,
        help=(
            "manifest from build_repair_test_split.py; verifies that a derived "
            "hidden scoring dataset came from the source-provenance dataset"
        ),
    )
    repair_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="maximum repair generations after the base attempt",
    )
    repair_parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help=(
            "sampled repair chains per task when --generation_budget_per_task "
            "is not set"
        ),
    )
    repair_parser.add_argument(
        "--generation_budget_per_task",
        type=int,
        default=None,
        help=(
            "exact model-generation calls per task; starts replacement chains "
            "after early stops until the budget is exhausted"
        ),
    )
    repair_parser.add_argument("--max_new_tokens", type=int, default=None)
    repair_parser.add_argument(
        "--decoder_prompt_max_length",
        type=int,
        default=None,
    )
    repair_parser.add_argument("--dart", default=None)
    repair_parser.add_argument("--timeout", type=int, default=30)
    repair_parser.add_argument("--limit", type=int, default=None)
    repair_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test(args))
    if args.cmd == "analyze":
        raise SystemExit(cmd_analyze(args))
    if args.cmd == "repair":
        if args.rounds < 0:
            parser.error("--rounds must be non-negative")
        if args.num_seeds <= 0:
            parser.error("--num_seeds must be positive")
        if (
            args.generation_budget_per_task is not None
            and args.generation_budget_per_task <= 0
        ):
            parser.error("--generation_budget_per_task must be positive")
        raise SystemExit(cmd_repair(args))
    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
