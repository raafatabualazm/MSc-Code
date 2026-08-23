"""Offline, fail-closed analysis for the fixed signature-scrub v3 experiment.

The script never invokes Dart.  It combines three already-scored candidate
pools (the frozen comparator, opaque neutral-exact, and opaque name-only),
checks that they are genuinely paired, and recomputes metrics from candidate
flags.  Candidate flags may live in the prediction JSON itself or in the
project's ``cand_N_compile``/``cand_N_pass`` statistics CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ARM_LABELS = ("comparator", "neutral_exact", "name_only")
METRICS = ("pass_at_1", "pass_at_5", "pass_at_10", "compile_at_1", "compile_at_5")
FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)
SCORING_PROVENANCE_SCHEMA = "fixed-scrub-v3-scoring-provenance-v1"


class AnalysisError(ValueError):
    """An integrity condition required for a paired analysis was not met."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Cannot read JSON {path}: {exc}") from exc


def json_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("rows", "data", "results", "tasks")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if rows is None:
            raise AnalysisError(f"{path} is not a row-list prediction JSON")
    else:
        raise AnalysisError(f"{path} is not a row-list prediction JSON")
    if not all(isinstance(row, dict) for row in rows):
        raise AnalysisError(f"{path} contains a non-object row")
    return rows


def row_id(row: dict[str, Any], index: int) -> str:
    for key in ("task_id", "id", "problem_id", "filename"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise AnalysisError(f"Prediction row {index + 1} has no task identifier")


def candidate_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("code", "prediction", "text", "candidate", "completion", "output"):
            if key in item:
                return str(item[key])
    raise AnalysisError("Candidate is neither text nor an object containing candidate text")


def candidate_items(row: dict[str, Any]) -> list[Any]:
    for key in ("predictions", "candidates", "completions", "outputs"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    if "prediction" in row:
        return [row["prediction"]]
    raise AnalysisError("Prediction row has no candidate list")


def as_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        raise AnalysisError("Missing candidate flag")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed"}:
            return 1
        if lowered in {"false", "no", "fail", "failed", ""}:
            return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"Invalid candidate flag: {value!r}") from exc
    if numeric not in (0.0, 1.0):
        raise AnalysisError(f"Candidate flag must be 0/1, got {value!r}")
    return int(numeric)


def dict_flag(obj: dict[str, Any], kind: str) -> int | None:
    keys = (
        ("compiled", "compile", "compile_ok", "did_compile")
        if kind == "compile"
        else ("passed", "pass", "tests_passed", "did_pass")
    )
    for container in (obj, obj.get("evaluation"), obj.get("eval"), obj.get("score")):
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key in container:
                return as_flag(container[key])
    return None


def flags_from_prediction_row(
    row: dict[str, Any], items: list[Any], expected_candidates: int
) -> tuple[list[int], list[int]] | None:
    if all(isinstance(item, dict) for item in items):
        compiled = [dict_flag(item, "compile") for item in items]
        passed = [dict_flag(item, "pass") for item in items]
        if all(value is not None for value in compiled + passed):
            return [int(value) for value in compiled], [int(value) for value in passed]

    for key in ("candidate_evaluations", "evaluations", "candidate_results"):
        evaluations = row.get(key)
        if isinstance(evaluations, list) and len(evaluations) == expected_candidates:
            if not all(isinstance(item, dict) for item in evaluations):
                raise AnalysisError(f"{key} must contain objects")
            compiled = [dict_flag(item, "compile") for item in evaluations]
            passed = [dict_flag(item, "pass") for item in evaluations]
            if all(value is not None for value in compiled + passed):
                return [int(value) for value in compiled], [int(value) for value in passed]

    compile_array = next(
        (row[key] for key in ("compile_flags", "compiled", "compile") if isinstance(row.get(key), list)),
        None,
    )
    pass_array = next(
        (row[key] for key in ("pass_flags", "passed", "pass") if isinstance(row.get(key), list)),
        None,
    )
    if compile_array is not None or pass_array is not None:
        if not (
            isinstance(compile_array, list)
            and isinstance(pass_array, list)
            and len(compile_array) == len(pass_array) == expected_candidates
        ):
            raise AnalysisError("Parallel compile/pass arrays are incomplete or mis-sized")
        return [as_flag(value) for value in compile_array], [as_flag(value) for value in pass_array]

    compiled: list[int] = []
    passed: list[int] = []
    for index in range(1, expected_candidates + 1):
        compile_key = f"cand_{index}_compile"
        pass_key = f"cand_{index}_pass"
        if compile_key not in row and pass_key not in row:
            return None
        if compile_key not in row or pass_key not in row:
            raise AnalysisError(f"Prediction row has only one of {compile_key}/{pass_key}")
        compiled.append(as_flag(row[compile_key]))
        passed.append(as_flag(row[pass_key]))
    return compiled, passed


def read_stats(path: Path, expected_candidates: int) -> dict[str, tuple[list[int], list[int]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise AnalysisError(f"Cannot read stats CSV {path}: {exc}") from exc
    result: dict[str, tuple[list[int], list[int]]] = {}
    for index, row in enumerate(rows):
        identifier = row_id(row, index)
        if identifier in result:
            raise AnalysisError(f"Duplicate stats problem_id {identifier!r} in {path}")
        compiled = []
        passed = []
        for candidate in range(1, expected_candidates + 1):
            c_key = f"cand_{candidate}_compile"
            p_key = f"cand_{candidate}_pass"
            if c_key not in row or p_key not in row:
                raise AnalysisError(f"{path} lacks {c_key}/{p_key}")
            compiled.append(as_flag(row[c_key]))
            passed.append(as_flag(row[p_key]))
        extra = [
            key
            for key in row
            if re.fullmatch(r"cand_(\d+)_(?:compile|pass)", key)
            and int(re.search(r"\d+", key).group()) > expected_candidates
        ]
        if extra:
            raise AnalysisError(f"{path} has more than {expected_candidates} candidates")
        result[identifier] = (compiled, passed)
    return result


def target_name(row: dict[str, Any]) -> str | None:
    for key in ("evaluation_only_dart_function_signature", "dart_function_signature"):
        signature = row.get(key)
        if isinstance(signature, str) and signature.strip():
            match = re.search(r"([A-Za-z_$][\w$]*)\s*\(", signature)
            if match:
                return match.group(1)
    protocol = row.get("benchmark_protocol")
    if isinstance(protocol, dict):
        for key in ("neutral_target_name", "target_name"):
            if isinstance(protocol.get(key), str) and protocol[key]:
                return protocol[key]
    for key in ("function", "camel_case_function_name"):
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    tests = str(row.get("tests") or "")
    # Benchmark harnesses historically used both ``final candidate = target``
    # and ``final implementation = target``.  The right-hand identifier is the
    # evaluated function; the left-hand alias is normalized separately.
    match = re.search(
        r"\bfinal\s+[A-Za-z_$][\w$]*\s*=\s*([A-Za-z_$][\w$]*)\s*;", tests
    )
    if match:
        return match.group(1)
    match = re.search(r"\bexpect\s*\(\s*([A-Za-z_$][\w$]*)\s*\(", tests)
    if match:
        return match.group(1)
    reference = str(row.get("reference") or row.get("dart_source") or "")
    match = re.search(
        r"(?m)^\s*(?:@[\w:'()\-]+\s*)*(?:[\w?<>,\[\] ]+\s+)?([A-Za-z_$][\w$]*)\s*\(",
        reference,
    )
    return match.group(1) if match else None


def canonical_source(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Preserve semantically meaningful whitespace inside string literals.  The
    # scrubber is expected to change identifiers, not reformat hidden tests.
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def normalized_tests(row: dict[str, Any]) -> str:
    tests = row.get("tests")
    if not isinstance(tests, str) or not tests.strip():
        raise AnalysisError("Every prediction row must carry hidden tests for pairing")
    name = target_name(row)
    if not name:
        raise AnalysisError("Cannot infer the evaluated target name from a prediction row")
    binding = re.search(
        r"\bfinal\s+([A-Za-z_$][\w$]*)\s*=\s*([A-Za-z_$][\w$]*)\s*;", tests
    )
    normalized = re.sub(rf"\b{re.escape(name)}\b", "__TARGET__", tests)
    if binding:
        alias = binding.group(1)
        if alias != name:
            normalized = re.sub(rf"\b{re.escape(alias)}\b", "__CALLABLE__", normalized)
    return canonical_source(normalized)


def task_digest(row: dict[str, Any]) -> str:
    # Tests are the stable cross-arm content: references deliberately differ
    # after identifier neutralization, while callers should differ only in the
    # target identifier.  A full SHA-256 avoids trusting row order or public ID.
    return sha256_text("fixed-scrub-v3-task\0" + normalized_tests(row))


@dataclass
class Task:
    identifier: str
    index: int
    row: dict[str, Any]
    candidates: list[str]
    compiled: list[int]
    passed: list[int]
    digest: str
    normalized_tests: str


@dataclass
class Arm:
    label: str
    prediction_path: Path
    stats_path: Path | None
    tasks: list[Task]
    by_id: dict[str, Task]
    provenance_docs: list[dict[str, Any]]


def auto_provenance_paths(prediction: Path, stats: Path | None) -> list[Path]:
    paths = [Path(str(prediction) + ".provenance.json")]
    if stats:
        paths.append(Path(str(stats) + ".provenance.json"))
    return [path for path in paths if path.is_file()]


def load_arm(
    label: str,
    prediction_path: Path,
    stats_path: Path | None,
    provenance_paths: list[Path],
    expected_tasks: int,
    expected_candidates: int,
) -> Arm:
    rows = json_rows(read_json(prediction_path), prediction_path)
    if len(rows) != expected_tasks:
        raise AnalysisError(
            f"{label}: expected {expected_tasks} tasks, found {len(rows)} in {prediction_path}"
        )
    stats = read_stats(stats_path, expected_candidates) if stats_path else None
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        identifier = row_id(row, index)
        if identifier in seen_ids:
            raise AnalysisError(f"{label}: duplicate prediction id {identifier!r}")
        seen_ids.add(identifier)
        items = candidate_items(row)
        if len(items) != expected_candidates:
            raise AnalysisError(
                f"{label}/{identifier}: expected {expected_candidates} candidates, found {len(items)}"
            )
        candidates = [candidate_text(item) for item in items]
        embedded = flags_from_prediction_row(row, items, expected_candidates)
        from_stats = stats.get(identifier) if stats is not None else None
        if stats is not None and from_stats is None:
            raise AnalysisError(f"{label}: stats CSV has no row for prediction id {identifier!r}")
        if embedded is None and from_stats is None:
            raise AnalysisError(
                f"{label}/{identifier}: no per-candidate flags; provide --{label.replace('_', '-')}-stats"
            )
        if embedded is not None and from_stats is not None and embedded != from_stats:
            raise AnalysisError(f"{label}/{identifier}: embedded flags disagree with stats CSV")
        compiled, passed = embedded or from_stats  # type: ignore[misc]
        if any(passed_i and not compiled_i for compiled_i, passed_i in zip(compiled, passed)):
            raise AnalysisError(f"{label}/{identifier}: pass flag is true while compile flag is false")
        norm_tests = normalized_tests(row)
        tasks.append(
            Task(
                identifier=identifier,
                index=index,
                row=row,
                candidates=candidates,
                compiled=compiled,
                passed=passed,
                digest=sha256_text("fixed-scrub-v3-task\0" + norm_tests),
                normalized_tests=norm_tests,
            )
        )
    if stats is not None and set(stats) != seen_ids:
        extras = sorted(set(stats) - seen_ids)
        raise AnalysisError(f"{label}: stats/prediction ID sets differ; extra stats IDs: {extras[:5]}")
    docs: list[dict[str, Any]] = []
    all_provenance_paths = list(
        dict.fromkeys([*provenance_paths, *auto_provenance_paths(prediction_path, stats_path)])
    )
    for path in all_provenance_paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise AnalysisError(f"{label}: provenance {path} is not a JSON object")
        payload = dict(payload)
        payload["__path__"] = str(path)
        docs.append(payload)
    if not docs:
        raise AnalysisError(f"{label}: no provenance sidecar found or supplied")
    return Arm(label, prediction_path, stats_path, tasks, {task.identifier: task for task in tasks}, docs)


def nested_values(value: Any, wanted: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in wanted:
                found.append(child)
            found.extend(nested_values(child, wanted))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_values(child, wanted))
    return found


def checkpoint_hashes(docs: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for doc in docs:
        checkpoint = doc.get("checkpoint")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("sha256"), str):
            result.add(checkpoint["sha256"].lower())
        for value in nested_values(doc, {"checkpoint_sha256", "model_checkpoint_sha256"}):
            if isinstance(value, str):
                result.add(value.lower())
    return result


def scalar_set(docs: list[dict[str, Any]], keys: set[str]) -> set[str]:
    values: set[str] = set()
    for doc in docs:
        for value in nested_values(doc, keys):
            if isinstance(value, (str, int, float, bool)):
                values.add(str(value))
    return values


def get_path(obj: Any, *parts: str) -> Any:
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def generation_seeds(docs: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for doc in docs:
        for path in (
            ("generation_seed",),
            ("seed",),
            ("generation", "seed"),
            ("generation", "generation_seed"),
            ("provenance_identifiers", "generation_seed"),
        ):
            value = get_path(doc, *path)
            if isinstance(value, (str, int, float)):
                values.add(str(value))
    return values


def scoring_source_hashes(docs: list[dict[str, Any]]) -> set[str]:
    result = scalar_set(docs, {"scorer_sha256", "classifier_sha256", "harness_sha256"})
    for doc in docs:
        for source in nested_values(doc, {"source_files"}):
            if not isinstance(source, list):
                continue
            for item in source:
                if not isinstance(item, dict):
                    continue
                name = Path(str(item.get("path", ""))).name
                if name in {"graph_compile_at_k_antigravity.py", "compile_statistical_results_antigravity.py"}:
                    if isinstance(item.get("sha256"), str):
                        result.add(item["sha256"].lower())
    return result


def one_value(values: set[str], label: str, field: str, required: bool = False) -> str | None:
    if len(values) > 1:
        raise AnalysisError(f"{label}: conflicting {field} values in provenance: {sorted(values)}")
    if required and not values:
        raise AnalysisError(f"{label}: provenance lacks required {field}")
    return next(iter(values), None)


def validate_bound_file_record(
    record: Any,
    actual_path: Path,
    label: str,
    field: str,
) -> None:
    if not isinstance(record, dict):
        raise AnalysisError(f"{label}: scoring provenance lacks an {field} file record")
    recorded_path = record.get("path")
    recorded_sha = record.get("sha256")
    recorded_size = record.get("size_bytes")
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise AnalysisError(f"{label}: scoring provenance {field} path is malformed")
    if not isinstance(recorded_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_sha):
        raise AnalysisError(f"{label}: scoring provenance {field} SHA-256 is malformed")
    if not isinstance(recorded_size, int) or isinstance(recorded_size, bool) or recorded_size < 0:
        raise AnalysisError(f"{label}: scoring provenance {field} size is malformed")
    actual_sha = sha256_file(actual_path)
    if recorded_sha != actual_sha:
        raise AnalysisError(
            f"{label}: scoring provenance {field} SHA-256 mismatch: "
            f"recorded={recorded_sha}, actual={actual_sha}"
        )
    actual_size = actual_path.stat().st_size
    if recorded_size != actual_size:
        raise AnalysisError(
            f"{label}: scoring provenance {field} size mismatch: "
            f"recorded={recorded_size}, actual={actual_size}"
        )


def validate_scoring_input_bindings(arm: Arm) -> None:
    scoring_docs = [
        doc
        for doc in arm.provenance_docs
        if doc.get("schema_version") == SCORING_PROVENANCE_SCHEMA
    ]
    if len(scoring_docs) != 1:
        raise AnalysisError(
            f"{arm.label}: expected exactly one {SCORING_PROVENANCE_SCHEMA} sidecar, "
            f"found {len(scoring_docs)}"
        )
    if arm.stats_path is None:
        raise AnalysisError(f"{arm.label}: scoring provenance requires a stats CSV")
    scoring = scoring_docs[0]
    if scoring.get("arm") != arm.label:
        raise AnalysisError(
            f"{arm.label}: scoring provenance arm mismatch: {scoring.get('arm')!r}"
        )
    inputs = scoring.get("inputs")
    if not isinstance(inputs, dict):
        raise AnalysisError(f"{arm.label}: scoring provenance inputs are missing or malformed")
    validate_bound_file_record(
        inputs.get("predictions"), arm.prediction_path, arm.label, "predictions"
    )
    validate_bound_file_record(inputs.get("stats"), arm.stats_path, arm.label, "stats")


def validate_provenance(arms: dict[str, Arm], expected_tasks: int, expected_candidates: int) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for label, arm in arms.items():
        validate_scoring_input_bindings(arm)
        checkpoint = one_value(checkpoint_hashes(arm.provenance_docs), label, "checkpoint SHA-256", True)
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint or ""):
            raise AnalysisError(f"{label}: malformed checkpoint SHA-256 {checkpoint!r}")
        seed = one_value(generation_seeds(arm.provenance_docs), label, "generation seed", True)
        compile_mode = one_value(
            scalar_set(arm.provenance_docs, {"compile_mode", "metric_mode"}),
            label,
            "compile mode",
            True,
        )
        scorer = one_value(
            scoring_source_hashes(arm.provenance_docs), label, "scorer SHA-256", True
        )
        if not re.fullmatch(r"[0-9a-f]{64}", scorer or ""):
            raise AnalysisError(f"{label}: malformed scorer SHA-256 {scorer!r}")
        if compile_mode != "jit_tests":
            raise AnalysisError(
                f"{label}: expected aligned compile mode 'jit_tests', found {compile_mode!r}"
            )
        dart = one_value(
            scalar_set(arm.provenance_docs, {"dart_sdk_version", "dart_version"}),
            label,
            "Dart SDK",
            True,
        )
        prompt_schema = one_value(
            scalar_set(arm.provenance_docs, {"prompt_schema_version"}),
            label,
            "prompt schema",
            True,
        )
        row_counts = scalar_set(arm.provenance_docs, {"row_count"})
        if row_counts and any(int(float(value)) != expected_tasks for value in row_counts):
            raise AnalysisError(f"{label}: provenance row_count does not equal {expected_tasks}")
        sample_counts = scalar_set(arm.provenance_docs, {"samples_per_row", "num_samples"})
        if sample_counts and any(int(float(value)) != expected_candidates for value in sample_counts):
            raise AnalysisError(f"{label}: provenance sample count does not equal {expected_candidates}")
        visibility_value = one_value(
            scalar_set(arm.provenance_docs, {"scoring_tests_visible_to_policy"}),
            label,
            "test-visibility declaration",
            True,
        )
        if visibility_value.lower() != "false":
            raise AnalysisError(f"{label}: provenance says scoring tests were visible to policy")
        if label in {"neutral_exact", "name_only"}:
            public_only = one_value(
                scalar_set(arm.provenance_docs, {"policy_input_verified_public_only"}),
                label,
                "public-only policy-input declaration",
                True,
            )
            if public_only.lower() != "true":
                raise AnalysisError(f"{label}: policy input was not verified public-only")
        summaries[label] = {
            "checkpoint_sha256": checkpoint,
            "seed": seed,
            "compile_mode": compile_mode,
            "scorer_sha256": scorer,
            "dart_sdk_version": dart,
            "prompt_schema_version": prompt_schema,
            "sidecars": [doc["__path__"] for doc in arm.provenance_docs],
        }

    for field in ("checkpoint_sha256", "seed", "compile_mode", "scorer_sha256", "dart_sdk_version"):
        present = {label: summary[field] for label, summary in summaries.items() if summary[field] is not None}
        if present and len(present) != len(arms):
            raise AnalysisError(f"Provenance field {field} is present for only some arms: {present}")
        if len(set(present.values())) > 1:
            raise AnalysisError(f"Provenance mismatch for {field}: {present}")
    # The two v3 prompts must share a schema; the frozen comparator is allowed
    # to retain its historical exact-signature schema.
    v3_schemas = {summaries[label]["prompt_schema_version"] for label in ("neutral_exact", "name_only")}
    v3_schemas.discard(None)
    if len(v3_schemas) > 1:
        raise AnalysisError(f"v3 prompt schema mismatch: {sorted(v3_schemas)}")
    return summaries


def manifest_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise AnalysisError(f"Pair manifest {path} must be a list or contain a rows list")
    if not all(isinstance(row, dict) for row in rows):
        raise AnalysisError(f"Pair manifest {path} contains a non-object row")
    return rows


def task_from_manifest(arm: Arm, item: dict[str, Any], prefix: str) -> Task:
    id_key = f"{prefix}_id"
    index_key = f"{prefix}_index"
    if id_key in item:
        identifier = str(item[id_key])
        if identifier not in arm.by_id:
            raise AnalysisError(f"Pair manifest references absent {prefix} id {identifier!r}")
        return arm.by_id[identifier]
    if index_key in item:
        index = int(item[index_key])
        if not 0 <= index < len(arm.tasks):
            raise AnalysisError(f"Pair manifest {index_key} {index} is out of range")
        return arm.tasks[index]
    raise AnalysisError(f"Pair manifest row needs {id_key} or {index_key}")


def build_pairs(
    arms: dict[str, Arm], pair_manifest: Path | None, expected_tasks: int
) -> tuple[list[dict[str, Task]], dict[str, Any]]:
    pairs: list[dict[str, Task]] = []
    if pair_manifest:
        rows = manifest_rows(read_json(pair_manifest), pair_manifest)
        if len(rows) != expected_tasks:
            raise AnalysisError(f"Pair manifest has {len(rows)} rows, expected {expected_tasks}")
        for item in rows:
            pair = {
                "comparator": task_from_manifest(arms["comparator"], item, "comparator"),
                "neutral_exact": task_from_manifest(arms["neutral_exact"], item, "neutral_exact"),
                "name_only": task_from_manifest(arms["name_only"], item, "name_only"),
            }
            declared_digest = item.get("task_digest")
            if declared_digest is not None and any(
                task.digest != str(declared_digest) for task in pair.values()
            ):
                raise AnalysisError("Pair manifest task_digest disagrees with normalized hidden tests")
            pairs.append(pair)
        method = "explicit_manifest"
    else:
        by_digest: dict[str, dict[str, Task]] = {}
        for label, arm in arms.items():
            local: dict[str, Task] = {}
            for task in arm.tasks:
                if task.digest in local:
                    raise AnalysisError(
                        f"{label}: normalized hidden-test digest is not unique; use an explicit pair manifest"
                    )
                local[task.digest] = task
            by_digest[label] = local
        digest_sets = {label: set(mapping) for label, mapping in by_digest.items()}
        if not (digest_sets["comparator"] == digest_sets["neutral_exact"] == digest_sets["name_only"]):
            counts = {label: len(value) for label, value in digest_sets.items()}
            raise AnalysisError(
                "Cross-arm stable test digests do not match; supply a verified explicit pair manifest "
                f"(unique counts: {counts})"
            )
        for digest in sorted(digest_sets["comparator"]):
            pairs.append({label: by_digest[label][digest] for label in ARM_LABELS})
        method = "normalized_hidden_test_sha256"

    for pair in pairs:
        tests = {task.normalized_tests for task in pair.values()}
        if len(tests) != 1:
            raise AnalysisError("Paired tasks do not have equal hidden tests after target-name normalization")
        arities: set[int] = set()
        for task in pair.values():
            name = target_name(task.row)
            arity = expected_arity(task.row, name or "") if name else None
            if arity is None:
                raise AnalysisError(
                    f"Cannot recover evaluation arity for paired task {task.identifier!r}"
                )
            arities.add(arity)
        if len(arities) != 1:
            raise AnalysisError(
                "Paired tasks have different evaluation arities: "
                + repr({label: expected_arity(task.row, target_name(task.row) or "") for label, task in pair.items()})
            )
        v3_references = set()
        for label in ("neutral_exact", "name_only"):
            task = pair[label]
            reference = task.row.get("reference") or task.row.get("dart_source")
            name = target_name(task.row)
            if not isinstance(reference, str) or not reference.strip() or not name:
                raise AnalysisError(f"{label}/{task.identifier}: missing reference for pair validation")
            v3_references.add(
                canonical_source(re.sub(rf"\b{re.escape(name)}\b", "__TARGET__", reference))
            )
        if len(v3_references) != 1:
            raise AnalysisError("The two v3 arms have different normalized hidden references")
    for label in ARM_LABELS:
        identifiers = [pair[label].identifier for pair in pairs]
        if len(set(identifiers)) != expected_tasks:
            raise AnalysisError(f"Pairing reuses or omits a {label} task")
    return pairs, {
        "method": method,
        "task_count": len(pairs),
        "manifest": str(pair_manifest) if pair_manifest else None,
        "manifest_sha256": sha256_file(pair_manifest) if pair_manifest else None,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or c < 0 or c > n:
        raise AnalysisError(f"Invalid pass@k inputs n={n}, c={c}")
    k = min(k, n)
    if n - c < k:
        return 1.0
    # 1 - C(n-c,k)/C(n,k), evaluated as a stable product.
    miss = 1.0
    for index in range(k):
        miss *= (n - c - index) / (n - index)
    return 1.0 - miss


def task_metric(task: Task, metric: str) -> float:
    kind, _, k_text = metric.partition("_at_")
    flags = task.passed if kind == "pass" else task.compiled
    return pass_at_k(len(flags), sum(flags), int(k_text))


def metric_summary(tasks: Iterable[Task]) -> dict[str, Any]:
    task_list = list(tasks)
    if not task_list:
        return {
            "task_count": 0,
            "candidate_count": 0,
            "candidate_compile_count": 0,
            "candidate_pass_count": 0,
            "metrics": {metric: None for metric in METRICS},
        }
    return {
        "task_count": len(task_list),
        "candidate_count": sum(len(task.candidates) for task in task_list),
        "candidate_compile_count": sum(sum(task.compiled) for task in task_list),
        "candidate_pass_count": sum(sum(task.passed) for task in task_list),
        "tasks_with_any_compile": sum(any(task.compiled) for task in task_list),
        "tasks_with_any_pass": sum(any(task.passed) for task in task_list),
        "metrics": {
            metric: statistics.fmean(task_metric(task, metric) for task in task_list)
            for metric in METRICS
        },
    }


def compare_tasks(candidate: list[Task], comparator: list[Task]) -> dict[str, Any]:
    if len(candidate) != len(comparator):
        raise AnalysisError("Internal error: comparison vectors are not paired")
    output: dict[str, Any] = {}
    for metric in METRICS:
        arm_values = [task_metric(task, metric) for task in candidate]
        baseline_values = [task_metric(task, metric) for task in comparator]
        k = int(metric.rsplit("_", 1)[1])
        flag_kind = "passed" if metric.startswith("pass") else "compiled"
        # "Solved task" follows the project's paired tables: at least one of
        # the n generated candidates succeeds.  Also expose the literal first-k
        # observation, which is order-sensitive and distinct from pass@k.
        arm_solved = [any(getattr(task, flag_kind)) for task in candidate]
        baseline_solved = [any(getattr(task, flag_kind)) for task in comparator]
        arm_first_k = [any(getattr(task, flag_kind)[:k]) for task in candidate]
        baseline_first_k = [any(getattr(task, flag_kind)[:k]) for task in comparator]
        output[metric] = {
            "mean_delta": statistics.fmean(a - b for a, b in zip(arm_values, baseline_values)),
            "metric_gains": sum(a > b for a, b in zip(arm_values, baseline_values)),
            "metric_losses": sum(a < b for a, b in zip(arm_values, baseline_values)),
            "metric_ties": sum(math.isclose(a, b, abs_tol=1e-15) for a, b in zip(arm_values, baseline_values)),
            "solved_task_gains": sum(a and not b for a, b in zip(arm_solved, baseline_solved)),
            "solved_task_losses": sum(b and not a for a, b in zip(arm_solved, baseline_solved)),
            "solved_task_ties": sum(a == b for a, b in zip(arm_solved, baseline_solved)),
            "both_solved": sum(a and b for a, b in zip(arm_solved, baseline_solved)),
            "both_unsolved": sum(not a and not b for a, b in zip(arm_solved, baseline_solved)),
            "observed_first_k_gains": sum(a and not b for a, b in zip(arm_first_k, baseline_first_k)),
            "observed_first_k_losses": sum(b and not a for a, b in zip(arm_first_k, baseline_first_k)),
            "observed_first_k_ties": sum(a == b for a, b in zip(arm_first_k, baseline_first_k)),
        }
    return output


def strip_non_code(text: str) -> str:
    output = list(text)
    index = 0
    quote: str | None = None
    while index < len(text):
        if quote:
            if text.startswith(quote, index):
                for offset in range(len(quote)):
                    output[index + offset] = " "
                index += len(quote)
                quote = None
            elif text[index] == "\\":
                output[index] = " "
                if index + 1 < len(text):
                    output[index + 1] = " "
                index += 2
            else:
                if text[index] != "\n":
                    output[index] = " "
                index += 1
            continue
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for pos in range(index, end):
                output[pos] = " "
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) - 2 if end < 0 else end
            for pos in range(index, min(len(text), end + 2)):
                if text[pos] != "\n":
                    output[pos] = " "
            index = end + 2
        elif text.startswith("'''", index) or text.startswith('\"\"\"', index):
            quote = text[index : index + 3]
            for offset in range(3):
                output[index + offset] = " "
            index += 3
        elif text[index] in {"'", '"'}:
            quote = text[index]
            output[index] = " "
            index += 1
        else:
            index += 1
    return "".join(output)


def extract_candidate_code(text: str) -> str:
    """Mirror the scorer's candidate extraction before static inspection."""
    if not text:
        return ""
    fenced = FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    lines = text.splitlines()
    starters = ("@pragma", "import ", "library ", "void ", "Future", "main(")
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(starters) or re.match(
            r"^[\w<>\[\],\?\s]+\s+\w+\s*\(", stripped
        ):
            return "\n".join(lines[index:]).strip()
    return text.strip()


def matching_paren(code: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == "(":
            depth += 1
        elif code[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def parameter_arity(parameters: str, allow_optional_groups: bool = True) -> int:
    parameters = parameters.strip()
    if not parameters:
        return 0
    paren_depth = 0
    angle_depth = 0
    square_depth = 0
    brace_depth = 0
    optional_groups: list[str] = []
    count = 0
    segment_has_token = False
    for char in parameters:
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char in "[{":
            collection_depth = square_depth + brace_depth
            if (
                allow_optional_groups
                and not segment_has_token
                and paren_depth == angle_depth == collection_depth == 0
            ):
                optional_groups.append(char)
                continue
            if char == "[":
                square_depth += 1
            else:
                brace_depth += 1
        elif char == "]":
            if square_depth:
                square_depth -= 1
            elif optional_groups and optional_groups[-1] == "[":
                optional_groups.pop()
                continue
        elif char == "}":
            if brace_depth:
                brace_depth -= 1
            elif optional_groups and optional_groups[-1] == "{":
                optional_groups.pop()
                continue
        elif (
            char == ","
            and paren_depth == angle_depth == square_depth == brace_depth == 0
        ):
            count += int(segment_has_token)
            segment_has_token = False
            continue
        if not char.isspace() and char not in "[]{}":
            segment_has_token = True
    return count + int(segment_has_token)


def top_level_target_arities(candidate: str, target: str) -> list[int]:
    code = strip_non_code(extract_candidate_code(candidate))
    depths: list[int] = [0] * (len(code) + 1)
    depth = 0
    for index, char in enumerate(code):
        depths[index] = depth
        if char == "{":
            depth += 1
        elif char == "}" and depth:
            depth -= 1
    pattern = re.compile(rf"(?<![\w$.]){re.escape(target)}\s*\(")
    arities: list[int] = []
    for match in pattern.finditer(code):
        if depths[match.start()] != 0:
            continue
        opening = code.find("(", match.start(), match.end() + 1)
        closing = matching_paren(code, opening)
        if closing is None:
            continue
        suffix = code[closing + 1 : closing + 80]
        if not re.match(r"\s*(?:(?:async|sync)\s*\*?\s*)?(?:=>|\{)", suffix):
            continue
        arities.append(parameter_arity(code[opening + 1 : closing]))
    return arities


def expected_arity(row: dict[str, Any], target: str) -> int | None:
    for key in ("evaluation_only_dart_function_signature", "dart_function_signature"):
        signature = row.get(key)
        if isinstance(signature, str) and signature.strip():
            code = strip_non_code(signature + " => null;")
            match = re.search(rf"\b{re.escape(target)}\s*\(", code)
            if match:
                opening = code.find("(", match.start())
                closing = matching_paren(code, opening)
                if closing is not None:
                    return parameter_arity(code[opening + 1 : closing])
    reference = str(row.get("reference") or row.get("dart_source") or "")
    arities = top_level_target_arities(reference, target)
    if arities:
        return arities[0]
    # One inherited benchmark row binds a stale wrapper name that is absent
    # from the stored reference.  The hidden caller still fixes its arity, so
    # recover that contract without pretending the reference is well formed.
    tests = strip_non_code(str(row.get("tests") or ""))
    binding = re.search(
        r"\bfinal\s+([A-Za-z_$][\w$]*)\s*=\s*([A-Za-z_$][\w$]*)\s*;", tests
    )
    callable_name = binding.group(1) if binding and binding.group(2) == target else target
    for match in re.finditer(rf"\b{re.escape(callable_name)}\s*\(", tests):
        opening = tests.find("(", match.start(), match.end() + 1)
        closing = matching_paren(tests, opening)
        if closing is not None:
            return parameter_arity(tests[opening + 1 : closing], allow_optional_groups=False)
    return None


def static_shape(tasks: list[Task], target: str) -> dict[str, Any]:
    candidate_total = 0
    declaring = 0
    matching = 0
    tasks_declaring = 0
    tasks_matching = 0
    missing_expected: list[str] = []
    for task in tasks:
        arity = expected_arity(task.row, target)
        if arity is None:
            missing_expected.append(task.identifier)
            continue
        task_declares = False
        task_matches = False
        for candidate in task.candidates:
            candidate_total += 1
            arities = top_level_target_arities(candidate, target)
            if arities:
                declaring += 1
                task_declares = True
            if arity in arities:
                matching += 1
                task_matches = True
        tasks_declaring += int(task_declares)
        tasks_matching += int(task_matches)
    if missing_expected:
        raise AnalysisError(
            f"Cannot recover expected {target} arity for tasks: {missing_expected[:5]}"
        )
    return {
        "target": target,
        "candidate_count": candidate_total,
        "top_level_target_definitions": declaring,
        "top_level_target_arity_matches": matching,
        "tasks_with_target_definition": tasks_declaring,
        "tasks_with_target_arity_match": tasks_matching,
    }


def load_broken_tokens(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = read_json(path)
    if isinstance(payload, dict):
        entries = payload.get("tasks", payload.get("broken_tasks", payload.get("rows")))
    else:
        entries = payload
    if not isinstance(entries, list):
        raise AnalysisError("Broken-task file must be a list or contain tasks/broken_tasks/rows")
    tokens: set[str] = set()
    for entry in entries:
        if isinstance(entry, (str, int)):
            tokens.add(str(entry))
        elif isinstance(entry, dict):
            for key in (
                "task_digest",
                "comparator_id",
                "neutral_exact_id",
                "name_only_id",
                "task_id",
                "id",
            ):
                if key in entry:
                    tokens.add(str(entry[key]))
        else:
            raise AnalysisError(f"Invalid broken-task entry: {entry!r}")
    return tokens


def pair_is_broken(pair: dict[str, Task], tokens: set[str]) -> bool:
    return any(task.identifier in tokens or task.digest in tokens for task in pair.values())


def make_report(
    arms: dict[str, Arm],
    pairs: list[dict[str, Task]],
    pairing: dict[str, Any],
    provenance: dict[str, Any],
    broken_tokens: set[str],
    target: str,
) -> dict[str, Any]:
    valid_pairs = [pair for pair in pairs if not pair_is_broken(pair, broken_tokens)]
    broken_pairs = [pair for pair in pairs if pair_is_broken(pair, broken_tokens)]
    report: dict[str, Any] = {
        "schema_version": "fixed-scrub-v3-analysis-v1",
        "validation": {
            "passed": True,
            "task_count": len(pairs),
            "candidates_per_task": len(pairs[0]["comparator"].candidates) if pairs else 0,
            "hidden_tests_equal_after_target_normalization": True,
            "evaluation_arities_equal": True,
            "v3_hidden_references_equal_after_target_normalization": True,
            "provenance_aligned": True,
        },
        "pairing": pairing,
        "provenance": provenance,
        "inputs": {
            label: {
                "predictions": str(arm.prediction_path),
                "predictions_sha256": sha256_file(arm.prediction_path),
                "stats": str(arm.stats_path) if arm.stats_path else None,
                "stats_sha256": sha256_file(arm.stats_path) if arm.stats_path else None,
            }
            for label, arm in arms.items()
        },
        "denominators": {
            "all_tasks": len(pairs),
            "valid_tasks": len(valid_pairs),
            "excluded_broken_tasks": len(broken_pairs),
            "excluded": [
                {
                    "task_digest": pair["comparator"].digest,
                    **{f"{label}_id": pair[label].identifier for label in ARM_LABELS},
                }
                for pair in broken_pairs
            ],
        },
        "arms": {},
        "comparisons_to_comparator": {},
    }
    for label in ARM_LABELS:
        all_tasks = [pair[label] for pair in pairs]
        valid_tasks = [pair[label] for pair in valid_pairs]
        arm_report: dict[str, Any] = {
            "all_tasks": metric_summary(all_tasks),
            "valid_tasks": metric_summary(valid_tasks),
        }
        if label in {"neutral_exact", "name_only"}:
            arm_report["static_output_shape"] = static_shape(all_tasks, target)
        report["arms"][label] = arm_report
    comparator_all = [pair["comparator"] for pair in pairs]
    comparator_valid = [pair["comparator"] for pair in valid_pairs]
    for label in ("neutral_exact", "name_only"):
        report["comparisons_to_comparator"][label] = {
            "all_tasks": compare_tasks([pair[label] for pair in pairs], comparator_all),
            "valid_tasks": compare_tasks([pair[label] for pair in valid_pairs], comparator_valid)
            if valid_pairs
            else {},
        }
    report["neutral_exact_vs_name_only"] = {
        "all_tasks": compare_tasks(
            [pair["name_only"] for pair in pairs], [pair["neutral_exact"] for pair in pairs]
        ),
        "direction": "name_only_minus_neutral_exact",
    }
    return report


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.4f}%"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed signature-scrub v3 analysis",
        "",
        f"Integrity checks: **PASS**. Paired tasks: {report['denominators']['all_tasks']}; "
        f"valid sensitivity denominator: {report['denominators']['valid_tasks']}.",
        "",
        "## Metrics (all tasks)",
        "",
        "| Arm | pass@1 | pass@5 | pass@10 | aligned JIT compile@1 | aligned JIT compile@5 | compiled candidates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ARM_LABELS:
        summary = report["arms"][label]["all_tasks"]
        metrics = summary["metrics"]
        lines.append(
            f"| {label} | {pct(metrics['pass_at_1'])} | {pct(metrics['pass_at_5'])} | "
            f"{pct(metrics['pass_at_10'])} | {pct(metrics['compile_at_1'])} | "
            f"{pct(metrics['compile_at_5'])} | {summary['candidate_compile_count']}/{summary['candidate_count']} |"
        )
    lines.extend(["", "## Paired outcomes versus comparator", ""])
    for label in ("neutral_exact", "name_only"):
        lines.extend(
            [
                f"### {label}",
                "",
                "| Metric | Mean delta | metric gains/losses/ties | solved-task gains/losses/ties |",
                "|---|---:|---:|---:|",
            ]
        )
        comparison = report["comparisons_to_comparator"][label]["all_tasks"]
        for metric in METRICS:
            item = comparison[metric]
            lines.append(
                f"| {metric} | {100 * item['mean_delta']:+.4f} pp | "
                f"{item['metric_gains']}/{item['metric_losses']}/{item['metric_ties']} | "
                f"{item['solved_task_gains']}/{item['solved_task_losses']}/{item['solved_task_ties']} |"
            )
        shape = report["arms"][label].get("static_output_shape")
        if shape:
            lines.extend(
                [
                    "",
                    f"Static `{shape['target']}` shape: {shape['top_level_target_definitions']}/"
                    f"{shape['candidate_count']} candidates define it at top level; "
                    f"{shape['top_level_target_arity_matches']} match the hidden arity.",
                ]
            )
        lines.append("")
    valid_count = report["denominators"]["valid_tasks"]
    excluded_count = report["denominators"]["excluded_broken_tasks"]
    lines.extend(
        [
            f"## Sensitivity metrics (valid tasks only; n={valid_count})",
            "",
            f"This sensitivity excludes {excluded_count} inherited benchmark contract defects; "
            "the all-task results above remain primary.",
            "",
            "| Arm | pass@1 | pass@5 | pass@10 | aligned JIT compile@1 | aligned JIT compile@5 | compiled candidates |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ARM_LABELS:
        summary = report["arms"][label]["valid_tasks"]
        metrics = summary["metrics"]
        lines.append(
            f"| {label} | {pct(metrics['pass_at_1'])} | {pct(metrics['pass_at_5'])} | "
            f"{pct(metrics['pass_at_10'])} | {pct(metrics['compile_at_1'])} | "
            f"{pct(metrics['compile_at_5'])} | {summary['candidate_compile_count']}/"
            f"{summary['candidate_count']} |"
        )
    lines.extend(["", "## Sensitivity paired outcomes versus comparator (valid tasks only)", ""])
    if valid_count:
        for label in ("neutral_exact", "name_only"):
            lines.extend(
                [
                    f"### {label} (valid tasks)",
                    "",
                    "| Metric | Mean delta | metric gains/losses/ties | solved-task gains/losses/ties |",
                    "|---|---:|---:|---:|",
                ]
            )
            comparison = report["comparisons_to_comparator"][label]["valid_tasks"]
            for metric in METRICS:
                item = comparison[metric]
                lines.append(
                    f"| {metric} | {100 * item['mean_delta']:+.4f} pp | "
                    f"{item['metric_gains']}/{item['metric_losses']}/{item['metric_ties']} | "
                    f"{item['solved_task_gains']}/{item['solved_task_losses']}/"
                    f"{item['solved_task_ties']} |"
                )
            lines.append("")
    else:
        lines.extend(["No valid tasks remain after exclusions.", ""])
    lines.extend(
        [
            "## Integrity/provenance",
            "",
            f"Pairing: `{report['pairing']['method']}`. Checkpoint, generation seed, and every "
            "available scorer/SDK identifier agree across arms. Hidden tests agree for every pair "
            "after target-name normalization.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator", type=Path, required=True, help="Frozen comparator predictions JSON")
    parser.add_argument("--neutral-exact", type=Path, required=True, help="Opaque neutral-exact predictions JSON")
    parser.add_argument("--name-only", type=Path, required=True, help="Opaque name-only predictions JSON")
    parser.add_argument("--comparator-stats", type=Path)
    parser.add_argument("--neutral-exact-stats", type=Path)
    parser.add_argument("--name-only-stats", type=Path)
    parser.add_argument("--comparator-provenance", type=Path, action="append", default=[])
    parser.add_argument("--neutral-exact-provenance", type=Path, action="append", default=[])
    parser.add_argument("--name-only-provenance", type=Path, action="append", default=[])
    parser.add_argument("--pair-manifest", type=Path, help="Explicit cross-arm task mapping JSON")
    parser.add_argument("--broken-tasks", type=Path, help="JSON list used for valid-denominator sensitivity")
    parser.add_argument("--target-name", default="fn0")
    parser.add_argument("--expected-tasks", type=int, default=154)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args(argv)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_tasks <= 0 or args.expected_candidates <= 0:
        raise AnalysisError("Expected task/candidate counts must be positive")
    config = {
        "comparator": (args.comparator, args.comparator_stats, args.comparator_provenance),
        "neutral_exact": (args.neutral_exact, args.neutral_exact_stats, args.neutral_exact_provenance),
        "name_only": (args.name_only, args.name_only_stats, args.name_only_provenance),
    }
    arms = {
        label: load_arm(
            label,
            prediction,
            stats,
            provenance,
            args.expected_tasks,
            args.expected_candidates,
        )
        for label, (prediction, stats, provenance) in config.items()
    }
    provenance = validate_provenance(arms, args.expected_tasks, args.expected_candidates)
    pairs, pairing = build_pairs(arms, args.pair_manifest, args.expected_tasks)
    report = make_report(
        arms,
        pairs,
        pairing,
        provenance,
        load_broken_tokens(args.broken_tasks),
        args.target_name,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = analyze(args)
    except AnalysisError as exc:
        raise SystemExit(f"INTEGRITY FAILURE: {exc}") from exc
    if not args.output_json and not args.output_markdown:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"PASS: analyzed {args.expected_tasks} paired tasks x {args.expected_candidates} candidates; "
            f"pairing={report['pairing']['method']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
