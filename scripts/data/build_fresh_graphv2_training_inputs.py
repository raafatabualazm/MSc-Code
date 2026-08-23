#!/usr/bin/env python3
"""Build holdout-bound Graph-v2 training inputs without mutating source pools."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.data.seal_fresh_graphv2_holdout import (  # noqa: E402
    Candidate,
    Fingerprint,
    fingerprint,
    load_candidates,
    row_names,
    sha256_bytes,
    sha256_file,
    similarity_match,
    stratum_for,
)


class TrainingInputError(RuntimeError):
    """Raised when the requested split cannot be proven leakage-clean."""


@dataclass(frozen=True)
class RawRow:
    path: Path
    line_number: int
    raw_line: str
    row: dict[str, Any]
    task_id: str
    fp: Fingerprint
    source_hashes: dict[str, str]


@dataclass
class MatchIndex:
    fingerprints: list[Fingerprint]
    names: set[str]
    source_hashes: dict[str, dict[str, list[str]]]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingInputError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingInputError(f"expected a JSON object in {path}")
    return value


def load_source_fingerprints(
    controls_path: Path,
) -> Callable[[dict[str, Any] | str], dict[str, str]]:
    spec = importlib.util.spec_from_file_location(
        "hybrid_data_controls_v2_3",
        controls_path,
    )
    if spec is None or spec.loader is None:
        raise TrainingInputError(f"could not load hybrid controls from {controls_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source_fingerprints = getattr(module, "source_fingerprints", None)
    if not callable(source_fingerprints):
        raise TrainingInputError(f"{controls_path} does not export source_fingerprints")
    return source_fingerprints


def source_text(row: dict[str, Any]) -> str:
    return str(row.get("dart_source", row.get("source", "")) or "")


def task_identity(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("task_id", row.get("id", fallback)) or fallback)


def iter_raw_jsonl(path: Path) -> Iterable[tuple[int, str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TrainingInputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TrainingInputError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, raw, row


def load_raw_rows(
    path: Path,
    source_fingerprints: Callable[[dict[str, Any] | str], dict[str, str]],
) -> list[RawRow]:
    rows: list[RawRow] = []
    seen: set[str] = set()
    for line_number, raw, row in iter_raw_jsonl(path):
        identity = task_identity(row, str(line_number))
        if identity in seen:
            raise TrainingInputError(f"{path}:{line_number}: duplicate task identity {identity!r}")
        seen.add(identity)
        source = source_text(row)
        if not source:
            raise TrainingInputError(f"{path}:{line_number}: missing source text")
        name = str(row.get("function", "") or "")
        rows.append(
            RawRow(
                path=path,
                line_number=line_number,
                raw_line=raw,
                row=row,
                task_id=identity,
                fp=fingerprint(source, name, identity),
                source_hashes=source_fingerprints(row),
            )
        )
    return rows


def empty_index() -> MatchIndex:
    return MatchIndex(
        fingerprints=[],
        names=set(),
        source_hashes={
            "neutral_sha256": defaultdict(list),
            "alpha_structural_sha256": defaultdict(list),
        },
    )


def add_to_index(index: MatchIndex, raw: RawRow, *, include_names: bool) -> None:
    index.fingerprints.append(raw.fp)
    if include_names:
        index.names.update(row_names(raw.row))
    for kind, digest in raw.source_hashes.items():
        index.source_hashes.setdefault(kind, defaultdict(list))[digest].append(raw.task_id)


def exact_hash_match(raw: RawRow, index: MatchIndex) -> dict[str, Any] | None:
    for kind in ("neutral_sha256", "alpha_structural_sha256"):
        digest = raw.source_hashes.get(kind, "")
        matched = index.source_hashes.get(kind, {}).get(digest, [])
        if digest and matched:
            return {
                "reason": "exact_neutral_source" if kind == "neutral_sha256" else "alpha_structural_source",
                "fingerprint_kind": kind,
                "fingerprint": digest,
                "matched_task_ids": matched,
            }
    return None


def build_index(rows: Iterable[RawRow], *, include_names: bool) -> MatchIndex:
    index = empty_index()
    for row in rows:
        add_to_index(index, row, include_names=include_names)
    return index


def row_from_candidate(
    candidate: Candidate,
    source_fingerprints: Callable[[dict[str, Any] | str], dict[str, str]],
) -> RawRow:
    return RawRow(
        path=candidate.path,
        line_number=candidate.source_line,
        raw_line=candidate.raw_line,
        row=candidate.row,
        task_id=candidate.task_id,
        fp=candidate.fingerprint,
        source_hashes=source_fingerprints(candidate.row),
    )


def match_row(
    raw: RawRow,
    index: MatchIndex,
    *,
    check_names: bool,
    jac_threshold: float,
    sequence_threshold: float,
) -> dict[str, Any] | None:
    if check_names:
        names = sorted(row_names(raw.row) & index.names)
        if names:
            return {"reason": "name_collision", "matches": names}

    exact_match = exact_hash_match(raw, index)
    if exact_match:
        return exact_match

    fuzzy = similarity_match(
        raw.fp,
        index.fingerprints,
        jac_threshold,
        sequence_threshold,
    )
    if fuzzy:
        reason, other, jac, sequence = fuzzy
        return {
            "reason": reason,
            "matched_task_id": other.task_id,
            "matched_function": other.name,
            "jaccard": jac,
            "sequence_ratio": sequence,
        }
    return None


def validate_source_manifest(
    source_manifest: dict[str, Any],
    source_path: Path,
    rows: int,
) -> None:
    outputs = source_manifest.get("outputs") or {}
    expected = outputs.get(source_path.name)
    if not isinstance(expected, dict):
        raise TrainingInputError(
            f"source manifest does not contain an output record for {source_path.name}"
        )
    actual_hash = sha256_file(source_path)
    if actual_hash != expected.get("sha256"):
        raise TrainingInputError(
            f"scrubbed master hash mismatch: expected {expected.get('sha256')} got {actual_hash}"
        )
    expected_rows = (source_manifest.get("counts") or {}).get("final_retained")
    if rows != expected_rows:
        raise TrainingInputError(
            f"scrubbed master row mismatch: expected {expected_rows} got {rows}"
        )
    if not (source_manifest.get("audit") or {}).get("passed"):
        raise TrainingInputError("scrubbed master source audit is not marked passed")


def validate_seal(
    seal_manifest: dict[str, Any],
    holdout_path: Path,
    holdout_rows: list[RawRow],
    candidates: list[Candidate],
    pool_specs: list[str],
) -> tuple[set[str], set[str], dict[str, Candidate]]:
    output = seal_manifest.get("output") or {}
    actual_holdout_hash = sha256_file(holdout_path)
    if output.get("sha256") != actual_holdout_hash:
        raise TrainingInputError(
            f"sealed holdout hash mismatch: expected {output.get('sha256')} got {actual_holdout_hash}"
        )
    if output.get("rows") != len(holdout_rows):
        raise TrainingInputError(
            f"sealed holdout row mismatch: expected {output.get('rows')} got {len(holdout_rows)}"
        )

    manifest_pools = {
        str(record.get("name")): record for record in seal_manifest.get("input_pools", [])
    }
    supplied_pools: dict[str, Path] = {}
    for spec in pool_specs:
        name, path_text = spec.split("=", 1)
        supplied_pools[name.strip()] = Path(path_text).expanduser().resolve()
    if set(manifest_pools) != set(supplied_pools):
        raise TrainingInputError(
            f"pool names differ from seal manifest: manifest={sorted(manifest_pools)} "
            f"supplied={sorted(supplied_pools)}"
        )
    for name, path in supplied_pools.items():
        record = manifest_pools[name]
        actual_hash = sha256_file(path)
        actual_rows = count_jsonl(path)
        if actual_hash != record.get("sha256") or actual_rows != record.get("rows"):
            raise TrainingInputError(
                f"pool {name} differs from sealed provenance: "
                f"hash={actual_hash} rows={actual_rows}"
            )

    by_id = {candidate.task_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise TrainingInputError("candidate task IDs are not unique")
    expected_candidate_rows = (seal_manifest.get("candidate_audit") or {}).get("rows")
    if expected_candidate_rows != len(candidates):
        raise TrainingInputError(
            f"candidate row mismatch: expected {expected_candidate_rows} got {len(candidates)}"
        )

    selected_tasks = (seal_manifest.get("selected") or {}).get("tasks") or []
    selected_ids = [str(item.get("task_id", "")) for item in selected_tasks]
    holdout_ids = [row.task_id for row in holdout_rows]
    if selected_ids != holdout_ids:
        raise TrainingInputError("holdout row order does not match the sealed selection manifest")
    for raw, selected_record in zip(holdout_rows, selected_tasks):
        expected_row_hash = str(selected_record.get("input_row_sha256", ""))
        actual_row_hash = sha256_bytes(raw.raw_line.encode("utf-8"))
        candidate = by_id.get(raw.task_id)
        if candidate is None or candidate.row_sha256 != expected_row_hash:
            raise TrainingInputError(f"selected task {raw.task_id} is missing or changed in its pool")
        if actual_row_hash != expected_row_hash:
            raise TrainingInputError(f"selected task {raw.task_id} differs from its sealed source row")

    external_records = (seal_manifest.get("candidate_audit") or {}).get(
        "external_exclusions", []
    )
    external_ids = {str(item.get("task_id", "")) for item in external_records}
    selected_set = set(selected_ids)
    if selected_set & external_ids:
        raise TrainingInputError("sealed and externally excluded candidate sets overlap")
    if not selected_set <= set(by_id) or not external_ids <= set(by_id):
        raise TrainingInputError("seal manifest references unknown candidate task IDs")
    return selected_set, external_ids, by_id


def rejection_record(
    raw: RawRow,
    *,
    scope: str,
    stage: str,
    match: dict[str, Any],
    pool: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": scope,
        "stage": stage,
        "task_id": raw.task_id,
        "source_path": str(raw.path.resolve()),
        "source_line": raw.line_number,
        **match,
    }
    if pool is not None:
        result["pool"] = pool
    return result


def rows_bytes(rows: Iterable[RawRow]) -> bytes:
    lines = [row.raw_line for row in rows]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def rejection_counts(rejections: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(f"{row['scope']}:{row['stage']}:{row['reason']}" for row in rejections)
    return dict(sorted(counts.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    holdout_path = Path(args.sealed_holdout).expanduser().resolve()
    seal_manifest_path = Path(args.seal_manifest).expanduser().resolve()
    existing_train_path = Path(args.existing_train).expanduser().resolve()
    existing_manifest_path = Path(args.existing_train_manifest).expanduser().resolve()
    controls_path = Path(args.hybrid_controls).expanduser().resolve()
    master_output = Path(args.master_output).expanduser().resolve()
    extra_output = Path(args.extra_output).expanduser().resolve()
    rejected_output = Path(args.rejected_output).expanduser().resolve()
    manifest_output = Path(args.manifest_output).expanduser().resolve()

    required = [
        holdout_path,
        seal_manifest_path,
        existing_train_path,
        existing_manifest_path,
        controls_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise TrainingInputError(f"required files do not exist: {missing}")
    output_paths = {master_output, extra_output, rejected_output, manifest_output}
    if len(output_paths) != 4:
        raise TrainingInputError("output paths must be distinct")
    input_paths = set(required)
    input_paths.update(Path(spec.split("=", 1)[1]).expanduser().resolve() for spec in args.pool)
    if output_paths & input_paths:
        raise TrainingInputError("an output path aliases an immutable input")

    source_fingerprints = load_source_fingerprints(controls_path)
    seal_manifest = load_json(seal_manifest_path)
    existing_manifest = load_json(existing_manifest_path)
    holdout_rows = load_raw_rows(holdout_path, source_fingerprints)
    existing_rows = load_raw_rows(existing_train_path, source_fingerprints)
    validate_source_manifest(existing_manifest, existing_train_path, len(existing_rows))

    candidates, pool_records = load_candidates(
        args.pool,
        args.seed,
        args.low_max,
        args.mid_max,
    )
    selected_ids, external_ids, candidates_by_id = validate_seal(
        seal_manifest,
        holdout_path,
        holdout_rows,
        candidates,
        args.pool,
    )
    holdout_index = build_index(holdout_rows, include_names=True)

    rejections: list[dict[str, Any]] = []
    clean_master: list[RawRow] = []
    clean_master_index = empty_index()
    for index, raw in enumerate(existing_rows, start=1):
        match = match_row(
            raw,
            holdout_index,
            check_names=False,
            jac_threshold=args.jac_threshold,
            sequence_threshold=args.sequence_threshold,
        )
        if match:
            rejections.append(
                rejection_record(
                    raw,
                    scope="existing_train",
                    stage="sealed_holdout",
                    match=match,
                )
            )
        else:
            duplicate = exact_hash_match(raw, clean_master_index)
            if duplicate:
                rejections.append(
                    rejection_record(
                        raw,
                        scope="existing_train",
                        stage="existing_train_set",
                        match=duplicate,
                    )
                )
            else:
                clean_master.append(raw)
                add_to_index(clean_master_index, raw, include_names=False)
        if index % 250 == 0 or index == len(existing_rows):
            print(f"master holdout audit {index}/{len(existing_rows)}", flush=True)

    external_by_id = {
        str(item.get("task_id", "")): item
        for item in (seal_manifest.get("candidate_audit") or {}).get(
            "external_exclusions", []
        )
    }
    for task_id in sorted(external_ids):
        candidate = candidates_by_id[task_id]
        raw = row_from_candidate(candidate, source_fingerprints)
        source_match = dict(external_by_id[task_id])
        source_match.pop("task_id", None)
        source_match.pop("pool", None)
        rejections.append(
            rejection_record(
                raw,
                scope="fresh_candidate",
                stage="external_corpora",
                match=source_match,
                pool=candidate.pool,
            )
        )

    leftovers = [
        candidate
        for candidate in candidates
        if candidate.task_id not in selected_ids and candidate.task_id not in external_ids
    ]
    leftovers.sort(key=lambda candidate: candidate.rank)
    clean_extra: list[RawRow] = []
    clean_extra_index = empty_index()
    for index, candidate in enumerate(leftovers, start=1):
        if index % 100 == 0 or index == len(leftovers):
            print(f"fresh extra audit {index}/{len(leftovers)}", flush=True)
        raw = row_from_candidate(candidate, source_fingerprints)
        match = match_row(
            raw,
            holdout_index,
            check_names=True,
            jac_threshold=args.jac_threshold,
            sequence_threshold=args.sequence_threshold,
        )
        if match:
            rejections.append(
                rejection_record(
                    raw,
                    scope="fresh_candidate",
                    stage="sealed_holdout",
                    match=match,
                    pool=candidate.pool,
                )
            )
            continue
        match = match_row(
            raw,
            clean_master_index,
            check_names=False,
            jac_threshold=args.jac_threshold,
            sequence_threshold=args.sequence_threshold,
        )
        if match:
            rejections.append(
                rejection_record(
                    raw,
                    scope="fresh_candidate",
                    stage="existing_train",
                    match=match,
                    pool=candidate.pool,
                )
            )
            continue
        match = match_row(
            raw,
            clean_extra_index,
            check_names=True,
            jac_threshold=args.jac_threshold,
            sequence_threshold=args.sequence_threshold,
        )
        if match:
            rejections.append(
                rejection_record(
                    raw,
                    scope="fresh_candidate",
                    stage="fresh_training_set",
                    match=match,
                    pool=candidate.pool,
                )
            )
            continue
        clean_extra.append(raw)
        add_to_index(clean_extra_index, raw, include_names=True)

    master_bytes = rows_bytes(clean_master)
    extra_bytes = rows_bytes(clean_extra)
    rejection_bytes = b"".join(json_bytes(row) for row in rejections)
    holdout_hash_before = sha256_file(holdout_path)
    atomic_write(master_output, master_bytes)
    atomic_write(extra_output, extra_bytes)
    atomic_write(rejected_output, rejection_bytes)
    holdout_hash_after = sha256_file(holdout_path)
    if holdout_hash_before != holdout_hash_after:
        raise TrainingInputError("sealed holdout changed while building training inputs")

    fresh_rejections = [row for row in rejections if row["scope"] == "fresh_candidate"]
    existing_rejections = [row for row in rejections if row["scope"] == "existing_train"]
    existing_holdout_rejections = [
        row for row in existing_rejections if row["stage"] == "sealed_holdout"
    ]
    existing_internal_rejections = [
        row for row in existing_rejections if row["stage"] == "existing_train_set"
    ]
    candidate_partition = len(selected_ids) + len(clean_extra) + len(fresh_rejections)
    if candidate_partition != len(candidates):
        raise TrainingInputError(
            f"fresh candidate partition is incomplete: {candidate_partition} != {len(candidates)}"
        )
    if len(clean_master) + len(existing_rejections) != len(existing_rows):
        raise TrainingInputError("existing training partition is incomplete")

    candidate_lookup = {candidate.task_id: candidate for candidate in candidates}
    retained_pools = Counter(candidate_lookup[row.task_id].pool for row in clean_extra)
    retained_strata = Counter(
        stratum_for(len(row.row.get("cfg") or []), args.low_max, args.mid_max)
        for row in clean_extra
    )
    retained_providers = Counter(
        f"{row.row.get('generator_provider', 'unknown')}:"
        f"{row.row.get('generator_model', 'unknown')}"
        for row in clean_extra
    )
    manifest = {
        "schema_version": "fresh-graphv2-training-inputs-v1",
        "stage": "holdout_bound_training_input_split",
        "created_utc": utc_now(),
        "policy": {
            "sealed_holdout_is_immutable": True,
            "source_pools_are_immutable": True,
            "training_interpretation": "same-corpus held-out split",
            "fresh_rows_are_raw_phase0_inputs": True,
            "checks": [
                "exact neutral source fingerprint",
                "alpha-structural source fingerprint",
                "v2.3-compatible exact/alpha deduplication within the scrubbed master",
                "holdout name collision for fresh candidates",
                "token Jaccard plus SequenceMatcher near-source match",
            ],
        },
        "parameters": {
            "seed": args.seed,
            "low_max": args.low_max,
            "mid_max": args.mid_max,
            "token_jaccard": args.jac_threshold,
            "sequence_matcher": args.sequence_threshold,
        },
        "inputs": {
            "sealed_holdout": file_record(holdout_path, len(holdout_rows)),
            "seal_manifest": file_record(seal_manifest_path),
            "existing_train": file_record(existing_train_path, len(existing_rows)),
            "existing_train_manifest": file_record(existing_manifest_path),
            "hybrid_data_controls": file_record(controls_path),
            "fresh_pools": pool_records,
        },
        "audit": {
            "existing_train_input_rows": len(existing_rows),
            "existing_train_holdout_exclusions": len(existing_holdout_rejections),
            "existing_train_internal_duplicates": len(existing_internal_rejections),
            "existing_train_retained_rows": len(clean_master),
            "fresh_candidate_rows": len(candidates),
            "sealed_holdout_rows": len(selected_ids),
            "manifest_external_exclusions": len(external_ids),
            "fresh_leftovers_considered": len(leftovers),
            "fresh_training_rejections": len(fresh_rejections) - len(external_ids),
            "fresh_training_retained_rows": len(clean_extra),
            "fresh_training_retained_pools": dict(sorted(retained_pools.items())),
            "fresh_training_retained_strata": dict(sorted(retained_strata.items())),
            "fresh_training_retained_providers": dict(sorted(retained_providers.items())),
            "rejection_counts": rejection_counts(rejections),
        },
        "outputs": {
            "holdout_bound_master_train_input": file_record(master_output, len(clean_master)),
            "fresh_extra_train_input": file_record(extra_output, len(clean_extra)),
            "rejections": file_record(rejected_output, len(rejections)),
        },
        "invariants": {
            "sealed_holdout_hash_unchanged": holdout_hash_before == holdout_hash_after,
            "sealed_holdout_order_matches_manifest": True,
            "fresh_candidate_partition_exact": candidate_partition == len(candidates),
            "existing_train_partition_exact": (
                len(clean_master) + len(existing_rejections) == len(existing_rows)
            ),
            "master_output_has_zero_holdout_overlap": True,
            "master_output_has_zero_exact_or_alpha_internal_duplicates": True,
            "fresh_output_has_zero_holdout_overlap": True,
            "fresh_output_has_zero_master_overlap": True,
            "fresh_output_has_zero_internal_overlap": True,
        },
    }
    atomic_write(manifest_output, json_bytes(manifest, pretty=True))
    manifest["outputs"]["manifest"] = file_record(manifest_output)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--seal-manifest", required=True)
    result.add_argument("--sealed-holdout", required=True)
    result.add_argument("--pool", action="append", required=True, help="NAME=PATH")
    result.add_argument("--existing-train", required=True)
    result.add_argument("--existing-train-manifest", required=True)
    result.add_argument("--hybrid-controls", required=True)
    result.add_argument("--master-output", required=True)
    result.add_argument("--extra-output", required=True)
    result.add_argument("--rejected-output", required=True)
    result.add_argument("--manifest-output", required=True)
    result.add_argument("--seed", type=int, default=44)
    result.add_argument("--low-max", type=int, default=14)
    result.add_argument("--mid-max", type=int, default=25)
    result.add_argument("--jac-threshold", type=float, default=0.55)
    result.add_argument("--sequence-threshold", type=float, default=0.70)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.low_max >= args.mid_max:
        raise SystemExit("--low-max must be less than --mid-max")
    if not 0.0 <= args.jac_threshold <= 1.0:
        raise SystemExit("--jac-threshold must be in [0, 1]")
    if not 0.0 <= args.sequence_threshold <= 1.0:
        raise SystemExit("--sequence-threshold must be in [0, 1]")
    try:
        manifest = build(args)
    except TrainingInputError as exc:
        raise SystemExit(f"training input split failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "passed",
                "existing_train_rows": manifest["audit"]["existing_train_retained_rows"],
                "fresh_extra_rows": manifest["audit"]["fresh_training_retained_rows"],
                "sealed_holdout_rows": manifest["audit"]["sealed_holdout_rows"],
                "rejections": manifest["outputs"]["rejections"]["rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
