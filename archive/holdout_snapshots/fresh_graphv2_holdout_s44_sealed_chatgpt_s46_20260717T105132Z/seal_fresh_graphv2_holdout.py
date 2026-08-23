"""Seal a deterministic, leakage-audited Graph-v2 fresh holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
MAIN_SPLIT_RE = re.compile(
    r"@pragma\('vm:entry-point'\)\s*\nvoid\s+main\s*\(", re.MULTILINE
)


class HoldoutError(RuntimeError):
    """Raised when sealing invariants cannot be satisfied."""


@dataclass(frozen=True)
class Fingerprint:
    joined: str
    tokens: frozenset[str]
    sha256: str
    name: str
    task_id: str


@dataclass(frozen=True)
class Candidate:
    pool: str
    path: Path
    source_line: int
    raw_line: str
    row: dict[str, Any]
    task_id: str
    function: str
    block_count: int
    stratum: str
    fingerprint: Fingerprint
    rank: str
    row_sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_body(source: str) -> str:
    return MAIN_SPLIT_RE.split(source, maxsplit=1)[0]


def normalized_tokens(source: str) -> list[str]:
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return [token.lower() for token in TOKEN_RE.findall(source)]


def fingerprint(source: str, name: str, task_id: str) -> Fingerprint:
    joined = " ".join(normalized_tokens(reference_body(source)))
    return Fingerprint(
        joined=joined,
        tokens=frozenset(joined.split()),
        sha256=sha256_bytes(joined.encode("utf-8")),
        name=name,
        task_id=task_id,
    )


def snakeify(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def row_names(row: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("function", "camel_case_function_name", "python_function_name"):
        value = str(row.get(key, "") or "").strip().lower()
        if value:
            names.add(value)
            names.add(snakeify(value))
    return names


def load_jsonl(path: Path) -> Iterable[tuple[int, str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            if raw.strip():
                yield line_number, raw, json.loads(raw)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def similarity_match(
    candidate: Fingerprint,
    pool: Iterable[Fingerprint],
    jac_threshold: float,
    sequence_threshold: float,
) -> tuple[str, Fingerprint, float, float] | None:
    for other in pool:
        if candidate.joined == other.joined:
            return "exact_source", other, 1.0, 1.0
        jac = jaccard(candidate.tokens, other.tokens)
        if jac < jac_threshold:
            continue
        sequence = SequenceMatcher(None, candidate.joined, other.joined).ratio()
        if sequence >= sequence_threshold:
            return "near_source", other, jac, sequence
    return None


def stratum_for(block_count: int, low_max: int, mid_max: int) -> str:
    if block_count <= low_max:
        return "low"
    if block_count <= mid_max:
        return "mid"
    return "high"


def parse_pool(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise HoldoutError(f"pool must be NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_file():
        raise HoldoutError(f"invalid pool {spec!r}: expected a name and existing file")
    return name, path


def file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def tool_record(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise HoldoutError(f"tool must be NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    requested_path = Path(raw_path).expanduser()
    resolved_path = requested_path.resolve()
    if not name.strip() or not resolved_path.is_file():
        raise HoldoutError(f"invalid tool {spec!r}: expected a name and executable file")
    completed = subprocess.run(
        [str(requested_path), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode != 0 or not version:
        raise HoldoutError(
            f"could not capture {name.strip()} version from {requested_path}: "
            f"returncode={completed.returncode}"
        )
    return {
        "name": name.strip(),
        "requested_path": str(requested_path),
        "resolved_path": str(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": sha256_file(resolved_path),
        "version": version,
    }


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[Fingerprint], list[dict[str, Any]]]:
    banned_names: set[str] = set()
    fingerprints: list[Fingerprint] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        path = path.resolve()
        rows = 0
        for line_number, _raw, row in load_jsonl(path):
            rows += 1
            banned_names.update(row_names(row))
            source = str(row.get("dart_source", row.get("source", "")) or "")
            if source:
                fingerprints.append(
                    fingerprint(
                        source,
                        str(row.get("function", "") or ""),
                        str(row.get("task_id", line_number)),
                    )
                )
        records.append(file_record(path, rows))
    return banned_names, fingerprints, records


def load_candidates(
    pool_specs: list[str], seed: int, low_max: int, mid_max: int
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    candidates: list[Candidate] = []
    pool_records: list[dict[str, Any]] = []
    seen_task_ids: dict[str, tuple[str, int]] = {}
    for pool_name, path in map(parse_pool, pool_specs):
        counts: Counter[str] = Counter()
        extractor_hashes: set[str] = set()
        schema_versions: set[str] = set()
        rows = 0
        for line_number, raw, row in load_jsonl(path):
            rows += 1
            task_id = str(row.get("task_id", "") or "").strip()
            function = str(row.get("function", "") or "").strip()
            source = str(row.get("dart_source", "") or "")
            cfg = row.get("cfg")
            integrity = row.get("integrity") or {}
            graph_v2 = row.get("graph_v2") or {}
            if not task_id or not function or not source:
                raise HoldoutError(f"{path}:{line_number}: missing task/function/source")
            if task_id in seen_task_ids:
                previous = seen_task_ids[task_id]
                raise HoldoutError(
                    f"duplicate task_id {task_id!r}: {previous[0]}:{previous[1]} and "
                    f"{path}:{line_number}"
                )
            seen_task_ids[task_id] = (str(path), line_number)
            if not isinstance(cfg, list) or not cfg or integrity.get("valid") is not True:
                raise HoldoutError(f"{path}:{line_number}: invalid or empty Graph-v2 row")
            block_count = len(cfg)
            stratum = stratum_for(block_count, low_max, mid_max)
            counts[stratum] += 1
            fp = fingerprint(source, function, task_id)
            rank = sha256_bytes(
                f"{seed}|{pool_name}|{task_id}|{fp.sha256}".encode("utf-8")
            )
            candidates.append(
                Candidate(
                    pool=pool_name,
                    path=path,
                    source_line=line_number,
                    raw_line=raw,
                    row=row,
                    task_id=task_id,
                    function=function,
                    block_count=block_count,
                    stratum=stratum,
                    fingerprint=fp,
                    rank=rank,
                    row_sha256=sha256_bytes(raw.encode("utf-8")),
                )
            )
            extractor = str(graph_v2.get("extractor_sha256", "") or "")
            schema = str(graph_v2.get("schema", "") or "")
            if extractor:
                extractor_hashes.add(extractor)
            if schema:
                schema_versions.add(schema)
        pool_records.append(
            {
                **file_record(path, rows),
                "name": pool_name,
                "realized_strata": dict(sorted(counts.items())),
                "extractor_sha256": sorted(extractor_hashes),
                "graph_schema": sorted(schema_versions),
            }
        )
    return candidates, pool_records


def externally_eligible(
    candidates: list[Candidate],
    banned_names: set[str],
    exclusion_fingerprints: list[Fingerprint],
    jac_threshold: float,
    sequence_threshold: float,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    eligible: list[Candidate] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_names = row_names(candidate.row)
        name_overlap = sorted(candidate_names & banned_names)
        if name_overlap:
            excluded.append(
                {
                    "task_id": candidate.task_id,
                    "pool": candidate.pool,
                    "reason": "banned_name",
                    "matches": name_overlap,
                }
            )
            continue
        match = similarity_match(
            candidate.fingerprint,
            exclusion_fingerprints,
            jac_threshold,
            sequence_threshold,
        )
        if match:
            reason, other, jac, sequence = match
            excluded.append(
                {
                    "task_id": candidate.task_id,
                    "pool": candidate.pool,
                    "reason": reason,
                    "matched_task_id": other.task_id,
                    "matched_function": other.name,
                    "jaccard": jac,
                    "sequence_ratio": sequence,
                }
            )
            continue
        eligible.append(candidate)
    return eligible, excluded


def select_candidates(
    eligible: list[Candidate],
    quotas: dict[str, int],
    jac_threshold: float,
    sequence_threshold: float,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    selected: list[Candidate] = []
    selected_fingerprints: list[Fingerprint] = []
    selected_names: set[str] = set()
    internal_exclusions: list[dict[str, Any]] = []
    for stratum in ("low", "mid", "high"):
        ranked = sorted(
            (candidate for candidate in eligible if candidate.stratum == stratum),
            key=lambda candidate: candidate.rank,
        )
        for candidate in ranked:
            if sum(item.stratum == stratum for item in selected) >= quotas[stratum]:
                break
            names = row_names(candidate.row)
            duplicate_names = sorted(names & selected_names)
            if duplicate_names:
                internal_exclusions.append(
                    {
                        "task_id": candidate.task_id,
                        "pool": candidate.pool,
                        "reason": "selected_name_collision",
                        "matches": duplicate_names,
                    }
                )
                continue
            match = similarity_match(
                candidate.fingerprint,
                selected_fingerprints,
                jac_threshold,
                sequence_threshold,
            )
            if match:
                reason, other, jac, sequence = match
                internal_exclusions.append(
                    {
                        "task_id": candidate.task_id,
                        "pool": candidate.pool,
                        "reason": f"selected_{reason}",
                        "matched_task_id": other.task_id,
                        "matched_function": other.name,
                        "jaccard": jac,
                        "sequence_ratio": sequence,
                    }
                )
                continue
            selected.append(candidate)
            selected_fingerprints.append(candidate.fingerprint)
            selected_names.update(names)
    realized = Counter(candidate.stratum for candidate in selected)
    missing = {
        stratum: quotas[stratum] - realized[stratum]
        for stratum in quotas
        if realized[stratum] < quotas[stratum]
    }
    if missing:
        supply = Counter(candidate.stratum for candidate in eligible)
        raise HoldoutError(
            "insufficient leakage-clean realized supply: "
            f"missing={dict(missing)} eligible={dict(supply)} selected={dict(realized)}"
        )
    return selected, internal_exclusions


def verify_selected(
    selected: list[Candidate],
    banned_names: set[str],
    exclusion_fingerprints: list[Fingerprint],
    jac_threshold: float,
    sequence_threshold: float,
) -> dict[str, Any]:
    seen_names: set[str] = set()
    seen_fingerprints: list[Fingerprint] = []
    for candidate in selected:
        names = row_names(candidate.row)
        external_names = names & banned_names
        if external_names:
            raise HoldoutError(
                f"selected task {candidate.task_id} overlaps excluded names: "
                f"{sorted(external_names)}"
            )
        external_match = similarity_match(
            candidate.fingerprint,
            exclusion_fingerprints,
            jac_threshold,
            sequence_threshold,
        )
        if external_match:
            raise HoldoutError(
                f"selected task {candidate.task_id} overlaps an exclusion source"
            )
        internal_names = names & seen_names
        if internal_names:
            raise HoldoutError(
                f"selected task {candidate.task_id} repeats selected names: "
                f"{sorted(internal_names)}"
            )
        internal_match = similarity_match(
            candidate.fingerprint,
            seen_fingerprints,
            jac_threshold,
            sequence_threshold,
        )
        if internal_match:
            raise HoldoutError(
                f"selected task {candidate.task_id} overlaps a prior selected source"
            )
        seen_names.update(names)
        seen_fingerprints.append(candidate.fingerprint)
    return {
        "selected_rows_checked": len(selected),
        "external_name_overlaps": 0,
        "external_source_overlaps": 0,
        "internal_name_overlaps": 0,
        "internal_source_overlaps": 0,
    }


def seal(args: argparse.Namespace) -> dict[str, Any]:
    if args.low_max >= args.mid_max:
        raise HoldoutError("--low-max must be less than --mid-max")
    quotas = {"low": args.low, "mid": args.mid, "high": args.high}
    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) <= 0:
        raise HoldoutError("stratum quotas must be non-negative and sum to a positive value")

    exclude_paths = [Path(path).expanduser().resolve() for path in args.exclude]
    for path in exclude_paths:
        if not path.is_file():
            raise HoldoutError(f"exclusion file does not exist: {path}")
    candidates, pool_records = load_candidates(
        args.pool, args.seed, args.low_max, args.mid_max
    )
    banned_names, exclusion_fingerprints, exclusion_records = load_exclusions(
        exclude_paths
    )
    eligible, external_exclusions = externally_eligible(
        candidates,
        banned_names,
        exclusion_fingerprints,
        args.jac_threshold,
        args.sequence_threshold,
    )
    eligible_counts = Counter(candidate.stratum for candidate in eligible)
    missing = {
        stratum: quota - eligible_counts[stratum]
        for stratum, quota in quotas.items()
        if eligible_counts[stratum] < quota
    }
    if missing:
        raise HoldoutError(
            "insufficient externally leakage-clean realized supply: "
            f"missing={missing} eligible={dict(eligible_counts)}"
        )
    selected, internal_exclusions = select_candidates(
        eligible,
        quotas,
        args.jac_threshold,
        args.sequence_threshold,
    )
    selected_leakage_audit = verify_selected(
        selected,
        banned_names,
        exclusion_fingerprints,
        args.jac_threshold,
        args.sequence_threshold,
    )

    output_bytes = ("\n".join(candidate.raw_line for candidate in selected) + "\n").encode(
        "utf-8"
    )
    selected_counts = Counter(candidate.stratum for candidate in selected)
    pool_counts = Counter(candidate.pool for candidate in selected)
    provider_counts = Counter(
        f"{candidate.row.get('generator_provider', 'unknown')}:"
        f"{candidate.row.get('generator_model', 'unknown')}"
        for candidate in selected
    )
    target_realized = Counter(
        f"{candidate.row.get('target_stratum', 'unknown')}->{candidate.stratum}"
        for candidate in selected
    )
    selected_ids = [candidate.task_id for candidate in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise HoldoutError("selected task IDs are not unique")

    provenance_records = []
    for raw_path in args.provenance:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise HoldoutError(f"provenance file does not exist: {path}")
        provenance_records.append(file_record(path))
    tool_records = [tool_record(spec) for spec in args.tool]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)
    output_record = file_record(args.output, len(selected))
    manifest = {
        "schema_version": 1,
        "stage": "fresh_graphv2_holdout_seal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "selection_rule": (
            "For each realized stratum in low,mid,high order, sort by "
            "SHA256(seed|pool|task_id|normalized-reference-SHA256); take the first "
            "quota rows that pass external and selected-set leakage checks."
        ),
        "realized_strata": {
            "low": f"cfg_block_count <= {args.low_max}",
            "mid": f"{args.low_max} < cfg_block_count <= {args.mid_max}",
            "high": f"cfg_block_count > {args.mid_max}",
        },
        "quotas": quotas,
        "leakage_thresholds": {
            "token_jaccard": args.jac_threshold,
            "sequence_matcher": args.sequence_threshold,
            "near_match_rule": "both thresholds must be met",
            "comparison_scope": "reference function before generated main",
        },
        "input_pools": pool_records,
        "exclusion_corpora": exclusion_records,
        "provenance_files": provenance_records,
        "toolchain": tool_records,
        "source_files": [file_record(Path(__file__).resolve())],
        "candidate_audit": {
            "rows": len(candidates),
            "eligible_rows": len(eligible),
            "eligible_realized_strata": dict(sorted(eligible_counts.items())),
            "external_exclusions": external_exclusions,
            "selected_set_exclusions": internal_exclusions,
        },
        "selected_leakage_audit": selected_leakage_audit,
        "selected": {
            "rows": len(selected),
            "realized_strata": dict(sorted(selected_counts.items())),
            "source_pools": dict(sorted(pool_counts.items())),
            "providers": dict(sorted(provider_counts.items())),
            "target_to_realized_strata": dict(sorted(target_realized.items())),
            "tasks": [
                {
                    "task_id": candidate.task_id,
                    "function": candidate.function,
                    "pool": candidate.pool,
                    "source_line": candidate.source_line,
                    "cfg_block_count": candidate.block_count,
                    "realized_stratum": candidate.stratum,
                    "normalized_reference_sha256": candidate.fingerprint.sha256,
                    "input_row_sha256": candidate.row_sha256,
                    "selection_rank": candidate.rank,
                }
                for candidate in selected
            ],
        },
        "output": output_record,
        "invariants": {
            "exact_quota_counts": dict(selected_counts) == quotas,
            "external_name_overlap_zero": selected_leakage_audit[
                "external_name_overlaps"
            ]
            == 0,
            "external_source_overlap_zero": selected_leakage_audit[
                "external_source_overlaps"
            ]
            == 0,
            "selected_task_ids_unique": len(selected_ids) == len(set(selected_ids)),
            "selected_name_overlap_zero": selected_leakage_audit[
                "internal_name_overlaps"
            ]
            == 0,
            "selected_source_overlap_zero": selected_leakage_audit[
                "internal_source_overlaps"
            ]
            == 0,
            "output_rows_match": len(selected) == sum(quotas.values()),
            "input_rows_preserved_verbatim": True,
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = file_record(args.manifest)
    print(
        json.dumps(
            {
                "output": output_record,
                "manifest": manifest["manifest"],
                "selected_realized_strata": dict(selected_counts),
                "selected_source_pools": dict(pool_counts),
            },
            indent=2,
        )
    )
    return manifest


def self_test() -> int:
    def fake_row(task_id: str, blocks: int) -> dict[str, Any]:
        function = f"function{task_id}"
        return {
            "task_id": task_id,
            "function": function,
            "dart_source": f"int {function}(int value) {{ return value + {task_id}; }}",
            "assembly": "Dump of assembler code\nEnd of assembler dump.",
            "cfg": [{"id": index} for index in range(blocks)],
            "edges": [],
            "integrity": {"valid": True},
            "graph_v2": {"schema": "antigravity-graph-v2.1", "extractor_sha256": "x"},
            "target_stratum": "test",
        }

    with tempfile.TemporaryDirectory(prefix="seal_holdout_selftest_") as raw_tmp:
        tmp = Path(raw_tmp)
        pool_a = tmp / "a.jsonl"
        pool_b = tmp / "b.jsonl"
        exclusion = tmp / "exclude.jsonl"
        rows_a = [fake_row(str(index), blocks) for index, blocks in enumerate((3, 8, 15, 20, 26))]
        rows_b = [
            fake_row(str(index), blocks)
            for index, blocks in enumerate((10, 14, 18, 25, 30, 40), start=10)
        ]
        pool_a.write_text(
            "".join(json.dumps(row) + "\n" for row in rows_a), encoding="utf-8"
        )
        pool_b.write_text(
            "".join(json.dumps(row) + "\n" for row in rows_b), encoding="utf-8"
        )
        exclusion.write_text(
            json.dumps(
                {
                    "task_id": "excluded",
                    "function": "excludedFunction",
                    "dart_source": "String excludedFunction(String text) { return text.trim(); }",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            pool=[f"old={pool_a}", f"topup={pool_b}"],
            exclude=[str(exclusion)],
            provenance=[],
            tool=[],
            output=tmp / "holdout.jsonl",
            manifest=tmp / "holdout.manifest.json",
            seed=44,
            low=2,
            mid=2,
            high=2,
            low_max=14,
            mid_max=25,
            jac_threshold=1.0,
            sequence_threshold=1.0,
        )
        first = seal(args)
        first_hash = first["output"]["sha256"]
        second = seal(args)
        assert second["output"]["sha256"] == first_hash
        assert first["selected"]["realized_strata"] == {"high": 2, "low": 2, "mid": 2}
        assert first["invariants"]["exact_quota_counts"]
        assert len(args.output.read_text(encoding="utf-8").splitlines()) == 6
    print("self_test OK: deterministic selection, exact realized quotas, and manifest hashes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", action="append", default=[], help="NAME=graphv2.jsonl")
    parser.add_argument("--exclude", action="append", default=[], help="JSONL corpus to exclude")
    parser.add_argument(
        "--provenance", action="append", default=[], help="Additional immutable source artifact"
    )
    parser.add_argument(
        "--tool", action="append", default=[], help="NAME=executable used for the build"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--low", type=int, default=170)
    parser.add_argument("--mid", type=int, default=170)
    parser.add_argument("--high", type=int, default=160)
    parser.add_argument("--low-max", type=int, default=14)
    parser.add_argument("--mid-max", type=int, default=25)
    parser.add_argument("--jac-threshold", type=float, default=0.55)
    parser.add_argument("--sequence-threshold", type=float, default=0.70)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.pool or not args.exclude or args.output is None or args.manifest is None:
        parser.error("--pool, --exclude, --output, and --manifest are required")
    try:
        seal(args)
    except HoldoutError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
