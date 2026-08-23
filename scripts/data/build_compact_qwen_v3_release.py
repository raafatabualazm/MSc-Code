#!/usr/bin/env python3
"""Build a sealed encoder-free compact-Qwen v3 release.

The input rows are private, source-blind AOT graph/pool receipts emitted by
``build_phase0_binary_pool_graphs.py``.  Only the four-field compact model row
is public.  The private alignment sidecar contains hashes and counts, never a
pool payload, compact text, assembly, graph, or supervised Dart source.

Instruction atoms are fitted from ``--fit`` only.  Every ``--measure`` row is
canonicalized and measured with that frozen train-side codebook, so fallback
statistics are honest generalization measurements rather than a fit-on-all
estimate.

The release accepts only the finalizer's single ``binary_build_manifest.json``
as AOT provenance.  Split build manifests are deliberately insufficient: they
do not prove that every retained AOT was rehashed immediately before sealing.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import build_compact_qwen_v2 as graph_v2
from scripts.data import build_compact_qwen_v3 as codec
from scripts.data import build_phase0_binary_pool_graphs as aot_builder


POOL_ALIGNMENT_SCHEMA = "dart-aot-target-pool-alignment-v1"
FINAL_BINARY_BUILD_SCHEMA = "phase0-s44-binary-pool-build-seal-v1"
PUBLIC_FIELDS = frozenset(
    {
        "compact_input_ids",
        "compact_codec_sha256",
        "compact_codebook_sha256",
        "compact_tokenizer_sha256",
    }
)
POOL_METADATA_FIELDS = frozenset(
    {
        "schema",
        "receipt_sha256",
        "projection_sha256",
        "use_count",
        "source_blind",
        "target_function",
    }
)
FORBIDDEN_ALIGNMENT_KEYS = frozenset(
    {
        "analysis_program",
        "assembly",
        "binary_pool",
        "binary_pool_private_receipt",
        "binary_pool_uses",
        "cfg",
        "compact_text",
        "dart_source",
        "edges",
        "function_source",
        "payload",
        "tests",
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def require_sha256(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(result):
        raise ValueError(f"{label}_must_be_lowercase_sha256")
    return result


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_{label}:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label}_must_be_json_object")
    return value


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank_jsonl_line:{path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non_object_jsonl_row:{path}:{line_number}")
            yield line_number, value


def write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n",
    )


def write_jsonl_atomic(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _rate(counter: Mapping[str, int]) -> float:
    total = int(counter.get("instructions", 0))
    return float(counter.get("fallback", 0)) / total if total else 0.0


def _with_rate(value: Mapping[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(value)
    result["fallback_rate"] = _rate(value)
    return result


def decode_compact_ids(
    ids: Sequence[int], base: Tokenizer, atom_ids: Mapping[str, int]
) -> str:
    """Invert ``compact_ids`` exactly, including base-token pool JSON spans."""
    atoms = {int(token_id): token for token, token_id in atom_ids.items()}
    if len(atoms) != len(atom_ids):
        raise ValueError("source_atom_id_collision")
    output: list[str] = []
    base_segment: list[int] = []

    def flush() -> None:
        if base_segment:
            output.append(base.decode(base_segment, skip_special_tokens=False))
            base_segment.clear()

    for token_id in ids:
        token_id = int(token_id)
        atom = atoms.get(token_id)
        if atom is None:
            base_segment.append(token_id)
        else:
            flush()
            output.append(atom)
    flush()
    return "".join(output)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def assert_alignment_source_free(value: Mapping[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_ALIGNMENT_KEYS & set(_walk_keys(value)))
    if forbidden:
        raise ValueError("alignment_contains_private_payload:" + ",".join(forbidden))


def pool_alignment_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact six-field source-free pool receipt projection."""
    uses = row.get("binary_pool_uses")
    if not isinstance(uses, list):
        raise ValueError("missing_binary_pool_uses")
    receipt = row.get("binary_pool_private_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("missing_binary_pool_private_receipt")
    projection = canonical_sha256(uses)
    bound_projection = require_sha256(
        receipt.get("projection_sha256"), "pool_projection_sha256"
    )
    if projection != bound_projection:
        raise ValueError(
            f"pool_projection_sha256_mismatch:{projection}!={bound_projection}"
        )
    static = receipt.get("static")
    runtime = receipt.get("runtime")
    if not isinstance(static, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("pool_receipt_missing_static_or_runtime_component")
    if static.get("source_blind") is not True or runtime.get("source_blind") is not True:
        raise ValueError("pool_receipt_not_source_blind")
    if static.get("target_function") != codec.TARGET_FUNCTION:
        raise ValueError("static_pool_receipt_target_mismatch")
    # Empty-runtime receipts remain source blind and are explicitly emitted by
    # the AOT builder.  They still bind the candidate target.
    if runtime.get("target_function") != codec.TARGET_FUNCTION:
        raise ValueError("runtime_pool_receipt_target_mismatch")
    result = {
        "schema": POOL_ALIGNMENT_SCHEMA,
        "receipt_sha256": canonical_sha256(receipt),
        "projection_sha256": projection,
        # The trainer/inference validator compares this with len(pool["uses"]).
        "use_count": len(uses),
        "source_blind": True,
        "target_function": codec.TARGET_FUNCTION,
    }
    if set(result) != POOL_METADATA_FIELDS:
        raise AssertionError("pool_metadata_schema_drift")
    return result


def _nested_kind_counts(
    kind: str,
    payload: Mapping[str, Any],
    counts: collections.Counter[str],
    descriptor_pairs: collections.Counter[str],
) -> None:
    counts[kind] += 1
    if kind == "nonliteral":
        profile_type = str(payload.get("profile_type") or "")
        nonliteral_kind = str(payload.get("nonliteral_kind") or "")
        expected = codec.NESTED_NONLITERAL_PROFILE_KIND.get(profile_type)
        if expected != nonliteral_kind:
            raise ValueError(
                "nested_nonliteral_descriptor_not_in_pinned_allowlist:"
                f"{profile_type}:{nonliteral_kind}"
            )
        descriptor_pairs[f"{profile_type}->{nonliteral_kind}"] += 1
        return
    if kind != "composite":
        return
    for element in payload.get("elements") or []:
        nested = element["value"]
        _nested_kind_counts(
            str(nested["kind"]),
            nested["payload"],
            counts,
            descriptor_pairs,
        )


def pool_statistics(canonical: Mapping[str, Any]) -> dict[str, Any]:
    pool = canonical["binary_pool"]
    records = pool["uses"]
    top = collections.Counter(str(record["kind"]) for record in records)
    nested: collections.Counter[str] = collections.Counter()
    descriptor_pairs: collections.Counter[str] = collections.Counter()
    for record in records:
        _nested_kind_counts(
            str(record["kind"]),
            record["payload"],
            nested,
            descriptor_pairs,
        )
    return {
        "records": len(records),
        "use_sites": sum(len(record["use_sites"]) for record in records),
        "top_level_kinds": dict(top),
        "all_node_kinds": dict(nested),
        "nested_nonliteral_descriptor_pairs": dict(descriptor_pairs),
    }


def reconcile_raw_pool_xrefs(
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    pool_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Exhaustively classify every exact-target raw AOT pool xref.

    The model-side pool deliberately covers canonical-graph-retained literal
    uses only.  This private record proves that every raw target xref is either
    represented, retained-but-nonliteral, or deterministically excluded because
    its instruction was pruned from the canonical graph.  Literal values never
    enter this manifest.
    """
    accounting = row.get("pool_projection_accounting")
    expected_accounting_fields = {
        "scope",
        "target_exact_xrefs",
        "graph_retained_xrefs",
        "excluded_non_graph_xrefs",
        "excluded_non_graph_xref_count",
        "all_target_xrefs_accounted",
    }
    if not isinstance(accounting, Mapping) or set(accounting) != expected_accounting_fields:
        raise ValueError("pool_projection_accounting_schema_mismatch")
    if accounting.get("scope") != "canonical_graph_retained_fixed_r15_xrefs":
        raise ValueError("pool_projection_accounting_scope_mismatch")
    for field in (
        "target_exact_xrefs",
        "graph_retained_xrefs",
        "excluded_non_graph_xref_count",
    ):
        value = accounting.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"pool_projection_accounting_{field}_invalid")
    if accounting.get("all_target_xrefs_accounted") is not True:
        raise ValueError("pool_projection_accounting_not_exhaustive")

    receipt = row["binary_pool_private_receipt"]
    static = receipt["static"]
    runtime = receipt["runtime"]
    static_entries = {
        aot_builder._entry_offset(entry): entry for entry in static.get("entries") or []
    }
    runtime_entries = {
        aot_builder._entry_offset(entry): entry for entry in runtime.get("entries") or []
    }
    if len(static_entries) != len(static.get("entries") or []):
        raise ValueError("duplicate_static_pool_receipt_offset")
    if len(runtime_entries) != len(runtime.get("entries") or []):
        raise ValueError("duplicate_runtime_pool_receipt_offset")
    if set(static_entries) != set(runtime_entries):
        raise ValueError("static_runtime_pool_receipt_offset_set_mismatch")

    target_scope = [
        item
        for item in runtime.get("target_scope") or []
        if item.get("function_id") == codec.TARGET_FUNCTION
    ]
    raw_runtime_use_count = sum(
        use.get("function_id") == codec.TARGET_FUNCTION
        for entry in runtime_entries.values()
        for use in entry.get("uses") or []
    )
    if raw_runtime_use_count:
        if len(target_scope) != 1:
            raise ValueError("runtime_candidate_scope_not_unique")
        candidate_base = int(str(target_scope[0].get("aot_address")), 16)
    else:
        candidate_base = 0
    static_xrefs = aot_builder._exact_static_xrefs(static)
    runtime_xrefs = aot_builder._exact_runtime_xrefs(
        runtime, candidate_base=candidate_base
    )
    if static_xrefs != runtime_xrefs:
        raise ValueError("static_runtime_raw_xrefs_do_not_match")
    if len(runtime_xrefs) != accounting["target_exact_xrefs"]:
        raise ValueError("raw_xref_count_does_not_match_accounting")

    raw_excluded = accounting["excluded_non_graph_xrefs"]
    if not isinstance(raw_excluded, list):
        raise ValueError("excluded_non_graph_xrefs_must_be_list")
    if len(raw_excluded) != accounting["excluded_non_graph_xref_count"]:
        raise ValueError("excluded_non_graph_xref_count_mismatch")
    excluded_counter: collections.Counter[tuple[int, int]] = collections.Counter()
    excluded_detail: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for index, item in enumerate(raw_excluded):
        if not isinstance(item, Mapping) or set(item) != {
            "pp_offset",
            "function_offset",
            "reason",
            "static_category",
            "runtime_category",
        }:
            raise ValueError(f"excluded_non_graph_xref_schema_mismatch:{index}")
        if item.get("reason") != "deterministically_pruned_non_graph_instruction":
            raise ValueError(f"excluded_non_graph_xref_reason_mismatch:{index}")
        key = (int(item["pp_offset"]), int(item["function_offset"]))
        if key not in runtime_xrefs:
            raise ValueError(f"excluded_non_graph_xref_not_in_raw_receipt:{index}")
        excluded_counter[key] += 1
        excluded_detail[key].append(dict(item))

    encoded_records = canonical["binary_pool"]["uses"]
    record_offsets = [int(record["pp_offset"]) for record in encoded_records]
    if record_offsets != sorted(record_offsets) or len(record_offsets) != len(set(record_offsets)):
        raise ValueError("producer_pool_record_offsets_not_unique_sorted")
    records_by_offset = {
        int(record["pp_offset"]): record for record in encoded_records
    }

    xref_rows: list[dict[str, Any]] = []
    retained_indices: dict[int, list[int]] = collections.defaultdict(list)
    excluded_seen: collections.Counter[tuple[int, int]] = collections.Counter()
    for ordinal, (pp_offset, function_offset) in enumerate(runtime_xrefs):
        key = (pp_offset, function_offset)
        static_entry = static_entries[pp_offset]
        runtime_entry = runtime_entries[pp_offset]
        base = {
            "ordinal": ordinal,
            "pp_offset": pp_offset,
            "function_offset": function_offset,
            "static_category": static_entry.get("category"),
            "runtime_category": runtime_entry.get("category"),
        }
        if excluded_seen[key] < excluded_counter[key]:
            detail = excluded_detail[key][excluded_seen[key]]
            excluded_seen[key] += 1
            if (
                detail.get("static_category") != base["static_category"]
                or detail.get("runtime_category") != base["runtime_category"]
            ):
                raise ValueError("excluded_non_graph_xref_category_mismatch")
            xref_rows.append(
                {
                    **base,
                    "classification": "excluded_non_graph_xref",
                    "reason": detail["reason"],
                }
            )
        else:
            retained_indices[pp_offset].append(len(xref_rows))
            xref_rows.append({**base, "classification": None})
    if excluded_seen != excluded_counter:
        raise ValueError("excluded_non_graph_xref_multiset_not_consumed")

    retained_count = sum(len(indices) for indices in retained_indices.values())
    if retained_count != accounting["graph_retained_xrefs"]:
        raise ValueError("graph_retained_xref_count_mismatch")
    included_literals = 0
    retained_nonliterals = 0
    for pp_offset in sorted(static_entries):
        indices = retained_indices.get(pp_offset, [])
        expected_value = aot_builder._reconciled_value(
            static_entries[pp_offset], runtime_entries[pp_offset]
        )
        record = records_by_offset.get(pp_offset)
        if expected_value is not None and indices:
            if record is None:
                raise ValueError(f"omitted_graph_retained_literal:0x{pp_offset:x}")
            if (
                record["kind"] != expected_value["kind"]
                or record["payload"] != expected_value["payload"]
            ):
                raise ValueError(f"graph_retained_literal_payload_drift:0x{pp_offset:x}")
            sites = record["use_sites"]
            if len(sites) != len(indices):
                raise ValueError(f"graph_retained_literal_use_count_drift:0x{pp_offset:x}")
            for xref_index, site in zip(indices, sites):
                xref_rows[xref_index].update(
                    {
                        "classification": "included_graph_retained_literal",
                        "literal_kind": record["kind"],
                        "use_site": dict(site),
                    }
                )
                included_literals += 1
        elif expected_value is not None:
            if record is not None:
                raise ValueError(f"encoded_literal_has_no_graph_retained_xref:0x{pp_offset:x}")
        else:
            if record is not None:
                raise ValueError(f"encoded_nonliteral_pool_entry:0x{pp_offset:x}")
            for xref_index in indices:
                xref_rows[xref_index].update(
                    {
                        "classification": "graph_retained_nonliteral",
                        "reason": "reconciled_nonliteral_pool_entry",
                    }
                )
                retained_nonliterals += 1
    if set(records_by_offset) - set(static_entries):
        raise ValueError("encoded_pool_offset_missing_from_receipts")
    if any(item["classification"] is None for item in xref_rows):
        raise ValueError("unclassified_raw_target_xref")
    if included_literals != sum(
        len(record["use_sites"]) for record in encoded_records
    ):
        raise ValueError("encoded_literal_use_site_accounting_mismatch")
    if included_literals + retained_nonliterals != retained_count:
        raise ValueError("graph_retained_xref_classification_mismatch")

    result = {
        "schema": "dart-aot-target-pool-reconciliation-row-v1",
        "status": "included",
        "task_id": row.get("task_id"),
        "split": row.get("split"),
        "split_row": row.get("split_row"),
        "family": family_of(row),
        "target_function": codec.TARGET_FUNCTION,
        "aot_sha256": require_sha256((row.get("aot") or {}).get("sha256"), "aot_sha256"),
        "receipt_sha256": pool_metadata["receipt_sha256"],
        "projection_sha256": pool_metadata["projection_sha256"],
        "counts": {
            "raw_target_xrefs": len(runtime_xrefs),
            "included_graph_retained_literal": included_literals,
            "graph_retained_nonliteral": retained_nonliterals,
            "excluded_non_graph_xref": len(raw_excluded),
            "encoded_pool_records": len(encoded_records),
            "encoded_pool_use_sites": sum(
                len(record["use_sites"]) for record in encoded_records
            ),
        },
        "gates": {
            "static_runtime_xrefs_match": True,
            "all_raw_target_xrefs_accounted": True,
            "all_graph_retained_supported_literals_encoded": True,
            "all_encoded_uses_reference_canonical_graph_instructions": True,
        },
        "xrefs": xref_rows,
    }
    assert_alignment_source_free(result)
    return result


def family_of(row: Mapping[str, Any]) -> str:
    direct = str(row.get("family") or "").strip()
    if direct:
        return direct
    private = row.get("compact_private_metadata")
    if isinstance(private, Mapping):
        nested = str(private.get("family") or "").strip()
        if nested:
            return nested
    return "unspecified"


def source_pool_of(row: Mapping[str, Any]) -> str | None:
    """Project the audit-only Phase-0 source pool into the private sidecar.

    The strict supervised join requires this field for top-up families.  It is
    never written to the four-field model input, but dropping it here makes an
    otherwise valid sealed release impossible to join bijectively.
    """

    observed: list[str] = []
    direct = row.get("source_pool")
    nested = row.get("compact_private_metadata")
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError("compact_private_metadata_must_be_object")
    for value in (direct, (nested or {}).get("source_pool")):
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source_pool_must_be_nonempty_string_or_null")
        observed.append(value.strip())
    unique = sorted(set(observed))
    if len(unique) > 1:
        raise ValueError(f"conflicting_source_pool_metadata:{unique}")
    return unique[0] if unique else None


def _manifest_binding(path: Path, label: str) -> dict[str, Any]:
    value = read_json_object(path, label)
    return {
        # Artifact seals are path independent so the release can be reproduced
        # on Linux and verified after syncing back to Windows.
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": value.get("schema"),
    }


def _input_dataset_binding(path: Path, role: str) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    observed_splits = {str(row.get("split") or "") for _, row in rows}
    if len(observed_splits) != 1 or "" in observed_splits:
        raise ValueError(f"{role}_dataset_must_have_one_nonempty_split")
    split = next(iter(observed_splits))
    for position, (_, row) in enumerate(rows):
        split_row = row.get("split_row")
        if isinstance(split_row, bool) or split_row != position:
            raise ValueError(
                f"{role}_dataset_split_row_not_position_aligned:{position}"
            )
    return {
        "role": role,
        "name": path.name,
        "split": split,
        "rows": len(rows),
        "sha256": sha256_file(path),
    }


def _sealed_aot_dataset_bindings(
    manifest_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Read the dataset bindings carried by finalized/split binary seals.

    Merely hashing an arbitrary JSON object as an ``aot_manifest`` does not
    prove that the compact inputs came from the AOT build it names.  This gate
    extracts the sealed codec-private SHA, row count, and split so main can
    compare them exactly with ``--fit``/``--measure``.
    """

    bindings: list[dict[str, Any]] = []
    linked_pool_manifests: set[str] = set()
    linked_toolchain_manifests: set[str] = set()
    for manifest_index, path in enumerate(manifest_paths):
        value = read_json_object(path, f"aot_manifest_{manifest_index}")
        schema = value.get("schema")
        gates = value.get("gates")
        if not isinstance(gates, Mapping) or not gates or not all(
            gate is True for gate in gates.values()
        ):
            raise ValueError(f"aot_manifest_gates_not_all_true:{path.name}")
        if schema == FINAL_BINARY_BUILD_SCHEMA:
            if len(manifest_paths) != 1:
                raise ValueError("combined_binary_build_seal_must_be_the_only_manifest")
            if gates.get("all_aots_present_and_hash_valid") is not True:
                raise ValueError("combined_binary_build_seal_lacks_aot_hash_gate")
            splits = value.get("splits")
            if not isinstance(splits, Mapping) or not splits:
                raise ValueError("combined_binary_build_seal_has_no_splits")
            for split, detail in sorted(splits.items()):
                if not isinstance(detail, Mapping):
                    raise ValueError(f"combined_split_not_object:{split}")
                codec_private = detail.get("codec_private")
                rows = detail.get("rows")
                if not isinstance(codec_private, Mapping):
                    raise ValueError(f"combined_split_has_no_codec_private:{split}")
                if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
                    raise ValueError(f"combined_split_rows_invalid:{split}")
                bindings.append(
                    {
                        "manifest": path.name,
                        "split": str(split),
                        "rows": rows,
                        "sha256": require_sha256(
                            codec_private.get("sha256"),
                            f"combined_{split}_codec_private_sha256",
                        ),
                    }
                )
            artifacts = value.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError("combined_binary_build_seal_has_no_artifacts")
            for field, target in (
                ("pool_extractor_manifest", linked_pool_manifests),
                ("dart_toolchain_manifest", linked_toolchain_manifests),
            ):
                artifact = artifacts.get(field)
                if not isinstance(artifact, Mapping):
                    raise ValueError(f"combined_binary_build_seal_missing_{field}")
                target.add(
                    require_sha256(
                        artifact.get("sha256"), f"combined_{field}_sha256"
                    )
                )
        else:
            raise ValueError(
                "aot_manifest_must_be_finalized_binary_build_seal:"
                f"{schema!r}"
            )
    if len({item["split"] for item in bindings}) != len(bindings):
        raise ValueError("duplicate_split_across_aot_manifests")
    return bindings, linked_pool_manifests, linked_toolchain_manifests


def _require_exact_dataset_bindings(
    input_bindings: Sequence[Mapping[str, Any]],
    sealed_bindings: Sequence[Mapping[str, Any]],
) -> None:
    expected = collections.Counter(
        (item["split"], item["rows"], item["sha256"])
        for item in input_bindings
    )
    observed = collections.Counter(
        (item["split"], item["rows"], item["sha256"])
        for item in sealed_bindings
    )
    if observed != expected:
        raise ValueError(
            "aot_manifest_dataset_binding_mismatch:"
            f"observed={sorted(observed.elements())}:"
            f"expected={sorted(expected.elements())}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--measure", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--combined-pool-extractor-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--aot-manifest", required=True, action="append", type=Path,
        help=(
            "Exactly one finalized binary_build_manifest.json whose finalizer "
            "rehash-verified every AOT."
        ),
    )
    parser.add_argument("--dart-toolchain-manifest", required=True, type=Path)
    parser.add_argument(
        "--legacy-cfg-extractor",
        type=Path,
        default=ROOT / "scrubbed_master_v2_release/extractors/cfg_extractor.py",
    )
    parser.add_argument(
        "--legacy-dfg-extractor",
        type=Path,
        default=ROOT / "scrubbed_master_v2_release/extractors/dfg_extractor.py",
    )
    parser.add_argument(
        "--current-cfg-extractor",
        type=Path,
        default=ROOT / "scripts/data/cfg_extractor.py",
    )
    parser.add_argument(
        "--current-dfg-extractor",
        type=Path,
        default=ROOT / "scripts/data/dfg_extractor.py",
    )
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--max-source-tokens", type=int, default=9000)
    parser.add_argument("--max-target-tokens", type=int, default=3072)
    parser.add_argument("--max-total-tokens", type=int, default=12288)
    parser.add_argument("--tokenizer-fingerprint-sha256", required=True)
    parser.add_argument("--decoder-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--decoder-revision", required=True)
    parser.add_argument("--target-function", default=codec.TARGET_FUNCTION)
    parser.add_argument(
        "--symbol-policy",
        choices=("runtime_aware", "strict_all_alias"),
        default="runtime_aware",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.target_function != codec.TARGET_FUNCTION:
        raise ValueError("compact-Qwen v3 requires target_function=candidate")
    if args.symbol_policy != "runtime_aware":
        raise ValueError("compact-Qwen v3 requires runtime-symbol-policy-v1")
    if not 0 < args.codebook_size:
        raise ValueError("codebook_size_must_be_positive")
    if not 0 < args.max_blocks:
        raise ValueError("max_blocks_must_be_positive")
    if not 0 < args.max_source_tokens <= 9000:
        raise ValueError("max_source_tokens_must_be_in_1_to_9000")
    if args.max_target_tokens <= 0 or args.max_total_tokens < args.max_source_tokens:
        raise ValueError("invalid_target_or_total_token_limit")
    tokenizer_fingerprint = require_sha256(
        args.tokenizer_fingerprint_sha256, "tokenizer_fingerprint_sha256"
    )
    resolved_measure = [path.resolve() for path in args.measure]
    if len(resolved_measure) != len(set(resolved_measure)):
        raise ValueError("duplicate_measure_input")
    if args.fit.resolve() in set(resolved_measure):
        raise ValueError("fit_input_may_not_also_be_measure_input")
    input_dataset_bindings = [
        _input_dataset_binding(args.fit, "fit"),
        *(
            _input_dataset_binding(path, f"measure_{index}")
            for index, path in enumerate(args.measure)
        ),
    ]
    if input_dataset_bindings[0]["split"] != "train":
        raise ValueError("fit_dataset_must_bind_the_train_split")
    if any(
        item["split"] == "train" for item in input_dataset_bindings[1:]
    ):
        raise ValueError("measure_dataset_may_not_bind_the_train_split")

    expected_files = {
        "codebook.json",
        "compact_contract.json",
        "compact_model_inputs.jsonl",
        "alignment_private.jsonl",
        "pool_reconciliation_private.jsonl",
        "quarantine.jsonl",
        "failures.jsonl",
        "preflight_report.json",
        "SHA256SUMS.txt",
    }
    if len(args.aot_manifest) != 1:
        raise ValueError("exactly_one_finalized_binary_build_manifest_is_required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        item.name for item in args.output_dir.iterdir() if item.name not in expected_files
    )
    if unexpected:
        raise ValueError("unexpected_output_dir_entries:" + ",".join(unexpected))

    resolved_aot_manifests = [path.resolve() for path in args.aot_manifest]
    if len(resolved_aot_manifests) != len(set(resolved_aot_manifests)):
        raise ValueError("duplicate_aot_manifest_argument")
    aot_bindings = [
        _manifest_binding(path, f"aot_manifest_{index}")
        for index, path in enumerate(args.aot_manifest)
    ]
    provenance = {
        "combined_pool_extractor_manifest": _manifest_binding(
            args.combined_pool_extractor_manifest,
            "combined_pool_extractor_manifest",
        ),
        "aot_manifests": aot_bindings,
        "dart_toolchain_manifest": _manifest_binding(
            args.dart_toolchain_manifest, "dart_toolchain_manifest"
        ),
    }
    pool_extractor_sha = provenance["combined_pool_extractor_manifest"]["sha256"]
    (
        sealed_dataset_bindings,
        linked_pool_manifests,
        linked_toolchain_manifests,
    ) = _sealed_aot_dataset_bindings(args.aot_manifest)
    _require_exact_dataset_bindings(
        input_dataset_bindings, sealed_dataset_bindings
    )
    dart_toolchain_sha = provenance["dart_toolchain_manifest"]["sha256"]
    if linked_pool_manifests and linked_pool_manifests != {pool_extractor_sha}:
        raise ValueError("binary_build_seal_pool_extractor_manifest_mismatch")
    if linked_toolchain_manifests != {dart_toolchain_sha}:
        raise ValueError("binary_build_seal_dart_toolchain_manifest_mismatch")
    provenance["input_dataset_bindings"] = input_dataset_bindings
    provenance["sealed_dataset_bindings"] = sealed_dataset_bindings
    aot_manifest_sha = aot_bindings[0]["sha256"]
    provenance["aot_manifest_contract_binding"] = {
        "mode": "finalized_binary_build_seal",
        "sha256": aot_manifest_sha,
    }

    registry = graph_v2.load_route_registry(
        args.legacy_cfg_extractor,
        args.legacy_dfg_extractor,
        args.current_cfg_extractor,
        args.current_dfg_extractor,
    )
    extractor_routes = graph_v2.route_contract(registry)

    quarantine: list[dict[str, Any]] = []
    total_input_rows = 0
    fit_good: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    frequencies: collections.Counter[str] = collections.Counter()
    for line_number, row in read_jsonl(args.fit):
        total_input_rows += 1
        try:
            canonical = codec.canonicalize(row, args.symbol_policy)
            metadata = pool_alignment_metadata(row)
            fit_good.append((line_number, row, canonical, metadata))
            frequencies.update(
                instruction
                for block in canonical["blocks"]
                for instruction in block["instructions"]
            )
        except Exception as error:
            quarantine.append(
                {
                    "dataset": str(args.fit),
                    "line": line_number,
                    "task_id": row.get("task_id"),
                    "reason": f"{type(error).__name__}:{error}",
                }
            )
    expansions = [value for value, _ in frequencies.most_common(args.codebook_size)]
    instruction_code = {value: index for index, value in enumerate(expansions)}

    model_config = args.model_config or args.tokenizer_json.with_name("config.json")
    model_cfg = read_json_object(model_config, "model_config")
    model_vocab_size = int(model_cfg["vocab_size"])
    base_tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tokenizer_vocab_size, source_expansions, atom_ids = codec.source_token_contract(
        args.tokenizer_json, model_vocab_size, expansions, args.max_blocks
    )
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            f"base_tokenizer_vocab_exceeds_model_vocab:{tokenizer_vocab_size}>{model_vocab_size}"
        )
    source_ids = sorted(map(int, source_expansions))
    if not source_ids or source_ids != list(
        range(model_vocab_size, model_vocab_size + len(source_ids))
    ):
        raise ValueError("source_token_ids_not_contiguous_after_base_vocab")

    datasets: list[
        tuple[str, Path, list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]]]
    ] = [("fit", args.fit, fit_good)]
    for path in args.measure:
        good: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for line_number, row in read_jsonl(path):
            total_input_rows += 1
            try:
                good.append(
                    (
                        line_number,
                        row,
                        codec.canonicalize(row, args.symbol_policy),
                        pool_alignment_metadata(row),
                    )
                )
            except Exception as error:
                quarantine.append(
                    {
                        "dataset": str(path),
                        "line": line_number,
                        "task_id": row.get("task_id"),
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
        datasets.append(("measure", path, good))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lengths: list[int] = []
    rows_by_role: collections.Counter[str] = collections.Counter()
    rows_by_family: collections.Counter[str] = collections.Counter()
    rows_by_route: collections.Counter[str] = collections.Counter()
    cfg_types: collections.Counter[str] = collections.Counter()
    dfg_by_route: collections.Counter[str] = collections.Counter()
    fallback_by_role: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    fallback_by_family: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    fallback_by_role_family: dict[
        tuple[str, str], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    fallback_by_dataset: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    pool_top_kinds: collections.Counter[str] = collections.Counter()
    pool_all_kinds: collections.Counter[str] = collections.Counter()
    pool_nested_nonliteral_pairs: collections.Counter[str] = collections.Counter()
    pool_record_count = 0
    pool_use_site_count = 0
    pool_rows_with_uses = 0
    compact_id_roundtrip_rows = 0

    for role, path, good in datasets:
        for line_number, row, canonical, pool_metadata in good:
            task_id = str(row.get("task_id") or "")
            if not task_id:
                failures.append({"dataset": str(path), "line": line_number, "reason": "missing_task_id"})
                continue
            if len(canonical["blocks"]) > args.max_blocks:
                failures.append(
                    {"task_id": task_id, "reason": "block_vocab_overflow", "blocks": len(canonical["blocks"])}
                )
                continue
            text = codec.encode(canonical, instruction_code)
            decoded = codec.regenerate_dfg(codec.decode(text, expansions), registry)
            if decoded != canonical:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": "canonical_graph_or_pool_roundtrip_mismatch",
                        "expected_sha256": canonical_sha256(canonical),
                        "observed_sha256": canonical_sha256(decoded),
                    }
                )
                continue
            ids = codec.compact_ids(text, base_tokenizer, atom_ids)
            recovered_text = decode_compact_ids(ids, base_tokenizer, atom_ids)
            if recovered_text != text:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": "compact_id_stream_not_exactly_reversible",
                        "expected_sha256": sha256_bytes(text.encode("utf-8")),
                        "observed_sha256": sha256_bytes(
                            recovered_text.encode("utf-8")
                        ),
                    }
                )
            else:
                compact_id_roundtrip_rows += 1
            source_tokens = len(ids)
            lengths.append(source_tokens)
            if source_tokens > args.max_source_tokens:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": "source_token_overflow",
                        "tokens": source_tokens,
                        "limit": args.max_source_tokens,
                    }
                )

            instructions = [
                instruction
                for block in canonical["blocks"]
                for instruction in block["instructions"]
            ]
            instruction_count = len(instructions)
            fallback = sum(instruction not in instruction_code for instruction in instructions)
            family = family_of(row)
            source_pool = source_pool_of(row)
            if family != "master" and source_pool is None:
                raise ValueError(
                    f"topup_row_missing_source_pool:{task_id}:{family}"
                )
            dataset_key = str(path)
            for key, counter in (
                (role, fallback_by_role),
                (family, fallback_by_family),
                (dataset_key, fallback_by_dataset),
            ):
                counter[key]["rows"] += 1
                counter[key]["instructions"] += instruction_count
                counter[key]["fallback"] += fallback
            role_family = fallback_by_role_family[(role, family)]
            role_family["rows"] += 1
            role_family["instructions"] += instruction_count
            role_family["fallback"] += fallback

            stats = pool_statistics(canonical)
            if pool_metadata["use_count"] != stats["records"]:
                raise AssertionError("pool_metadata_use_count_drift")
            pool_record_count += stats["records"]
            pool_use_site_count += stats["use_sites"]
            pool_rows_with_uses += int(stats["records"] > 0)
            pool_top_kinds.update(stats["top_level_kinds"])
            pool_all_kinds.update(stats["all_node_kinds"])
            pool_nested_nonliteral_pairs.update(
                stats["nested_nonliteral_descriptor_pairs"]
            )
            route = str(canonical["dfg_route"])
            rows_by_role[role] += 1
            rows_by_family[family] += 1
            rows_by_route[route] += 1
            cfg_types.update(edge["edge_type"] for edge in canonical["cfg_edges"])
            dfg_by_route[route] += len(canonical["dfg_edges"])
            receipt = row["binary_pool_private_receipt"]
            reconciliation = reconcile_raw_pool_xrefs(row, canonical, pool_metadata)
            records.append(
                {
                    "role": role,
                    "dataset": str(path),
                    "line": line_number,
                    "task_id": task_id,
                    "split": row.get("split"),
                    "split_row": row.get("split_row"),
                    "family": family,
                    "source_pool": source_pool,
                    "compact_input_ids": ids,
                    "source_tokens": source_tokens,
                    "canonical_sha256": canonical_sha256(canonical),
                    "compact_sha256": sha256_bytes(text.encode("utf-8")),
                    "fallback_instructions": fallback,
                    "instruction_count": instruction_count,
                    "block_count": len(canonical["blocks"]),
                    "cfg_edge_count": len(canonical["cfg_edges"]),
                    "dfg_edge_count": len(canonical["dfg_edges"]),
                    "dfg_route": route,
                    "graph_extractor_sha256": codec.ROUTE_SPECS[route].combined_sha256,
                    "aot_sha256": (row.get("aot") or {}).get("sha256"),
                    "pool_metadata": pool_metadata,
                    "_pool_reconciliation": reconciliation,
                    # This internal assertion value is dropped before writing.
                    "_receipt_projection_sha256": receipt["projection_sha256"],
                }
            )

    reconciliation_rows: list[dict[str, Any]] = []
    for model_row, record in enumerate(records):
        reconciliation = dict(record["_pool_reconciliation"])
        reconciliation.update(
            {
                "model_row": model_row,
                "role": record["role"],
                "dataset": record["dataset"],
                "line": record["line"],
            }
        )
        reconciliation["row_sha256"] = canonical_sha256(
            {key: value for key, value in reconciliation.items() if key != "row_sha256"}
        )
        reconciliation_rows.append(reconciliation)
    for item in quarantine:
        reconciliation_rows.append(
            {
                "schema": "dart-aot-target-pool-reconciliation-row-v1",
                "status": "quarantined",
                "model_row": None,
                "role": "fit" if item["dataset"] == str(args.fit) else "measure",
                "dataset": item["dataset"],
                "line": item["line"],
                "task_id": item.get("task_id"),
                "reason": item["reason"],
            }
        )
    reconciliation_rows.sort(
        key=lambda value: (
            0 if value["dataset"] == str(args.fit) else 1,
            [str(path) for path in args.measure].index(value["dataset"])
            if value["dataset"] != str(args.fit)
            else -1,
            int(value["line"]),
        )
    )
    if len(reconciliation_rows) != total_input_rows:
        raise AssertionError(
            f"pool_reconciliation_input_coverage_drift:{len(reconciliation_rows)}!={total_input_rows}"
        )
    reconciliation_path = args.output_dir / "pool_reconciliation_private.jsonl"
    write_jsonl_atomic(reconciliation_path, reconciliation_rows)
    pool_reconciliation_sha = sha256_file(reconciliation_path)

    codec_sha = sha256_file(Path(codec.__file__))
    graph_codec_sha = codec.graph_codec_sha256()
    tokenizer_sha = sha256_file(args.tokenizer_json)
    model_config_sha = sha256_file(model_config)
    runtime_policy_sha = canonical_sha256(codec.RUNTIME_POLICY)
    pool_contract = codec.codec_contract(
        codec_sha256=codec_sha,
        codebook_sha256="0" * 64,
        tokenizer_json_sha256=tokenizer_sha,
        pool_extractor_sha256=pool_extractor_sha,
        aot_manifest_sha256=aot_manifest_sha,
        dart_toolchain_manifest_sha256=dart_toolchain_sha,
        pool_reconciliation_manifest_sha256=pool_reconciliation_sha,
    )
    # codebook_sha256 is self-referential in the helper template; the real
    # value is added only to the final contract after codebook serialization.
    pool_contract.pop("codebook_sha256")
    pool_contract.pop("codec_sha256")
    pool_contract.pop("tokenizer_json_sha256")
    pool_contract.pop("schema")

    codebook = {
        "schema": codec.CODEBOOK_SCHEMA,
        "fit_public_sha256": sha256_file(args.fit),
        "fit_retained": len(fit_good),
        "fit_quarantined": sum(item["dataset"] == str(args.fit) for item in quarantine),
        "fit_scope": "train_only",
        "measure_excluded_from_fit": True,
        "codebook_size": len(expansions),
        "expansions": expansions,
        "added_token_scheme": {
            "instruction": "<I{index}>",
            "block": "<B{index}>",
            "control_tokens": codec.CONTROL,
            "edge_tokens": codec.EDGE_TOKEN,
            "extractor_route_tokens": {
                route: spec.atom for route, spec in codec.ROUTE_SPECS.items()
            },
            "pool_payload": "base_tokenizer_canonical_positional_json_delta_v2",
        },
        "tokenizer_json_sha256": tokenizer_sha,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_config_sha256": model_config_sha,
        "decoder_model": args.decoder_model.strip(),
        "decoder_revision": args.decoder_revision.strip(),
        "model_vocab_size": model_vocab_size,
        "base_vocab_size": model_vocab_size,
        "source_token_expansions": source_expansions,
        "source_atom_ids": atom_ids,
        "extractor_routes": extractor_routes,
        "max_blocks": args.max_blocks,
        "symbol_policy": args.symbol_policy,
        "runtime_symbol_policy": (
            codec.RUNTIME_POLICY if args.symbol_policy == "runtime_aware" else None
        ),
        "runtime_symbol_policy_sha256": (
            runtime_policy_sha if args.symbol_policy == "runtime_aware" else None
        ),
        **pool_contract,
        "provenance_manifests": provenance,
    }
    codebook_bytes = (
        json.dumps(codebook, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    write_bytes_atomic(args.output_dir / "codebook.json", codebook_bytes)
    codebook_sha = sha256_bytes(codebook_bytes)

    stream_marker_ids = {
        marker: atom_ids[marker]
        for marker in (codec.STREAM_START, "<CFG>", codec.POOL_START, codec.POOL_END, codec.STREAM_END)
    }
    contract = {
        "schema": codec.CONTRACT_SCHEMA,
        "codec_sha256": codec_sha,
        "codebook_sha256": codebook_sha,
        "tokenizer_json_sha256": tokenizer_sha,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint,
        "model_config_sha256": model_config_sha,
        "decoder_model": args.decoder_model.strip(),
        "decoder_revision": args.decoder_revision.strip(),
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
        "max_total_tokens": args.max_total_tokens,
        "target_function": codec.TARGET_FUNCTION,
        "target_language": "Dart",
        "extractor_routes": extractor_routes,
        "runtime_symbol_policy_sha256": (
            runtime_policy_sha if args.symbol_policy == "runtime_aware" else None
        ),
        "base_vocab_size": model_vocab_size,
        "source_token_ids": source_ids,
        "source_token_expansions": source_expansions,
        "source_embedding_init": "codebook_mean",
        "stream_marker_ids": stream_marker_ids,
        **pool_contract,
        "provenance_manifests": provenance,
        "release_builder_sha256": sha256_file(Path(__file__)),
    }
    # Make the trainer's own strict contract class the executable schema gate
    # when the hybrid patch is available in this checkout.
    from hybrid_training_patch_v2_3.models.direct_compact_causal import DirectCompactContract

    DirectCompactContract.from_mapping(contract)
    write_json_atomic(args.output_dir / "compact_contract.json", contract)

    model_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    reconciliation_by_model_row = {
        int(row["model_row"]): row
        for row in reconciliation_rows
        if row.get("status") == "included"
    }
    for model_row, record in enumerate(records):
        public = {
            "compact_input_ids": record["compact_input_ids"],
            "compact_codec_sha256": codec_sha,
            "compact_codebook_sha256": codebook_sha,
            "compact_tokenizer_sha256": tokenizer_sha,
        }
        if set(public) != PUBLIC_FIELDS:
            raise AssertionError("strict_four_field_model_schema_drift")
        alignment = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "compact_input_ids",
                "_receipt_projection_sha256",
                "_pool_reconciliation",
            }
        }
        alignment["model_row"] = model_row
        alignment["model_row_sha256"] = canonical_sha256(public)
        alignment["pool_reconciliation_manifest_sha256"] = pool_reconciliation_sha
        alignment["pool_reconciliation_row_sha256"] = reconciliation_by_model_row[
            model_row
        ]["row_sha256"]
        if alignment["pool_metadata"]["projection_sha256"] != record["_receipt_projection_sha256"]:
            raise AssertionError("alignment_pool_projection_binding_drift")
        assert_alignment_source_free(alignment)
        model_rows.append(public)
        alignment_rows.append(alignment)
    if len(model_rows) != len(alignment_rows) or any(
        row["model_row"] != index for index, row in enumerate(alignment_rows)
    ):
        raise AssertionError("public_private_alignment_not_bijective")
    write_jsonl_atomic(args.output_dir / "compact_model_inputs.jsonl", model_rows)
    write_jsonl_atomic(args.output_dir / "alignment_private.jsonl", alignment_rows)
    write_jsonl_atomic(args.output_dir / "quarantine.jsonl", quarantine)
    write_jsonl_atomic(args.output_dir / "failures.jsonl", failures)

    leakage = {
        "candidate": sum("candidate" in value.lower() for value in expansions),
        "file_uri": sum("file://" in value.lower() for value in expansions),
        "absolute_symbol_address": sum(
            bool(re.search(r"0x[0-9a-fA-F]+\s*<", value)) for value in expansions
        ),
        "private_field_terms": sum(
            bool(re.search(r"dart_source|semantic_function|original_source|\btests\b", value, re.I))
            for value in expansions
        ),
    }
    report = {
        "schema": codec.PREFLIGHT_SCHEMA,
        "rows_retained": len(records),
        "rows_by_role": dict(rows_by_role),
        "rows_by_family": dict(rows_by_family),
        "rows_by_route": dict(rows_by_route),
        "quarantined": len(quarantine),
        "quarantine_reasons": dict(collections.Counter(item["reason"] for item in quarantine)),
        "failures_count": len(failures),
        "failure_examples": failures[:50],
        "tokens": {
            "kind": "compact_v3_graph_plus_binary_pool_source_only",
            "min": min(lengths) if lengths else 0,
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            "limit": args.max_source_tokens,
            "rows_over_limit": sum(value > args.max_source_tokens for value in lengths),
        },
        "fallback_by_role": {
            key: _with_rate(value) for key, value in fallback_by_role.items()
        },
        "fallback_by_family": {
            key: _with_rate(value) for key, value in fallback_by_family.items()
        },
        "fallback_by_role_and_family": {
            role: {
                family: _with_rate(fallback_by_role_family[(role, family)])
                for observed_role, family in sorted(fallback_by_role_family)
                if observed_role == role
            }
            for role in sorted({item[0] for item in fallback_by_role_family})
        },
        "fallback_by_dataset": {
            key: _with_rate(value) for key, value in fallback_by_dataset.items()
        },
        "cfg_edge_types": dict(cfg_types),
        "pool": {
            "schema": codec.POOL_SCHEMA,
            "rows_validated": len(records),
            "rows_with_uses": pool_rows_with_uses,
            "records": pool_record_count,
            "use_sites": pool_use_site_count,
            "top_level_kind_counts": dict(pool_top_kinds),
            "all_node_kind_counts": dict(pool_all_kinds),
            "nested_nonliteral_descriptors": {
                "count": sum(pool_nested_nonliteral_pairs.values()),
                "pair_counts": dict(pool_nested_nonliteral_pairs),
                "profile_type_to_nonliteral_kind_allowlist": dict(
                    sorted(codec.NESTED_NONLITERAL_PROFILE_KIND.items())
                ),
            },
        },
        "lossless_invariants": {
            "lossless_domain": pool_contract["lossless_domain"],
            "privacy_scrub_is_only_intentional_irreversibility": True,
            "exact_graph_and_pool_roundtrip_rows": len(records),
            "compact_id_stream_roundtrip_rows": compact_id_roundtrip_rows,
            "dfg_regenerated_and_matched_rows": len(records),
            "dfg_edges_matched_edge_for_edge": sum(dfg_by_route.values()),
            "dfg_edges_by_route": dict(dfg_by_route),
            "pool_records_roundtripped": pool_record_count,
            "nested_nonliteral_descriptors_preserved": sum(
                pool_nested_nonliteral_pairs.values()
            ),
            "pool_use_sites_validated": pool_use_site_count,
            "pool_projection_hashes_matched": len(records),
            "pool_receipts_hash_bound": len(records),
            "raw_target_xrefs_reconciled": sum(
                row.get("counts", {}).get("raw_target_xrefs", 0)
                for row in reconciliation_rows
                if row.get("status") == "included"
            ),
            "pool_reconciliation_rows_cover_all_inputs": len(reconciliation_rows),
            "pool_reconciliation_manifest_sha256": pool_reconciliation_sha,
            "source_blind_pool_rows": len(records),
            "unknown_tokens": 0,
            "truncated_rows": 0,
            "raw_instruction_fallback_is_reversible": True,
            "call_edges_encoded_explicitly": True,
            "alignment_rows_without_payload_or_source": len(alignment_rows),
            "public_private_rows_bijective_and_hash_bound": len(model_rows),
        },
        "codebook_fit": {
            "role": "fit_only",
            "input_sha256": sha256_file(args.fit),
            "retained_rows": len(fit_good),
            "measure_rows_in_fit": 0,
        },
        "codebook_expansion_leakage_scan": leakage,
        "provenance_manifests": provenance,
        "contract": {
            "codec_sha256": codec_sha,
            "graph_codec_dependency_sha256": graph_codec_sha,
            "codebook_sha256": codebook_sha,
            "tokenizer_json_sha256": tokenizer_sha,
            "pool_extractor_sha256": pool_extractor_sha,
            "aot_manifest_sha256": aot_manifest_sha,
            "dart_toolchain_manifest_sha256": dart_toolchain_sha,
            "pool_reconciliation_manifest_sha256": pool_reconciliation_sha,
            "target_function": codec.TARGET_FUNCTION,
            "stream_marker_ids": stream_marker_ids,
        },
        "passed": (
            bool(records)
            and not quarantine
            and not failures
            and not any(leakage.values())
            and len(reconciliation_rows) == total_input_rows
        ),
    }
    write_json_atomic(args.output_dir / "preflight_report.json", report)

    sealed_names = sorted(expected_files - {"SHA256SUMS.txt"})
    checksum_text = "".join(
        f"{sha256_file(args.output_dir / name)}  {name}\n" for name in sealed_names
    )
    write_bytes_atomic(args.output_dir / "SHA256SUMS.txt", checksum_text.encode("ascii"))
    observed_files = {item.name for item in args.output_dir.iterdir() if item.is_file()}
    if observed_files != expected_files or any(item.is_dir() for item in args.output_dir.iterdir()):
        raise ValueError(
            f"sealed_output_file_set_mismatch:{sorted(observed_files)}!={sorted(expected_files)}"
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
