#!/usr/bin/env python3
"""Fail-closed umbrella seal for the Phase-0 compact-Qwen v2 release.

The finalizer is intentionally stricter than the individual builders.  It
reconciles the canonical input and Phase-0 manifest, validates the strict
public/private joins and aggregate audits, checks every nested checksum, then
writes one deterministic release manifest and one recursive checksum file.

It refuses to emit either umbrella artifact while any required producer output
is missing or while an unexpected file is present in a sealed subdirectory.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = (
    WORKSPACE / "scrubbed_master_v2_release/direct_compact_phase0_s44_v2"
)

PUBLIC_FIELDS = {
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
}
SUPERVISED_FIELDS = {
    "lang",
    "function",
    "dart_source",
    *PUBLIC_FIELDS,
}
LABEL_FIELDS = {"task_id", "lang", "function", "dart_source", "family"}
FAMILIES = {"master", "topup_s45", "topup_s46"}
ROUTES = {"legacy_release_v1", "current_combined_v2"}
EXPECTED_STATUS_COUNTS = {
    "included-train": 2951,
    "included-dev": 326,
    "quarantined": 14,
    "excluded": 15,
}
EXPECTED_ROLE_COUNTS = {"fit": 2951, "measure": 326}
EXPECTED_ROUTE_COUNTS = {
    "legacy_release_v1": 2172,
    "current_combined_v2": 1105,
}
EXPECTED_ROLE_FAMILY_COUNTS = {
    "fit": {"master": 1953, "topup_s45": 894, "topup_s46": 104},
    "measure": {"master": 219, "topup_s45": 100, "topup_s46": 7},
}
EXPECTED_FILES = {
    "prepared": {
        "train_codec_private.jsonl",
        "dev_codec_private.jsonl",
        "train_private_labels.jsonl",
        "dev_private_labels.jsonl",
        "reconciliation.jsonl",
        "quarantine.jsonl",
        "forbidden_overlap_audit.jsonl",
        "preparation_manifest.json",
        "SHA256SUMS.txt",
    },
    "compact_qwen_phase0_s44_v2": {
        "compact_model_inputs.jsonl",
        "alignment_private.jsonl",
        "codebook.json",
        "compact_contract.json",
        "preflight_report.json",
        "quarantine.jsonl",
        "failures.jsonl",
        "SHA256SUMS.txt",
    },
    "supervised": {
        "train.jsonl",
        "train.join_seal.json",
        "dev.jsonl",
        "dev.join_seal.json",
    },
    "audits": {
        "dev_fallback_token_audit.json",
        "scrubbed_humaneval_fallback_token_audit.json",
        "topup_family_fallback_audit.json",
        "generalization_audit_seal.json",
        "SHA256SUMS.txt",
    },
}
EXPECTED_ROOT_FILES = {"README.md", "release_manifest.json", "SHA256SUMS.txt"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseValidationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    require(bool(SHA256_RE.fullmatch(result)), f"{label}: invalid SHA-256")
    return result


def stable_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing required JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing required JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(
                isinstance(value, dict),
                f"{path}:{line_number}: expected JSON object",
            )
            rows.append(value)
    require(allow_empty or bool(rows), f"{path}: unexpectedly empty")
    return rows


def resolve_bound_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    require(bool(str(path)), "empty bound path")
    return path if path.is_absolute() else WORKSPACE / path


def verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = resolve_bound_path(binding.get("path"))
    require(path.is_file(), f"{label}: bound file is missing: {path}")
    expected = require_digest(binding.get("sha256"), f"{label}.sha256")
    observed = sha256_file(path)
    require(observed == expected, f"{label}: SHA drift: {observed} != {expected}")
    if "size_bytes" in binding:
        require(
            path.stat().st_size == int(binding["size_bytes"]),
            f"{label}: size drift",
        )
    if "rows" in binding:
        rows = sum(1 for line in path.open("rb") if line.strip())
        require(rows == int(binding["rows"]), f"{label}: row-count drift")
    return path


def verify_checksum_file(path: Path, expected_names: set[str]) -> dict[str, str]:
    require(path.is_file(), f"missing nested checksum: {path}")
    found: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split("  ", 1)
        require(len(parts) == 2, f"{path}:{line_number}: malformed checksum line")
        digest, name = parts
        require_digest(digest, f"{path}:{line_number}")
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"{path}:{line_number}: unsafe checksum path",
        )
        require(name not in found, f"{path}: duplicate checksum path {name!r}")
        target = path.parent / relative
        require(target.is_file(), f"{path}: checksum target missing: {name}")
        observed = sha256_file(target)
        require(observed == digest, f"{path}: checksum mismatch for {name}")
        found[name] = digest
    require(
        set(found) == expected_names - {path.name},
        f"{path}: coverage drift; found={sorted(found)}, "
        f"expected={sorted(expected_names - {path.name})}",
    )
    return found


def require_exact_directory_contents(release: Path) -> None:
    missing_directories = [name for name in EXPECTED_FILES if not (release / name).is_dir()]
    require(
        not missing_directories,
        "producer outputs are not complete; missing directories: "
        + ", ".join(missing_directories),
    )
    for directory, expected in EXPECTED_FILES.items():
        observed = {
            item.name for item in (release / directory).iterdir() if item.is_file()
        }
        missing = expected - observed
        unexpected = observed - expected
        require(
            not missing and not unexpected,
            f"{directory}: file-set mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}",
        )
        require(
            not any(item.is_dir() for item in (release / directory).iterdir()),
            f"{directory}: unexpected nested directory",
        )
    observed_dirs = {item.name for item in release.iterdir() if item.is_dir()}
    require(
        observed_dirs == set(EXPECTED_FILES),
        f"release directory-set mismatch: {sorted(observed_dirs)}",
    )
    observed_root_files = {item.name for item in release.iterdir() if item.is_file()}
    unexpected_root = observed_root_files - EXPECTED_ROOT_FILES
    missing_stable_root = {"README.md"} - observed_root_files
    require(
        not unexpected_root and not missing_stable_root,
        f"release root file-set mismatch; missing={sorted(missing_stable_root)}, "
        f"unexpected={sorted(unexpected_root)}",
    )


def verify_preparation(release: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = release / "prepared"
    verify_checksum_file(prepared / "SHA256SUMS.txt", EXPECTED_FILES["prepared"])
    manifest = read_json(prepared / "preparation_manifest.json")
    require(manifest.get("schema") == "phase0-compact-qwen-v2-preparation", "bad preparation schema")
    require(manifest.get("gates", {}).get("passed") is True, "preparation did not pass")
    require(
        all(value is True for value in manifest["gates"].values()),
        "one or more preparation gates are false",
    )
    require(manifest.get("target_function") == "candidate", "target is not candidate")
    require(
        manifest.get("family_policy", {}).get("version")
        == "pre-s46-phase-umbrella-v1",
        "unexpected family policy",
    )
    counts = manifest.get("counts") or {}
    require(counts.get("input") == 3306, "canonical input count drift")
    require(counts.get("phase0_manifest") == 3305, "Phase-0 count drift")
    require(counts.get("statuses") == EXPECTED_STATUS_COUNTS, "status-count drift")
    require(counts.get("train_codec") == 2951, "train codec count drift")
    require(counts.get("dev_codec") == 326, "dev codec count drift")
    require(counts.get("long_dev_included") == 138, "retained long-dev count drift")
    require(counts.get("long_dev_quarantined") == 2, "quarantined long-dev count drift")

    # Revalidate all preparation bindings, not just their manifest text.
    inputs = manifest.get("inputs") or {}
    canonical_path = verify_binding(inputs["canonical_corpus"], "canonical corpus")
    phase0_path = verify_binding(inputs["phase0_split_manifest"], "Phase-0 manifest")
    verify_binding(inputs["source_pool_manifest"], "source-pool manifest")
    for index, binding in enumerate(inputs.get("source_pools") or []):
        verify_binding(binding, f"source pool {index}")
    for index, binding in enumerate(inputs.get("forbidden") or []):
        verify_binding(binding, f"forbidden family {index}")
    for name, binding in (manifest.get("scripts") or {}).items():
        verify_binding(binding, f"preparation script {name}")
    for name, binding in (manifest.get("outputs") or {}).items():
        verify_binding(binding, f"preparation output {name}")

    reconciliation = read_jsonl(prepared / "reconciliation.jsonl")
    require(len(reconciliation) == 3306, "reconciliation must contain 3,306 rows")
    statuses = collections.Counter(str(row.get("status")) for row in reconciliation)
    require(dict(statuses) == EXPECTED_STATUS_COUNTS, "reconciliation status drift")
    task_ids = [str(row.get("task_id") or "") for row in reconciliation]
    require(all(task_ids), "reconciliation contains an empty task_id")
    require(len(task_ids) == len(set(task_ids)), "reconciliation task IDs are not unique")
    require(
        [row.get("input_line") for row in reconciliation] == list(range(1, 3307)),
        "reconciliation is not in canonical input order",
    )

    # Bind each reconciliation line to the literal canonical JSONL line.
    with canonical_path.open("rb") as handle:
        canonical_lines = [raw.rstrip(b"\r\n") for raw in handle if raw.strip()]
    require(len(canonical_lines) == 3306, "canonical corpus line count changed")
    for index, (raw, reconciled) in enumerate(zip(canonical_lines, reconciliation, strict=True), 1):
        source = json.loads(raw.decode("utf-8"))
        require(isinstance(source, dict), f"canonical row {index} is not an object")
        require(str(source.get("task_id") or "") == reconciled["task_id"], f"canonical/reconciliation task mismatch at row {index}")
        require(sha256_bytes(raw) == reconciled["input_row_sha256"], f"canonical/reconciliation hash mismatch at row {index}")

    phase0_rows = read_jsonl(phase0_path)
    require(len(phase0_rows) == 3305, "Phase-0 manifest must contain 3,305 rows")
    phase0_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for line_number, row in enumerate(phase0_rows, 1):
        task_id = str(row.get("task_id") or "")
        require(task_id and task_id not in phase0_by_id, "invalid/duplicate Phase-0 task ID")
        require(row.get("split") in {"train", "dev"}, f"invalid Phase-0 split for {task_id}")
        phase0_by_id[task_id] = (line_number, row)
    require(
        collections.Counter(row["split"] for row in phase0_rows)
        == {"train": 2975, "dev": 330},
        "Phase-0 split totals drift",
    )
    require(
        sum(bool(row.get("in_long_dev_ge200")) for row in phase0_rows) == 140,
        "Phase-0 long-dev count drift",
    )
    unlisted: list[dict[str, Any]] = []
    model_rows: set[int] = set()
    for row in reconciliation:
        assigned = phase0_by_id.get(row["task_id"])
        if assigned is None:
            unlisted.append(row)
            require(row.get("phase0_manifest_present") is False, "unlisted row marked present")
            require(row.get("phase0_manifest_line") is None, "unlisted row has manifest line")
            require(row.get("phase0_split") is None, "unlisted row has a split")
            continue
        line_number, assignment = assigned
        require(row.get("phase0_manifest_present") is True, "listed row marked absent")
        require(row.get("phase0_manifest_line") == line_number, "Phase-0 line mapping drift")
        require(row.get("phase0_split") == assignment["split"], "Phase-0 split mapping drift")
        require(
            bool(row.get("in_long_dev_ge200"))
            == bool(assignment.get("in_long_dev_ge200")),
            "Phase-0 long-dev mapping drift",
        )
        if str(row.get("status", "")).startswith("included-"):
            model_row = row.get("model_row")
            require(isinstance(model_row, int) and not isinstance(model_row, bool), "included row lacks model_row")
            require(model_row not in model_rows, "duplicate reconciliation model_row")
            model_rows.add(model_row)
        else:
            require(row.get("model_row") is None, "excluded/quarantined row has model_row")
    require(len(unlisted) == 1, "expected exactly one unlisted canonical row")
    require(unlisted[0]["task_id"] == "sigless_4901067c13b9", "unexpected unlisted row")
    require(unlisted[0]["status"] == "excluded", "unlisted row must be excluded")
    require(str(unlisted[0]["reason"]).startswith("not_in_phase0_manifest:"), "unlisted exclusion reason drift")
    require(model_rows == set(range(3277)), "model_row is not a 0..3276 bijection")

    quarantine = read_jsonl(prepared / "quarantine.jsonl")
    quarantined_rows = [row for row in reconciliation if row["status"] == "quarantined"]
    require(quarantine == quarantined_rows, "quarantine sidecar is not the reconciliation subset")
    require(
        all(re.fullmatch(r"unknown_or_corrupt_mnemonic:local_\d+", str(row.get("reason"))) for row in quarantine),
        "quarantine contains a non-local_N reason",
    )
    quarantine_ids = {row["task_id"] for row in quarantine}
    require(
        {"sigless_179318026249", "sigless_b51cb0fef9eb"} <= quarantine_ids,
        "required named corrupt rows are not quarantined",
    )
    overlap = read_jsonl(prepared / "forbidden_overlap_audit.jsonl")
    require(len(overlap) == 14, "expected 14 forbidden near-clone exclusions")
    require(all(row.get("reason") == "forbidden_near_clone" for row in overlap), "non-near-clone overlap reason")
    excluded_near = [row for row in reconciliation if row.get("reason") == "forbidden_near_clone"]
    require(overlap == excluded_near, "overlap audit is not the reconciliation subset")
    return manifest, reconciliation


def validate_public_and_alignment(
    release: Path,
    preparation: Mapping[str, Any],
    reconciliation: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    bundle = release / "compact_qwen_phase0_s44_v2"
    verify_checksum_file(bundle / "SHA256SUMS.txt", EXPECTED_FILES[bundle.name])
    contract = read_json(bundle / "compact_contract.json")
    codebook = read_json(bundle / "codebook.json")
    preflight = read_json(bundle / "preflight_report.json")
    require(contract.get("schema") == "direct-compact-causal-v2", "bad compact contract schema")
    require(contract.get("target_language") == "Dart", "contract language drift")
    require(contract.get("target_function") == "candidate", "contract target drift")
    require(contract.get("lossless_domain") == "scrubbed_canonical_graph_v2", "lossless domain drift")
    require(contract.get("max_source_tokens") == 9000, "source token gate drift")
    require(codebook.get("schema") == "compact-qwen-v2-codebook", "bad codebook schema")
    require(
        sha256_file(bundle / "codebook.json") == contract.get("codebook_sha256"),
        "contract/codebook hash mismatch",
    )
    require(
        codebook.get("fit_public_sha256")
        == preparation["outputs"]["train_codec_private.jsonl"]["sha256"],
        "codebook fit input is not the sealed train-only graph file",
    )
    require(codebook.get("fit_retained") == 2951 and codebook.get("fit_quarantined") == 0, "codebook fit population drift")
    require(codebook.get("runtime_symbol_policy", {}).get("version") == "runtime-symbol-policy-v1", "runtime symbol policy drift")
    require(codebook.get("runtime_symbol_policy_sha256") == contract.get("runtime_symbol_policy_sha256"), "runtime policy hash mismatch")
    require(codebook.get("extractor_routes") == contract.get("extractor_routes"), "contract/codebook route mismatch")
    route_specs = contract.get("extractor_routes") or {}
    require(set(route_specs) == ROUTES, "extractor route set drift")
    require(route_specs["legacy_release_v1"].get("route_atom") == "<DX0>", "legacy route atom drift")
    require(route_specs["current_combined_v2"].get("route_atom") == "<DX1>", "current route atom drift")
    require(route_specs["legacy_release_v1"].get("allow_call_edges") is False, "legacy route silently permits call edges")
    require(route_specs["current_combined_v2"].get("allow_call_edges") is True, "current route does not permit call edges")
    for route, values in route_specs.items():
        for field in ("graph_extractor_sha256", "cfg_extractor_sha256", "dfg_extractor_sha256"):
            require_digest(values.get(field), f"{route}.{field}")
    scheme = codebook.get("added_token_scheme") or {}
    require(scheme.get("extractor_route_tokens") == {"legacy_release_v1": "<DX0>", "current_combined_v2": "<DX1>"}, "codebook route atoms drift")
    require((scheme.get("edge_tokens") or {}).get("call") == "<CC>", "call-edge atom is absent")

    require(preflight.get("schema") == "compact-qwen-v2-preflight", "bad preflight schema")
    require(preflight.get("passed") is True, "preflight failed")
    require(preflight.get("rows_retained") == 3277, "preflight retained-row drift")
    require(preflight.get("rows_by_role") == EXPECTED_ROLE_COUNTS, "preflight role drift")
    require(preflight.get("rows_by_route") == EXPECTED_ROUTE_COUNTS, "preflight route drift")
    require(preflight.get("quarantined") == 0 and preflight.get("failures_count") == 0, "codec-stage quarantine/failure is nonzero")
    require(preflight.get("exploratory_full_release_fit") is False, "dev was included in codebook fit")
    require((preflight.get("fallback_by_role") or {}).get("fit", {}).get("fallback") == 0, "fit side has fallback despite train-only fit")
    lossless = preflight.get("lossless_invariants") or {}
    require(lossless.get("exact_instruction_entry_cfg_route_roundtrip_rows") == 3277, "not all rows round-trip")
    require(lossless.get("dfg_regenerated_and_matched_rows") == 3277, "not all DFGs were regenerated")
    require(lossless.get("dfg_edges_matched_edge_for_edge") == 334849, "DFG edge total drift")
    require(lossless.get("dfg_edges_by_route") == {"current_combined_v2": 134518, "legacy_release_v1": 200331}, "DFG route totals drift")
    require(lossless.get("call_edges_encoded_explicitly") is True, "call edges were not encoded explicitly")
    require(lossless.get("raw_fallback_is_reversible") is True, "raw fallback is not reversible")
    require(lossless.get("privacy_scrub_is_only_intentional_irreversibility") is True, "unexpected lossy transform")
    require(lossless.get("truncated_rows") == 0, "preflight truncation is nonzero")
    require(lossless.get("unknown_tokens") == 0, "preflight unknown-token count is nonzero")
    tokens = preflight.get("tokens") or {}
    require(tokens.get("limit") == 9000 and int(tokens.get("max", 9001)) <= 9000, "9,000-token gate failed")
    require((preflight.get("cfg_edge_types") or {}).get("call") == 43, "call-edge count drift")
    require(read_jsonl(bundle / "quarantine.jsonl", allow_empty=True) == [], "codec quarantine file is not empty")
    require(read_jsonl(bundle / "failures.jsonl", allow_empty=True) == [], "codec failure file is not empty")

    public = read_jsonl(bundle / "compact_model_inputs.jsonl")
    alignment = read_jsonl(bundle / "alignment_private.jsonl")
    require(len(public) == len(alignment) == 3277, "public/alignment row count drift")
    by_model_row = {
        int(row["model_row"]): row
        for row in reconciliation
        if str(row.get("status", "")).startswith("included-")
    }
    seen_ids: set[str] = set()
    role_counts: collections.Counter[str] = collections.Counter()
    route_counts: collections.Counter[str] = collections.Counter()
    role_family_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for index, (model, private) in enumerate(zip(public, alignment, strict=True)):
        require(set(model) == PUBLIC_FIELDS, f"public row {index} is not strict four-field")
        ids = model.get("compact_input_ids")
        require(isinstance(ids, list) and 0 < len(ids) <= 9000, f"public row {index} has invalid token list")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in ids), f"public row {index} has a non-integer token ID")
        require(model["compact_codec_sha256"] == contract["codec_sha256"], f"public row {index} codec hash drift")
        require(model["compact_codebook_sha256"] == contract["codebook_sha256"], f"public row {index} codebook hash drift")
        require(model["compact_tokenizer_sha256"] == contract["tokenizer_json_sha256"], f"public row {index} tokenizer hash drift")
        require(private.get("model_row") == index, f"alignment row {index} is not positional")
        task_id = str(private.get("task_id") or "")
        require(task_id and task_id not in seen_ids, f"duplicate/empty alignment task at row {index}")
        seen_ids.add(task_id)
        expected = by_model_row[index]
        require(task_id == expected["task_id"], f"alignment/reconciliation task mismatch at row {index}")
        require(private.get("input_line") == expected["input_line"], f"alignment/reconciliation line mismatch at row {index}")
        require(private.get("input_row_sha256") == expected["input_row_sha256"], f"alignment/reconciliation input hash mismatch at row {index}")
        require(private.get("canonical_sha256") == expected["canonical_sha256"], f"alignment/reconciliation canonical hash mismatch at row {index}")
        require(
            private.get("original_target_sha256")
            == sha256_bytes(str(expected["original_target"]).encode("utf-8")),
            f"alignment/reconciliation original-target hash mismatch at row {index}",
        )
        role = str(private.get("role") or "")
        expected_role = "fit" if expected["status"] == "included-train" else "measure"
        require(role == expected_role, f"alignment role drift at row {index}")
        require(private.get("phase0_split") == ("train" if role == "fit" else "dev"), f"alignment split drift at row {index}")
        require(private.get("target_function") == "candidate", f"alignment target drift at row {index}")
        require(private.get("source_tokens") == len(ids), f"alignment token count drift at row {index}")
        family = str(private.get("family") or "")
        route = str(private.get("dfg_route") or "")
        require(family in FAMILIES, f"alignment family drift at row {index}")
        require(route in ROUTES, f"alignment extractor route drift at row {index}")
        role_counts[role] += 1
        route_counts[route] += 1
        role_family_counts[role][family] += 1
    require(dict(role_counts) == EXPECTED_ROLE_COUNTS, "alignment role counts drift")
    require(dict(route_counts) == EXPECTED_ROUTE_COUNTS, "alignment route counts drift")
    require({role: dict(values) for role, values in role_family_counts.items()} == EXPECTED_ROLE_FAMILY_COUNTS, "alignment family counts drift")

    # The scan covers all top-up identities, including excluded rows, so a
    # target symbol cannot enter the learned vocabulary through an excluded
    # row or survive in an included compact stream.  A single literal hit is a
    # release-blocking leak.
    original_targets = [
        str(row.get("original_target") or "")
        for row in reconciliation
        if row.get("original_target") != "candidate"
    ]
    require(len(original_targets) == 1117, "non-candidate original-target count drift")
    require(
        len(set(original_targets)) == 1117 and all(original_targets),
        "non-candidate original targets are empty or non-unique",
    )
    target_matcher = re.compile(
        "|".join(re.escape(value) for value in sorted(original_targets, key=len, reverse=True))
    )
    expansion_hits = [
        value for value in codebook.get("expansions", [])
        if target_matcher.search(str(value))
    ]
    stream_hits = [
        index for index, row in enumerate(alignment)
        if target_matcher.search(str(row.get("compact_text") or ""))
    ]
    require(not expansion_hits, "an original target identifier leaked into codebook expansions")
    require(not stream_hits, "an original target identifier leaked into compact_text")
    leakage = {
        "non_candidate_original_targets": 1117,
        "identifier_set_sha256": sha256_bytes(stable_bytes(sorted(original_targets))),
        "codebook_expansions_scanned": len(codebook.get("expansions", [])),
        "alignment_compact_streams_scanned": len(alignment),
        "codebook_literal_hits": 0,
        "compact_stream_literal_hits": 0,
    }
    return contract, public, alignment, leakage


def validate_private_target_relabeling(
    release: Path,
    reconciliation: list[dict[str, Any]],
) -> dict[str, int]:
    labels = (
        read_jsonl(release / "prepared/train_private_labels.jsonl")
        + read_jsonl(release / "prepared/dev_private_labels.jsonl")
    )
    included = sorted(
        (
            row for row in reconciliation
            if str(row.get("status", "")).startswith("included-")
        ),
        key=lambda row: int(row["model_row"]),
    )
    require(len(labels) == len(included) == 3277, "private target population drift")
    candidate_rows = 0
    topup_rows = 0
    old_target_hits = 0
    for index, (label, source) in enumerate(zip(labels, included, strict=True)):
        require(label.get("task_id") == source.get("task_id"), f"private target order drift at row {index}")
        if label.get("function") == "candidate":
            candidate_rows += 1
        original = str(source.get("original_target") or "")
        if original != "candidate":
            topup_rows += 1
            if original in str(label.get("dart_source") or ""):
                old_target_hits += 1
    require(candidate_rows == 3277, "not every private-label function is candidate")
    require(topup_rows == 1105, "included top-up target count drift")
    require(old_target_hits == 0, "an included top-up old target survived in dart_source")
    return {
        "private_label_candidate_rows": candidate_rows,
        "included_topup_rows_scanned": topup_rows,
        "old_target_literal_hits_in_own_dart_source": old_target_hits,
    }


def validate_join(
    release: Path,
    role: str,
    public: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    supervised_dir = release / "supervised"
    expected_role = "fit" if role == "train" else "measure"
    expected_rows = EXPECTED_ROLE_COUNTS[expected_role]
    output_path = supervised_dir / f"{role}.jsonl"
    seal_path = supervised_dir / f"{role}.join_seal.json"
    label_path = release / "prepared" / f"{role}_private_labels.jsonl"
    rows = read_jsonl(output_path)
    labels = read_jsonl(label_path)
    seal = read_json(seal_path)
    require(len(rows) == len(labels) == expected_rows, f"{role}: joined/label row count drift")
    require(seal.get("schema") == "compact-public-private-join-seal-v1", f"{role}: bad join seal schema")
    require(seal.get("rows") == expected_rows, f"{role}: seal row count drift")
    require(seal.get("source_rows") == 3277, f"{role}: source row count drift")
    require(seal.get("selected_role") == expected_role, f"{role}: selected role drift")
    require(seal.get("skipped_rows") == 3277 - expected_rows, f"{role}: skipped count drift")
    require(seal.get("unused_private_rows") == 0, f"{role}: unused private labels")
    bijection = seal.get("private_bijection") or {}
    require(bijection.get("required") is True and bijection.get("verified") is True, f"{role}: private bijection was not required and verified")
    require(bijection.get("unused_private_rows") == 0, f"{role}: private bijection has unused rows")
    bundle = release / "compact_qwen_phase0_s44_v2"
    require(seal.get("public_sha256") == sha256_file(bundle / "compact_model_inputs.jsonl"), f"{role}: public hash drift")
    require(seal.get("alignment_sha256") == sha256_file(bundle / "alignment_private.jsonl"), f"{role}: alignment hash drift")
    require(seal.get("private_sha256") == sha256_file(label_path), f"{role}: label hash drift")
    require(seal.get("contract_sha256") == sha256_file(bundle / "compact_contract.json"), f"{role}: contract hash drift")
    require(seal.get("output_sha256") == sha256_file(output_path), f"{role}: output hash drift")
    require(seal.get("output_size_bytes") == output_path.stat().st_size, f"{role}: output size drift")
    selected_indices = [index for index, row in enumerate(alignment) if row["role"] == expected_role]
    require(len(selected_indices) == expected_rows, f"{role}: selected public population drift")
    expected_mapping: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    metadata_counts = {field: collections.Counter() for field in ("family", "source_pool", "extractor_route")}
    missing = collections.Counter()
    for private_index, (public_index, label, joined) in enumerate(zip(selected_indices, labels, rows, strict=True)):
        align = alignment[public_index]
        task_id = str(label.get("task_id") or "")
        require(set(label) == LABEL_FIELDS, f"{role}: private label {private_index} schema drift")
        require(task_id == align["task_id"], f"{role}: private/alignment task mismatch")
        require(
            str(label.get("lang") or "").lower() == "dart"
            and label.get("function") == "candidate",
            f"{role}: label target drift",
        )
        require(label.get("family") == align.get("family"), f"{role}: label family drift")
        require(set(joined) == SUPERVISED_FIELDS, f"{role}: joined row {private_index} schema drift")
        expected_joined = {
            "lang": "Dart",
            "function": "candidate",
            "dart_source": str(label["dart_source"]).strip(),
            **public[public_index],
        }
        require(joined == expected_joined, f"{role}: joined row {private_index} is not exact public+label")
        identity_sha = sha256_bytes(task_id.encode("utf-8"))
        expected_mapping.append({
            "public_line": public_index,
            "alignment_line": public_index,
            "private_line": private_index,
            "identity_sha256": identity_sha,
        })
        projection = {
            "identity_sha256": identity_sha,
            "family": align.get("family"),
            "source_pool": align.get("source_pool"),
            "extractor_route": align.get("dfg_route"),
        }
        metadata_rows.append(projection)
        for field in ("family", "source_pool", "extractor_route"):
            value = projection[field]
            if value is None:
                missing[field] += 1
            else:
                metadata_counts[field][str(value)] += 1
    require(seal.get("mapping") == expected_mapping, f"{role}: join mapping drift")
    require(seal.get("mapping_sha256") == sha256_bytes(stable_bytes(expected_mapping)), f"{role}: mapping seal drift")
    require(seal.get("private_metadata_projection_sha256") == sha256_bytes(stable_bytes(metadata_rows)), f"{role}: metadata projection seal drift")
    expected_metadata_counts = {
        "rows": expected_rows,
        **{
            field: {
                "counts": dict(sorted(metadata_counts[field].items())),
                "missing_rows": int(missing[field]),
            }
            for field in ("family", "source_pool", "extractor_route")
        },
    }
    require(seal.get("private_metadata_counts") == expected_metadata_counts, f"{role}: metadata counts drift")
    require(seal.get("model_visible_fields") == sorted(SUPERVISED_FIELDS), f"{role}: model-visible field declaration drift")
    return seal


def validate_audits(release: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    audits = release / "audits"
    verify_checksum_file(audits / "SHA256SUMS.txt", EXPECTED_FILES["audits"])
    paths = {
        "dev_fallback_token_audit.json": "phase0_dev_fallback_and_tokens",
        "scrubbed_humaneval_fallback_token_audit.json": "scrubbed_humaneval_fallback_and_tokens",
        "topup_family_fallback_audit.json": "phase0_family_and_source_pool_fallback",
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, audit_name in paths.items():
        report = read_json(audits / name)
        require(report.get("schema") == "compact-qwen-v2-generalization-audit-v1", f"{name}: bad audit schema")
        require(report.get("audit") == audit_name, f"{name}: audit kind drift")
        require(report.get("passed") is True, f"{name}: audit did not pass")
        require(all(value is True for value in (report.get("gates") or {}).values()), f"{name}: one or more gates are false")
        require((report.get("route_binding") or {}).get("extractor_routes") == contract.get("extractor_routes"), f"{name}: extractor binding drift")
        require((report.get("route_binding") or {}).get("target_function") == "candidate", f"{name}: target binding drift")
        for label, binding in (report.get("input_bindings") or {}).items():
            verify_binding(binding, f"{name} input {label}")
        reports[name] = report
    dev = reports["dev_fallback_token_audit.json"]
    require(dev.get("rows") == 326, "dev audit row count drift")
    require(dev.get("exact_canonical_and_dfg_roundtrip_rows") == 326, "dev audit round-trip count drift")
    require((dev.get("tokens") or {}).get("rows_over_limit") == 0, "dev audit exceeds 9,000 tokens")
    require((dev.get("tokens") or {}).get("limit") == 9000, "dev audit token limit drift")
    human = reports["scrubbed_humaneval_fallback_token_audit.json"]
    require(human.get("rows") == 154, "HumanEval audit row count drift")
    require(human.get("exact_canonical_and_dfg_roundtrip_rows") == 154, "HumanEval audit round-trip count drift")
    require((human.get("tokens") or {}).get("rows_over_limit") == 0, "HumanEval audit exceeds 9,000 tokens")
    require((human.get("route_override_authorization") or {}).get("route_atom") == "<DX0>", "HumanEval route override is not legacy/hash-authorized")
    family = reports["topup_family_fallback_audit.json"]
    require(family.get("population") == {
        "train_rows": 2951,
        "dev_rows": 326,
        "rows": 3277,
        "requested_family_policy": {
            "master": [None],
            "topup_s45": ["base_llm", "topup_s44", "topup_s45"],
            "topup_s46": ["topup_s46"],
        },
    }, "family audit population/policy drift")
    train_only = family.get("sealed_train_only_codebook") or {}
    require(train_only.get("fit_population") == "all included Phase-0 train rows only", "family audit codebook fit population drift")
    require(train_only.get("deterministically_refit_and_matched_seal") is True, "codebook was not deterministically refit")
    require(((train_only.get("coverage") or {}).get("by_phase0_split") or {}).get("train", {}).get("fallback_instructions") == 0, "train-only codebook has train fallback")
    seal = read_json(audits / "generalization_audit_seal.json")
    require(seal.get("schema") == "compact-qwen-v2-generalization-audit-seal-v1", "bad audit seal schema")
    require(seal.get("all_passed") is True, "audit umbrella seal did not pass")
    verify_binding(seal.get("script") or {}, "generalization audit producer")
    for label, binding in (seal.get("additional_input_bindings") or {}).items():
        verify_binding(binding, f"generalization audit additional input {label}")
    for name in paths:
        entry = (seal.get("reports") or {}).get(name) or {}
        require(entry.get("passed") is True, f"audit seal does not mark {name} passed")
        require(entry.get("sha256") == sha256_file(audits / name), f"audit seal hash drift for {name}")
    return {name: report for name, report in reports.items()}


def file_record(path: Path, release: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(release).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        record["rows"] = sum(1 for line in path.open("rb") if line.strip())
    return record


def producer_record(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing release producer: {path}")
    return {
        "path": path.relative_to(WORKSPACE).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def finalize(release: Path) -> dict[str, Any]:
    release = release.resolve()
    require(release.is_dir(), f"release directory does not exist: {release}")
    require_exact_directory_contents(release)
    preparation, reconciliation = verify_preparation(release)
    contract, public, alignment, leakage = validate_public_and_alignment(
        release, preparation, reconciliation
    )
    train_seal = validate_join(release, "train", public, alignment, contract)
    dev_seal = validate_join(release, "dev", public, alignment, contract)
    target_relabeling = validate_private_target_relabeling(release, reconciliation)
    audits = validate_audits(release, contract)

    # The manifest lists payloads; its own digest is provided by SHA256SUMS.txt.
    payload_paths = sorted(
        (
            path
            for path in release.rglob("*")
            if path.is_file()
            and path.relative_to(release).as_posix()
            not in {"release_manifest.json", "SHA256SUMS.txt"}
        ),
        key=lambda path: path.relative_to(release).as_posix(),
    )
    manifest = {
        "schema": "compact-qwen-phase0-s44-v2-release-seal-v1",
        "release": "direct_compact_phase0_s44_v2",
        "deterministic": True,
        "self_binding": "release_manifest.json is bound by the root SHA256SUMS.txt",
        "counts": {
            "canonical_input": 3306,
            "phase0_manifest": 3305,
            "included_train": 2951,
            "included_dev": 326,
            "quarantined_local_mnemonic": 14,
            "excluded_forbidden_near_clone": 14,
            "excluded_not_in_phase0_manifest": 1,
            "model_rows": 3277,
            "long_dev_retained": 138,
            "long_dev_quarantined": 2,
            "non_candidate_original_targets_leak_scanned": 1117,
            "included_topup_targets_relabel_checked": 1105,
        },
        "contract": {
            "schema": contract["schema"],
            "sha256": sha256_file(release / "compact_qwen_phase0_s44_v2/compact_contract.json"),
            "codebook_sha256": contract["codebook_sha256"],
            "codec_sha256": contract["codec_sha256"],
            "tokenizer_json_sha256": contract["tokenizer_json_sha256"],
            "runtime_symbol_policy_sha256": contract["runtime_symbol_policy_sha256"],
            "extractor_routes": contract["extractor_routes"],
            "target_function": "candidate",
            "lossless_domain": "scrubbed_canonical_graph_v2",
            "max_source_tokens": 9000,
        },
        "gates": {
            "every_canonical_row_reconciled_once": True,
            "phase0_assignment_preserved": True,
            "train_only_codebook_refit_verified": True,
            "strict_four_field_public_schema": True,
            "public_alignment_bijective": True,
            "private_joins_bijective_and_hash_bound": True,
            "all_private_label_functions_are_candidate": True,
            "included_topup_old_targets_absent_from_own_dart_source": True,
            "original_targets_absent_from_codebook_expansions": True,
            "original_targets_absent_from_compact_streams": True,
            "all_3277_rows_roundtrip": True,
            "dfg_regenerated_and_matched_edge_for_edge": True,
            "call_edges_explicit": True,
            "zero_unknown_tokens": True,
            "zero_truncation": True,
            "all_source_rows_within_9000_tokens": True,
            "dev_generalization_audit_passed": audits["dev_fallback_token_audit.json"]["passed"],
            "scrubbed_humaneval_generalization_audit_passed": audits["scrubbed_humaneval_fallback_token_audit.json"]["passed"],
            "topup_family_generalization_audit_passed": audits["topup_family_fallback_audit.json"]["passed"],
            "all_nested_checksums_verified": True,
            "passed": True,
        },
        "nested_seals": {
            "preparation_manifest_sha256": sha256_file(release / "prepared/preparation_manifest.json"),
            "preparation_checksums_sha256": sha256_file(release / "prepared/SHA256SUMS.txt"),
            "compact_preflight_sha256": sha256_file(release / "compact_qwen_phase0_s44_v2/preflight_report.json"),
            "compact_checksums_sha256": sha256_file(release / "compact_qwen_phase0_s44_v2/SHA256SUMS.txt"),
            "train_join_seal_sha256": sha256_file(release / "supervised/train.join_seal.json"),
            "dev_join_seal_sha256": sha256_file(release / "supervised/dev.join_seal.json"),
            "generalization_audit_seal_sha256": sha256_file(release / "audits/generalization_audit_seal.json"),
            "generalization_checksums_sha256": sha256_file(release / "audits/SHA256SUMS.txt"),
        },
        "join_rows": {
            "train": train_seal["rows"],
            "dev": dev_seal["rows"],
        },
        "target_leakage_audit": leakage,
        "private_target_relabeling": target_relabeling,
        "producer_bindings": {
            "codec": producer_record(WORKSPACE / "scripts/data/build_compact_qwen_v2.py"),
            "preparer": producer_record(WORKSPACE / "scrubbed_master_v2_release/prepare_phase0_compact_qwen_v2.py"),
            "generalization_auditor": producer_record(WORKSPACE / "scrubbed_master_v2_release/audit_phase0_compact_qwen_v2_generalization.py"),
            "release_finalizer": producer_record(Path(__file__).resolve()),
            "trainer_contract": producer_record(WORKSPACE / "hybrid_training_patch_v2_3/models/direct_compact_causal.py"),
            "private_joiner": producer_record(WORKSPACE / "hybrid_training_patch_v2_3/scripts/training/join_compact_public_private.py"),
        },
        "files": [file_record(path, release) for path in payload_paths],
    }
    require(all(manifest["gates"].values()), "umbrella gate failure")
    write_json_atomic(release / "release_manifest.json", manifest)

    shipped = sorted(
        (
            path
            for path in release.rglob("*")
            if path.is_file()
            and path.relative_to(release).as_posix() != "SHA256SUMS.txt"
        ),
        key=lambda path: path.relative_to(release).as_posix(),
    )
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(release).as_posix()}\n"
        for path in shipped
    )
    checksum_tmp = release / "SHA256SUMS.txt.tmp"
    checksum_tmp.write_text(checksum_text, encoding="utf-8", newline="\n")
    os.replace(checksum_tmp, release / "SHA256SUMS.txt")

    # Verify recursive coverage after the atomic replace.  The root checksum is
    # the only shipped file intentionally excluded from its own contents.
    observed: dict[str, str] = {}
    for line in (release / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        observed[name] = digest
    expected_paths = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.relative_to(release).as_posix() != "SHA256SUMS.txt"
    }
    require(set(observed) == expected_paths, "umbrella checksum coverage is incomplete")
    for name, digest in observed.items():
        require(sha256_file(release / name) == digest, f"umbrella checksum mismatch: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    try:
        manifest = finalize(args.release_dir)
    except (KeyError, TypeError, json.JSONDecodeError, ReleaseValidationError) as error:
        raise SystemExit(f"REFUSED TO SEAL: {error}") from error
    print(json.dumps({
        "release_dir": str(args.release_dir.resolve()),
        "manifest_sha256": sha256_file(args.release_dir / "release_manifest.json"),
        "checksums_sha256": sha256_file(args.release_dir / "SHA256SUMS.txt"),
        "shipped_files": len(manifest["files"]) + 2,
        "passed": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
