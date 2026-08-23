#!/usr/bin/env python3
"""Assemble and seal the encoder-free Phase-0 compact-Qwen v3 release.

This is an umbrella *packager*, not another data builder.  It accepts only
already-sealed producer outputs, revalidates their hashes and cross-component
bindings, and copies the finite release payload into a new directory.  AOT
ELFs are deliberately not copied: the verified binary-build seal and its AOT
manifest are shipped instead.

The generated release manifest is deterministic and path-independent.  It
contains only release-relative paths, hashes, sizes, counts, and policy facts;
host paths and timestamps are never recorded.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "compact-qwen-phase0-s44-v3-release-seal-v1"
INHERITANCE_SCHEMA = "phase0-s44-v3-canonical-membership-inheritance-v1"
EXPECTED_SPLITS = {"train": 2951, "dev": 326}
EXPECTED_ROLES = {"fit": 2951, "measure": 326}
EXPECTED_STATUS_COUNTS = {
    "included-train": 2951,
    "included-dev": 326,
    "quarantined": 14,
    "excluded": 15,
}
EXPECTED_MODEL_ROWS = 3277
EXPECTED_CANONICAL_ROWS = 3306
TARGET_FUNCTION = "candidate"
PUBLIC_FIELDS = frozenset(
    {
        "compact_input_ids",
        "compact_codec_sha256",
        "compact_codebook_sha256",
        "compact_tokenizer_sha256",
    }
)
SUPERVISED_FIELDS = frozenset(
    {
        "lang",
        "function",
        "dart_source",
        *PUBLIC_FIELDS,
    }
)
SOURCE_FILES = frozenset(
    {
        "private_build_inputs/train.jsonl",
        "private_build_inputs/dev.jsonl",
        "prepared/train_private_labels.jsonl",
        "prepared/dev_private_labels.jsonl",
        "source_preparation_manifest.json",
    }
)
BINARY_FILES = frozenset(
    {
        "binary_build_manifest.json",
        "aot_manifest.jsonl",
        "pool_reconciliation_private.jsonl",
        "pool_extractor_manifest.json",
        "dart_toolchain_manifest.json",
        "manifests/train.json",
        "manifests/dev.json",
        "prepared/train_codec_private.jsonl",
        "prepared/dev_codec_private.jsonl",
        "quarantine/train.jsonl",
        "quarantine/dev.jsonl",
    }
)
COMPACT_REQUIRED = frozenset(
    {
        "codebook.json",
        "compact_contract.json",
        "compact_model_inputs.jsonl",
        "alignment_private.jsonl",
        "pool_reconciliation_private.jsonl",
        "preflight_report.json",
        "quarantine.jsonl",
        "failures.jsonl",
    }
)
SUPERVISED_FILES = frozenset(
    {
        "train.jsonl",
        "train.join_seal.json",
        "dev.jsonl",
        "dev.join_seal.json",
    }
)
AUDIT_REPORTS = frozenset(
    {
        "dev_fallback_audit.json",
        "scrubbed_humaneval_instruction_codebook_audit.json",
        "topup_family_source_pool_fallback_audit.json",
    }
)
AUDIT_FILES = frozenset({*AUDIT_REPORTS, "generalization_audit_seal.json"})
V2_PREP_FILES = frozenset(
    {
        "train_codec_private.jsonl",
        "dev_codec_private.jsonl",
        "train_private_labels.jsonl",
        "dev_private_labels.jsonl",
        "reconciliation.jsonl",
        "quarantine.jsonl",
        "forbidden_overlap_audit.jsonl",
        "preparation_manifest.json",
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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AOT_SUFFIXES = frozenset({".aot", ".elf", ".so", ".dylib", ".dll"})


class ReleaseValidationError(ValueError):
    """Raised for any fail-closed release validation error."""


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


def require_sha256(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    require(bool(SHA256_RE.fullmatch(result)), f"{label}: invalid SHA-256")
    return result


def private_target(row: Mapping[str, Any], label: str) -> str:
    """Mirror the strict joiner's target normalization exactly."""
    value = row.get("supervised_target") or row.get("dart_source") or row.get("source")
    result = str(value or "").strip()
    require(bool(result), f"{label}: missing private target")
    return result


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


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"{path}:{line_number}: blank JSONL line")
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    require(allow_empty or bool(rows), f"{path}: unexpectedly empty")
    return rows


def _safe_relative(name: str, label: str) -> str:
    normalized = name.replace("\\", "/")
    path = Path(normalized)
    require(
        bool(normalized)
        and not path.is_absolute()
        and ".." not in path.parts
        and normalized == path.as_posix(),
        f"{label}: unsafe/noncanonical path {name!r}",
    )
    return normalized


def verify_checksum(
    root: Path,
    checksum_name: str,
    expected: set[str] | frozenset[str],
) -> dict[str, str]:
    """Verify an exact, non-self-referential component checksum."""
    checksum = root / checksum_name
    require(
        checksum.is_file() and not checksum.is_symlink(),
        f"missing/unsafe checksum: {checksum}",
    )
    found: dict[str, str] = {}
    for line_number, raw in enumerate(
        checksum.read_text(encoding="utf-8").splitlines(), 1
    ):
        require(bool(raw.strip()), f"{checksum}:{line_number}: blank line")
        parts = raw.split("  ", 1)
        require(len(parts) == 2, f"{checksum}:{line_number}: malformed line")
        digest = require_sha256(parts[0], f"{checksum}:{line_number}")
        name = _safe_relative(parts[1], f"{checksum}:{line_number}")
        require(name not in found, f"{checksum}: duplicate entry {name}")
        target = root / name
        require(
            target.is_file() and not target.is_symlink(),
            f"{checksum}: missing/unsafe target {name}",
        )
        require(sha256_file(target) == digest, f"{checksum}: hash drift for {name}")
        found[name] = digest
    require(
        set(found) == set(expected),
        f"{checksum}: coverage drift; found={sorted(found)}, expected={sorted(expected)}",
    )
    return found


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _walk_keys(nested)


def _task_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    values = [str(row.get("task_id") or "") for row in rows]
    require(all(values), f"{label}: empty task ID")
    require(len(values) == len(set(values)), f"{label}: duplicate task ID")
    return values


def _all_gates(value: Mapping[str, Any], label: str) -> None:
    gates = value.get("gates")
    require(isinstance(gates, Mapping) and bool(gates), f"{label}: missing gates")
    require(all(gate is True for gate in gates.values()), f"{label}: false gate")


def _require_file_set(root: Path, expected: set[str] | frozenset[str]) -> None:
    for name in expected:
        target = root / name
        require(target.is_file() and not target.is_symlink(), f"missing/unsafe: {target}")


def validate_source_preparation(root: Path) -> dict[str, Any]:
    _require_file_set(root, SOURCE_FILES | {"SOURCE_SHA256SUMS.txt"})
    hashes = verify_checksum(root, "SOURCE_SHA256SUMS.txt", SOURCE_FILES)
    manifest = read_json(root / "source_preparation_manifest.json")
    require(
        manifest.get("schema") == "phase0-s44-binary-pool-v3-source-preparation-v1",
        "source preparation schema drift",
    )
    _all_gates(manifest, "source preparation")
    split_entries = manifest.get("splits")
    require(isinstance(split_entries, list), "source preparation splits missing")
    entries = {str(item.get("split")): item for item in split_entries}
    require(set(entries) == set(EXPECTED_SPLITS), "source preparation split drift")
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, expected in EXPECTED_SPLITS.items():
        build = read_jsonl(root / f"private_build_inputs/{split}.jsonl")
        labels = read_jsonl(root / f"prepared/{split}_private_labels.jsonl")
        require(len(build) == expected and len(labels) == expected, f"{split}: source count drift")
        build_ids = _task_ids(build, f"{split} build inputs")
        label_ids = _task_ids(labels, f"{split} private labels")
        require(build_ids == label_ids, f"{split}: build/label task order drift")
        require(
            all(str(row.get("function") or "") == TARGET_FUNCTION for row in labels),
            f"{split}: private labels are not uniformly candidate",
        )
        entry = entries[split]
        require(entry.get("rows") == expected, f"{split}: source manifest count drift")
        for field, name in (
            ("private_build_inputs", f"private_build_inputs/{split}.jsonl"),
            ("private_labels", f"prepared/{split}_private_labels.jsonl"),
        ):
            binding = entry.get(field) or {}
            require(
                require_sha256(binding.get("sha256"), f"source {split} {field}")
                == hashes[name],
                f"source {split} {field}: manifest/checksum drift",
            )
        split_rows[split] = {"build": build, "labels": labels}
    return {
        "manifest": manifest,
        "hashes": hashes,
        "splits": split_rows,
    }


def validate_binary_build(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    _require_file_set(root, BINARY_FILES | {"BINARY_BUILD_SHA256SUMS.txt"})
    hashes = verify_checksum(root, "BINARY_BUILD_SHA256SUMS.txt", BINARY_FILES)
    manifest = read_json(root / "binary_build_manifest.json")
    require(
        manifest.get("schema") == "phase0-s44-binary-pool-build-seal-v1",
        "binary build schema drift",
    )
    require(manifest.get("rows") == EXPECTED_MODEL_ROWS, "binary build row drift")
    _all_gates(manifest, "binary build")
    require(
        (manifest.get("gates") or {}).get("all_aots_present_and_hash_valid") is True,
        "binary build was not finalized with full AOT hash verification",
    )
    source_binding = (manifest.get("artifacts") or {}).get(
        "source_preparation_manifest"
    ) or {}
    require(
        require_sha256(source_binding.get("sha256"), "binary source preparation")
        == source["hashes"]["source_preparation_manifest.json"],
        "binary build does not bind the supplied source preparation",
    )
    artifact_names = {
        "aot_manifest": "aot_manifest.jsonl",
        "pool_reconciliation_private": "pool_reconciliation_private.jsonl",
        "pool_extractor_manifest": "pool_extractor_manifest.json",
        "dart_toolchain_manifest": "dart_toolchain_manifest.json",
    }
    for field, name in artifact_names.items():
        binding = (manifest.get("artifacts") or {}).get(field) or {}
        require(
            require_sha256(binding.get("sha256"), f"binary artifact {field}")
            == hashes[name],
            f"binary artifact binding drift: {field}",
        )
    aot_rows = read_jsonl(root / "aot_manifest.jsonl")
    reconciliation = read_jsonl(root / "pool_reconciliation_private.jsonl")
    require(
        len(aot_rows) == len(reconciliation) == EXPECTED_MODEL_ROWS,
        "binary AOT/reconciliation count drift",
    )
    aot_bytes = 0
    split_codecs: dict[str, list[dict[str, Any]]] = {}
    ordered_ids: list[str] = []
    for split, expected in EXPECTED_SPLITS.items():
        codec_rows = read_jsonl(root / f"prepared/{split}_codec_private.jsonl")
        require(len(codec_rows) == expected, f"binary {split} codec count drift")
        source_ids = _task_ids(source["splits"][split]["build"], f"source {split}")
        codec_ids = _task_ids(codec_rows, f"binary {split}")
        require(codec_ids == source_ids, f"binary/source {split} order drift")
        for position, row in enumerate(codec_rows):
            require(
                row.get("split") == split
                and row.get("split_row") == position
                and row.get("function") == TARGET_FUNCTION,
                f"binary {split}:{position}: split/target drift",
            )
        split_manifest = read_json(root / f"manifests/{split}.json")
        _all_gates(split_manifest, f"binary {split} manifest")
        counts = split_manifest.get("counts") or {}
        require(
            counts.get("built_or_resumed") == expected and counts.get("failed") == 0,
            f"binary {split} manifest count/failure drift",
        )
        require(
            (manifest.get("splits") or {}).get(split, {}).get("rows") == expected,
            f"binary umbrella {split} count drift",
        )
        split_codecs[split] = codec_rows
        ordered_ids.extend(codec_ids)
    aot_ids = _task_ids(aot_rows, "AOT manifest")
    recon_ids = _task_ids(reconciliation, "binary pool reconciliation")
    require(aot_ids == ordered_ids, "AOT manifest task order drift")
    require(recon_ids == ordered_ids, "binary reconciliation task order drift")
    for index, row in enumerate(aot_rows):
        digest = require_sha256(row.get("aot_sha256"), f"AOT row {index}")
        size = row.get("aot_size_bytes")
        require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"AOT row {index}: bad size")
        relative = str(row.get("aot_path") or "").replace("\\", "/")
        require(relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, f"AOT row {index}: unsafe path")
        require(bool(digest), f"AOT row {index}: missing digest")
        aot_bytes += size
    extractor = read_json(root / "pool_extractor_manifest.json")
    toolchain = read_json(root / "dart_toolchain_manifest.json")
    require(extractor.get("target_function") == TARGET_FUNCTION, "pool extractor target drift")
    require(extractor.get("source_blind_after_aot") is True, "pool extractor is not source blind")
    return {
        "manifest": manifest,
        "hashes": hashes,
        "aot_rows": aot_rows,
        "aot_bytes": aot_bytes,
        "reconciliation": reconciliation,
        "split_codecs": split_codecs,
        "ordered_ids": ordered_ids,
        "extractor": extractor,
        "toolchain": toolchain,
    }


def validate_compact(root: Path, binary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {item.name for item in root.iterdir() if item.is_file()}
    expected = set(COMPACT_REQUIRED) | {"SHA256SUMS.txt"}
    if "aot_manifest_bundle.json" in observed:
        expected.add("aot_manifest_bundle.json")
    require(observed == expected, f"compact file-set drift: {sorted(observed)}")
    require(
        not any(item.is_dir() or item.is_symlink() for item in root.iterdir()),
        "compact root contains nested directories or symlinks",
    )
    hashes = verify_checksum(root, "SHA256SUMS.txt", expected - {"SHA256SUMS.txt"})
    contract = read_json(root / "compact_contract.json")
    codebook = read_json(root / "codebook.json")
    preflight = read_json(root / "preflight_report.json")
    require(contract.get("schema") == "direct-compact-causal-v3", "compact contract is not v3")
    require(codebook.get("schema") == "compact-qwen-v3-codebook", "compact codebook schema drift")
    require(contract.get("target_function") == TARGET_FUNCTION, "compact target drift")
    require(str(contract.get("target_language") or "").lower() == "dart", "compact language drift")
    require(contract.get("target_architecture") == "x86_64", "compact architecture drift")
    require(contract.get("max_source_tokens") == 9000, "compact source-token gate drift")
    require(codebook.get("fit_scope") == "train_only", "codebook was not fit train-only")
    require(codebook.get("measure_excluded_from_fit") is True, "measure rows entered codebook fit")
    require(codebook.get("fit_retained") == EXPECTED_ROLES["fit"], "codebook fit count drift")
    require(
        contract.get("codebook_sha256") == hashes["codebook.json"],
        "compact contract/codebook hash drift",
    )
    for field in ("codec_sha256", "tokenizer_json_sha256"):
        require_sha256(contract.get(field), f"compact contract {field}")
    require(preflight.get("passed") is True, "compact preflight did not pass")
    require(preflight.get("rows_retained") == EXPECTED_MODEL_ROWS, "compact retained-row drift")
    require(preflight.get("rows_by_role") == EXPECTED_ROLES, "compact role-count drift")
    require(preflight.get("quarantined") == 0, "compact release contains quarantined rows")
    require(preflight.get("failures_count") == 0, "compact release contains failures")
    tokens = preflight.get("tokens") or {}
    require(
        tokens.get("limit") == 9000
        and tokens.get("rows_over_limit") == 0
        and isinstance(tokens.get("max"), int)
        and tokens.get("max") <= 9000,
        "compact 9,000-token gate failed",
    )
    lossless = preflight.get("lossless_invariants") or {}
    for field in (
        "exact_graph_and_pool_roundtrip_rows",
        "compact_id_stream_roundtrip_rows",
        "dfg_regenerated_and_matched_rows",
        "source_blind_pool_rows",
        "public_private_rows_bijective_and_hash_bound",
    ):
        require(lossless.get(field) == EXPECTED_MODEL_ROWS, f"compact lossless count drift: {field}")
    require(lossless.get("unknown_tokens") == 0, "compact unknown tokens present")
    require(lossless.get("truncated_rows") == 0, "compact truncation present")
    require(lossless.get("call_edges_encoded_explicitly") is True, "call-edge atom gate failed")
    require(
        hashes["quarantine.jsonl"] == sha256_bytes(b"")
        and hashes["failures.jsonl"] == sha256_bytes(b""),
        "compact failure/quarantine files are not empty",
    )
    public = read_jsonl(root / "compact_model_inputs.jsonl")
    alignment = read_jsonl(root / "alignment_private.jsonl")
    reconciliation = read_jsonl(root / "pool_reconciliation_private.jsonl")
    require(
        len(public) == len(alignment) == len(reconciliation) == EXPECTED_MODEL_ROWS,
        "compact public/alignment/reconciliation row drift",
    )
    expected_ids = binary["ordered_ids"]
    observed_ids: list[str] = []
    role_counts: collections.Counter[str] = collections.Counter()
    for index, (model_row, sidecar) in enumerate(zip(public, alignment, strict=True)):
        require(set(model_row) == PUBLIC_FIELDS, f"compact public row {index}: schema drift")
        require(
            model_row.get("compact_codec_sha256") == contract.get("codec_sha256")
            and model_row.get("compact_codebook_sha256") == hashes["codebook.json"]
            and model_row.get("compact_tokenizer_sha256") == contract.get("tokenizer_json_sha256"),
            f"compact public row {index}: contract binding drift",
        )
        ids = model_row.get("compact_input_ids")
        require(
            isinstance(ids, list)
            and bool(ids)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in ids)
            and len(ids) <= 9000,
            f"compact public row {index}: invalid token IDs",
        )
        require(sidecar.get("model_row") == index, f"alignment row {index}: model-row drift")
        require(
            sidecar.get("model_row_sha256") == canonical_sha256(model_row),
            f"alignment row {index}: public-row hash drift",
        )
        role = str(sidecar.get("role") or "")
        require(role in EXPECTED_ROLES, f"alignment row {index}: invalid role")
        role_counts[role] += 1
        task_id = str(sidecar.get("task_id") or "")
        require(task_id == expected_ids[index], f"alignment row {index}: task order drift")
        observed_ids.append(task_id)
        require(
            not (set(_walk_keys(sidecar)) & FORBIDDEN_ALIGNMENT_KEYS),
            f"alignment row {index}: source/payload field leaked",
        )
        pool = sidecar.get("pool_metadata") or {}
        require(
            pool.get("schema") == "dart-aot-target-pool-alignment-v1"
            and pool.get("source_blind") is True
            and pool.get("target_function") == TARGET_FUNCTION,
            f"alignment row {index}: pool metadata drift",
        )
    require(dict(role_counts) == EXPECTED_ROLES, "alignment role-count drift")
    require(len(set(observed_ids)) == EXPECTED_MODEL_ROWS, "alignment duplicate task IDs")
    for index, row in enumerate(reconciliation):
        require(row.get("status") == "included", f"compact reconciliation row {index}: not included")
        require(row.get("model_row") == index, f"compact reconciliation row {index}: index drift")
        require(row.get("task_id") == expected_ids[index], f"compact reconciliation row {index}: task drift")
        gates = row.get("gates") or {}
        require(gates and all(value is True for value in gates.values()), f"compact reconciliation row {index}: gate failure")
    require(
        contract.get("pool_reconciliation_manifest_sha256")
        == hashes["pool_reconciliation_private.jsonl"],
        "compact contract/reconciliation hash drift",
    )
    require(
        contract.get("pool_extractor_sha256")
        == binary["hashes"]["pool_extractor_manifest.json"],
        "compact/binary pool extractor binding drift",
    )
    require(
        contract.get("dart_toolchain_manifest_sha256")
        == binary["hashes"]["dart_toolchain_manifest.json"],
        "compact/binary Dart toolchain binding drift",
    )
    binary_manifest_sha = binary["hashes"]["binary_build_manifest.json"]
    aot_contract_sha = require_sha256(contract.get("aot_manifest_sha256"), "compact AOT manifest")
    if "aot_manifest_bundle.json" in hashes:
        bundle = read_json(root / "aot_manifest_bundle.json")
        require(aot_contract_sha == hashes["aot_manifest_bundle.json"], "compact AOT bundle hash drift")
        bindings = bundle.get("manifests")
        require(isinstance(bindings, list) and bool(bindings), "compact AOT bundle is empty")
        observed_hashes = {
            require_sha256(item.get("sha256"), "compact AOT bundle entry")
            for item in bindings
        }
        accepted_hashes = {
            binary_manifest_sha,
            binary["hashes"]["manifests/train.json"],
            binary["hashes"]["manifests/dev.json"],
        }
        require(observed_hashes <= accepted_hashes, "compact AOT bundle contains an unsealed manifest")
        require(
            binary_manifest_sha in observed_hashes
            or observed_hashes
            == {
                binary["hashes"]["manifests/train.json"],
                binary["hashes"]["manifests/dev.json"],
            },
            "compact AOT bundle does not cover the finalized binary build",
        )
    else:
        require(aot_contract_sha == binary_manifest_sha, "compact does not bind finalized binary-build seal")
    return {
        "hashes": hashes,
        "contract": contract,
        "codebook": codebook,
        "preflight": preflight,
        "public": public,
        "alignment": alignment,
        "reconciliation": reconciliation,
        "task_ids": observed_ids,
    }


def validate_supervised(
    root: Path,
    source: Mapping[str, Any],
    compact: Mapping[str, Any],
) -> dict[str, Any]:
    observed = {item.name for item in root.iterdir() if item.is_file()}
    require(observed == set(SUPERVISED_FILES), f"supervised file-set drift: {sorted(observed)}")
    require(
        not any(item.is_dir() or item.is_symlink() for item in root.iterdir()),
        "supervised root contains nested directories or symlinks",
    )
    public_sha = compact["hashes"]["compact_model_inputs.jsonl"]
    alignment_sha = compact["hashes"]["alignment_private.jsonl"]
    contract_sha = compact["hashes"]["compact_contract.json"]
    results: dict[str, Any] = {}
    role_for_split = {"train": "fit", "dev": "measure"}
    compact_by_role: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        role: [
            (public, alignment)
            for public, alignment in zip(
                compact["public"], compact["alignment"], strict=True
            )
            if alignment.get("role") == role
        ]
        for role in EXPECTED_ROLES
    }
    for split, expected in EXPECTED_SPLITS.items():
        rows = read_jsonl(root / f"{split}.jsonl")
        seal = read_json(root / f"{split}.join_seal.json")
        require(len(rows) == expected, f"supervised {split} count drift")
        require(seal.get("schema") == "compact-public-private-join-seal-v2", f"supervised {split}: seal schema drift")
        require(seal.get("contract_schema") == "direct-compact-causal-v3", f"supervised {split}: contract schema drift")
        require(
            seal.get("rows") == expected
            and seal.get("source_rows") == EXPECTED_MODEL_ROWS
            and seal.get("selected_role") == role_for_split[split]
            and seal.get("skipped_rows") == EXPECTED_MODEL_ROWS - expected,
            f"supervised {split}: row/role seal drift",
        )
        require(
            seal.get("public_sha256") == public_sha
            and seal.get("alignment_sha256") == alignment_sha
            and seal.get("contract_sha256") == contract_sha,
            f"supervised {split}: compact binding drift",
        )
        private_name = f"prepared/{split}_private_labels.jsonl"
        require(
            seal.get("private_sha256") == source["hashes"][private_name],
            f"supervised {split}: private-label binding drift",
        )
        require(
            seal.get("output_sha256") == sha256_file(root / f"{split}.jsonl")
            and seal.get("output_size_bytes") == (root / f"{split}.jsonl").stat().st_size,
            f"supervised {split}: output binding drift",
        )
        bijection = seal.get("private_bijection") or {}
        require(
            bijection.get("required") is True
            and bijection.get("verified") is True
            and bijection.get("private_rows") == expected
            and bijection.get("unused_private_rows") == 0,
            f"supervised {split}: private bijection failed",
        )
        require(
            set(seal.get("model_visible_fields") or []) == SUPERVISED_FIELDS,
            f"supervised {split}: visible-field declaration drift",
        )
        pool = seal.get("pool_metadata") or {}
        require(
            pool.get("schema") == "dart-aot-target-pool-alignment-v1"
            and pool.get("rows") == expected
            and pool.get("source_blind_rows") == expected
            and pool.get("target_function") == TARGET_FUNCTION,
            f"supervised {split}: pool metadata seal drift",
        )
        labels = source["splits"][split]["labels"]
        selected = compact_by_role[role_for_split[split]]
        require(len(selected) == len(labels) == len(rows), f"supervised {split}: selection drift")
        for index, (row, label, (public, alignment)) in enumerate(
            zip(rows, labels, selected, strict=True)
        ):
            require(set(row) == SUPERVISED_FIELDS, f"supervised {split}:{index}: schema drift")
            require(row.get("function") == TARGET_FUNCTION, f"supervised {split}:{index}: target drift")
            require(str(row.get("lang") or "").lower() == "dart", f"supervised {split}:{index}: language drift")
            require(
                row.get("dart_source")
                == private_target(label, f"supervised {split}:{index}"),
                f"supervised {split}:{index}: label drift",
            )
            for field in PUBLIC_FIELDS:
                require(row.get(field) == public.get(field), f"supervised {split}:{index}: compact input drift")
            require(alignment.get("task_id") == label.get("task_id"), f"supervised {split}:{index}: task alignment drift")
        mapping = seal.get("mapping")
        require(isinstance(mapping, list) and len(mapping) == expected, f"supervised {split}: mapping drift")
        expected_mapping = []
        for private_index, (_public, alignment) in enumerate(selected):
            model_row = int(alignment["model_row"])
            task_id = str(alignment["task_id"])
            expected_mapping.append(
                {
                    "public_line": model_row,
                    "alignment_line": model_row,
                    "private_line": private_index,
                    "identity_sha256": sha256_bytes(task_id.encode("utf-8")),
                }
            )
        require(mapping == expected_mapping, f"supervised {split}: mapping content drift")
        require(
            seal.get("mapping_sha256")
            == sha256_bytes(
                json.dumps(
                    expected_mapping, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ),
            f"supervised {split}: mapping hash drift",
        )
        results[split] = {"rows": rows, "seal": seal}
    return results


def validate_audits(root: Path, compact: Mapping[str, Any]) -> dict[str, Any]:
    observed = {item.name for item in root.iterdir() if item.is_file()}
    expected = set(AUDIT_FILES) | {"SHA256SUMS.txt"}
    require(observed == expected, f"audit file-set drift: {sorted(observed)}")
    require(
        not any(item.is_dir() or item.is_symlink() for item in root.iterdir()),
        "audit root contains nested directories or symlinks",
    )
    hashes = verify_checksum(root, "SHA256SUMS.txt", AUDIT_FILES)
    reports: dict[str, dict[str, Any]] = {}
    for name in sorted(AUDIT_REPORTS):
        report = read_json(root / name)
        require(
            report.get("schema") == "compact-qwen-v3-generalization-audit-v1",
            f"{name}: audit schema drift",
        )
        require(report.get("passed") is True, f"{name}: audit did not pass")
        require(
            report.get("bundle_contract_sha256")
            == compact["hashes"]["compact_contract.json"],
            f"{name}: compact-contract binding drift",
        )
        require(
            report.get("codebook_sha256") == compact["hashes"]["codebook.json"],
            f"{name}: codebook binding drift",
        )
        reports[name] = report
    dev = reports["dev_fallback_audit.json"]
    require(
        dev.get("audit") == "sealed_v3_dev_instruction_codebook_fallback"
        and dev.get("scope")
        == "full-v3-release-alignment-derived-instruction-fallback",
        "dev audit kind/scope drift",
    )
    dev_coverage = dev.get("coverage") or {}
    require(
        dev_coverage.get("rows") == EXPECTED_SPLITS["dev"],
        "dev audit row drift",
    )
    require(
        (dev.get("cross_checks") or {})
        and all(value is True for value in dev["cross_checks"].values()),
        "dev audit cross-check failure",
    )
    preflight_measure = (compact["preflight"].get("fallback_by_role") or {}).get(
        "measure"
    ) or {}
    for field in ("instructions", "fallback"):
        require(
            dev_coverage.get(field) == preflight_measure.get(field),
            f"dev audit/preflight {field} drift",
        )

    topup = reports["topup_family_source_pool_fallback_audit.json"]
    require(
        topup.get("audit")
        == "sealed_v3_topup_family_and_source_pool_instruction_fallback"
        and topup.get("scope")
        == "full-v3-release-alignment-derived-instruction-fallback",
        "top-up audit kind/scope drift",
    )
    require(
        (topup.get("cross_checks") or {})
        and all(value is True for value in topup["cross_checks"].values()),
        "top-up audit cross-check failure",
    )
    family_rows = sum(
        int((value or {}).get("rows", -1))
        for value in (topup.get("coverage_by_family") or {}).values()
    )
    require(
        family_rows == (topup.get("coverage") or {}).get("rows"),
        "top-up family audit partition drift",
    )

    human = reports["scrubbed_humaneval_instruction_codebook_audit.json"]
    require(
        human.get("audit")
        == "scrubbed_humaneval_instruction_codebook_coverage",
        "HumanEval audit kind drift",
    )
    require(human.get("rows") == 154, "HumanEval audit row drift")
    require(
        human.get("exact_canonical_and_dfg_roundtrip_rows") == 154,
        "HumanEval exact graph/DFG round-trip count drift",
    )
    require(
        (human.get("fallback_representation") or {}).get("reversible") is True,
        "HumanEval fallback is not reversible",
    )
    full_measurement = human.get("full_v3_source_token_measurement")
    require(
        isinstance(full_measurement, bool),
        "HumanEval audit must explicitly declare full_v3_source_token_measurement",
    )
    if full_measurement is False:
        require(
            human.get("scope") == "instruction_codebook_coverage_only",
            "partial HumanEval audit must use instruction_codebook_coverage_only scope",
        )
        # Instruction coverage has no v3 pool projection.  A source-token or
        # 9K claim in this mode would silently turn a partial audit into a
        # false full-stream measurement.
        forbidden_claim_fields = {
            "tokens",
            "source_tokens",
            "max_source_tokens",
            "rows_over_limit",
            "truncated_rows",
        }
        require(
            not (set(human) & forbidden_claim_fields),
            "partial HumanEval audit makes a forbidden full-token/9K claim",
        )
        non_claims = human.get("non_claims") or {}
        require(
            non_claims.get("full_v3_binary_pool_stream_was_available") is False
            and non_claims.get("full_v3_source_token_count_was_measured") is False
            and non_claims.get("human_eval_9000_token_gate_was_evaluated") is False,
            "partial HumanEval audit does not explicitly disclaim full-v3/9K measurement",
        )
        require(
            non_claims.get("human_eval_truncation_count") is None,
            "partial HumanEval audit claims a truncation count",
        )
    else:
        require(
            human.get("scope") == "full_v3_source_token_measurement",
            "full HumanEval audit must use full_v3_source_token_measurement scope",
        )
        human_tokens = human.get("tokens")
        require(
            isinstance(human_tokens, Mapping)
            and human_tokens.get("limit") == 9000
            and human_tokens.get("rows_over_limit") == 0
            and isinstance(human_tokens.get("max"), int)
            and human_tokens.get("max") <= 9000,
            "full-v3 HumanEval audit lacks a real 9,000-token gate",
        )
    seal = read_json(root / "generalization_audit_seal.json")
    require(
        seal.get("schema") == "compact-qwen-v3-generalization-audit-seal-v1",
        "generalization seal schema drift",
    )
    require(
        seal.get("all_passed") is True or seal.get("passed") is True,
        "generalization umbrella seal did not pass",
    )
    entries = seal.get("reports")
    require(isinstance(entries, Mapping), "generalization seal has no report bindings")
    for name in AUDIT_REPORTS:
        entry = entries.get(name) or {}
        require(entry.get("passed") is True, f"generalization seal does not pass {name}")
        require(entry.get("sha256") == hashes[name], f"generalization seal hash drift: {name}")
    return {"hashes": hashes, "reports": reports, "seal": seal}


def _verify_v2_root_binding(v2_root: Path) -> dict[str, str]:
    checksum = v2_root / "SHA256SUMS.txt"
    require(checksum.is_file() and not checksum.is_symlink(), "v2 root checksum missing/unsafe")
    found: dict[str, str] = {}
    for line_number, raw in enumerate(checksum.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split("  ", 1)
        require(len(parts) == 2, f"v2 root checksum line {line_number}: malformed")
        digest = require_sha256(parts[0], f"v2 root checksum line {line_number}")
        name = _safe_relative(parts[1], f"v2 root checksum line {line_number}")
        require(name not in found, f"v2 root checksum duplicate: {name}")
        found[name] = digest
    for name in (
        "release_manifest.json",
        "prepared/SHA256SUMS.txt",
        "prepared/preparation_manifest.json",
        "prepared/reconciliation.jsonl",
    ):
        require(name in found, f"v2 root checksum omits {name}")
        target = v2_root / name
        require(target.is_file() and not target.is_symlink(), f"v2 inherited file missing/unsafe: {name}")
        require(sha256_file(target) == found[name], f"v2 root hash drift: {name}")
    return found


def validate_v2_membership(v2_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    root_hashes = _verify_v2_root_binding(v2_root)
    prepared = v2_root / "prepared"
    prep_hashes = verify_checksum(prepared, "SHA256SUMS.txt", V2_PREP_FILES)
    manifest = read_json(prepared / "preparation_manifest.json")
    require(manifest.get("schema") == "phase0-compact-qwen-v2-preparation", "v2 preparation schema drift")
    _all_gates(manifest, "v2 preparation")
    counts = manifest.get("counts") or {}
    require(counts.get("input") == EXPECTED_CANONICAL_ROWS, "v2 canonical input count drift")
    require(counts.get("statuses") == EXPECTED_STATUS_COUNTS, "v2 status-count manifest drift")
    rows = read_jsonl(prepared / "reconciliation.jsonl")
    require(len(rows) == EXPECTED_CANONICAL_ROWS, "v2 reconciliation row drift")
    statuses = collections.Counter(str(row.get("status") or "") for row in rows)
    require(dict(statuses) == EXPECTED_STATUS_COUNTS, "v2 reconciliation status drift")
    task_ids = _task_ids(rows, "v2 reconciliation")
    require(
        [row.get("input_line") for row in rows] == list(range(1, EXPECTED_CANONICAL_ROWS + 1)),
        "v2 reconciliation is not in canonical input order",
    )
    included = {
        "train": [row for row in rows if row.get("status") == "included-train"],
        "dev": [row for row in rows if row.get("status") == "included-dev"],
    }
    for split, expected in EXPECTED_SPLITS.items():
        require(len(included[split]) == expected, f"v2 inherited {split} count drift")
        inherited_ids = [str(row.get("task_id")) for row in included[split]]
        source_ids = [str(row.get("task_id")) for row in source["splits"][split]["build"]]
        # v2 reconciliation is canonical-corpus order, while source preparation
        # is Phase-0 split order.  Membership must be exact; ordering is proved
        # separately by source/binary/compact joins.
        require(set(inherited_ids) == set(source_ids), f"v2/source {split} membership drift")
    release_manifest = read_json(v2_root / "release_manifest.json")
    require(
        release_manifest.get("schema") == "compact-qwen-phase0-s44-v2-release-seal-v1",
        "v2 release schema drift",
    )
    _all_gates(release_manifest, "v2 release")
    require(
        source["manifest"].get("source_release_manifest_sha256")
        == root_hashes["release_manifest.json"],
        "source preparation does not inherit the supplied canonical v2 release",
    )
    return {
        "root_hashes": root_hashes,
        "prep_hashes": prep_hashes,
        "manifest": manifest,
        "release_manifest": release_manifest,
        "rows": rows,
        "task_ids": task_ids,
    }


def _copy_files(source: Path, destination: Path, names: Iterable[str]) -> None:
    for name in sorted(names):
        src = source / name
        dst = destination / name
        require(src.is_file() and not src.is_symlink(), f"copy source missing/unsafe: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        require(sha256_file(dst) == sha256_file(src), f"post-copy hash drift: {name}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_checksum(root: Path, name: str, relative_names: Iterable[str]) -> None:
    lines = []
    for relative in sorted(relative_names):
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"checksum target missing/unsafe: {relative}")
        lines.append(f"{sha256_file(path)}  {relative}\n")
    (root / name).write_text("".join(lines), encoding="ascii", newline="\n")


def _file_record(path: Path, release: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(release).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        record["rows"] = sum(1 for line in path.open("rb") if line.strip())
    return record


def _readme(manifest: Mapping[str, Any]) -> str:
    return f"""# Encoder-free compact-Qwen Phase-0 s44 v3

This is the fail-closed, path-independent release of the direct compact-Qwen
v3 representation.  It preserves the canonical Phase-0 assignment exactly:
2,951 supervised training rows and 326 supervised development rows.  The
inherited reconciliation accounts for all 3,306 canonical input rows.

The model sees only the strict four-field compact input plus the supervised
Dart target after the private join.  Task IDs, family/source-pool metadata,
assembly, graphs, binary-pool receipts, tests, and join mappings remain in
private sidecars or seals.  The target function is uniformly `candidate`.

The compact stream is lossless over the contract's scrubbed graph-plus-pool
domain: canonical graph/pool and token-ID round-trips passed for all 3,277
model rows, DFG edges were regenerated and matched, call edges are explicit,
and no row exceeds the 9,000-source-token gate.

## AOT payload policy

The multi-gigabyte AOT/ELF payload is intentionally **not shipped**.  The
release includes the finalized binary-build seal and its complete 3,277-row
AOT manifest, which bind every external AOT by SHA-256 and byte size.  The
external payload totals {manifest['external_aot_payload']['size_bytes']} bytes.

## Verification

Verify `SHA256SUMS.txt` recursively.  `release_manifest.json` records every
shipped payload with a release-relative path, SHA-256, size, and JSONL count.
Nested producer checksums remain included.  No absolute host path or timestamp
is used by the umbrella manifest.
"""


def _validate_output_file_set(release: Path) -> None:
    expected_dirs = {
        "source_preparation",
        "binary_build",
        "compact",
        "supervised",
        "audits",
        "canonical_membership",
    }
    require(
        {item.name for item in release.iterdir() if item.is_dir()} == expected_dirs,
        "packaged release directory-set drift",
    )
    require(
        {item.name for item in release.iterdir() if item.is_file()}
        == {"README.md", "release_manifest.json", "SHA256SUMS.txt"},
        "packaged release root file-set drift",
    )
    require(not any(item.is_symlink() for item in release.rglob("*")), "release contains symlink")
    forbidden = [
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.suffix.lower() in AOT_SUFFIXES
    ]
    require(not forbidden, f"release improperly ships AOT/binary payloads: {forbidden[:5]}")


def package_release(
    *,
    source_prep_root: Path,
    binary_build_root: Path,
    compact_root: Path,
    supervised_root: Path,
    audits_root: Path,
    v2_release: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate all producers, then atomically materialize one v3 release."""
    roots = [
        source_prep_root,
        binary_build_root,
        compact_root,
        supervised_root,
        audits_root,
        v2_release,
    ]
    for root in roots:
        require(root.is_dir() and not root.is_symlink(), f"missing/unsafe input root: {root}")
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    require(
        all(
            output_dir != root.resolve()
            and output_dir not in root.resolve().parents
            and root.resolve() not in output_dir.parents
            for root in roots
        ),
        "output directory aliases/contains an input root",
    )

    # Validate everything before writing even the staging tree.
    source = validate_source_preparation(source_prep_root.resolve())
    binary = validate_binary_build(binary_build_root.resolve(), source)
    compact = validate_compact(compact_root.resolve(), binary)
    supervised = validate_supervised(supervised_root.resolve(), source, compact)
    audits = validate_audits(audits_root.resolve(), compact)
    inherited = validate_v2_membership(v2_release.resolve(), source)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        source_dst = staging / "source_preparation"
        binary_dst = staging / "binary_build"
        compact_dst = staging / "compact"
        supervised_dst = staging / "supervised"
        audits_dst = staging / "audits"
        membership_dst = staging / "canonical_membership"

        _copy_files(
            source_prep_root,
            source_dst,
            SOURCE_FILES | {"SOURCE_SHA256SUMS.txt"},
        )
        _copy_files(
            binary_build_root,
            binary_dst,
            BINARY_FILES | {"BINARY_BUILD_SHA256SUMS.txt"},
        )
        compact_names = set(COMPACT_REQUIRED) | {"SHA256SUMS.txt"}
        if (compact_root / "aot_manifest_bundle.json").is_file():
            compact_names.add("aot_manifest_bundle.json")
        _copy_files(compact_root, compact_dst, compact_names)
        _copy_files(supervised_root, supervised_dst, SUPERVISED_FILES)
        _write_checksum(
            supervised_dst,
            "SHA256SUMS.txt",
            SUPERVISED_FILES,
        )
        _copy_files(audits_root, audits_dst, AUDIT_FILES | {"SHA256SUMS.txt"})
        _copy_files(
            v2_release / "prepared",
            membership_dst,
            {"preparation_manifest.json", "reconciliation.jsonl"},
        )
        inheritance_seal = {
            "schema": INHERITANCE_SCHEMA,
            "canonical_rows": EXPECTED_CANONICAL_ROWS,
            "status_counts": EXPECTED_STATUS_COUNTS,
            "phase0_rows": EXPECTED_MODEL_ROWS,
            "phase0_split_counts": EXPECTED_SPLITS,
            "canonical_reconciliation_sha256": inherited["prep_hashes"]["reconciliation.jsonl"],
            "v2_preparation_manifest_sha256": inherited["prep_hashes"]["preparation_manifest.json"],
            "v2_prepared_checksums_sha256": sha256_file(v2_release / "prepared/SHA256SUMS.txt"),
            "v2_release_manifest_sha256": inherited["root_hashes"]["release_manifest.json"],
            "v2_root_checksums_sha256": sha256_file(v2_release / "SHA256SUMS.txt"),
            "gates": {
                "all_3306_rows_reconciled_once": True,
                "status_counts_exact": True,
                "included_membership_matches_source_preparation": True,
                "source_preparation_binds_v2_release": True,
            },
        }
        _write_json(membership_dst / "inheritance_seal.json", inheritance_seal)
        _write_checksum(
            membership_dst,
            "SHA256SUMS.txt",
            {
                "preparation_manifest.json",
                "reconciliation.jsonl",
                "inheritance_seal.json",
            },
        )

        # The README is written before the manifest so both become ordinary
        # root-sealed payloads; no file contains its own digest.
        preliminary = {
            "external_aot_payload": {
                "rows": EXPECTED_MODEL_ROWS,
                "size_bytes": binary["aot_bytes"],
            }
        }
        (staging / "README.md").write_text(
            _readme(preliminary), encoding="utf-8", newline="\n"
        )
        payload_paths = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        contract = compact["contract"]
        preflight = compact["preflight"]
        manifest = {
            "schema": SCHEMA,
            "release": "direct_compact_phase0_s44_v3",
            "deterministic": True,
            "path_independent": True,
            "self_binding": "release_manifest.json is bound by root SHA256SUMS.txt",
            "counts": {
                "canonical_input": EXPECTED_CANONICAL_ROWS,
                "included_train": EXPECTED_SPLITS["train"],
                "included_dev": EXPECTED_SPLITS["dev"],
                "model_rows": EXPECTED_MODEL_ROWS,
                "supervised_train": len(supervised["train"]["rows"]),
                "supervised_dev": len(supervised["dev"]["rows"]),
                "canonical_statuses": EXPECTED_STATUS_COUNTS,
            },
            "contract": {
                "schema": contract["schema"],
                "sha256": compact["hashes"]["compact_contract.json"],
                "codebook_sha256": compact["hashes"]["codebook.json"],
                "codec_sha256": contract["codec_sha256"],
                "tokenizer_json_sha256": contract["tokenizer_json_sha256"],
                "target_function": TARGET_FUNCTION,
                "target_architecture": "x86_64",
                "max_source_tokens": 9000,
                "lossless_domain": contract.get("lossless_domain"),
            },
            "external_aot_payload": {
                "shipped": False,
                "policy": "external-hash-bound-only",
                "rows": EXPECTED_MODEL_ROWS,
                "size_bytes": binary["aot_bytes"],
                "aot_manifest": {
                    "path": "binary_build/aot_manifest.jsonl",
                    "sha256": binary["hashes"]["aot_manifest.jsonl"],
                },
                "binary_build_seal": {
                    "path": "binary_build/binary_build_manifest.json",
                    "sha256": binary["hashes"]["binary_build_manifest.json"],
                },
            },
            "nested_seals": {
                "source_preparation": sha256_file(source_dst / "SOURCE_SHA256SUMS.txt"),
                "binary_build": sha256_file(binary_dst / "BINARY_BUILD_SHA256SUMS.txt"),
                "compact": sha256_file(compact_dst / "SHA256SUMS.txt"),
                "supervised": sha256_file(supervised_dst / "SHA256SUMS.txt"),
                "audits": sha256_file(audits_dst / "SHA256SUMS.txt"),
                "canonical_membership": sha256_file(membership_dst / "SHA256SUMS.txt"),
            },
            "gates": {
                "canonical_3306_reconciliation_inherited_and_hash_bound": True,
                "exact_2951_train_and_326_dev_model_rows": True,
                "exact_2951_train_and_326_dev_supervised_rows": True,
                "source_binary_compact_task_order_identical": True,
                "strict_four_field_model_inputs": True,
                "strict_supervised_schema": True,
                "private_joins_bijective_and_hash_bound": True,
                "all_targets_uniform_candidate": True,
                "train_only_codebook": True,
                "all_3277_graph_pool_and_id_roundtrips": True,
                "dfg_regenerated_and_matched_edge_for_edge": True,
                "call_edges_explicit": True,
                "zero_unknown_tokens": preflight["lossless_invariants"]["unknown_tokens"] == 0,
                "zero_truncation": preflight["lossless_invariants"]["truncated_rows"] == 0,
                "all_rows_within_9000_tokens": preflight["tokens"]["rows_over_limit"] == 0,
                "generalization_audits_passed": all(
                    report["passed"] is True for report in audits["reports"].values()
                ),
                "binary_aots_previously_fully_hash_verified": binary["manifest"]["gates"]["all_aots_present_and_hash_valid"],
                "aot_payload_not_shipped": True,
                "all_nested_checksums_verified": True,
                "passed": True,
            },
            "files": [_file_record(path, staging) for path in payload_paths],
        }
        require(all(manifest["gates"].values()), "umbrella release gate failure")
        # Explicitly prove the umbrella document itself is path-independent.
        serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        for root in roots:
            require(str(root.resolve()) not in serialized, "absolute input path leaked into release manifest")
        _write_json(staging / "release_manifest.json", manifest)
        _write_checksum(
            staging,
            "SHA256SUMS.txt",
            {
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*")
                if path.is_file()
            },
        )
        _validate_output_file_set(staging)
        root_hashes = verify_checksum(
            staging,
            "SHA256SUMS.txt",
            {
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*")
                if path.is_file()
                and path.relative_to(staging).as_posix() != "SHA256SUMS.txt"
            },
        )
        require(
            root_hashes["release_manifest.json"] == sha256_file(staging / "release_manifest.json"),
            "root manifest seal drift",
        )
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-prep-root", type=Path, required=True)
    parser.add_argument("--binary-build-root", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--supervised-root", type=Path, required=True)
    parser.add_argument("--audits-root", type=Path, required=True)
    parser.add_argument("--v2-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = package_release(
            source_prep_root=args.source_prep_root,
            binary_build_root=args.binary_build_root,
            compact_root=args.compact_root,
            supervised_root=args.supervised_root,
            audits_root=args.audits_root,
            v2_release=args.v2_release,
            output_dir=args.output_dir,
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise SystemExit(f"REFUSED TO PACKAGE: {error}") from error
    result = {
        "release": str(args.output_dir.resolve()),
        "manifest_sha256": sha256_file(args.output_dir / "release_manifest.json"),
        "checksums_sha256": sha256_file(args.output_dir / "SHA256SUMS.txt"),
        "model_rows": manifest["counts"]["model_rows"],
        "aot_payload_shipped": manifest["external_aot_payload"]["shipped"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
