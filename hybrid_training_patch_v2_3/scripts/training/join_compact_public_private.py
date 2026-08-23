#!/usr/bin/env python3
"""Join compact public inputs to private labels through a sealed mapping.

The emitted JSONL contains no source task ID, line number, tests, signatures,
assembly, CFG, or join key.  Those remain only in the separately written seal.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V2,
    CONTRACT_SCHEMA_V3,
    JOIN_SEAL_SCHEMA_V1,
    JOIN_SEAL_SCHEMA_V2,
    POOL_ALIGNMENT_SCHEMA_V1,
    DirectCompactContract,
    sha256_file,
    validate_v3_pool_alignment_metadata,
)


MODEL_PUBLIC_FIELDS = frozenset(
    {
        "compact_input_ids",
        "compact_codec_sha256",
        "compact_codebook_sha256",
        "compact_tokenizer_sha256",
    }
)
NEUTRAL_FUNCTION_RE = re.compile(r"^(?:candidate|fn\d+)$")
PRIVATE_METADATA_FIELDS = ("family", "source_pool", "extractor_route")
STRICT_PHASE0_SCHEMAS = frozenset((CONTRACT_SCHEMA_V2, CONTRACT_SCHEMA_V3))


def _identity(row: Mapping[str, Any], path: Path, index: int) -> str:
    value = row.get("task_id") or row.get("id")
    if value in (None, ""):
        raise ValueError(f"{path}: row {index + 1} has no task identity")
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: row {index + 1} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: contains no rows")
    return rows


def _private_target(row: Mapping[str, Any], identity: str) -> str:
    value = row.get("supervised_target") or row.get("dart_source") or row.get("source")
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"private row {identity!r} has no target source")
    return result


def _private_metadata_value(
    alignment: Mapping[str, Any],
    field: str,
    path: Path,
    index: int,
) -> str | None:
    """Read one audit-only metadata value without exposing it to model rows.

    The v2 codec writes flattened alignment metadata and calls its extractor
    routing field ``dfg_route``.  The nested fallback keeps this helper usable
    with pre-materialization alignment rows while rejecting ambiguous aliases.
    """

    nested = alignment.get("compact_private_metadata")
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError(
            f"{path}: row {index + 1} compact_private_metadata must be an object"
        )
    aliases = ("extractor_route", "dfg_route") if field == "extractor_route" else (field,)
    observed: list[str] = []
    for container in (alignment, nested or {}):
        for alias in aliases:
            if alias not in container or container[alias] is None:
                continue
            raw = container[alias]
            if not isinstance(raw, str):
                raise ValueError(
                    f"{path}: row {index + 1} private metadata {alias!r} "
                    "must be a string or null"
                )
            value = raw.strip()
            if value:
                observed.append(value)
    unique = sorted(set(observed))
    if len(unique) > 1:
        raise ValueError(
            f"{path}: row {index + 1} has conflicting private metadata "
            f"aliases for {field!r}: {unique}"
        )
    return unique[0] if unique else None


def build_join(
    public_path: str | Path,
    alignment_path: str | Path,
    private_path: str | Path,
    output_path: str | Path,
    seal_path: str | Path,
    contract_path: str | Path,
    role: str = "all",
    require_bijective_private: bool = False,
) -> dict[str, Any]:
    role = str(role).strip().lower()
    if role not in {"fit", "measure", "all"}:
        raise ValueError("role must be one of: fit, measure, all")
    public_path = Path(public_path).resolve()
    alignment_path = Path(alignment_path).resolve()
    private_path = Path(private_path).resolve()
    output_path = Path(output_path).resolve()
    seal_path = Path(seal_path).resolve()
    contract_path = Path(contract_path).resolve()
    contract = DirectCompactContract.load(contract_path)
    strict_private_bijection = bool(
        require_bijective_private or contract.schema in STRICT_PHASE0_SCHEMAS
    )
    public_rows = _read_jsonl(public_path)
    alignment_rows = _read_jsonl(alignment_path)
    private_rows = _read_jsonl(private_path)
    if len(public_rows) != len(alignment_rows):
        raise ValueError(
            "compact public/alignment row-count mismatch: "
            f"{len(public_rows)} != {len(alignment_rows)}"
        )

    private_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(private_rows):
        identity = _identity(row, private_path, index)
        if identity in private_by_id:
            raise ValueError(f"duplicate private identity {identity!r}")
        private_by_id[identity] = (index, row)

    output_rows: list[dict[str, Any]] = []
    sealed_mapping: list[dict[str, Any]] = []
    selected_private_metadata: list[dict[str, str | None]] = []
    private_metadata_counts: dict[str, collections.Counter[str]] = {
        field: collections.Counter() for field in PRIVATE_METADATA_FIELDS
    }
    private_metadata_missing = collections.Counter()
    selected_pool_metadata: list[dict[str, Any]] = []
    seen_alignment: set[str] = set()
    seen_public: set[str] = set()
    for public_index, (public, alignment) in enumerate(
        zip(public_rows, alignment_rows, strict=True)
    ):
        public_fields = frozenset(public)
        if public_fields != MODEL_PUBLIC_FIELDS:
            raise ValueError(
                f"{public_path}: row {public_index + 1} must contain exactly "
                f"{sorted(MODEL_PUBLIC_FIELDS)}; observed {sorted(public_fields)}"
            )
        model_row = alignment.get("model_row")
        if isinstance(model_row, bool) or not isinstance(model_row, int):
            raise ValueError(
                f"{alignment_path}: row {public_index + 1} has invalid model_row"
            )
        if model_row != public_index:
            raise ValueError(
                f"{alignment_path}: row {public_index + 1} maps model_row "
                f"{model_row}, expected {public_index}"
            )
        alignment_role = str(alignment.get("role") or "").strip().lower()
        if alignment_role not in {"fit", "measure"}:
            raise ValueError(
                f"{alignment_path}: row {public_index + 1} has invalid role "
                f"{alignment_role!r}"
            )
        # Validate every strict public row before selecting a role so corruption
        # in the combined codec artifact cannot hide in the skipped partition.
        identity = _identity(alignment, alignment_path, public_index)
        if identity in seen_alignment:
            raise ValueError(f"duplicate alignment identity {identity!r}")
        seen_alignment.add(identity)
        compact_ids = contract.validate_row(public, f"public:{identity}")
        pool_metadata = (
            validate_v3_pool_alignment_metadata(
                alignment, f"alignment row {identity!r}"
            )
            if contract.schema == CONTRACT_SCHEMA_V3
            else None
        )
        if role != "all" and alignment_role != role:
            continue
        if identity in seen_public:
            raise ValueError(f"duplicate public identity {identity!r}")
        seen_public.add(identity)
        if identity not in private_by_id:
            raise ValueError(f"public identity {identity!r} has no private label")
        private_index, private = private_by_id[identity]
        function_name = str(private.get("function") or "").strip()
        if not NEUTRAL_FUNCTION_RE.fullmatch(function_name):
            raise ValueError(
                f"private row {identity!r} exposes non-neutral function name "
                f"{function_name!r}"
            )
        if function_name != contract.target_function:
            raise ValueError(
                f"private row {identity!r} function {function_name!r} does not "
                f"match contract target {contract.target_function!r}"
            )
        private_language = str(private.get("language") or private.get("lang") or "").strip()
        if private_language and private_language.lower() != contract.target_language.lower():
            raise ValueError(
                f"private row {identity!r} language {private_language!r} does not "
                f"match contract target {contract.target_language!r}"
            )
        identity_sha = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        metadata_projection: dict[str, str | None] = {}
        for field in PRIVATE_METADATA_FIELDS:
            value = _private_metadata_value(
                alignment, field, alignment_path, public_index
            )
            metadata_projection[field] = value
            if value is None:
                private_metadata_missing[field] += 1
            else:
                private_metadata_counts[field][value] += 1
        private_family = private.get("family")
        if private_family is not None:
            if not isinstance(private_family, str) or not private_family.strip():
                raise ValueError(
                    f"private row {identity!r} family must be a non-empty string or null"
                )
            if (
                contract.schema in STRICT_PHASE0_SCHEMAS
                or metadata_projection["family"] is not None
            ) and metadata_projection["family"] != private_family.strip():
                raise ValueError(
                    f"private/alignment family mismatch for {identity!r}: "
                    f"{private_family.strip()!r} != {metadata_projection['family']!r}"
                )
        if contract.schema in STRICT_PHASE0_SCHEMAS:
            if metadata_projection["family"] is None:
                raise ValueError(
                    f"{contract.schema} alignment row {identity!r} has no "
                    "family metadata"
                )
            if metadata_projection["extractor_route"] is None:
                raise ValueError(
                    f"{contract.schema} alignment row {identity!r} has no "
                    "extractor-route metadata"
                )
            if (
                metadata_projection["family"] != "master"
                and metadata_projection["source_pool"] is None
            ):
                raise ValueError(
                    f"{contract.schema} top-up alignment row {identity!r} has no "
                    "source_pool metadata"
                )
        if contract.schema == CONTRACT_SCHEMA_V3:
            if pool_metadata is None:
                raise AssertionError("validated v3 row lost pool metadata")
            selected_pool_metadata.append(pool_metadata)
        selected_private_metadata.append(
            {"identity_sha256": identity_sha, **metadata_projection}
        )
        sealed_mapping.append(
            {
                "public_line": public_index,
                "alignment_line": public_index,
                "private_line": private_index,
                "identity_sha256": identity_sha,
            }
        )
        output_rows.append(
            {
                "lang": contract.target_language,
                "function": function_name,
                "dart_source": _private_target(private, identity),
                "compact_input_ids": compact_ids,
                "compact_codec_sha256": contract.codec_sha256,
                "compact_codebook_sha256": contract.codebook_sha256,
                "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
            }
        )

    extra_private = set(private_by_id) - seen_public
    if not output_rows:
        raise ValueError(f"role {role!r} selected no compact rows")
    if strict_private_bijection and extra_private:
        raise ValueError(
            "strict private-label bijection failed: "
            f"{len(extra_private)} private rows are not selected by role {role!r}"
        )
    mapping_bytes = json.dumps(
        sealed_mapping, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata_bytes = json.dumps(
        selected_private_metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    seal = {
        "schema": (
            JOIN_SEAL_SCHEMA_V2
            if contract.schema == CONTRACT_SCHEMA_V3
            else JOIN_SEAL_SCHEMA_V1
        ),
        "contract_schema": contract.schema,
        "rows": len(output_rows),
        "source_rows": len(public_rows),
        "selected_role": role,
        "skipped_rows": len(public_rows) - len(output_rows),
        "public_sha256": sha256_file(public_path),
        "alignment_sha256": sha256_file(alignment_path),
        "private_sha256": sha256_file(private_path),
        "contract_sha256": sha256_file(contract_path),
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "mapping": sealed_mapping,
        "unused_private_rows": len(extra_private),
        "private_bijection": {
            "required": strict_private_bijection,
            "required_by_contract": contract.schema in STRICT_PHASE0_SCHEMAS,
            "requested_explicitly": bool(require_bijective_private),
            "verified": not extra_private,
            "private_rows": len(private_rows),
            "selected_public_rows": len(output_rows),
            "selected_private_rows": len(seen_public),
            "unused_private_rows": len(extra_private),
        },
        "private_metadata_projection_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "private_metadata_counts": {
            "rows": len(selected_private_metadata),
            **{
                field: {
                    "counts": dict(sorted(private_metadata_counts[field].items())),
                    "missing_rows": int(private_metadata_missing[field]),
                }
                for field in PRIVATE_METADATA_FIELDS
            },
        },
        "model_visible_fields": sorted(output_rows[0]),
        "withheld_from_model": [
            "source task identity",
            "public/private line indices",
            "tests",
            "signatures",
            "assembly",
            "cfg",
            "edges",
            "join mapping",
            "family",
            "source_pool",
            "extractor route",
            "binary-pool receipts and sidecar metadata",
        ],
    }
    if contract.schema == CONTRACT_SCHEMA_V3:
        if len(selected_pool_metadata) != len(output_rows):
            raise AssertionError("v3 pool metadata selection lost row alignment")
        pool_projection_bytes = json.dumps(
            selected_pool_metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        seal["pool_metadata"] = {
            "schema": POOL_ALIGNMENT_SCHEMA_V1,
            "rows": len(selected_pool_metadata),
            "source_blind_rows": sum(
                metadata["source_blind"] for metadata in selected_pool_metadata
            ),
            "target_function": contract.target_function,
            "projection_sha256": hashlib.sha256(pool_projection_bytes).hexdigest(),
            "total_use_count": sum(
                metadata["use_count"] for metadata in selected_pool_metadata
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    seal_tmp = seal_path.with_suffix(seal_path.suffix + ".tmp")
    with output_tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    seal["output_sha256"] = sha256_file(output_tmp)
    seal["output_size_bytes"] = output_tmp.stat().st_size
    seal_tmp.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(output_tmp, output_path)
    os.replace(seal_tmp, seal_path)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--role", choices=["fit", "measure", "all"], default="all")
    parser.add_argument(
        "--require-bijective-private",
        action="store_true",
        help=(
            "fail unless every private-label row belongs to the selected public role; "
            "use split-specific private files with this release gate"
        ),
    )
    args = parser.parse_args()
    seal = build_join(
        args.public,
        args.alignment,
        args.private,
        args.output,
        args.seal,
        args.contract,
        args.role,
        args.require_bijective_private,
    )
    print(json.dumps({key: value for key, value in seal.items() if key != "mapping"}, indent=2))


if __name__ == "__main__":
    main()
