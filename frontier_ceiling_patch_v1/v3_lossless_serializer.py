#!/usr/bin/env python3
"""Fail-closed Opus-v3 held-out serializer for black-box frontier APIs.

The Opus v3 release does not contain readable model inputs.  Its public rows
contain private Qwen source-token IDs whose meanings are bound by a sealed
codebook, tokenizer, codec, and alignment sidecar.  This module reconstructs
the exact codec-domain value and renders it with the tokenizer-aware F2 text
grammar used by the audited frontier runner.

No model API is imported or called here.  A row is returned only after all of
the following have been proved:

* the public/alignment/tests join is positional, unique, and hash-bound;
* the selected private row is the alignment's exact source row;
* the release, contract, codebook, tokenizer, codec, base pool codec, graph
  codec, extractor routes, and pool-reconciliation artifacts match their
  seals;
* the cfgtypes-v2 codec encodes, decodes, regenerates DFG, and retokenizes to
  the exact public ``compact_input_ids``;
* the API-readable representation decodes to the exact normalized blocks,
  ordered CFG (including ``call``), binary-pool values, and pool use-sites;
* the complete system+user prompt plus a declared reserve fits the requested
  token budget.

The F2 graph grammar predates explicit call edges.  The v3 wrapper therefore
stores call edges in a byte-framed canonical-JSON prefix, including each
edge's ordinal in the complete CFG list.  Non-call edges remain in the F2 CFG
stream.  Decoding merges the two streams by ordinal and proves exact equality.
The same prefix stores the canonical binary pool and its exact use-sites.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

try:
    from frontier_f2 import (
        F2_SYSTEM_PROMPT,
        decode_f2,
        serialize_f2,
        visible_one_token_symbols,
    )
except ImportError:  # Package import.
    from .frontier_f2 import (
        F2_SYSTEM_PROMPT,
        decode_f2,
        serialize_f2,
        visible_one_token_symbols,
    )


SERIALIZER_SCHEMA = "opus-v3-api-readable-f2-v1"
PREFIX_SCHEMA = "opus-v3-lossless-prefix-v1"
ROW_SCHEMA = "opus-v3-frontier-input-v1"
MANIFEST_SCHEMA = "opus-v3-frontier-input-manifest-v1"
EXPECTED_CFG_ENCODING = "inline-source-implicit-next-fallthrough-targets-v2"
EXPECTED_LOSSLESS_DOMAIN = (
    "scrubbed-canonical-graph-v2-plus-complete-source-blind-pool-values-"
    "at-canonical-graph-retained-fixed-r15-uses-v1"
)
PUBLIC_FIELDS = frozenset(
    {
        "compact_input_ids",
        "compact_codec_sha256",
        "compact_codebook_sha256",
        "compact_tokenizer_sha256",
    }
)
PROJECTION_FIELDS = (
    "architecture",
    "dfg_route",
    "entry_blocks",
    "blocks",
    "cfg_edges",
    "binary_pool",
)
PREFIX_FIELDS = frozenset(
    {"schema", "dfg_route", "call_edges", "binary_pool"}
)
CALL_FIELDS = frozenset({"ordinal", "source", "target"})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# These are the exact artifacts used to create the 175-row Opus-v3 held-out
# evaluation.  Different artifacts must use a separately reviewed seal set.
KNOWN_OPUS_V3_175_SEALS: dict[str, str | int] = {
    "public_sha256": (
        "bad3a869bfc3373d6a1e7cf1ae42050efaf743b9a7f4c7c895e0408b212eb699"
    ),
    "alignment_sha256": (
        "17d9b1ffba286da227278cac56790aa35c784e7213df5b9e41f7cacb47ff8526"
    ),
    "tests_sha256": (
        "4b599859e466165038dea8a16abda192e994d4ab2eb9e0878fa1d9b10dcedeb0"
    ),
    "private_sha256": (
        "90a39e089a104ae30526f0b043da28a7931e6aaf59302c11ef4fda82df41d216"
    ),
    "pool_reconciliation_sha256": (
        "1d6aa5d5075ff0bcd6ac944bc2f9e31faa2c8a796888f137f6951a3b819c2a79"
    ),
    "release_manifest_sha256": (
        "5d527dbe733df1a98fe55bd3d2d7e6403d465627674b27ba20dc596129cab8f1"
    ),
    "contract_sha256": (
        "bdfc16373fd55b708a6edb8210067238faeb975bd921c86016cbe9a74ab02dda"
    ),
    "codebook_sha256": (
        "5197f9007c001686572d1efbcc672a455d29c4917538f30ba2255ea8ef7b591f"
    ),
    "codec_sha256": (
        "b30f531761fbc0497e20eea8873f787bbfe3fa8e9fbdbc5ad97c8ad78e55325e"
    ),
    "tokenizer_sha256": (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    ),
    "expected_rows": 175,
    "expected_private_rows": 2951,
    "expected_reconciliation_rows": 3277,
}

_OLD_F2_PREFIX_GUIDANCE = (
    "F2: Cn+n UTF8 prefix; strings/numbers=JSON; external J[n]=@Xn. "
)
_V3_PREFIX_GUIDANCE = (
    "F2: Cn+n UTF8 prefix is canonical JSON with dfg_route, binary_pool, "
    "and call_edges. Pool uses preserve pp_offset, typed payload, and "
    "zero-based block/instruction use_sites. Each call edge has its ordinal "
    "in the complete ordered CFG; merge it into the B2 CFG at that ordinal. "
)
if _OLD_F2_PREFIX_GUIDANCE not in F2_SYSTEM_PROMPT:
    raise RuntimeError("frontier F2 prompt contract changed unexpectedly")
V3_SYSTEM_PROMPT = F2_SYSTEM_PROMPT.replace(
    _OLD_F2_PREFIX_GUIDANCE,
    _V3_PREFIX_GUIDANCE,
    1,
)


class V3SealError(RuntimeError):
    """An Opus-v3 artifact or semantic invariant failed closed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise V3SealError(f"value_is_not_canonical_json:{error}") from error


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def canonical_json_text(value: Any) -> str:
    return canonical_bytes(value).decode("ascii")


def _plain_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise V3SealError(f"{label}_must_be_integer")
    return value


def _digest(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if SHA256_RE.fullmatch(result) is None:
        raise V3SealError(f"{label}_must_be_lowercase_sha256")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3SealError(message)


def _require_file_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise V3SealError(f"missing_{label}:{path}")
    expected = _digest(expected, f"expected_{label}_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise V3SealError(
            f"{label}_sha256_mismatch:expected={expected}:actual={actual}:path={path}"
        )
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise V3SealError(f"invalid_{label}_json:{path}:{error}") from error
    if not isinstance(value, dict):
        raise V3SealError(f"{label}_must_be_json_object:{path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise V3SealError(f"blank_{label}_row:{path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise V3SealError(
                    f"invalid_{label}_row:{path}:{line_number}:{error}"
                ) from error
            if not isinstance(row, dict):
                raise V3SealError(
                    f"non_object_{label}_row:{path}:{line_number}"
                )
            rows.append(row)
    return rows


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    _atomic_write_bytes(path, payload)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _load_codec(path: Path) -> ModuleType:
    module_name = (
        "_frontier_opus_v3_cfgtypes_"
        + sha256_text(str(path.expanduser().resolve()))[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise V3SealError(f"cannot_import_cfgtypes_codec:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise V3SealError(f"cannot_import_cfgtypes_codec:{path}:{error}") from error
    return module


def _projection(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in PROJECTION_FIELDS if key not in value]
    if missing:
        raise V3SealError(
            "canonical_projection_missing_fields:" + ",".join(missing)
        )
    return {key: value[key] for key in PROJECTION_FIELDS}


def _prefix_for_projection(
    canonical: Mapping[str, Any],
) -> tuple[str, list[dict[str, int]], list[dict[str, Any]]]:
    calls: list[dict[str, int]] = []
    non_calls: list[dict[str, Any]] = []
    for ordinal, raw_edge in enumerate(canonical["cfg_edges"]):
        edge = {
            "source": _plain_int(raw_edge.get("source"), "cfg_source"),
            "target": _plain_int(raw_edge.get("target"), "cfg_target"),
            "edge_type": str(raw_edge.get("edge_type") or ""),
        }
        if edge["edge_type"] == "call":
            calls.append(
                {
                    "ordinal": ordinal,
                    "source": edge["source"],
                    "target": edge["target"],
                }
            )
        else:
            non_calls.append(edge)
    prefix = {
        "schema": PREFIX_SCHEMA,
        "dfg_route": str(canonical["dfg_route"]),
        "call_edges": calls,
        "binary_pool": canonical["binary_pool"],
    }
    return canonical_json_text(prefix), calls, non_calls


def _merge_call_edges(
    non_calls: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    total = len(non_calls) + len(calls)
    call_by_ordinal: dict[int, dict[str, Any]] = {}
    for index, raw_call in enumerate(calls):
        if not isinstance(raw_call, Mapping) or set(raw_call) != CALL_FIELDS:
            raise V3SealError(f"malformed_call_edge:{index}")
        ordinal = _plain_int(raw_call["ordinal"], f"call_ordinal_{index}")
        source = _plain_int(raw_call["source"], f"call_source_{index}")
        target = _plain_int(raw_call["target"], f"call_target_{index}")
        if not 0 <= ordinal < total:
            raise V3SealError(f"call_ordinal_out_of_range:{ordinal}:{total}")
        if ordinal in call_by_ordinal:
            raise V3SealError(f"duplicate_call_ordinal:{ordinal}")
        call_by_ordinal[ordinal] = {
            "source": source,
            "target": target,
            "edge_type": "call",
        }
    output: list[dict[str, Any]] = []
    non_call_position = 0
    for ordinal in range(total):
        if ordinal in call_by_ordinal:
            output.append(call_by_ordinal[ordinal])
            continue
        if non_call_position >= len(non_calls):
            raise V3SealError("call_edge_ordinals_leave_missing_non_call")
        raw_edge = non_calls[non_call_position]
        non_call_position += 1
        output.append(
            {
                "source": _plain_int(
                    raw_edge.get("source"), "decoded_cfg_source"
                ),
                "target": _plain_int(
                    raw_edge.get("target"), "decoded_cfg_target"
                ),
                "edge_type": str(raw_edge.get("edge_type") or ""),
            }
        )
    if non_call_position != len(non_calls):
        raise V3SealError("call_edge_ordinals_leave_extra_non_calls")
    return output


def decode_v3_api_readable(text: str) -> dict[str, Any]:
    """Decode a v3 F2 API input to its exact codec-domain projection."""
    try:
        prefix_text, graph = decode_f2(text)
    except Exception as error:
        raise V3SealError(f"v3_f2_decode_failed:{error}") from error
    try:
        prefix = json.loads(prefix_text)
    except json.JSONDecodeError as error:
        raise V3SealError(f"invalid_v3_prefix_json:{error}") from error
    if not isinstance(prefix, dict) or set(prefix) != PREFIX_FIELDS:
        raise V3SealError("v3_prefix_schema_fields_mismatch")
    if prefix.get("schema") != PREFIX_SCHEMA:
        raise V3SealError("v3_prefix_schema_mismatch")
    if canonical_json_text(prefix) != prefix_text:
        raise V3SealError("v3_prefix_is_not_canonical_json")
    calls = prefix.get("call_edges")
    if not isinstance(calls, list):
        raise V3SealError("v3_call_edges_must_be_list")
    binary_pool = prefix.get("binary_pool")
    if not isinstance(binary_pool, dict):
        raise V3SealError("v3_binary_pool_must_be_object")
    non_calls = graph.get("cfg_edges")
    if not isinstance(non_calls, list):
        raise V3SealError("v3_f2_cfg_edges_must_be_list")
    return {
        "architecture": graph["architecture"],
        "dfg_route": str(prefix.get("dfg_route") or ""),
        "entry_blocks": graph["entry_blocks"],
        "blocks": graph["blocks"],
        "cfg_edges": _merge_call_edges(non_calls, calls),
        "binary_pool": binary_pool,
    }


def serialize_v3_api_readable(
    canonical: Mapping[str, Any],
    *,
    tokenizer: Any,
    visible_symbols: Sequence[str] | None = None,
) -> str:
    """Serialize and internally prove a lossless API-readable v3 row."""
    expected = _projection(canonical)
    prefix, _calls, non_calls = _prefix_for_projection(expected)
    f2_graph = {
        "architecture": expected["architecture"],
        "entry_blocks": expected["entry_blocks"],
        "blocks": expected["blocks"],
        "cfg_edges": non_calls,
    }
    try:
        text = serialize_f2(
            prefix,
            f2_graph,
            tokenizer=tokenizer,
            visible_symbols=visible_symbols,
        )
    except Exception as error:
        raise V3SealError(f"v3_f2_encode_failed:{error}") from error
    observed = decode_v3_api_readable(text)
    if observed != expected:
        raise V3SealError(
            "v3_api_semantic_roundtrip_mismatch:"
            f"expected={canonical_sha256(expected)}:"
            f"observed={canonical_sha256(observed)}"
        )
    return text


def decode_compact_ids(
    ids: Sequence[int],
    tokenizer: Any,
    atom_ids: Mapping[str, int],
) -> str:
    """Invert cfgtypes-v2 ``compact_ids`` including base-token pool spans."""
    reverse = {int(token_id): token for token, token_id in atom_ids.items()}
    if len(reverse) != len(atom_ids):
        raise V3SealError("source_atom_id_collision")
    output: list[str] = []
    base_segment: list[int] = []

    def flush() -> None:
        if base_segment:
            output.append(
                str(
                    tokenizer.decode(
                        base_segment,
                        skip_special_tokens=False,
                    )
                )
            )
            base_segment.clear()

    for raw_token_id in ids:
        token_id = _plain_int(raw_token_id, "compact_input_token_id")
        atom = reverse.get(token_id)
        if atom is None:
            base_segment.append(token_id)
        else:
            flush()
            output.append(atom)
    flush()
    return "".join(output)


@dataclass(frozen=True)
class V3ArtifactPaths:
    public: Path
    alignment: Path
    tests: Path
    private: Path
    pool_reconciliation: Path
    release_manifest: Path
    contract: Path
    codebook: Path
    codec: Path
    tokenizer: Path
    legacy_cfg_extractor: Path
    legacy_dfg_extractor: Path
    current_cfg_extractor: Path
    current_dfg_extractor: Path


class V3ArtifactSerializer:
    """Verified loader and materializer for the exact Opus-v3 held-out set."""

    def __init__(
        self,
        paths: V3ArtifactPaths,
        *,
        seals: Mapping[str, str | int] = KNOWN_OPUS_V3_175_SEALS,
        max_prompt_tokens: int = 12_000,
        chat_overhead_reserve: int = 256,
    ) -> None:
        if max_prompt_tokens <= 0:
            raise V3SealError("max_prompt_tokens_must_be_positive")
        if chat_overhead_reserve < 0:
            raise V3SealError("chat_overhead_reserve_must_be_nonnegative")
        self.paths = V3ArtifactPaths(
            **{
                name: value.expanduser().resolve()
                for name, value in paths.__dict__.items()
            }
        )
        self.seals = dict(seals)
        self.max_prompt_tokens = max_prompt_tokens
        self.chat_overhead_reserve = chat_overhead_reserve
        self.file_records: dict[str, dict[str, Any]] = {}
        self._verify_file_seals()
        self._load_and_verify_release()

    def _seal_text(self, name: str) -> str:
        if name not in self.seals:
            raise V3SealError(f"missing_expected_seal:{name}")
        return _digest(self.seals[name], name)

    def _seal_count(self, name: str) -> int:
        if name not in self.seals:
            raise V3SealError(f"missing_expected_count:{name}")
        return _plain_int(self.seals[name], name)

    def _verify_file_seals(self) -> None:
        pairs = {
            "public": (self.paths.public, "public_sha256"),
            "alignment": (self.paths.alignment, "alignment_sha256"),
            "tests": (self.paths.tests, "tests_sha256"),
            "private": (self.paths.private, "private_sha256"),
            "pool_reconciliation": (
                self.paths.pool_reconciliation,
                "pool_reconciliation_sha256",
            ),
            "release_manifest": (
                self.paths.release_manifest,
                "release_manifest_sha256",
            ),
            "contract": (self.paths.contract, "contract_sha256"),
            "codebook": (self.paths.codebook, "codebook_sha256"),
            "codec": (self.paths.codec, "codec_sha256"),
            "tokenizer": (self.paths.tokenizer, "tokenizer_sha256"),
        }
        for label, (path, seal_name) in pairs.items():
            self.file_records[label] = _require_file_hash(
                path,
                self._seal_text(seal_name),
                label,
            )

    def _load_and_verify_release(self) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise V3SealError("tokenizers_package_is_required") from error

        self.release = _read_json(
            self.paths.release_manifest, "release_manifest"
        )
        self.contract = _read_json(self.paths.contract, "compact_contract")
        self.codebook = _read_json(self.paths.codebook, "codebook")
        self.codec = _load_codec(self.paths.codec)

        _require(
            self.release.get("schema")
            == "compact-qwen-phase0-s44-v3-release-seal-v1",
            "release_manifest_schema_mismatch",
        )
        _require(
            (self.release.get("gates") or {}).get("passed") is True,
            "release_manifest_gates_not_passed",
        )
        release_contract = self.release.get("contract") or {}
        _require(
            release_contract.get("sha256")
            == self.file_records["contract"]["sha256"],
            "release_contract_sha256_mismatch",
        )
        _require(
            release_contract.get("codec_sha256")
            == self.file_records["codec"]["sha256"],
            "release_codec_sha256_mismatch",
        )
        _require(
            release_contract.get("codebook_sha256")
            == self.file_records["codebook"]["sha256"],
            "release_codebook_sha256_mismatch",
        )
        _require(
            release_contract.get("tokenizer_json_sha256")
            == self.file_records["tokenizer"]["sha256"],
            "release_tokenizer_sha256_mismatch",
        )
        release_files = {
            str(record.get("path")): record
            for record in self.release.get("files") or []
            if isinstance(record, dict)
        }
        expected_release_files = {
            "binary_build/prepared/train_codec_private.jsonl": "private",
            "compact/pool_reconciliation_private.jsonl": "pool_reconciliation",
            "compact/compact_contract.json": "contract",
            "compact/codebook.json": "codebook",
        }
        for relative, label in expected_release_files.items():
            record = release_files.get(relative)
            _require(record is not None, f"release_missing_file_record:{relative}")
            _require(
                record.get("sha256") == self.file_records[label]["sha256"],
                f"release_file_sha256_mismatch:{relative}",
            )

        _require(
            self.contract.get("codec_sha256")
            == self.file_records["codec"]["sha256"],
            "contract_codec_sha256_mismatch",
        )
        for value, label in (
            (self.contract, "contract"),
            (self.codebook, "codebook"),
        ):
            _require(
                value.get("tokenizer_json_sha256")
                == self.file_records["tokenizer"]["sha256"],
                f"{label}_tokenizer_sha256_mismatch",
            )
            _require(
                value.get("cfg_encoding") == EXPECTED_CFG_ENCODING,
                f"{label}_cfg_encoding_mismatch",
            )
            _require(
                value.get("lossless_domain") == EXPECTED_LOSSLESS_DOMAIN,
                f"{label}_lossless_domain_mismatch",
            )
            _require(
                value.get("pool_reconciliation_manifest_sha256")
                == self.file_records["pool_reconciliation"]["sha256"],
                f"{label}_pool_reconciliation_sha256_mismatch",
            )
        _require(
            self.contract.get("codebook_sha256")
            == self.file_records["codebook"]["sha256"],
            "contract_codebook_sha256_mismatch",
        )
        _require(
            self.contract.get("target_architecture") == "x86_64",
            "contract_target_architecture_mismatch",
        )
        _require(
            self.contract.get("target_function") == "candidate",
            "contract_target_function_mismatch",
        )
        _require(
            self.codebook.get("measure_excluded_from_fit") is True,
            "codebook_measure_was_not_excluded_from_fit",
        )
        _require(
            self.codebook.get("fit_scope") == "train_only",
            "codebook_fit_scope_mismatch",
        )

        base_codec_path = Path(self.codec.base.__file__).resolve()
        observed_base_sha = sha256_file(base_codec_path)
        expected_base_sha = _digest(
            self.contract.get("base_pool_codec_sha256"),
            "contract_base_pool_codec_sha256",
        )
        _require(
            observed_base_sha == expected_base_sha,
            "base_pool_codec_sha256_mismatch",
        )
        self.file_records["base_pool_codec"] = {
            "path": str(base_codec_path),
            "sha256": observed_base_sha,
            "size_bytes": base_codec_path.stat().st_size,
        }
        observed_graph_sha = str(self.codec.graph_codec_sha256())
        expected_graph_sha = _digest(
            self.contract.get("graph_codec_dependency_sha256"),
            "contract_graph_codec_dependency_sha256",
        )
        _require(
            observed_graph_sha == expected_graph_sha,
            "graph_codec_dependency_sha256_mismatch",
        )

        self.expansions = self.codebook.get("expansions")
        _require(
            isinstance(self.expansions, list)
            and all(isinstance(value, str) for value in self.expansions),
            "codebook_expansions_must_be_string_list",
        )
        _require(
            len(self.expansions) == self.codebook.get("codebook_size"),
            "codebook_expansion_count_mismatch",
        )
        _require(
            len(self.expansions) == len(set(self.expansions)),
            "codebook_expansions_not_unique",
        )
        self.instruction_code = {
            instruction: index
            for index, instruction in enumerate(self.expansions)
        }
        self.atom_ids = self.codebook.get("source_atom_ids")
        _require(
            isinstance(self.atom_ids, dict)
            and all(
                isinstance(key, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
                for key, value in self.atom_ids.items()
            ),
            "codebook_source_atom_ids_invalid",
        )
        _require(
            self.codebook.get("source_token_expansions")
            == self.contract.get("source_token_expansions"),
            "contract_codebook_source_expansions_mismatch",
        )
        source_ids = sorted(
            int(value)
            for value in self.contract.get("source_token_ids") or []
        )
        _require(
            sorted(self.atom_ids.values()) == source_ids,
            "contract_codebook_source_ids_mismatch",
        )

        self.tokenizer = Tokenizer.from_file(str(self.paths.tokenizer))
        self.visible_symbols = visible_one_token_symbols(self.tokenizer)

        graph_v2 = self.codec.base.graph_v2
        self.registry = graph_v2.load_route_registry(
            self.paths.legacy_cfg_extractor,
            self.paths.legacy_dfg_extractor,
            self.paths.current_cfg_extractor,
            self.paths.current_dfg_extractor,
        )
        route_contract = graph_v2.route_contract(self.registry)
        _require(
            route_contract == self.contract.get("extractor_routes"),
            "extractor_route_contract_mismatch",
        )
        for route, record in self.registry.items():
            for kind in ("cfg", "dfg"):
                path = Path(record[f"{kind}_path"]).resolve()
                self.file_records[f"{route}_{kind}_extractor"] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }

        self.public_rows = _read_jsonl(self.paths.public, "v3_public")
        self.alignment_rows = _read_jsonl(
            self.paths.alignment, "v3_alignment"
        )
        self.test_rows = _read_jsonl(self.paths.tests, "v3_tests")
        self.private_rows = _read_jsonl(self.paths.private, "v3_private")
        self.reconciliation_rows = _read_jsonl(
            self.paths.pool_reconciliation,
            "v3_pool_reconciliation",
        )
        expected_rows = self._seal_count("expected_rows")
        _require(
            len(self.public_rows)
            == len(self.alignment_rows)
            == len(self.test_rows)
            == expected_rows,
            "v3_eval_row_count_mismatch",
        )
        _require(
            len(self.private_rows) == self._seal_count("expected_private_rows"),
            "v3_private_row_count_mismatch",
        )
        _require(
            len(self.reconciliation_rows)
            == self._seal_count("expected_reconciliation_rows"),
            "v3_reconciliation_row_count_mismatch",
        )
        self.reconciliation_by_sha: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(self.reconciliation_rows, 1):
            row_sha = _digest(
                row.get("row_sha256"),
                f"pool_reconciliation_row_{row_number}_sha256",
            )
            observed = canonical_sha256(
                {key: value for key, value in row.items() if key != "row_sha256"}
            )
            _require(
                observed == row_sha,
                f"pool_reconciliation_row_hash_mismatch:{row_number}",
            )
            _require(
                row_sha not in self.reconciliation_by_sha,
                f"duplicate_pool_reconciliation_row_sha256:{row_sha}",
            )
            self.reconciliation_by_sha[row_sha] = row

    def _verify_pool_binding(
        self,
        *,
        task_id: str,
        private: Mapping[str, Any],
        alignment: Mapping[str, Any],
        canonical: Mapping[str, Any],
    ) -> dict[str, int]:
        raw_uses = private.get("binary_pool_uses")
        receipt = private.get("binary_pool_private_receipt")
        _require(isinstance(raw_uses, list), f"missing_binary_pool_uses:{task_id}")
        _require(
            isinstance(receipt, dict),
            f"missing_binary_pool_private_receipt:{task_id}",
        )
        metadata = alignment.get("pool_metadata")
        _require(
            isinstance(metadata, dict),
            f"missing_alignment_pool_metadata:{task_id}",
        )
        projection_sha = canonical_sha256(raw_uses)
        receipt_sha = canonical_sha256(receipt)
        _require(
            metadata.get("projection_sha256") == projection_sha,
            f"pool_projection_sha256_mismatch:{task_id}",
        )
        _require(
            metadata.get("receipt_sha256") == receipt_sha,
            f"pool_receipt_sha256_mismatch:{task_id}",
        )
        pool = canonical.get("binary_pool")
        _require(isinstance(pool, dict), f"missing_canonical_pool:{task_id}")
        uses = pool.get("uses")
        _require(isinstance(uses, list), f"missing_canonical_pool_uses:{task_id}")
        _require(
            metadata.get("use_count") == len(uses) == len(raw_uses),
            f"pool_use_count_mismatch:{task_id}",
        )
        use_sites = sum(
            len(record.get("use_sites") or [])
            for record in uses
            if isinstance(record, dict)
        )
        reconciliation_sha = _digest(
            alignment.get("pool_reconciliation_row_sha256"),
            f"alignment_pool_reconciliation_row_sha256:{task_id}",
        )
        reconciliation = self.reconciliation_by_sha.get(reconciliation_sha)
        _require(
            reconciliation is not None,
            f"missing_pool_reconciliation_row:{task_id}",
        )
        _require(
            reconciliation.get("task_id") == task_id,
            f"pool_reconciliation_task_mismatch:{task_id}",
        )
        _require(
            reconciliation.get("projection_sha256") == projection_sha,
            f"pool_reconciliation_projection_mismatch:{task_id}",
        )
        _require(
            reconciliation.get("receipt_sha256") == receipt_sha,
            f"pool_reconciliation_receipt_mismatch:{task_id}",
        )
        _require(
            reconciliation.get("line") == alignment.get("line"),
            f"pool_reconciliation_line_mismatch:{task_id}",
        )
        _require(
            all((reconciliation.get("gates") or {}).values()),
            f"pool_reconciliation_gate_failed:{task_id}",
        )
        counts = reconciliation.get("counts") or {}
        _require(
            counts.get("encoded_pool_records") == len(uses),
            f"pool_reconciliation_record_count_mismatch:{task_id}",
        )
        _require(
            counts.get("encoded_pool_use_sites") == use_sites,
            f"pool_reconciliation_use_site_count_mismatch:{task_id}",
        )
        return {"records": len(uses), "use_sites": use_sites}

    def _prepare_row(self, index: int) -> dict[str, Any]:
        public = self.public_rows[index]
        alignment = self.alignment_rows[index]
        test_row = self.test_rows[index]
        if set(public) != PUBLIC_FIELDS:
            raise V3SealError(f"public_schema_mismatch:{index}")
        task_id = str(alignment.get("task_id") or "")
        _require(bool(task_id), f"missing_task_id:{index}")
        _require(
            alignment.get("model_row") == index,
            f"alignment_model_row_mismatch:{task_id}",
        )
        _require(
            test_row.get("task_id") == task_id,
            f"test_alignment_task_mismatch:{task_id}",
        )
        tests = test_row.get("tests")
        _require(
            isinstance(tests, str) and bool(tests.strip()),
            f"missing_test_harness:{task_id}",
        )
        _require(
            canonical_sha256(public) == alignment.get("model_row_sha256"),
            f"public_model_row_sha256_mismatch:{task_id}",
        )
        expected_public_hashes = {
            "compact_codec_sha256": self.file_records["codec"]["sha256"],
            "compact_codebook_sha256": self.file_records["codebook"]["sha256"],
            "compact_tokenizer_sha256": self.file_records["tokenizer"]["sha256"],
        }
        for field, expected in expected_public_hashes.items():
            _require(
                public.get(field) == expected,
                f"public_{field}_mismatch:{task_id}",
            )
        input_ids = public.get("compact_input_ids")
        _require(
            isinstance(input_ids, list)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in input_ids
            ),
            f"public_compact_input_ids_invalid:{task_id}",
        )
        _require(
            len(input_ids) == alignment.get("source_tokens"),
            f"public_source_token_count_mismatch:{task_id}",
        )
        line_number = _plain_int(
            alignment.get("line"),
            f"alignment_private_line:{task_id}",
        )
        _require(
            1 <= line_number <= len(self.private_rows),
            f"alignment_private_line_out_of_range:{task_id}",
        )
        private = self.private_rows[line_number - 1]
        _require(
            private.get("task_id") == task_id,
            f"private_alignment_task_mismatch:{task_id}",
        )
        _require(
            private.get("split") == alignment.get("split"),
            f"private_alignment_split_mismatch:{task_id}",
        )
        _require(
            private.get("split_row") == alignment.get("split_row"),
            f"private_alignment_split_row_mismatch:{task_id}",
        )
        _require(
            private.get("family") == alignment.get("family"),
            f"private_alignment_family_mismatch:{task_id}",
        )
        _require(
            (private.get("aot") or {}).get("sha256")
            == alignment.get("aot_sha256"),
            f"private_alignment_aot_mismatch:{task_id}",
        )
        _require(
            (private.get("graph_v2") or {}).get("extractor_sha256")
            == alignment.get("graph_extractor_sha256"),
            f"private_alignment_graph_extractor_mismatch:{task_id}",
        )

        try:
            canonical = self.codec.canonicalize(private)
        except Exception as error:
            raise V3SealError(
                f"private_canonicalize_failed:{task_id}:{error}"
            ) from error
        _require(
            canonical_sha256(canonical) == alignment.get("canonical_sha256"),
            f"private_canonical_sha256_mismatch:{task_id}",
        )
        _require(
            canonical.get("dfg_route") == alignment.get("dfg_route"),
            f"private_dfg_route_mismatch:{task_id}",
        )
        _require(
            len(canonical.get("blocks") or []) == alignment.get("block_count"),
            f"private_block_count_mismatch:{task_id}",
        )
        _require(
            sum(
                len(block.get("instructions") or [])
                for block in canonical.get("blocks") or []
            )
            == alignment.get("instruction_count"),
            f"private_instruction_count_mismatch:{task_id}",
        )
        _require(
            len(canonical.get("cfg_edges") or [])
            == alignment.get("cfg_edge_count"),
            f"private_cfg_edge_count_mismatch:{task_id}",
        )
        _require(
            len(canonical.get("dfg_edges") or [])
            == alignment.get("dfg_edge_count"),
            f"private_dfg_edge_count_mismatch:{task_id}",
        )
        pool_counts = self._verify_pool_binding(
            task_id=task_id,
            private=private,
            alignment=alignment,
            canonical=canonical,
        )

        try:
            compact_text = self.codec.encode(
                canonical,
                self.instruction_code,
            )
            compact_decoded = self.codec.decode(
                compact_text,
                self.expansions,
            )
            regenerated = self.codec.regenerate_dfg(
                compact_decoded,
                self.registry,
            )
            retokenized = self.codec.compact_ids(
                compact_text,
                self.tokenizer,
                self.atom_ids,
            )
            recovered_text = decode_compact_ids(
                input_ids,
                self.tokenizer,
                self.atom_ids,
            )
        except Exception as error:
            raise V3SealError(
                f"cfgtypes_codec_roundtrip_failed:{task_id}:{error}"
            ) from error
        _require(
            regenerated == canonical,
            f"cfgtypes_codec_semantic_roundtrip_mismatch:{task_id}",
        )
        _require(
            sha256_text(compact_text) == alignment.get("compact_sha256"),
            f"compact_text_sha256_mismatch:{task_id}",
        )
        _require(
            retokenized == input_ids,
            f"compact_retokenization_mismatch:{task_id}",
        )
        _require(
            recovered_text == compact_text,
            f"compact_id_decode_mismatch:{task_id}",
        )
        fallback_count = sum(
            instruction not in self.instruction_code
            for block in canonical["blocks"]
            for instruction in block["instructions"]
        )
        _require(
            fallback_count == alignment.get("fallback_instructions"),
            f"fallback_instruction_count_mismatch:{task_id}",
        )

        readable = serialize_v3_api_readable(
            canonical,
            tokenizer=self.tokenizer,
            visible_symbols=self.visible_symbols,
        )
        readable_decoded = decode_v3_api_readable(readable)
        _require(
            readable_decoded == compact_decoded,
            f"api_and_cfgtypes_projection_mismatch:{task_id}",
        )
        system_tokens = len(
            self.tokenizer.encode(
                V3_SYSTEM_PROMPT,
                add_special_tokens=False,
            ).ids
        )
        user_tokens = len(
            self.tokenizer.encode(
                readable,
                add_special_tokens=False,
            ).ids
        )
        estimated = (
            system_tokens + user_tokens + self.chat_overhead_reserve
        )
        _require(
            estimated <= self.max_prompt_tokens,
            f"complete_prompt_over_budget:{task_id}:{estimated}>"
            f"{self.max_prompt_tokens}",
        )
        edge_counts = Counter(
            str(edge["edge_type"]) for edge in canonical["cfg_edges"]
        )
        return {
            "schema": ROW_SCHEMA,
            "task_id": task_id,
            "api_readable_input": readable,
            "api_readable_sha256": sha256_text(readable),
            "private_test_harness": tests,
            "private_test_harness_sha256": sha256_text(tests),
            "bindings": {
                "alignment_model_row": index,
                "alignment_model_row_sha256": alignment[
                    "model_row_sha256"
                ],
                "private_line": line_number,
                "canonical_sha256": alignment["canonical_sha256"],
                "compact_sha256": alignment["compact_sha256"],
                "pool_reconciliation_row_sha256": alignment[
                    "pool_reconciliation_row_sha256"
                ],
                "aot_sha256": alignment["aot_sha256"],
            },
            "counts": {
                "blocks": len(canonical["blocks"]),
                "instructions": sum(
                    len(block["instructions"]) for block in canonical["blocks"]
                ),
                "cfg_edges": len(canonical["cfg_edges"]),
                "cfg_edge_types": dict(sorted(edge_counts.items())),
                "binary_pool_records": pool_counts["records"],
                "binary_pool_use_sites": pool_counts["use_sites"],
            },
            "prompt_preflight": {
                "tokenizer_sha256": self.file_records["tokenizer"]["sha256"],
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "chat_overhead_reserve": self.chat_overhead_reserve,
                "estimated_prompt_tokens": estimated,
                "max_prompt_tokens": self.max_prompt_tokens,
            },
            "invariants": {
                "cfgtypes_codec_roundtrip_exact": True,
                "public_ids_retokenized_exactly": True,
                "api_semantic_roundtrip_exact": True,
                "ordered_cfg_including_call_preserved": True,
                "binary_pool_values_and_use_sites_preserved": True,
                "tests_are_private_and_not_in_api_readable_input": True,
            },
        }

    def prepare_all(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = [self._prepare_row(index) for index in range(len(self.public_rows))]
        task_ids = [row["task_id"] for row in rows]
        _require(
            len(task_ids) == len(set(task_ids)),
            "duplicate_v3_eval_task_ids",
        )
        token_counts = [
            row["prompt_preflight"]["estimated_prompt_tokens"] for row in rows
        ]
        cfg_types: Counter[str] = Counter()
        total_pool_records = 0
        total_pool_use_sites = 0
        for row in rows:
            cfg_types.update(row["counts"]["cfg_edge_types"])
            total_pool_records += row["counts"]["binary_pool_records"]
            total_pool_use_sites += row["counts"]["binary_pool_use_sites"]
        sorted_counts = sorted(token_counts)

        def percentile(fraction: float) -> int:
            index = min(
                len(sorted_counts) - 1,
                int(fraction * len(sorted_counts)),
            )
            return sorted_counts[index]

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "representation_schema": SERIALIZER_SCHEMA,
            "rows": len(rows),
            "task_sequence_sha256": canonical_sha256(task_ids),
            "row_binding_sequence_sha256": canonical_sha256(
                [
                    {
                        "task_id": row["task_id"],
                        "api_readable_sha256": row[
                            "api_readable_sha256"
                        ],
                        **row["bindings"],
                    }
                    for row in rows
                ]
            ),
            "system_prompt": V3_SYSTEM_PROMPT,
            "system_prompt_sha256": sha256_text(V3_SYSTEM_PROMPT),
            "files": self.file_records,
            "prompt_tokens": {
                "tokenizer_sha256": self.file_records["tokenizer"]["sha256"],
                "chat_overhead_reserve": self.chat_overhead_reserve,
                "limit": self.max_prompt_tokens,
                "min": min(sorted_counts),
                "p50": percentile(0.50),
                "p90": percentile(0.90),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "max": max(sorted_counts),
                "rows_over_limit": 0,
            },
            "cohort_totals": {
                "cfg_edge_types": dict(sorted(cfg_types.items())),
                "binary_pool_records": total_pool_records,
                "binary_pool_use_sites": total_pool_use_sites,
            },
            "invariants": {
                "exact_175_public_alignment_test_rows": True,
                "private_rows_selected_by_hash_bound_line": True,
                "release_contract_codebook_codec_tokenizer_sealed": True,
                "pool_reconciliation_rows_hash_bound": True,
                "all_cfgtypes_codec_dfg_and_id_roundtrips_exact": True,
                "all_normalized_blocks_and_instructions_preserved": True,
                "all_ordered_cfg_edges_including_call_preserved": True,
                "all_binary_pool_values_and_use_sites_preserved": True,
                "all_api_readable_roundtrips_exact": True,
                "all_complete_prompts_within_limit": True,
                "no_api_calls": True,
                "no_truncation": True,
            },
        }
        return rows, manifest

    def materialize(
        self,
        output_path: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows, manifest = self.prepare_all()
        output_path = output_path.expanduser().resolve()
        _write_jsonl(output_path, rows)
        manifest["output"] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "rows": len(rows),
        }
        manifest_path = output_path.with_suffix(
            output_path.suffix + ".manifest.json"
        )
        _write_json(manifest_path, manifest)
        return rows, manifest


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _default_workspace() -> Path:
    return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    workspace = _default_workspace()
    v3_clean = workspace / "pod_sync_20260723" / "artifacts" / "v3_clean"
    release = (
        workspace
        / "scrubbed_master_v2_release"
        / "direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release"
    )
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--public",
        type=_path,
        default=v3_clean / "v3eval_public.jsonl",
    )
    parser.add_argument(
        "--alignment",
        type=_path,
        default=v3_clean / "v3eval_align.jsonl",
    )
    parser.add_argument(
        "--tests",
        type=_path,
        default=v3_clean / "v3eval_tests.jsonl",
    )
    parser.add_argument(
        "--private",
        type=_path,
        default=release
        / "binary_build"
        / "prepared"
        / "train_codec_private.jsonl",
    )
    parser.add_argument(
        "--pool-reconciliation",
        type=_path,
        default=release / "compact" / "pool_reconciliation_private.jsonl",
    )
    parser.add_argument(
        "--release-manifest",
        type=_path,
        default=release / "release_manifest.json",
    )
    parser.add_argument(
        "--contract",
        type=_path,
        default=release / "compact" / "compact_contract.json",
    )
    parser.add_argument(
        "--codebook",
        type=_path,
        default=release / "compact" / "codebook.json",
    )
    parser.add_argument(
        "--codec",
        type=_path,
        default=workspace
        / "scripts"
        / "data"
        / "build_compact_qwen_v3_cfgtypes_v2.py",
    )
    parser.add_argument("--tokenizer", required=True, type=_path)
    parser.add_argument(
        "--legacy-cfg-extractor",
        type=_path,
        default=workspace
        / "scrubbed_master_v2_release"
        / "extractors"
        / "cfg_extractor.py",
    )
    parser.add_argument(
        "--legacy-dfg-extractor",
        type=_path,
        default=workspace
        / "scrubbed_master_v2_release"
        / "extractors"
        / "dfg_extractor.py",
    )
    parser.add_argument(
        "--current-cfg-extractor",
        type=_path,
        default=workspace / "scripts" / "data" / "cfg_extractor.py",
    )
    parser.add_argument(
        "--current-dfg-extractor",
        type=_path,
        default=workspace / "scripts" / "data" / "dfg_extractor.py",
    )
    parser.add_argument("--output", type=_path)
    parser.add_argument("--max-prompt-tokens", type=int, default=12_000)
    parser.add_argument("--chat-overhead-reserve", type=int, default=256)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify and report without writing a materialized JSONL.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_only and args.output is not None:
        raise SystemExit("--preflight-only and --output are mutually exclusive")
    if not args.preflight_only and args.output is None:
        raise SystemExit("--output is required unless --preflight-only is set")
    paths = V3ArtifactPaths(
        public=args.public,
        alignment=args.alignment,
        tests=args.tests,
        private=args.private,
        pool_reconciliation=args.pool_reconciliation,
        release_manifest=args.release_manifest,
        contract=args.contract,
        codebook=args.codebook,
        codec=args.codec,
        tokenizer=args.tokenizer,
        legacy_cfg_extractor=args.legacy_cfg_extractor,
        legacy_dfg_extractor=args.legacy_dfg_extractor,
        current_cfg_extractor=args.current_cfg_extractor,
        current_dfg_extractor=args.current_dfg_extractor,
    )
    serializer = V3ArtifactSerializer(
        paths,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_overhead_reserve=args.chat_overhead_reserve,
    )
    if args.preflight_only:
        _rows, manifest = serializer.prepare_all()
    else:
        _rows, manifest = serializer.materialize(args.output)
    tokens = manifest["prompt_tokens"]
    totals = manifest["cohort_totals"]
    print(
        "OPUS_V3_LOSSLESS_PREFLIGHT "
        f"rows={manifest['rows']} "
        f"prompt_max={tokens['max']}/{tokens['limit']} "
        f"call_edges={totals['cfg_edge_types'].get('call', 0)} "
        f"pool_records={totals['binary_pool_records']} "
        f"pool_use_sites={totals['binary_pool_use_sites']} "
        f"task_sequence_sha256={manifest['task_sequence_sha256']}",
        flush=True,
    )
    if "output" in manifest:
        print(
            "OPUS_V3_LOSSLESS_OUTPUT "
            f"sha256={manifest['output']['sha256']} "
            f"path={manifest['output']['path']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
