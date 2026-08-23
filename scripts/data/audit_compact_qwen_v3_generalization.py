#!/usr/bin/env python3
"""Seal compact-Qwen v3 instruction-codebook generalization audits.

The finalized v3 release already measured its train/dev graph-plus-pool rows.
This auditor verifies that release byte-for-byte, re-derives the aggregate dev
and top-up counters from its private alignment sidecar, and cross-checks those
counters against the sealed preflight report.

Scrubbed HumanEval is deliberately narrower.  Its canonical private dataset
predates the binary-pool-v3 extraction and therefore has no v3 pool receipt.
The audit applies the graph-v2 canonicalization shared by v3, regenerates the
DFG with the pinned current extractor, and measures *instruction-codebook
coverage only*.  Out-of-codebook instructions remain exactly representable as
native Qwen token IDs between the reserved ``<R>``/``<E>`` atoms.  The report
does not claim a full v3 pool-stream token count or a 9,000-token result for
HumanEval.

No task-level graph, instruction, source, label, or test payload is emitted.
All output reports are aggregate and are published transactionally with an
audit seal and SHA256SUMS.txt.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import build_compact_qwen_v2 as graph_codec
from scripts.data import build_compact_qwen_v3 as codec


CANONICAL_HUMANEVAL = (
    ROOT / "data/testing/grpo_data_graphv2_signature_scrubbed_private.jsonl"
)
CANONICAL_HUMANEVAL_SHA256 = (
    "91dc2eee2f06602e6cd95873c81802e0997fd3ff758b395aeab9c3da114df252"
)
CANONICAL_HUMANEVAL_ROWS = 154

AUDIT_SCHEMA = "compact-qwen-v3-generalization-audit-v1"
SEAL_SCHEMA = "compact-qwen-v3-generalization-audit-seal-v1"
REPORT_NAMES = frozenset(
    {
        "dev_fallback_audit.json",
        "topup_family_source_pool_fallback_audit.json",
        "scrubbed_humaneval_instruction_codebook_audit.json",
    }
)
OUTPUT_NAMES = REPORT_NAMES | {"generalization_audit_seal.json", "SHA256SUMS.txt"}
BUNDLE_NAMES = frozenset(
    {
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
)
PUBLIC_FIELDS = frozenset(
    {
        "compact_input_ids",
        "compact_codec_sha256",
        "compact_codebook_sha256",
        "compact_tokenizer_sha256",
    }
)
ALLOWED_FAMILIES = frozenset({"master", "topup_s45", "topup_s46"})
EXPECTED_SOURCE_POOLS: Mapping[str, frozenset[str | None]] = {
    "master": frozenset({None}),
    "topup_s45": frozenset({"base_llm", "topup_s44", "topup_s45"}),
    "topup_s46": frozenset({"topup_s46"}),
}
TOPUP_FAMILIES = frozenset({"topup_s45", "topup_s46"})
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


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_{label}:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label}_must_be_json_object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank_{label}_line:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non_object_{label}_row:{line_number}")
            result.append(value)
    return result


def _require_plain_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}_must_be_nonnegative_integer")
    return value


def _binding(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _parse_sha256s(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError("bundle_SHA256SUMS_must_be_ascii") from error
    if text and not text.endswith("\n"):
        raise ValueError("bundle_SHA256SUMS_missing_final_newline")
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match:
            raise ValueError(f"invalid_bundle_SHA256SUMS_line:{line_number}")
        digest, name = match.groups()
        if name == "SHA256SUMS.txt" or name in result:
            raise ValueError(f"invalid_bundle_SHA256SUMS_name:{name}")
        result[name] = digest
    return result


def verify_bundle_checksums(bundle: Path) -> dict[str, str]:
    if not bundle.is_dir():
        raise FileNotFoundError(f"missing_compact_v3_bundle:{bundle}")
    observed = {item.name for item in bundle.iterdir()}
    if observed != BUNDLE_NAMES or any(item.is_dir() for item in bundle.iterdir()):
        raise ValueError(
            "sealed_bundle_file_set_mismatch:"
            f"observed={sorted(observed)}:expected={sorted(BUNDLE_NAMES)}"
        )
    checksums = _parse_sha256s(bundle / "SHA256SUMS.txt")
    expected_checksum_names = BUNDLE_NAMES - {"SHA256SUMS.txt"}
    if set(checksums) != expected_checksum_names:
        raise ValueError(
            "bundle_SHA256SUMS_coverage_mismatch:"
            f"observed={sorted(checksums)}:expected={sorted(expected_checksum_names)}"
        )
    for name, expected in checksums.items():
        observed_digest = sha256_file(bundle / name)
        if observed_digest != expected:
            raise ValueError(
                f"bundle_checksum_mismatch:{name}:{observed_digest}!={expected}"
            )
    return checksums


@dataclasses.dataclass(frozen=True)
class BundleState:
    path: Path
    contract: dict[str, Any]
    codebook: dict[str, Any]
    preflight: dict[str, Any]
    model_rows: list[dict[str, Any]]
    alignment_rows: list[dict[str, Any]]
    expansions: list[str]
    tokenizer: Tokenizer
    registry: dict[str, dict[str, Any]]
    bindings: dict[str, dict[str, Any]]


def verify_bundle(
    bundle: Path,
    tokenizer_path: Path,
    *,
    codec_path: Path,
    graph_codec_path: Path,
    release_builder_path: Path,
    legacy_cfg_extractor: Path,
    legacy_dfg_extractor: Path,
    current_cfg_extractor: Path,
    current_dfg_extractor: Path,
) -> BundleState:
    """Verify every sealed release artifact and all executable dependencies."""
    verify_bundle_checksums(bundle)
    contract_path = bundle / "compact_contract.json"
    codebook_path = bundle / "codebook.json"
    preflight_path = bundle / "preflight_report.json"
    contract = read_json(contract_path, "compact_contract")
    codebook = read_json(codebook_path, "codebook")
    preflight = read_json(preflight_path, "preflight")
    if contract.get("schema") != codec.CONTRACT_SCHEMA:
        raise ValueError("bundle_contract_is_not_v3")
    if codebook.get("schema") != codec.CODEBOOK_SCHEMA:
        raise ValueError("bundle_codebook_is_not_v3")
    if preflight.get("schema") != codec.PREFLIGHT_SCHEMA or preflight.get("passed") is not True:
        raise ValueError("bundle_preflight_is_not_passing_v3")
    if preflight.get("quarantined") != 0 or preflight.get("failures_count") != 0:
        raise ValueError("passing_preflight_has_quarantine_or_failures")

    codec_sha = sha256_file(codec_path)
    imported_codec_sha = sha256_file(Path(codec.__file__).resolve())
    if codec_sha != imported_codec_sha:
        raise ValueError("selected_v3_codec_differs_from_imported_v3_codec")
    graph_codec_sha = sha256_file(graph_codec_path)
    imported_graph_sha = sha256_file(Path(graph_codec.__file__).resolve())
    if graph_codec_sha != imported_graph_sha:
        raise ValueError("selected_graph_codec_differs_from_imported_graph_codec")
    if codec.graph_codec_sha256() != graph_codec_sha:
        raise ValueError("v3_codec_transitive_graph_dependency_mismatch")
    tokenizer_sha = sha256_file(tokenizer_path)
    codebook_sha = sha256_file(codebook_path)
    expected_links = {
        "codec_sha256": codec_sha,
        "graph_codec_dependency_sha256": graph_codec_sha,
        "codebook_sha256": codebook_sha,
        "tokenizer_json_sha256": tokenizer_sha,
    }
    preflight_contract = preflight.get("contract")
    if not isinstance(preflight_contract, Mapping):
        raise ValueError("preflight_contract_missing")
    for field, expected in expected_links.items():
        if require_sha256(contract.get(field), f"contract_{field}") != expected:
            raise ValueError(f"contract_{field}_mismatch")
        if field != "codec_sha256" and field in codebook:
            if require_sha256(codebook.get(field), f"codebook_{field}") != expected:
                raise ValueError(f"codebook_{field}_mismatch")
        if require_sha256(preflight_contract.get(field), f"preflight_{field}") != expected:
            raise ValueError(f"preflight_{field}_mismatch")
    if require_sha256(codebook.get("tokenizer_json_sha256"), "codebook_tokenizer") != tokenizer_sha:
        raise ValueError("codebook_tokenizer_json_sha256_mismatch")
    if require_sha256(contract.get("release_builder_sha256"), "release_builder") != sha256_file(
        release_builder_path
    ):
        raise ValueError("release_builder_sha256_mismatch")
    if contract.get("target_function") != codec.TARGET_FUNCTION:
        raise ValueError("contract_target_function_is_not_candidate")
    if codebook.get("fit_scope") != "train_only" or codebook.get("measure_excluded_from_fit") is not True:
        raise ValueError("codebook_is_not_train_only")
    if codebook.get("symbol_policy") != "runtime_aware":
        raise ValueError("codebook_symbol_policy_is_not_runtime_aware")
    if codebook.get("runtime_symbol_policy") != graph_codec.RUNTIME_POLICY:
        raise ValueError("runtime_symbol_policy_drift")
    runtime_policy_sha = canonical_sha256(graph_codec.RUNTIME_POLICY)
    if require_sha256(
        contract.get("runtime_symbol_policy_sha256"),
        "contract_runtime_symbol_policy_sha256",
    ) != runtime_policy_sha:
        raise ValueError("contract_runtime_symbol_policy_sha256_mismatch")
    if require_sha256(
        codebook.get("runtime_symbol_policy_sha256"),
        "codebook_runtime_symbol_policy_sha256",
    ) != runtime_policy_sha:
        raise ValueError("codebook_runtime_symbol_policy_sha256_mismatch")

    registry = graph_codec.load_route_registry(
        legacy_cfg_extractor,
        legacy_dfg_extractor,
        current_cfg_extractor,
        current_dfg_extractor,
    )
    observed_routes = graph_codec.route_contract(registry)
    if contract.get("extractor_routes") != observed_routes:
        raise ValueError("contract_extractor_route_drift")
    if codebook.get("extractor_routes") != observed_routes:
        raise ValueError("codebook_extractor_route_drift")

    expansions = codebook.get("expansions")
    if not isinstance(expansions, list) or not all(isinstance(item, str) for item in expansions):
        raise ValueError("codebook_expansions_must_be_string_array")
    if len(expansions) != _require_plain_nonnegative_int(
        codebook.get("codebook_size"), "codebook_size"
    ):
        raise ValueError("codebook_expansion_count_mismatch")
    if len(expansions) != len(set(expansions)):
        raise ValueError("duplicate_instruction_codebook_expansion")

    model_rows = read_jsonl(bundle / "compact_model_inputs.jsonl", "model_inputs")
    alignment_rows = read_jsonl(bundle / "alignment_private.jsonl", "alignment")
    quarantine = read_jsonl(bundle / "quarantine.jsonl", "quarantine")
    failures = read_jsonl(bundle / "failures.jsonl", "failures")
    if quarantine or failures:
        raise ValueError("sealed_bundle_contains_quarantine_or_failures")
    if len(model_rows) != len(alignment_rows) or len(model_rows) != preflight.get("rows_retained"):
        raise ValueError("sealed_public_private_preflight_row_count_mismatch")
    for index, (public, alignment) in enumerate(zip(model_rows, alignment_rows)):
        if set(public) != PUBLIC_FIELDS:
            raise ValueError(f"model_row_{index}_is_not_strict_four_field")
        ids = public.get("compact_input_ids")
        if not isinstance(ids, list) or not ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            raise ValueError(f"model_row_{index}_has_invalid_compact_ids")
        if public.get("compact_codec_sha256") != codec_sha:
            raise ValueError(f"model_row_{index}_codec_binding_mismatch")
        if public.get("compact_codebook_sha256") != codebook_sha:
            raise ValueError(f"model_row_{index}_codebook_binding_mismatch")
        if public.get("compact_tokenizer_sha256") != tokenizer_sha:
            raise ValueError(f"model_row_{index}_tokenizer_binding_mismatch")
        if alignment.get("model_row") != index:
            raise ValueError(f"alignment_row_{index}_position_mismatch")
        if alignment.get("model_row_sha256") != canonical_sha256(public):
            raise ValueError(f"alignment_row_{index}_public_hash_mismatch")

    bindings = {
        name: _binding(bundle / name)
        for name in sorted(BUNDLE_NAMES)
    }
    bindings.update(
        {
            "tokenizer": _binding(tokenizer_path),
            "v3_codec": _binding(codec_path),
            "graph_codec_dependency": _binding(graph_codec_path),
            "release_builder": _binding(release_builder_path),
            "legacy_cfg_extractor": _binding(legacy_cfg_extractor),
            "legacy_dfg_extractor": _binding(legacy_dfg_extractor),
            "current_cfg_extractor": _binding(current_cfg_extractor),
            "current_dfg_extractor": _binding(current_dfg_extractor),
        }
    )
    return BundleState(
        path=bundle,
        contract=contract,
        codebook=codebook,
        preflight=preflight,
        model_rows=model_rows,
        alignment_rows=alignment_rows,
        expansions=expansions,
        tokenizer=Tokenizer.from_file(str(tokenizer_path)),
        registry=registry,
        bindings=bindings,
    )


@dataclasses.dataclass
class Coverage:
    rows: int = 0
    instructions: int = 0
    fallback: int = 0
    rows_with_fallback: int = 0

    def add(self, instruction_count: int, fallback_count: int) -> None:
        if fallback_count > instruction_count:
            raise ValueError("fallback_count_exceeds_instruction_count")
        self.rows += 1
        self.instructions += instruction_count
        self.fallback += fallback_count
        self.rows_with_fallback += int(fallback_count > 0)

    def report(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "instructions": self.instructions,
            "codebook_hits": self.instructions - self.fallback,
            "fallback": self.fallback,
            "fallback_rate": self.fallback / self.instructions if self.instructions else 0.0,
            "rows_with_fallback": self.rows_with_fallback,
            "rows_with_fallback_rate": self.rows_with_fallback / self.rows if self.rows else 0.0,
        }

    def preflight_projection(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "instructions": self.instructions,
            "fallback": self.fallback,
            "fallback_rate": self.fallback / self.instructions if self.instructions else 0.0,
        }


def _render(values: Mapping[str, Coverage]) -> dict[str, dict[str, Any]]:
    return {key: values[key].report() for key in sorted(values)}


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _token_report(values: Sequence[int], limit: int) -> dict[str, Any]:
    return {
        "kind": "full_v3_graph_plus_binary_pool_source_tokens",
        "rows": len(values),
        "min": min(values) if values else 0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else 0,
        "limit": limit,
        "rows_over_limit": sum(value > limit for value in values),
    }


def _source_key(value: str | None) -> str:
    return "null" if value is None else value


def _crosscheck_counter(observed: Any, expected: Coverage, label: str) -> None:
    if not isinstance(observed, Mapping):
        raise ValueError(f"preflight_{label}_missing")
    expected_value = expected.preflight_projection()
    for field in ("rows", "instructions", "fallback"):
        if observed.get(field) != expected_value[field]:
            raise ValueError(
                f"preflight_{label}_{field}_mismatch:"
                f"{observed.get(field)}!={expected_value[field]}"
            )
    rate = observed.get("fallback_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isclose(
        float(rate), expected_value["fallback_rate"], rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(f"preflight_{label}_fallback_rate_mismatch")


@dataclasses.dataclass(frozen=True)
class AlignmentAggregates:
    by_role: dict[str, Coverage]
    by_family: dict[str, Coverage]
    by_source: dict[str, Coverage]
    by_role_family: dict[str, Coverage]
    by_role_source: dict[str, Coverage]
    by_family_source: dict[str, Coverage]
    by_dataset: dict[str, Coverage]
    token_lengths_by_role: dict[str, tuple[int, ...]]


def aggregate_and_crosscheck_alignment(state: BundleState) -> AlignmentAggregates:
    by_role: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_family: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_source: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_role_family: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_role_source: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_family_source: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_dataset: dict[str, Coverage] = collections.defaultdict(Coverage)
    token_lengths_by_role: dict[str, list[int]] = collections.defaultdict(list)
    task_ids: set[str] = set()
    for index, row in enumerate(state.alignment_rows):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in task_ids:
            raise ValueError(f"alignment_task_id_missing_or_duplicate:{index}")
        task_ids.add(task_id)
        role = str(row.get("role") or "")
        split = str(row.get("split") or "")
        if (role, split) not in {("fit", "train"), ("measure", "dev")}:
            raise ValueError(f"alignment_role_split_mismatch:{index}:{role}:{split}")
        family = str(row.get("family") or "")
        if family not in ALLOWED_FAMILIES:
            raise ValueError(f"alignment_unknown_family:{index}:{family}")
        source_pool = row.get("source_pool")
        if source_pool is not None and (
            not isinstance(source_pool, str) or not source_pool.strip()
        ):
            raise ValueError(f"alignment_invalid_source_pool:{index}")
        if source_pool not in EXPECTED_SOURCE_POOLS[family]:
            raise ValueError(
                f"alignment_family_source_pool_mismatch:{index}:{family}:{source_pool}"
            )
        dataset = str(row.get("dataset") or "")
        if not dataset:
            raise ValueError(f"alignment_missing_dataset:{index}")
        instructions = _require_plain_nonnegative_int(
            row.get("instruction_count"), f"alignment_{index}_instruction_count"
        )
        fallback = _require_plain_nonnegative_int(
            row.get("fallback_instructions"), f"alignment_{index}_fallback"
        )
        source_tokens = _require_plain_nonnegative_int(
            row.get("source_tokens"), f"alignment_{index}_source_tokens"
        )
        if source_tokens != len(state.model_rows[index]["compact_input_ids"]):
            raise ValueError(f"alignment_{index}_source_token_count_mismatch")
        token_lengths_by_role[role].append(source_tokens)
        source = _source_key(source_pool)
        for key, table in (
            (role, by_role),
            (family, by_family),
            (source, by_source),
            (f"{role}:{family}", by_role_family),
            (f"{role}:{source}", by_role_source),
            (f"{family}:{source}", by_family_source),
            (dataset, by_dataset),
        ):
            table[key].add(instructions, fallback)

    preflight = state.preflight
    rows_by_role = preflight.get("rows_by_role")
    rows_by_family = preflight.get("rows_by_family")
    if not isinstance(rows_by_role, Mapping) or not isinstance(rows_by_family, Mapping):
        raise ValueError("preflight_row_breakdowns_missing")
    if dict(rows_by_role) != {key: value.rows for key, value in by_role.items()}:
        raise ValueError("preflight_rows_by_role_mismatch")
    if dict(rows_by_family) != {key: value.rows for key, value in by_family.items()}:
        raise ValueError("preflight_rows_by_family_mismatch")

    preflight_role = preflight.get("fallback_by_role")
    preflight_family = preflight.get("fallback_by_family")
    preflight_role_family = preflight.get("fallback_by_role_and_family")
    preflight_dataset = preflight.get("fallback_by_dataset")
    if not all(
        isinstance(value, Mapping)
        for value in (preflight_role, preflight_family, preflight_role_family, preflight_dataset)
    ):
        raise ValueError("preflight_fallback_breakdowns_missing")
    if set(preflight_role) != set(by_role) or set(preflight_family) != set(by_family):
        raise ValueError("preflight_role_or_family_key_drift")
    for key, value in by_role.items():
        _crosscheck_counter(preflight_role[key], value, f"role_{key}")
    for key, value in by_family.items():
        _crosscheck_counter(preflight_family[key], value, f"family_{key}")
    flattened_role_family: dict[str, Any] = {}
    for role, families in preflight_role_family.items():
        if not isinstance(families, Mapping):
            raise ValueError(f"preflight_role_family_{role}_must_be_object")
        for family, value in families.items():
            flattened_role_family[f"{role}:{family}"] = value
    if set(flattened_role_family) != set(by_role_family):
        raise ValueError("preflight_role_family_key_drift")
    for key, value in by_role_family.items():
        _crosscheck_counter(flattened_role_family[key], value, f"role_family_{key}")
    if set(preflight_dataset) != set(by_dataset):
        raise ValueError("preflight_dataset_key_drift")
    for key, value in by_dataset.items():
        _crosscheck_counter(preflight_dataset[key], value, "dataset")

    max_source_tokens = _require_plain_nonnegative_int(
        state.contract.get("max_source_tokens"), "contract_max_source_tokens"
    )
    all_token_lengths = [
        value for role in sorted(token_lengths_by_role) for value in token_lengths_by_role[role]
    ]
    expected_tokens = _token_report(all_token_lengths, max_source_tokens)
    preflight_tokens = preflight.get("tokens")
    if not isinstance(preflight_tokens, Mapping):
        raise ValueError("preflight_token_summary_missing")
    for field in ("min", "p50", "p95", "p99", "max", "limit", "rows_over_limit"):
        if preflight_tokens.get(field) != expected_tokens[field]:
            raise ValueError(f"preflight_token_{field}_mismatch")

    return AlignmentAggregates(
        by_role=dict(by_role),
        by_family=dict(by_family),
        by_source=dict(by_source),
        by_role_family=dict(by_role_family),
        by_role_source=dict(by_role_source),
        by_family_source=dict(by_family_source),
        by_dataset=dict(by_dataset),
        token_lengths_by_role={
            key: tuple(value) for key, value in token_lengths_by_role.items()
        },
    )


def build_alignment_reports(
    state: BundleState, aggregates: AlignmentAggregates
) -> tuple[dict[str, Any], dict[str, Any]]:
    dev = aggregates.by_role.get("measure")
    if dev is None or dev.rows <= 0:
        raise ValueError("sealed_bundle_has_no_dev_measure_rows")
    dev_family = {
        key.split(":", 1)[1]: value
        for key, value in aggregates.by_role_family.items()
        if key.startswith("measure:")
    }
    dev_source = {
        key.split(":", 1)[1]: value
        for key, value in aggregates.by_role_source.items()
        if key.startswith("measure:")
    }
    if sum(item.rows for item in dev_family.values()) != dev.rows:
        raise AssertionError("dev_family_partition_drift")
    if sum(item.rows for item in dev_source.values()) != dev.rows:
        raise AssertionError("dev_source_pool_partition_drift")
    dev_tokens = _token_report(
        aggregates.token_lengths_by_role["measure"],
        _require_plain_nonnegative_int(
            state.contract.get("max_source_tokens"), "contract_max_source_tokens"
        ),
    )
    common = {
        "schema": AUDIT_SCHEMA,
        "bundle_contract_sha256": state.bindings["compact_contract.json"]["sha256"],
        "bundle_preflight_sha256": state.bindings["preflight_report.json"]["sha256"],
        "alignment_private_sha256": state.bindings["alignment_private.jsonl"]["sha256"],
        "codebook_sha256": state.bindings["codebook.json"]["sha256"],
    }
    dev_report = {
        **common,
        "audit": "sealed_v3_dev_instruction_codebook_fallback",
        "scope": "full-v3-release-alignment-derived-instruction-fallback",
        "coverage": dev.report(),
        "coverage_by_family": _render(dev_family),
        "coverage_by_source_pool": _render(dev_source),
        "tokens": dev_tokens,
        "cross_checks": {
            "alignment_rows_match_preflight_role": True,
            "alignment_family_partitions_match_preflight": True,
            "alignment_dataset_partitions_match_preflight": True,
            "alignment_token_lengths_match_public_rows_and_preflight": True,
        },
        "gates": {"all_rows_within_sealed_token_limit": dev_tokens["rows_over_limit"] == 0},
        "passed": True,
    }

    topup_family = {
        key: value for key, value in aggregates.by_family.items() if key in TOPUP_FAMILIES
    }
    if not topup_family or sum(value.rows for value in topup_family.values()) <= 0:
        raise ValueError("sealed_bundle_has_no_topup_rows")
    topup_role_family = {
        key: value
        for key, value in aggregates.by_role_family.items()
        if key.split(":", 1)[1] in TOPUP_FAMILIES
    }
    topup_family_source = {
        key: value
        for key, value in aggregates.by_family_source.items()
        if key.split(":", 1)[0] in TOPUP_FAMILIES
    }
    topup_total = Coverage()
    for value in topup_family.values():
        topup_total.rows += value.rows
        topup_total.instructions += value.instructions
        topup_total.fallback += value.fallback
        topup_total.rows_with_fallback += value.rows_with_fallback
    if sum(value.rows for value in topup_role_family.values()) != topup_total.rows:
        raise AssertionError("topup_role_family_partition_drift")
    if sum(value.rows for value in topup_family_source.values()) != topup_total.rows:
        raise AssertionError("topup_family_source_pool_partition_drift")
    topup_report = {
        **common,
        "audit": "sealed_v3_topup_family_and_source_pool_instruction_fallback",
        "scope": "full-v3-release-alignment-derived-instruction-fallback",
        "coverage": topup_total.report(),
        "coverage_by_family": _render(topup_family),
        "coverage_by_role_and_family": _render(topup_role_family),
        "coverage_by_family_and_source_pool": _render(topup_family_source),
        "source_pool_policy": {
            key: sorted("null" if item is None else item for item in values)
            for key, values in EXPECTED_SOURCE_POOLS.items()
        },
        "cross_checks": {
            "alignment_family_totals_match_preflight": True,
            "alignment_role_family_totals_match_preflight": True,
            "source_pool_partitions_sum_to_preflight_family_totals": True,
        },
        "passed": True,
    }
    return dev_report, topup_report


def audit_humaneval_instruction_coverage(
    state: BundleState,
    humaneval_path: Path,
    *,
    expected_sha256: str = CANONICAL_HUMANEVAL_SHA256,
    expected_rows: int = CANONICAL_HUMANEVAL_ROWS,
) -> dict[str, Any]:
    observed_sha = sha256_file(humaneval_path)
    if observed_sha != require_sha256(expected_sha256, "expected_humaneval_sha256"):
        raise ValueError(
            f"canonical_humaneval_sha256_mismatch:{observed_sha}!={expected_sha256}"
        )
    rows = read_jsonl(humaneval_path, "humaneval")
    if len(rows) != expected_rows:
        raise ValueError(f"canonical_humaneval_row_count_mismatch:{len(rows)}!={expected_rows}")
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("canonical_humaneval_task_ids_missing_or_nonunique")

    vocabulary = set(state.expansions)
    coverage = Coverage()
    fallback_payload_token_lengths: list[int] = []
    fallback_unique: set[str] = set()
    dfg_edges = 0
    fallback_roundtrips = 0
    withheld_contract = {"return_type", "parameter_types", "parameter_names"}
    for index, row in enumerate(rows):
        protocol = row.get("benchmark_protocol")
        graph = row.get("graph_v2")
        if not isinstance(protocol, Mapping) or not isinstance(graph, Mapping):
            raise ValueError(f"humaneval_{index}_missing_protocol_or_graph")
        if row.get("function") != "candidate" or row.get("camel_case_function_name") != "candidate":
            raise ValueError(f"humaneval_{index}_target_is_not_candidate")
        if row.get("prompt_signature_mode") != "name_only" or row.get("dart_function_signature") != "":
            raise ValueError(f"humaneval_{index}_is_not_name_only")
        if protocol.get("neutral_target_name") != "candidate":
            raise ValueError(f"humaneval_{index}_protocol_target_is_not_candidate")
        withholds = protocol.get("prompt_withholds")
        if not isinstance(withholds, list) or not withheld_contract.issubset(set(withholds)):
            raise ValueError(f"humaneval_{index}_signature_withholding_contract_drift")
        if graph.get("extractor_sha256") != graph_codec.ROUTE_SPECS[
            graph_codec.ROUTE_CURRENT
        ].combined_sha256:
            raise ValueError(f"humaneval_{index}_is_not_current_extractor")

        canonical = graph_codec.canonicalize(
            row, state.codebook.get("symbol_policy", "runtime_aware")
        )
        if canonical.get("dfg_route") != graph_codec.ROUTE_CURRENT:
            raise ValueError(f"humaneval_{index}_canonical_route_is_not_current")
        regenerated = state.registry[graph_codec.ROUTE_CURRENT]["build_dfg"](
            canonical["blocks"], canonical["cfg_edges"], max_edges=100000
        )
        regenerated_dfg = graph_codec._sort_dfg(
            graph_codec._canonical_dfg_edge(edge, graph_codec.ROUTE_CURRENT)
            for edge in regenerated
        )
        if regenerated_dfg != canonical["dfg_edges"]:
            raise ValueError(f"humaneval_{index}_dfg_edge_for_edge_mismatch")
        dfg_edges += len(regenerated_dfg)
        instructions = [
            instruction
            for block in canonical["blocks"]
            for instruction in block["instructions"]
        ]
        fallback = [instruction for instruction in instructions if instruction not in vocabulary]
        coverage.add(len(instructions), len(fallback))
        for instruction in fallback:
            ids = state.tokenizer.encode(instruction, add_special_tokens=False).ids
            if not ids:
                raise ValueError(f"humaneval_{index}_empty_native_fallback_tokenization")
            recovered = state.tokenizer.decode(ids, skip_special_tokens=False)
            if recovered != instruction:
                raise ValueError(f"humaneval_{index}_native_fallback_not_reversible")
            fallback_payload_token_lengths.append(len(ids))
            fallback_unique.add(instruction)
            fallback_roundtrips += 1

    report = {
        "schema": AUDIT_SCHEMA,
        "audit": "scrubbed_humaneval_instruction_codebook_coverage",
        "scope": "instruction_codebook_coverage_only",
        "graph_representation": "v2-canonical-graph-shared-by-v3",
        "full_v3_source_token_measurement": False,
        "input_binding": _binding(humaneval_path),
        "bundle_contract_sha256": state.bindings["compact_contract.json"]["sha256"],
        "codebook_sha256": state.bindings["codebook.json"]["sha256"],
        "tokenizer_sha256": state.bindings["tokenizer"]["sha256"],
        "rows": len(rows),
        "unique_task_ids": len(task_ids),
        "coverage": coverage.report(),
        "current_extractor_route": graph_codec.ROUTE_CURRENT,
        "current_extractor_sha256": graph_codec.ROUTE_SPECS[
            graph_codec.ROUTE_CURRENT
        ].combined_sha256,
        "exact_canonical_and_dfg_roundtrip_rows": len(rows),
        "dfg_edges_regenerated_and_matched_edge_for_edge": dfg_edges,
        "fallback_representation": {
            "encoding": "native-qwen-token-ids-between-reserved-<R>-and-<E>-atoms",
            "instruction_occurrences": len(fallback_payload_token_lengths),
            "unique_instruction_strings": len(fallback_unique),
            "native_qwen_payload_tokens": sum(fallback_payload_token_lengths),
            "tokens_per_fallback_instruction": {
                "min": min(fallback_payload_token_lengths) if fallback_payload_token_lengths else 0,
                "p50": _percentile(fallback_payload_token_lengths, 0.50),
                "p95": _percentile(fallback_payload_token_lengths, 0.95),
                "max": max(fallback_payload_token_lengths) if fallback_payload_token_lengths else 0,
            },
            "exact_native_token_roundtrip_occurrences": fallback_roundtrips,
            "reversible": fallback_roundtrips == len(fallback_payload_token_lengths),
        },
        "non_claims": {
            "full_v3_binary_pool_stream_was_available": False,
            "full_v3_source_token_count_was_measured": False,
            "human_eval_9000_token_gate_was_evaluated": False,
        },
        "gates": {
            "canonical_154_row_dataset_hash_matched": True,
            "all_task_ids_unique": True,
            "all_targets_candidate": True,
            "all_prompts_name_only": True,
            "all_rows_current_extractor": True,
            "all_dfgs_regenerated_edge_for_edge": True,
            "all_fallback_payloads_native_token_roundtrip_exactly": True,
        },
        "passed": True,
    }
    return report


def _json_payload(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def emit_sealed_reports(
    output_dir: Path,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    state: BundleState,
    humaneval_path: Path,
    script_path: Path,
) -> None:
    if set(reports) != REPORT_NAMES:
        raise ValueError("audit_report_set_mismatch")
    if any(report.get("passed") is not True for report in reports.values()):
        raise ValueError("refusing_to_seal_failed_audit_report")
    if output_dir.exists():
        raise FileExistsError(f"audit_output_must_not_already_exist:{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        report_bindings: dict[str, dict[str, Any]] = {}
        for name in sorted(REPORT_NAMES):
            payload = _json_payload(reports[name])
            (staging / name).write_bytes(payload)
            report_bindings[name] = {
                "sha256": sha256_bytes(payload),
                "passed": True,
            }
        seal = {
            "schema": SEAL_SCHEMA,
            "all_passed": True,
            "bundle_bindings": state.bindings,
            "humaneval_binding": _binding(humaneval_path),
            "script": _binding(script_path),
            "reports": report_bindings,
        }
        (staging / "generalization_audit_seal.json").write_bytes(_json_payload(seal))
        checksum_names = sorted(OUTPUT_NAMES - {"SHA256SUMS.txt"})
        checksums = "".join(
            f"{sha256_file(staging / name)}  {name}\n" for name in checksum_names
        ).encode("ascii")
        (staging / "SHA256SUMS.txt").write_bytes(checksums)
        observed = {item.name for item in staging.iterdir()}
        if observed != OUTPUT_NAMES or any(item.is_dir() for item in staging.iterdir()):
            raise AssertionError("staged_audit_file_set_mismatch")
        parsed = _parse_sha256s(staging / "SHA256SUMS.txt")
        if set(parsed) != OUTPUT_NAMES - {"SHA256SUMS.txt"}:
            raise AssertionError("staged_audit_checksum_coverage_mismatch")
        for name, expected in parsed.items():
            if sha256_file(staging / name) != expected:
                raise AssertionError(f"staged_audit_checksum_mismatch:{name}")
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--humaneval", type=Path, default=CANONICAL_HUMANEVAL)
    parser.add_argument("--codec", type=Path, default=Path(codec.__file__).resolve())
    parser.add_argument(
        "--graph-codec", type=Path, default=Path(graph_codec.__file__).resolve()
    )
    parser.add_argument(
        "--release-builder",
        type=Path,
        default=ROOT / "scripts/data/build_compact_qwen_v3_release.py",
    )
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
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    expected_humaneval_sha256: str = CANONICAL_HUMANEVAL_SHA256,
    expected_humaneval_rows: int = CANONICAL_HUMANEVAL_ROWS,
) -> int:
    args = _build_parser().parse_args(argv)
    state = verify_bundle(
        args.bundle,
        args.tokenizer_json,
        codec_path=args.codec,
        graph_codec_path=args.graph_codec,
        release_builder_path=args.release_builder,
        legacy_cfg_extractor=args.legacy_cfg_extractor,
        legacy_dfg_extractor=args.legacy_dfg_extractor,
        current_cfg_extractor=args.current_cfg_extractor,
        current_dfg_extractor=args.current_dfg_extractor,
    )
    aggregates = aggregate_and_crosscheck_alignment(state)
    dev_report, topup_report = build_alignment_reports(state, aggregates)
    humaneval_report = audit_humaneval_instruction_coverage(
        state,
        args.humaneval,
        expected_sha256=expected_humaneval_sha256,
        expected_rows=expected_humaneval_rows,
    )
    reports = {
        "dev_fallback_audit.json": dev_report,
        "topup_family_source_pool_fallback_audit.json": topup_report,
        "scrubbed_humaneval_instruction_codebook_audit.json": humaneval_report,
    }
    emit_sealed_reports(
        args.output_dir,
        reports,
        state=state,
        humaneval_path=args.humaneval,
        script_path=Path(__file__).resolve(),
    )
    print(
        json.dumps(
            {
                "schema": SEAL_SCHEMA,
                "output_dir": str(args.output_dir),
                "reports": sorted(reports),
                "all_passed": True,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
