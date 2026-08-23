#!/usr/bin/env python3
"""Fail-closed generalization audits for the Phase-0 compact-Qwen v2 release.

This audit deliberately consumes the private, prepared graph rows.  It never
rewrites them and never emits task-level source, labels, instructions, or model
inputs.  The three emitted reports contain aggregate fallback/token statistics:

* the held-out Phase-0 dev split;
* the harmonized scrubbed HumanEval family; and
* family/source-pool coverage, including a master-train-only vocabulary
  counterfactual.

HumanEval's retained ``graph_v2.extractor_sha256`` describes the older binary
build rather than the harmonized DFG.  A legacy-route override is therefore
permitted only after the harmonization manifest, its output dataset, and its
frozen DFG extractor have all been hash-verified.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "scrubbed_master_v2_release/direct_compact_phase0_s44_v2"
DEFAULT_BUNDLE = RELEASE / "compact_qwen_phase0_s44_v2"
DEFAULT_PREPARED = RELEASE / "prepared"
DEFAULT_HUMANEVAL_DIR = (
    ROOT / "data/testing/humaneval_v2_nameonly_dfg_harmonized"
)

AUDIT_SCHEMA = "compact-qwen-v2-generalization-audit-v1"
SEAL_SCHEMA = "compact-qwen-v2-generalization-audit-seal-v1"
ALLOWED_FAMILIES = {"master", "topup_s45", "topup_s46"}
EXPECTED_SOURCE_POOLS = {
    "master": {None},
    "topup_s45": {"base_llm", "topup_s44", "topup_s45"},
    "topup_s46": {"topup_s46"},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def require_digest(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError(f"{label}: expected lowercase SHA-256 digest")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            result.append(value)
    return result


def load_codec(path: Path, expected_sha256: str) -> Any:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"codec SHA mismatch: {observed} != contract {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location("compact_qwen_v2_audit_codec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import compact codec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def token_summary(values: list[int], limit: int) -> dict[str, Any]:
    return {
        "kind": "compact_source_only",
        "rows": len(values),
        "min": min(values) if values else 0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
        "limit": limit,
        "rows_over_limit": sum(value > limit for value in values),
    }


@dataclass
class Coverage:
    rows: int = 0
    instructions: int = 0
    fallback_instructions: int = 0
    rows_with_fallback: int = 0

    def add(self, instructions: Iterable[str], vocabulary: set[str]) -> None:
        values = list(instructions)
        fallback = sum(value not in vocabulary for value in values)
        self.rows += 1
        self.instructions += len(values)
        self.fallback_instructions += fallback
        self.rows_with_fallback += int(fallback > 0)

    def report(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "instructions": self.instructions,
            "fallback_instructions": self.fallback_instructions,
            "fallback_rate": (
                self.fallback_instructions / self.instructions
                if self.instructions
                else 0.0
            ),
            "rows_with_fallback": self.rows_with_fallback,
            "rows_with_fallback_rate": (
                self.rows_with_fallback / self.rows if self.rows else 0.0
            ),
        }


def instructions(canonical: dict[str, Any]) -> list[str]:
    return [
        instruction
        for block in canonical["blocks"]
        for instruction in block["instructions"]
    ]


def source_pool_key(value: Any) -> str:
    if value is None:
        return "null"
    result = str(value)
    if not result:
        raise ValueError("source_pool must be null or a nonempty string")
    return result


def private_metadata(row: dict[str, Any], expected_role: str) -> dict[str, Any]:
    metadata = row.get("compact_private_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{row.get('task_id')}: missing compact_private_metadata")
    family = str(metadata.get("family") or "")
    source_pool = metadata.get("source_pool")
    if family not in ALLOWED_FAMILIES:
        raise ValueError(f"{row.get('task_id')}: invalid family {family!r}")
    if source_pool not in EXPECTED_SOURCE_POOLS[family]:
        raise ValueError(
            f"{row.get('task_id')}: family/source_pool mismatch: "
            f"{family!r}/{source_pool!r}"
        )
    if metadata.get("phase0_split") != expected_role:
        raise ValueError(
            f"{row.get('task_id')}: prepared role does not match Phase-0 metadata"
        )
    if metadata.get("target_function") != "candidate" or row.get("function") != "candidate":
        raise ValueError(f"{row.get('task_id')}: target function is not candidate")
    return metadata


def exact_roundtrip(
    codec: Any,
    registry: dict[str, dict[str, Any]],
    canonical: dict[str, Any],
    code: dict[str, int],
    expansions: list[str],
) -> tuple[str, int]:
    compact = codec.encode(canonical, code)
    decoded = codec.decode(compact, expansions)
    route = decoded["dfg_route"]
    regenerated = registry[route]["build_dfg"](
        decoded["blocks"], decoded["cfg_edges"], max_edges=100000
    )
    decoded["dfg_edges"] = codec._sort_dfg(
        codec._canonical_dfg_edge(edge, route) for edge in regenerated
    )
    if decoded != canonical:
        raise ValueError(
            "canonical/DFG round-trip mismatch: "
            f"expected={sha256_bytes(stable(canonical))} "
            f"observed={sha256_bytes(stable(decoded))}"
        )
    return compact, len(decoded["dfg_edges"])


def aggregate_breakdowns(
    entries: list[tuple[str, dict[str, Any], dict[str, Any]]],
    vocabulary: set[str],
) -> dict[str, Any]:
    overall = Coverage()
    by_role: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_family: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_source: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_role_family: dict[str, Coverage] = collections.defaultdict(Coverage)
    by_role_source: dict[str, Coverage] = collections.defaultdict(Coverage)
    for role, row, canonical in entries:
        metadata = private_metadata(row, role)
        family = metadata["family"]
        source = source_pool_key(metadata.get("source_pool"))
        values = instructions(canonical)
        overall.add(values, vocabulary)
        by_role[role].add(values, vocabulary)
        by_family[family].add(values, vocabulary)
        by_source[source].add(values, vocabulary)
        by_role_family[f"{role}:{family}"].add(values, vocabulary)
        by_role_source[f"{role}:{source}"].add(values, vocabulary)
    render = lambda values: {
        key: values[key].report() for key in sorted(values)
    }
    return {
        "overall": overall.report(),
        "by_phase0_split": render(by_role),
        "by_requested_family": render(by_family),
        "by_exact_source_pool": render(by_source),
        "by_phase0_split_and_family": render(by_role_family),
        "by_phase0_split_and_source_pool": render(by_role_source),
    }


def input_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": file_sha256(resolved)}


def write_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return sha256_bytes(payload)


def discover_tokenizer() -> Path | None:
    # A convenience only; every run still hash-binds the selected file.
    candidates = []
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    root = home / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots"
    if root.exists():
        candidates.extend(root.glob("*/tokenizer.json"))
    return sorted(candidates)[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--train", type=Path, default=DEFAULT_PREPARED / "train_codec_private.jsonl")
    parser.add_argument("--dev", type=Path, default=DEFAULT_PREPARED / "dev_codec_private.jsonl")
    parser.add_argument("--codebook", type=Path, default=DEFAULT_BUNDLE / "codebook.json")
    parser.add_argument("--contract", type=Path, default=DEFAULT_BUNDLE / "compact_contract.json")
    parser.add_argument("--preflight", type=Path, default=DEFAULT_BUNDLE / "preflight_report.json")
    parser.add_argument("--codec", type=Path, default=ROOT / "scripts/data/build_compact_qwen_v2.py")
    parser.add_argument(
        "--humaneval",
        type=Path,
        default=DEFAULT_HUMANEVAL_DIR / "humaneval_v2_nameonly_dfg_harmonized_private.jsonl",
    )
    parser.add_argument(
        "--harmonization-manifest",
        type=Path,
        default=DEFAULT_HUMANEVAL_DIR / "harmonization_manifest.json",
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
    parser.add_argument("--tokenizer-json", type=Path, default=discover_tokenizer())
    parser.add_argument("--output-dir", type=Path, default=RELEASE / "audits")
    args = parser.parse_args()
    if args.tokenizer_json is None:
        raise ValueError("--tokenizer-json is required (automatic discovery failed)")

    # Verify the sealed codec bundle before importing or interpreting any row.
    contract = read_json(args.contract)
    codebook = read_json(args.codebook)
    preflight = read_json(args.preflight)
    if contract.get("schema") != "direct-compact-causal-v2":
        raise ValueError("contract is not direct-compact-causal-v2")
    if codebook.get("schema") != "compact-qwen-v2-codebook":
        raise ValueError("codebook is not compact-qwen-v2-codebook")
    if preflight.get("schema") != "compact-qwen-v2-preflight" or not preflight.get("passed"):
        raise ValueError("compact-Qwen v2 preflight is absent or failed")
    contract_codebook_sha = require_digest(contract.get("codebook_sha256"), "contract.codebook_sha256")
    if file_sha256(args.codebook) != contract_codebook_sha:
        raise ValueError("codebook does not match compact contract")
    codec_sha = require_digest(contract.get("codec_sha256"), "contract.codec_sha256")
    codec = load_codec(args.codec, codec_sha)
    tokenizer_sha = file_sha256(args.tokenizer_json)
    if tokenizer_sha != require_digest(contract.get("tokenizer_json_sha256"), "contract.tokenizer_json_sha256"):
        raise ValueError("tokenizer does not match compact contract")
    if codebook.get("tokenizer_json_sha256") != tokenizer_sha:
        raise ValueError("tokenizer does not match codebook")
    if codebook.get("extractor_routes") != contract.get("extractor_routes"):
        raise ValueError("codebook/contract extractor-route drift")
    if codebook.get("runtime_symbol_policy_sha256") != contract.get("runtime_symbol_policy_sha256"):
        raise ValueError("codebook/contract runtime symbol policy drift")
    if codebook.get("fit_public_sha256") != file_sha256(args.train):
        raise ValueError("codebook was not fit on the supplied prepared train side")

    registry = codec.load_route_registry(
        args.legacy_cfg_extractor,
        args.legacy_dfg_extractor,
        args.current_cfg_extractor,
        args.current_dfg_extractor,
    )
    observed_routes = codec.route_contract(registry)
    if observed_routes != contract.get("extractor_routes"):
        raise ValueError("loaded extractor registry does not match sealed routes")

    expansions = codebook.get("expansions")
    if not isinstance(expansions, list) or not all(isinstance(x, str) for x in expansions):
        raise ValueError("invalid codebook expansions")
    if len(expansions) != int(codebook.get("codebook_size", -1)):
        raise ValueError("codebook expansion count drift")
    code = {value: index for index, value in enumerate(expansions)}
    if len(code) != len(expansions):
        raise ValueError("duplicate instruction in codebook")
    atom_ids = codebook.get("source_atom_ids")
    if not isinstance(atom_ids, dict):
        raise ValueError("missing source atom IDs")
    atom_ids = {str(key): int(value) for key, value in atom_ids.items()}
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    max_source_tokens = int(contract.get("max_source_tokens", -1))
    if not 0 < max_source_tokens <= 9000:
        raise ValueError("invalid sealed source-token limit")

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    if len(train_rows) != int(codebook.get("fit_retained", -1)):
        raise ValueError("prepared train count does not match codebook fit count")
    task_ids = [str(row.get("task_id") or "") for row in train_rows + dev_rows]
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("prepared train/dev task IDs are missing or non-bijective")

    entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    train_frequencies: collections.Counter[str] = collections.Counter()
    master_frequencies: collections.Counter[str] = collections.Counter()
    for role, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            metadata = private_metadata(row, role)
            canonical = codec.canonicalize(row, codebook.get("symbol_policy"))
            values = instructions(canonical)
            entries.append((role, row, canonical))
            if role == "train":
                train_frequencies.update(values)
                if metadata["family"] == "master":
                    master_frequencies.update(values)
    rebuilt_expansions = [
        value for value, _ in train_frequencies.most_common(len(expansions))
    ]
    if rebuilt_expansions != expansions:
        raise ValueError("sealed codebook is not the deterministic prepared-train fit")
    final_vocabulary = set(expansions)
    master_expansions = [
        value for value, _ in master_frequencies.most_common(len(expansions))
    ]
    master_vocabulary = set(master_expansions)

    # Dev audit: exact codec+DFG regeneration and actual compact tokenization.
    dev_lengths: list[int] = []
    dev_coverage = Coverage()
    dev_routes: collections.Counter[str] = collections.Counter()
    dev_dfg_edges: collections.Counter[str] = collections.Counter()
    dev_family: dict[str, Coverage] = collections.defaultdict(Coverage)
    dev_source: dict[str, Coverage] = collections.defaultdict(Coverage)
    for role, row, canonical in entries:
        if role != "dev":
            continue
        metadata = private_metadata(row, role)
        compact, edge_count = exact_roundtrip(codec, registry, canonical, code, expansions)
        ids = codec.compact_ids(compact, tokenizer, atom_ids)
        dev_lengths.append(len(ids))
        values = instructions(canonical)
        dev_coverage.add(values, final_vocabulary)
        dev_family[metadata["family"]].add(values, final_vocabulary)
        dev_source[source_pool_key(metadata.get("source_pool"))].add(values, final_vocabulary)
        dev_routes[canonical["dfg_route"]] += 1
        dev_dfg_edges[canonical["dfg_route"]] += edge_count
    dev_tokens = token_summary(dev_lengths, max_source_tokens)

    common_bindings = {
        "train": input_binding(args.train),
        "dev": input_binding(args.dev),
        "codebook": input_binding(args.codebook),
        "contract": input_binding(args.contract),
        "preflight": input_binding(args.preflight),
        "codec": input_binding(args.codec),
        "tokenizer": input_binding(args.tokenizer_json),
        "legacy_cfg_extractor": input_binding(args.legacy_cfg_extractor),
        "legacy_dfg_extractor": input_binding(args.legacy_dfg_extractor),
        "current_cfg_extractor": input_binding(args.current_cfg_extractor),
        "current_dfg_extractor": input_binding(args.current_dfg_extractor),
    }
    route_binding = {
        "extractor_routes": observed_routes,
        "runtime_symbol_policy_sha256": contract["runtime_symbol_policy_sha256"],
        "target_function": contract["target_function"],
    }
    dev_report = {
        "schema": AUDIT_SCHEMA,
        "audit": "phase0_dev_fallback_and_tokens",
        "input_bindings": common_bindings,
        "route_binding": route_binding,
        "rows": len(dev_rows),
        "coverage": dev_coverage.report(),
        "coverage_by_requested_family": {
            key: dev_family[key].report() for key in sorted(dev_family)
        },
        "coverage_by_exact_source_pool": {
            key: dev_source[key].report() for key in sorted(dev_source)
        },
        "tokens": dev_tokens,
        "exact_canonical_and_dfg_roundtrip_rows": len(dev_rows),
        "dfg_edges_matched_edge_for_edge_by_route": dict(sorted(dev_dfg_edges.items())),
        "rows_by_extractor_route": dict(sorted(dev_routes.items())),
        "gates": {
            "all_rows_roundtrip_exactly": len(dev_lengths) == len(dev_rows),
            "all_rows_within_9000_tokens": dev_tokens["rows_over_limit"] == 0,
        },
    }
    dev_report["passed"] = all(dev_report["gates"].values())

    # HumanEval route override is authorized by the separate, hash-bound
    # harmonization manifest.  The stale graph_v2 extractor field is never used.
    harmonization = read_json(args.harmonization_manifest)
    if harmonization.get("schema") != "scrubbed-humaneval-dfg-harmonization-v1":
        raise ValueError("unexpected HumanEval harmonization manifest schema")
    if harmonization.get("transform") != "replace_dataflow_edges_only":
        raise ValueError("HumanEval harmonization transform is not DFG-only")
    gates = harmonization.get("gates") or {}
    required_harmonization_gates = {
        "canonical_dfg_regenerated_exactly",
        "deterministic_serialization",
        "instructions_cfg_non_dataflow_preserved",
        "public_private_binary_parity",
        "task_id_sets_equal",
    }
    if not all(gates.get(key) is True for key in required_harmonization_gates):
        raise ValueError("HumanEval harmonization manifest gates are incomplete")
    human_sha = file_sha256(args.humaneval)
    if require_digest(
        ((harmonization.get("outputs") or {}).get("private") or {}).get("sha256"),
        "harmonization.outputs.private.sha256",
    ) != human_sha:
        raise ValueError("HumanEval private data does not match harmonization manifest")
    legacy_spec = codec.ROUTE_SPECS[codec.ROUTE_LEGACY]
    if require_digest(
        (harmonization.get("frozen_dfg_extractor") or {}).get("sha256"),
        "harmonization.frozen_dfg_extractor.sha256",
    ) != legacy_spec.dfg_sha256:
        raise ValueError("harmonized HumanEval does not authorize the sealed legacy DFG")
    human_rows = read_jsonl(args.humaneval)
    manifest_rows = harmonization.get("rows")
    if not isinstance(manifest_rows, list):
        raise ValueError("harmonization manifest rows missing")
    human_order = [str(row.get("task_id") or "") for row in human_rows]
    manifest_order = [str(row.get("task_id") or "") for row in manifest_rows]
    if (
        not all(human_order)
        or len(human_order) != len(set(human_order))
        or set(human_order) != set(manifest_order)
        or len(human_order) != int(harmonization.get("row_count", -1))
    ):
        raise ValueError("HumanEval task IDs do not match harmonization manifest")
    if sha256_bytes(stable(human_order)) != require_digest(
        harmonization.get("private_task_order_sha256"),
        "harmonization.private_task_order_sha256",
    ):
        raise ValueError("HumanEval private row order does not match its manifest seal")
    manifest_row_by_id = {str(row["task_id"]): row for row in manifest_rows}

    human_lengths: list[int] = []
    human_coverage = Coverage()
    human_dfg_edges = 0
    for row in human_rows:
        if row.get("function") != "candidate":
            raise ValueError(f"{row.get('task_id')}: HumanEval target is not candidate")
        canonical = codec.canonicalize(
            row, codebook.get("symbol_policy"), route_override=codec.ROUTE_LEGACY
        )
        compact, edge_count = exact_roundtrip(codec, registry, canonical, code, expansions)
        if edge_count != int(manifest_row_by_id[str(row["task_id"])]["new_dfg_edges"]):
            raise ValueError(f"{row.get('task_id')}: per-row harmonized DFG count mismatch")
        human_lengths.append(len(codec.compact_ids(compact, tokenizer, atom_ids)))
        human_coverage.add(instructions(canonical), final_vocabulary)
        human_dfg_edges += edge_count
    expected_human_dfg = int(
        (harmonization.get("edge_counts") or {}).get("harmonized_dataflow_each_side", -1)
    )
    if human_dfg_edges != expected_human_dfg:
        raise ValueError(
            f"HumanEval harmonized DFG total mismatch: {human_dfg_edges} != {expected_human_dfg}"
        )
    human_tokens = token_summary(human_lengths, max_source_tokens)
    human_report = {
        "schema": AUDIT_SCHEMA,
        "audit": "scrubbed_humaneval_fallback_and_tokens",
        "input_bindings": {
            **common_bindings,
            "humaneval_private": input_binding(args.humaneval),
            "harmonization_manifest": input_binding(args.harmonization_manifest),
        },
        "route_binding": route_binding,
        "route_override_authorization": {
            "route": codec.ROUTE_LEGACY,
            "route_atom": legacy_spec.atom,
            "reason": (
                "harmonization manifest is authoritative for replaced DFG; "
                "retained graph_v2 metadata describes the pre-harmonization build"
            ),
            "manifest_schema": harmonization["schema"],
            "manifest_output_private_sha256": human_sha,
            "frozen_dfg_extractor_sha256": legacy_spec.dfg_sha256,
            "required_manifest_gates": sorted(required_harmonization_gates),
        },
        "rows": len(human_rows),
        "coverage": human_coverage.report(),
        "tokens": human_tokens,
        "exact_canonical_and_dfg_roundtrip_rows": len(human_rows),
        "dfg_edges_matched_edge_for_edge": human_dfg_edges,
        "gates": {
            "manifest_authorizes_legacy_route_override": True,
            "all_rows_roundtrip_exactly": len(human_lengths) == len(human_rows),
            "dfg_total_matches_harmonization_manifest": human_dfg_edges == expected_human_dfg,
            "all_rows_within_9000_tokens": human_tokens["rows_over_limit"] == 0,
        },
    }
    human_report["passed"] = all(human_report["gates"].values())

    final_breakdown = aggregate_breakdowns(entries, final_vocabulary)
    master_breakdown = aggregate_breakdowns(entries, master_vocabulary)
    family_report = {
        "schema": AUDIT_SCHEMA,
        "audit": "phase0_family_and_source_pool_fallback",
        "input_bindings": common_bindings,
        "route_binding": route_binding,
        "population": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "rows": len(entries),
            "requested_family_policy": {
                "master": [None],
                "topup_s45": ["base_llm", "topup_s44", "topup_s45"],
                "topup_s46": ["topup_s46"],
            },
        },
        "sealed_train_only_codebook": {
            "fit_population": "all included Phase-0 train rows only",
            "capacity": len(expansions),
            "vocabulary_size": len(final_vocabulary),
            "deterministically_refit_and_matched_seal": True,
            "coverage": final_breakdown,
        },
        "master_train_only_vocabulary_counterfactual": {
            "fit_population": "included family=master Phase-0 train rows only",
            "capacity": len(expansions),
            "vocabulary_size": len(master_vocabulary),
            "fit_rows": sum(
                role == "train"
                and private_metadata(row, role)["family"] == "master"
                for role, row, _ in entries
            ),
            "fit_instructions": sum(master_frequencies.values()),
            "coverage": master_breakdown,
            "token_stats": None,
            "token_stats_note": (
                "Not computed: a master-only codebook changes instruction atom IDs "
                "and therefore requires a distinct tokenizer/embedding-init contract."
            ),
        },
        "gates": {
            "sealed_codebook_rebuilt_exactly_from_train": rebuilt_expansions == expansions,
            "sealed_train_side_has_zero_fallback": (
                final_breakdown["by_phase0_split"]["train"]["fallback_instructions"] == 0
            ),
            "family_and_source_pool_policy_validated_for_every_row": True,
        },
    }
    family_report["passed"] = all(family_report["gates"].values())

    reports = {
        "dev_fallback_token_audit.json": dev_report,
        "scrubbed_humaneval_fallback_token_audit.json": human_report,
        "topup_family_fallback_audit.json": family_report,
    }
    if not all(report["passed"] for report in reports.values()):
        raise ValueError("one or more aggregate audits failed; refusing to seal outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_hashes = {
        name: write_json(args.output_dir / name, report)
        for name, report in reports.items()
    }
    seal = {
        "schema": SEAL_SCHEMA,
        "script": input_binding(Path(__file__).resolve()),
        "input_bindings": common_bindings,
        "additional_input_bindings": {
            "humaneval_private": input_binding(args.humaneval),
            "harmonization_manifest": input_binding(args.harmonization_manifest),
        },
        "reports": {
            name: {"sha256": digest, "passed": reports[name]["passed"]}
            for name, digest in sorted(report_hashes.items())
        },
        "all_passed": True,
    }
    seal_hash = write_json(args.output_dir / "generalization_audit_seal.json", seal)
    checksum_lines = [
        f"{digest}  {name}" for name, digest in sorted(report_hashes.items())
    ] + [f"{seal_hash}  generalization_audit_seal.json"]
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "dev": {
            "rows": len(dev_rows),
            "fallback": dev_coverage.fallback_instructions,
            "instructions": dev_coverage.instructions,
            "max_tokens": dev_tokens["max"],
        },
        "humaneval": {
            "rows": len(human_rows),
            "fallback": human_coverage.fallback_instructions,
            "instructions": human_coverage.instructions,
            "max_tokens": human_tokens["max"],
        },
        "master_only_vocabulary_size": len(master_vocabulary),
        "reports": report_hashes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
