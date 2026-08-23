#!/usr/bin/env python3
"""Build and append the 1,196 missing Phase-0 multi-function representations.

The historical 1,580-row dataset, its compact IDs, its API-readable F2 rows,
and its representation contract are immutable inputs.  Supplemental tasks are
encoded with that exact frozen codebook.  Instructions absent from the frozen
dictionary use the codec's lossless ``<R>raw instruction<E>`` path; no token ID
is reassigned and no frozen embedding meaning changes.

The output includes both a standalone supplemental view (for an append-only
Qwen harvest) and a 2,776-row expanded view.  Expanded JSONL files are produced
by byte concatenation, so the complete historical file is an exact prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUILD_SCHEMA = "binary-multifunction-compact-expansion-v1"
EXPANSION_SEAL_SCHEMA = "multifunction-phase0-fit-expansion-seal-v1"
SPLIT_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
SELECTION_SCHEMA = "multifunction-phase0-fit-expansion-selection-v1"
SELECTION_ROW_SCHEMA = "multifunction-phase0-fit-membership-v1"
F2_MANIFEST_SCHEMA = "verified-api-readable-compact-v2"
F2_ROW_SCHEMA = "audited-frontier-passk-v1"
F2_REPRESENTATION_SCHEMA = "lossless-semantic-f2"
CONTRACT_SCHEMA = "direct-compact-causal-v1"
CODEBOOK_SCHEMA = "compact-qwen-inline-cfg-v2-codebook"
CONSTANT_SCHEMA = "dart-aot-attested-pool-constants-v1"

PARENT_ROWS = 1_580
SUPPLEMENTAL_ROWS = 1_196
EXPANDED_ROWS = 2_776
HELDOUT_ROWS = 175
API_PROMPT_TOKEN_LIMIT = 12_000
CHAT_OVERHEAD_RESERVE = 256
GRAPH_MARKER = "<G2C1>"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ExpansionBuildError(ValueError):
    """The append-only multi-function expansion failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def require_file_hash(
    path: str | Path, expected: str, label: str
) -> dict[str, Any]:
    expected = str(expected).strip().lower()
    if SHA256_RE.fullmatch(expected) is None:
        raise ExpansionBuildError(f"{label} expected SHA-256 is malformed")
    record = file_record(path)
    if record["sha256"] != expected:
        raise ExpansionBuildError(
            f"{label} hash mismatch: expected {expected}, "
            f"observed {record['sha256']}"
        )
    return record


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpansionBuildError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExpansionBuildError(f"{label} is not a JSON object")
    return value


def load_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ExpansionBuildError(
                    f"{label} has a blank row at line {line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExpansionBuildError(
                    f"{label} has invalid JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ExpansionBuildError(
                    f"{label} row {line_number} is not an object"
                )
            rows.append(value)
    if not rows:
        raise ExpansionBuildError(f"{label} is empty")
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def atomic_write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    _atomic_write_bytes(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def atomic_concat(path: Path, first: Path, second: Path) -> None:
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    if first_bytes and not first_bytes.endswith(b"\n"):
        raise ExpansionBuildError(f"{first} does not end with a newline")
    if second_bytes and not second_bytes.endswith(b"\n"):
        raise ExpansionBuildError(f"{second} does not end with a newline")
    _atomic_write_bytes(path, first_bytes + second_bytes)
    observed = path.read_bytes()
    if (
        observed[: len(first_bytes)] != first_bytes
        or observed[len(first_bytes) :] != second_bytes
    ):
        raise ExpansionBuildError(f"append-only byte proof failed for {path}")


def import_pinned_module(
    path: Path, expected_sha256: str, label: str
) -> Any:
    if sha256_file(path) != expected_sha256:
        raise ExpansionBuildError(f"{label} changed before import")
    name = "phase0_expansion_" + hashlib.sha256(
        f"{label}:{path}:{expected_sha256}".encode()
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExpansionBuildError(f"cannot import {label} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha256_file(path) != expected_sha256:
        raise ExpansionBuildError(f"{label} changed during import")
    return module


def ordered_ids(
    rows: Sequence[Mapping[str, Any]], label: str
) -> list[str]:
    result = [str(row.get("task_id") or "") for row in rows]
    if any(not value for value in result):
        raise ExpansionBuildError(f"{label} has an empty task_id")
    if len(set(result)) != len(result):
        raise ExpansionBuildError(f"{label} has duplicate task IDs")
    return result


def percentile(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0
    return ordered[round((len(ordered) - 1) * quantile)]


def validate_split_seal(
    *,
    path: Path,
    record: Mapping[str, Any],
    dataset_record: Mapping[str, Any],
    role: str,
    rows: int,
) -> dict[str, Any]:
    seal = load_json(path, f"{role} split seal")
    if (
        seal.get("schema") != SPLIT_SEAL_SCHEMA
        or seal.get("selected_role") != role
        or int(seal.get("rows", -1)) != rows
        or seal.get("output_sha256") != dataset_record.get("sha256")
        or record.get("sha256") != sha256_file(path)
    ):
        raise ExpansionBuildError(f"{role} split seal contract mismatch")
    return seal


def validate_f2_manifest(
    *,
    path: Path,
    manifest_record: Mapping[str, Any],
    f2_record: Mapping[str, Any],
    dataset_record: Mapping[str, Any],
    rows: int,
) -> dict[str, Any]:
    value = load_json(path, "parent F2 manifest")
    output = value.get("output")
    dataset = value.get("dataset")
    contract = value.get("f2_prompt_contract")
    if (
        value.get("schema") != F2_MANIFEST_SCHEMA
        or int(value.get("rows", -1)) != rows
        or not isinstance(output, Mapping)
        or output.get("sha256") != f2_record.get("sha256")
        or not isinstance(dataset, Mapping)
        or dataset.get("sha256") != dataset_record.get("sha256")
        or not isinstance(contract, Mapping)
        or contract.get("representation_schema") != F2_REPRESENTATION_SCHEMA
        or contract.get("all_rows_within_limit") is not True
        or manifest_record.get("sha256") != sha256_file(path)
    ):
        raise ExpansionBuildError("parent F2 manifest contract mismatch")
    return value


def validate_constants(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        task_id = str(value.get("task_id") or "")
        if not task_id or task_id in result:
            raise ExpansionBuildError(
                f"invalid/duplicate constants task_id {task_id!r}"
            )
        if (
            value.get("schema") != CONSTANT_SCHEMA
            or value.get("err") not in (None, "")
            or not isinstance(value.get("strings"), list)
            or not isinstance(value.get("numbers"), list)
            or not all(isinstance(item, str) for item in value["strings"])
            or not all(isinstance(item, str) for item in value["numbers"])
        ):
            raise ExpansionBuildError(
                f"{task_id}: attested binary constants are incomplete"
            )
        accounting = value.get("accounting")
        if (
            not isinstance(accounting, Mapping)
            or accounting.get("unreadable_entries") != 0
        ):
            raise ExpansionBuildError(
                f"{task_id}: binary constants have unreadable entries"
            )
        result[task_id] = dict(value)
    if set(result) != set(expected_ids):
        raise ExpansionBuildError(
            "constants task set differs from supplemental membership"
        )
    return result


def validate_bundles(
    rows: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
    *,
    extractor_sha256: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        task_id = str(value.get("task_id") or "")
        if not task_id or task_id in result:
            raise ExpansionBuildError(
                f"invalid/duplicate function bundle task_id {task_id!r}"
            )
        producer = value.get("producer")
        inputs = value.get("inputs")
        if (
            not isinstance(producer, Mapping)
            or producer.get("script_sha256") != extractor_sha256
            or not isinstance(inputs, Mapping)
            or SHA256_RE.fullmatch(str(inputs.get("aot_sha256") or "")) is None
            or SHA256_RE.fullmatch(
                str(value.get("model_projection_sha256") or "")
            )
            is None
        ):
            raise ExpansionBuildError(
                f"{task_id}: function bundle binding is incomplete"
            )
        result[task_id] = dict(value)
    if set(result) != set(expected_ids):
        raise ExpansionBuildError(
            "function-bundle task set differs from supplemental membership"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--supplemental-base", required=True, type=Path)
    parser.add_argument("--expected-supplemental-base-sha256", required=True)
    parser.add_argument("--supplemental-base-seal", required=True, type=Path)
    parser.add_argument(
        "--expected-supplemental-base-seal-sha256", required=True
    )
    parser.add_argument("--supplemental-manifest", required=True, type=Path)
    parser.add_argument("--expected-supplemental-manifest-sha256", required=True)
    parser.add_argument("--selection-seal", required=True, type=Path)
    parser.add_argument("--expected-selection-seal-sha256", required=True)
    parser.add_argument("--function-bundles", required=True, type=Path)
    parser.add_argument("--expected-function-bundles-sha256", required=True)
    parser.add_argument("--constants", required=True, type=Path)
    parser.add_argument("--expected-constants-sha256", required=True)
    parser.add_argument("--extractor-script", required=True, type=Path)
    parser.add_argument("--expected-extractor-script-sha256", required=True)
    parser.add_argument("--adapter-script", required=True, type=Path)
    parser.add_argument("--expected-adapter-script-sha256", required=True)
    parser.add_argument("--frozen-contract", required=True, type=Path)
    parser.add_argument("--expected-frozen-contract-sha256", required=True)
    parser.add_argument("--frozen-codebook", required=True, type=Path)
    parser.add_argument("--expected-frozen-codebook-sha256", required=True)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--inline-cfg-codec", required=True, type=Path)
    parser.add_argument("--expected-inline-cfg-codec-sha256", required=True)
    parser.add_argument("--frontier-f2", required=True, type=Path)
    parser.add_argument("--expected-frontier-f2-sha256", required=True)
    parser.add_argument("--parent-dataset", required=True, type=Path)
    parser.add_argument("--expected-parent-dataset-sha256", required=True)
    parser.add_argument("--parent-seal", required=True, type=Path)
    parser.add_argument("--expected-parent-seal-sha256", required=True)
    parser.add_argument("--parent-f2", required=True, type=Path)
    parser.add_argument("--expected-parent-f2-sha256", required=True)
    parser.add_argument("--parent-f2-manifest", required=True, type=Path)
    parser.add_argument("--expected-parent-f2-manifest-sha256", required=True)
    parser.add_argument("--heldout-dataset", required=True, type=Path)
    parser.add_argument("--expected-heldout-dataset-sha256", required=True)
    parser.add_argument("--heldout-seal", required=True, type=Path)
    parser.add_argument("--expected-heldout-seal-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--api-prompt-token-limit",
        type=int,
        default=API_PROMPT_TOKEN_LIMIT,
    )
    parser.add_argument(
        "--chat-overhead-reserve",
        type=int,
        default=CHAT_OVERHEAD_RESERVE,
    )
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.api_prompt_token_limit) != API_PROMPT_TOKEN_LIMIT:
        raise ExpansionBuildError("API prompt token limit must remain 12000")
    if int(args.chat_overhead_reserve) != CHAT_OVERHEAD_RESERVE:
        raise ExpansionBuildError("chat overhead reserve must remain 256")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "supplemental_base": args.supplemental_base,
            "supplemental_base_seal": args.supplemental_base_seal,
            "supplemental_manifest": args.supplemental_manifest,
            "selection_seal": args.selection_seal,
            "function_bundles": args.function_bundles,
            "constants": args.constants,
            "extractor_script": args.extractor_script,
            "adapter_script": args.adapter_script,
            "frozen_contract": args.frozen_contract,
            "frozen_codebook": args.frozen_codebook,
            "tokenizer": args.tokenizer_json,
            "inline_cfg_codec": args.inline_cfg_codec,
            "frontier_f2": args.frontier_f2,
            "parent_dataset": args.parent_dataset,
            "parent_seal": args.parent_seal,
            "parent_f2": args.parent_f2,
            "parent_f2_manifest": args.parent_f2_manifest,
            "heldout_dataset": args.heldout_dataset,
            "heldout_seal": args.heldout_seal,
        }.items()
    }
    expected = {
        "supplemental_base": args.expected_supplemental_base_sha256,
        "supplemental_base_seal": (
            args.expected_supplemental_base_seal_sha256
        ),
        "supplemental_manifest": args.expected_supplemental_manifest_sha256,
        "selection_seal": args.expected_selection_seal_sha256,
        "function_bundles": args.expected_function_bundles_sha256,
        "constants": args.expected_constants_sha256,
        "extractor_script": args.expected_extractor_script_sha256,
        "adapter_script": args.expected_adapter_script_sha256,
        "frozen_contract": args.expected_frozen_contract_sha256,
        "frozen_codebook": args.expected_frozen_codebook_sha256,
        "tokenizer": args.expected_tokenizer_sha256,
        "inline_cfg_codec": args.expected_inline_cfg_codec_sha256,
        "frontier_f2": args.expected_frontier_f2_sha256,
        "parent_dataset": args.expected_parent_dataset_sha256,
        "parent_seal": args.expected_parent_seal_sha256,
        "parent_f2": args.expected_parent_f2_sha256,
        "parent_f2_manifest": args.expected_parent_f2_manifest_sha256,
        "heldout_dataset": args.expected_heldout_dataset_sha256,
        "heldout_seal": args.expected_heldout_seal_sha256,
    }
    inputs = {
        name: require_file_hash(paths[name], expected_sha, name)
        for name, expected_sha in expected.items()
    }
    inputs["expansion_builder"] = file_record(Path(__file__).resolve())

    contract = load_json(paths["frozen_contract"], "frozen contract")
    codebook = load_json(paths["frozen_codebook"], "frozen codebook")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or codebook.get("schema") != CODEBOOK_SCHEMA
        or contract.get("codebook_sha256")
        != inputs["frozen_codebook"]["sha256"]
        or contract.get("codec_sha256")
        != inputs["inline_cfg_codec"]["sha256"]
        or contract.get("tokenizer_json_sha256")
        != inputs["tokenizer"]["sha256"]
        or codebook.get("tokenizer_json_sha256")
        != inputs["tokenizer"]["sha256"]
        or codebook.get("codec_sha256")
        != inputs["inline_cfg_codec"]["sha256"]
        or codebook.get("source_token_expansions")
        != contract.get("source_token_expansions")
    ):
        raise ExpansionBuildError("frozen contract/codebook binding mismatch")
    student_token_limit = int(contract.get("max_source_tokens", -1))
    if student_token_limit != 9_000:
        raise ExpansionBuildError("frozen source limit must remain 9000")

    parent_seal = validate_split_seal(
        path=paths["parent_seal"],
        record=inputs["parent_seal"],
        dataset_record=inputs["parent_dataset"],
        role="fit",
        rows=PARENT_ROWS,
    )
    heldout_seal = validate_split_seal(
        path=paths["heldout_seal"],
        record=inputs["heldout_seal"],
        dataset_record=inputs["heldout_dataset"],
        role="measure",
        rows=HELDOUT_ROWS,
    )
    supplemental_base_seal = validate_split_seal(
        path=paths["supplemental_base_seal"],
        record=inputs["supplemental_base_seal"],
        dataset_record=inputs["supplemental_base"],
        role="fit",
        rows=SUPPLEMENTAL_ROWS,
    )
    for seal, label in (
        (parent_seal, "parent"),
        (heldout_seal, "heldout"),
        (supplemental_base_seal, "supplemental"),
    ):
        if seal.get("contract_sha256") != inputs["frozen_contract"]["sha256"]:
            raise ExpansionBuildError(
                f"{label} seal does not bind the frozen contract"
            )
    parent_f2_manifest = validate_f2_manifest(
        path=paths["parent_f2_manifest"],
        manifest_record=inputs["parent_f2_manifest"],
        f2_record=inputs["parent_f2"],
        dataset_record=inputs["parent_dataset"],
        rows=PARENT_ROWS,
    )
    parent_f2_contract = parent_f2_manifest["f2_prompt_contract"]

    selection = load_json(paths["selection_seal"], "selection seal")
    selection_counts = selection.get("counts")
    selection_artifacts = selection.get("artifacts")
    selection_digests = selection.get("digests")
    heldout_commitment = selection.get("heldout_commitment")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("passed") is not True
        or not isinstance(selection_counts, Mapping)
        or int(selection_counts.get("parent_fit_rows", -1)) != PARENT_ROWS
        or int(selection_counts.get("supplemental_rows", -1))
        != SUPPLEMENTAL_ROWS
        or int(selection_counts.get("expanded_fit_rows", -1)) != EXPANDED_ROWS
        or int(selection_counts.get("heldout_rows", -1)) != HELDOUT_ROWS
        or not isinstance(selection_artifacts, Mapping)
        or selection_artifacts.get("parent_fit", {}).get("sha256")
        != inputs["parent_dataset"]["sha256"]
        or selection_artifacts.get("heldout", {}).get("sha256")
        != inputs["heldout_dataset"]["sha256"]
        or selection_artifacts.get("supplemental_task_manifest", {}).get(
            "sha256"
        )
        != inputs["supplemental_manifest"]["sha256"]
        or not isinstance(selection_digests, Mapping)
        or not isinstance(heldout_commitment, Mapping)
        or heldout_commitment.get("dataset_sha256")
        != inputs["heldout_dataset"]["sha256"]
        or heldout_commitment.get("seal_sha256")
        != inputs["heldout_seal"]["sha256"]
    ):
        raise ExpansionBuildError("selection seal binding mismatch")

    parent_rows = load_jsonl(paths["parent_dataset"], "parent dataset")
    parent_f2_rows = load_jsonl(paths["parent_f2"], "parent F2")
    heldout_rows = load_jsonl(paths["heldout_dataset"], "heldout dataset")
    base_rows = load_jsonl(paths["supplemental_base"], "supplemental base")
    membership = load_jsonl(
        paths["supplemental_manifest"], "supplemental membership"
    )
    parent_ids = ordered_ids(parent_rows, "parent dataset")
    parent_f2_ids = ordered_ids(parent_f2_rows, "parent F2")
    heldout_ids = ordered_ids(heldout_rows, "heldout dataset")
    supplemental_ids = ordered_ids(base_rows, "supplemental base")
    membership_ids = ordered_ids(membership, "supplemental membership")
    if (
        parent_ids != parent_f2_ids
        or supplemental_ids != membership_ids
        or set(parent_ids) & set(supplemental_ids)
        or set(parent_ids) & set(heldout_ids)
        or set(supplemental_ids) & set(heldout_ids)
        or stable_sha256(parent_ids)
        != selection_digests.get("parent_fit_ordered_task_ids_sha256")
        or stable_sha256(supplemental_ids)
        != selection_digests.get("supplemental_ordered_task_ids_sha256")
        or stable_sha256(heldout_ids)
        != selection_digests.get("heldout_ordered_task_ids_sha256")
        or stable_sha256(parent_ids + supplemental_ids)
        != selection_digests.get("expanded_fit_ordered_task_ids_sha256")
    ):
        raise ExpansionBuildError("parent/supplemental/heldout order mismatch")
    for index, member in enumerate(membership):
        if (
            member.get("schema") != SELECTION_ROW_SCHEMA
            or member.get("partition") != "supplemental"
            or int(member.get("supplemental_row", -1)) != index
        ):
            raise ExpansionBuildError(
                f"{supplemental_ids[index]}: membership row contract mismatch"
            )

    frozen_row_hashes = {
        "compact_codec_sha256": inputs["inline_cfg_codec"]["sha256"],
        "compact_codebook_sha256": inputs["frozen_codebook"]["sha256"],
        "compact_tokenizer_sha256": inputs["tokenizer"]["sha256"],
    }
    for row in parent_rows:
        task_id = str(row["task_id"])
        for field, expected_hash in frozen_row_hashes.items():
            if row.get(field) != expected_hash:
                raise ExpansionBuildError(
                    f"{task_id}: parent {field} is not frozen"
                )

    adapter = import_pinned_module(
        paths["adapter_script"],
        inputs["adapter_script"]["sha256"],
        "multi-function adapter",
    )
    extractor = import_pinned_module(
        paths["extractor_script"],
        inputs["extractor_script"]["sha256"],
        "user-function extractor",
    )
    codec = import_pinned_module(
        paths["inline_cfg_codec"],
        inputs["inline_cfg_codec"]["sha256"],
        "inline-CFG codec",
    )
    frontier_f2 = import_pinned_module(
        paths["frontier_f2"],
        inputs["frontier_f2"]["sha256"],
        "frontier F2",
    )
    if (
        getattr(frontier_f2, "F2_SCHEMA", None) != F2_REPRESENTATION_SCHEMA
        or tuple(getattr(codec, "FORMAT_SENTINEL", ()))
        != ("<G2C1>", "<CFG>")
        or not hasattr(adapter, "combine_user_function_bundle")
    ):
        raise ExpansionBuildError("adapter/codec/F2 interface mismatch")
    if (
        getattr(adapter, "ADAPTER_CONTRACT_SHA256", None)
        != parent_rows[0].get("binary_adapter_contract_sha256")
    ):
        raise ExpansionBuildError(
            "frozen parent adapter contract differs from adapter script"
        )

    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise ExpansionBuildError("tokenizers package is required") from exc
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    base_vocab_size = int(contract["base_vocab_size"])
    atom_ids = {
        str(key): int(value)
        for key, value in (codebook.get("source_atom_ids") or {}).items()
    }
    expansions = [str(value) for value in codebook.get("expansions") or []]
    code = {value: index for index, value in enumerate(expansions)}
    if (
        GRAPH_MARKER not in atom_ids
        or "<CFG>" not in atom_ids
        or not expansions
    ):
        raise ExpansionBuildError("frozen codebook is incomplete")

    bundle_rows = load_jsonl(paths["function_bundles"], "function bundles")
    constant_rows = load_jsonl(paths["constants"], "binary constants")
    bundles = validate_bundles(
        bundle_rows,
        supplemental_ids,
        extractor_sha256=inputs["extractor_script"]["sha256"],
    )
    constants = validate_constants(constant_rows, supplemental_ids)
    membership_by_task = {
        str(row["task_id"]): row for row in membership
    }
    for task_id in supplemental_ids:
        member_aot_sha = str(
            membership_by_task[task_id].get("aot_sha256") or ""
        )
        bundle_aot_sha = str(
            (bundles[task_id].get("inputs") or {}).get("aot_sha256") or ""
        )
        if (
            SHA256_RE.fullmatch(member_aot_sha) is None
            or bundle_aot_sha != member_aot_sha
        ):
            raise ExpansionBuildError(
                f"{task_id}: function bundle AOT differs from sealed membership"
            )

    system_prompt = str(frontier_f2.F2_SYSTEM_PROMPT)
    system_prompt_sha = sha256_text(system_prompt)
    if (
        parent_f2_contract.get("system_prompt_sha256") != system_prompt_sha
        or parent_f2_contract.get("tokenizer_sha256")
        != inputs["tokenizer"]["sha256"]
    ):
        raise ExpansionBuildError(
            "supplemental F2 system prompt differs from frozen parent"
        )
    system_tokens = len(adapter._tokenizer_encode(tokenizer, system_prompt))

    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for task_id in supplemental_ids:
        prepared[task_id] = adapter.combine_user_function_bundle(
            bundles[task_id], extractor
        )

    output_rows: list[dict[str, Any]] = []
    f2_rows: list[dict[str, Any]] = []
    student_lengths: list[int] = []
    api_lengths: list[int] = []
    graph_lengths: list[int] = []
    prefix_lengths: list[int] = []
    function_counts: list[int] = []
    block_counts: list[int] = []
    instruction_counts: list[int] = []
    transfer_counts: list[int] = []
    raw_fallback_counts: list[int] = []
    raw_fallback_unique: set[str] = set()
    attestation_files: set[str] = set()
    attestation_keys: set[str] = set()
    attestation_binding_sequence: list[str] = []

    for index, base_row in enumerate(base_rows):
        task_id = supplemental_ids[index]
        bundle = bundles[task_id]
        canonical, semantic_projection = prepared[task_id]
        external_symbols = semantic_projection["external_symbols"]
        source_attestation = semantic_projection["source_symbol_attestation"]
        attestation_binding = source_attestation["binding"]
        attestation_binding_sha = source_attestation["binding_sha256"]
        constant_record = constants[task_id]
        prefix_source = adapter.binary_enrichment_preamble(
            constant_record["strings"],
            constant_record["numbers"],
            external_symbols,
        )
        prefix_ids = adapter._tokenizer_encode(tokenizer, prefix_source)
        if any(
            token_id < 0 or token_id >= base_vocab_size
            for token_id in prefix_ids
        ):
            raise ExpansionBuildError(
                f"{task_id}: prefix uses a non-base token"
            )
        prefix_text = adapter._tokenizer_decode(tokenizer, prefix_ids)
        if (
            prefix_text != prefix_source
            or adapter._tokenizer_encode(tokenizer, prefix_text) != prefix_ids
            or adapter.parse_external_dictionary_from_preamble(prefix_text)
            != external_symbols
        ):
            raise ExpansionBuildError(
                f"{task_id}: enrichment prefix roundtrip failed"
            )

        row_instructions = [
            str(instruction)
            for block in canonical["blocks"]
            for instruction in block["instructions"]
        ]
        raw_values = [
            instruction
            for instruction in row_instructions
            if instruction not in code
        ]
        raw_fallback_counts.append(len(raw_values))
        raw_fallback_unique.update(raw_values)
        try:
            graph_text = codec.encode(canonical, code)
            decoded = codec.decode(graph_text, expansions)
            graph_ids = [
                int(value)
                for value in codec.compact_ids(
                    graph_text, tokenizer, atom_ids
                )
            ]
            reencoded = codec.encode(decoded, code)
            reencoded_ids = [
                int(value)
                for value in codec.compact_ids(
                    reencoded, tokenizer, atom_ids
                )
            ]
        except Exception as exc:
            raise ExpansionBuildError(
                f"{task_id}: frozen inline-CFG codec failed: {exc}"
            ) from exc
        if (
            decoded != canonical
            or reencoded != graph_text
            or reencoded_ids != graph_ids
            or graph_ids.count(atom_ids[GRAPH_MARKER]) != 1
            or graph_ids.count(atom_ids["<CFG>"]) != 1
        ):
            raise ExpansionBuildError(
                f"{task_id}: frozen compact semantic roundtrip failed"
            )
        compact_ids = prefix_ids + graph_ids
        if len(compact_ids) > student_token_limit:
            raise ExpansionBuildError(
                f"{task_id}: source has {len(compact_ids)} tokens, "
                f"exceeds frozen {student_token_limit}"
            )

        try:
            f2_text = frontier_f2.serialize_f2(
                prefix_text, canonical, tokenizer=tokenizer
            )
            decoded_prefix, decoded_f2 = frontier_f2.decode_f2(f2_text)
        except Exception as exc:
            raise ExpansionBuildError(
                f"{task_id}: F2 serialization failed: {exc}"
            ) from exc
        if decoded_prefix != prefix_text or decoded_f2 != canonical:
            raise ExpansionBuildError(
                f"{task_id}: F2 semantic roundtrip failed"
            )
        user_tokens = len(adapter._tokenizer_encode(tokenizer, f2_text))
        api_tokens = system_tokens + user_tokens + CHAT_OVERHEAD_RESERVE
        if api_tokens > API_PROMPT_TOKEN_LIMIT:
            raise ExpansionBuildError(
                f"{task_id}: F2 prompt has {api_tokens} tokens, exceeds 12000"
            )

        source_projection_sha = str(bundle["model_projection_sha256"])
        semantic_projection_sha = stable_sha256(semantic_projection)
        output = dict(base_row)
        output.pop("schema", None)
        output["compact_input_ids"] = compact_ids
        output.update(frozen_row_hashes)
        output.update(
            {
                "binary_multifunction_schema": adapter.ADAPTER_SCHEMA,
                "binary_adapter_contract_sha256": (
                    adapter.ADAPTER_CONTRACT_SHA256
                ),
                "binary_adapter_script_sha256": inputs["adapter_script"][
                    "sha256"
                ],
                "binary_expansion_builder_sha256": inputs[
                    "expansion_builder"
                ]["sha256"],
                "binary_aot_sha256": bundle["inputs"]["aot_sha256"],
                "binary_source_model_projection_sha256": source_projection_sha,
                "binary_semantic_projection_sha256": semantic_projection_sha,
                "binary_function_count": len(
                    semantic_projection["functions"]
                ),
                "binary_external_symbol_count": len(external_symbols),
                "binary_transfer_count": semantic_projection[
                    "transfer_semantics"
                ]["transfer_row_count"],
                "binary_source_symbol_attestation_used": True,
                "binary_source_symbol_attestation_is_keyed": True,
                "binary_source_symbol_attestation_binding": (
                    attestation_binding
                ),
                "binary_source_symbol_attestation_binding_sha256": (
                    attestation_binding_sha
                ),
                "binary_frozen_codebook_raw_fallback_instructions": len(
                    raw_values
                ),
            }
        )
        output_rows.append(output)

        constants_binding = {
            "constants_record": constant_record,
            "external_symbols": external_symbols,
        }
        f2_rows.append(
            {
                "schema": F2_ROW_SCHEMA,
                "representation_schema": F2_REPRESENTATION_SCHEMA,
                "system_prompt_sha256": system_prompt_sha,
                "task_id": task_id,
                "text": f2_text,
                "text_sha256": sha256_text(f2_text),
                "compact_ids_sha256": stable_sha256(compact_ids),
                "compact_text_sha256": sha256_text(graph_text),
                "canonical_sha256": stable_sha256(canonical),
                "constants_record_sha256": stable_sha256(constants_binding),
                "constants_extraction_error": None,
                "constant_prefix_tokens": len(prefix_ids),
                "graph_tokens": len(graph_ids),
                "source_model_projection_sha256": source_projection_sha,
                "semantic_projection_sha256": semantic_projection_sha,
                "binary_aot_sha256": bundle["inputs"]["aot_sha256"],
                "source_symbol_attestation_used": True,
                "source_symbol_attestation_is_keyed": True,
                "source_symbol_attestation_binding": attestation_binding,
                "source_symbol_attestation_binding_sha256": (
                    attestation_binding_sha
                ),
                "prompt_preflight": {
                    "system_tokens": system_tokens,
                    "user_tokens": user_tokens,
                    "chat_overhead_reserve": CHAT_OVERHEAD_RESERVE,
                    "estimated_prompt_tokens": api_tokens,
                },
                "verified": {
                    "artifact_hashes": True,
                    "row_contract_hashes": True,
                    "codec_text_roundtrip": True,
                    "codec_token_id_roundtrip": True,
                    "student_constant_prefix": True,
                    "per_task_instruction_dictionary_roundtrip": True,
                    "lossless_raw_instruction_fallback": True,
                    "compact_semantic_f2_roundtrip": True,
                    "branch_targets_reconstructed_from_cfg": True,
                    "visible_task_symbols_one_token": True,
                    "opaque_custom_ids_in_text": False,
                    "all_user_functions_retained": True,
                    "all_external_symbols_retained": True,
                    "transfer_table_redundancy_proven": True,
                    "keyed_source_symbol_attestation_bound": True,
                    "raw_source_names_not_serialized": True,
                    "frozen_parent_codebook_unchanged": True,
                },
            }
        )

        student_lengths.append(len(compact_ids))
        api_lengths.append(api_tokens)
        graph_lengths.append(len(graph_ids))
        prefix_lengths.append(len(prefix_ids))
        function_counts.append(len(semantic_projection["functions"]))
        block_counts.append(len(canonical["blocks"]))
        instruction_counts.append(len(row_instructions))
        transfer_counts.append(
            int(semantic_projection["transfer_semantics"]["transfer_row_count"])
        )
        attestation_files.add(
            str(attestation_binding["attestation_file_sha256"])
        )
        attestation_keys.add(str(attestation_binding["key_id_sha256"]))
        attestation_binding_sequence.append(attestation_binding_sha)

    if len(output_rows) != SUPPLEMENTAL_ROWS or len(f2_rows) != SUPPLEMENTAL_ROWS:
        raise ExpansionBuildError("supplemental zero-exclusion gate failed")
    if len(attestation_files) != 1 or len(attestation_keys) != 1:
        raise ExpansionBuildError(
            "supplemental bundles do not share one keyed attestation"
        )
    supplemental_attestation_file = next(iter(attestation_files))
    supplemental_attestation_key = next(iter(attestation_keys))
    parent_attestation = parent_f2_manifest.get("source_symbol_attestation")
    if (
        isinstance(parent_attestation, Mapping)
        and parent_attestation.get("key_id_sha256")
        != supplemental_attestation_key
    ):
        raise ExpansionBuildError(
            "parent and supplemental source attestations use different keys"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_paths = {
        "supplemental_dataset": (
            output_dir / "train_multifunction_binary_supplemental_1196.jsonl"
        ),
        "supplemental_seal": (
            output_dir
            / "train_multifunction_binary_supplemental_1196.seal.json"
        ),
        "supplemental_f2": (
            output_dir
            / "train_multifunction_binary_supplemental_1196_f2.jsonl"
        ),
        "supplemental_f2_manifest": (
            output_dir
            / "train_multifunction_binary_supplemental_1196_f2.jsonl.manifest.json"
        ),
        "expanded_dataset": (
            output_dir / "train_multifunction_binary_expanded_2776.jsonl"
        ),
        "expanded_seal": (
            output_dir / "train_multifunction_binary_expanded_2776.seal.json"
        ),
        "expanded_f2": (
            output_dir / "train_multifunction_binary_expanded_2776_f2.jsonl"
        ),
        "expanded_f2_manifest": (
            output_dir
            / "train_multifunction_binary_expanded_2776_f2.jsonl.manifest.json"
        ),
        "expansion_seal": output_dir / "expansion_build.seal.json",
        "report": output_dir / "build_report.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing expansion outputs: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_paths["supplemental_dataset"], output_rows)
    atomic_write_jsonl(output_paths["supplemental_f2"], f2_rows)
    atomic_concat(
        output_paths["expanded_dataset"],
        paths["parent_dataset"],
        output_paths["supplemental_dataset"],
    )
    atomic_concat(
        output_paths["expanded_f2"],
        paths["parent_f2"],
        output_paths["supplemental_f2"],
    )

    expanded_ids = parent_ids + supplemental_ids
    expanded_rows_check = load_jsonl(
        output_paths["expanded_dataset"], "expanded dataset"
    )
    expanded_f2_check = load_jsonl(
        output_paths["expanded_f2"], "expanded F2"
    )
    if (
        ordered_ids(expanded_rows_check, "expanded dataset") != expanded_ids
        or ordered_ids(expanded_f2_check, "expanded F2") != expanded_ids
    ):
        raise ExpansionBuildError("expanded append-only order drifted")

    supplemental_max_row = max(
        f2_rows,
        key=lambda row: row["prompt_preflight"]["estimated_prompt_tokens"],
    )
    parent_max = int(
        parent_f2_contract.get("maximum_estimated_prompt_tokens", -1)
    )
    parent_max_task = str(parent_f2_contract.get("maximum_task_id") or "")
    if parent_max < 0:
        raise ExpansionBuildError("parent F2 maximum token receipt is missing")
    if (
        supplemental_max_row["prompt_preflight"]["estimated_prompt_tokens"]
        > parent_max
    ):
        expanded_max = supplemental_max_row["prompt_preflight"][
            "estimated_prompt_tokens"
        ]
        expanded_max_task = str(supplemental_max_row["task_id"])
    else:
        expanded_max = parent_max
        expanded_max_task = parent_max_task

    common_invariants = {
        "all_artifact_hashes_verified": True,
        "all_row_contract_hashes_verified": True,
        "all_codec_roundtrips_verified": True,
        "all_student_constant_prefixes_verified": True,
        "all_f2_semantic_roundtrips_verified": True,
        "f2_system_prompt_self_contained_and_hashed": True,
        "all_complete_prompts_within_limit": True,
        "opaque_source_ids_expanded": True,
        "cfg_explicit": True,
        "all_user_functions_retained": True,
        "all_external_symbols_retained": True,
        "transfer_table_redundancy_proven": True,
        "train_dev_representation_contract_identical": True,
        "frozen_parent_codebook_unchanged": True,
        "lossless_raw_instruction_fallback_permitted": True,
        "keyed_private_source_symbol_attestation_used": True,
        "raw_source_names_not_serialized": True,
    }
    artifact_records = {
        **inputs,
        "representation_contract": inputs["frozen_contract"],
        "representation_codebook": inputs["frozen_codebook"],
    }

    def f2_manifest(
        *,
        dataset_path: Path,
        f2_path: Path,
        task_ids: Sequence[str],
        rows: int,
        maximum_tokens: int,
        maximum_task_id: str,
        partition: str,
    ) -> dict[str, Any]:
        return {
            "schema": F2_MANIFEST_SCHEMA,
            "rows": rows,
            "dataset": file_record(dataset_path),
            "task_set_sha256": stable_sha256(list(task_ids)),
            "ordered_task_ids_sha256": stable_sha256(list(task_ids)),
            "binary_constant_extraction_errors": {
                "count": 0,
                "task_ids": [],
            },
            "source_symbol_attestation": {
                "used": True,
                "is_keyed": True,
                "key_id_sha256": supplemental_attestation_key,
                "supplemental_attestation_file_sha256": (
                    supplemental_attestation_file
                ),
                "binding_sha256_sequence": stable_sha256(
                    attestation_binding_sequence
                ),
                "raw_names_serialized": False,
            },
            "artifacts": artifact_records,
            "f2_prompt_contract": {
                "representation_schema": F2_REPRESENTATION_SCHEMA,
                "system_prompt": system_prompt,
                "system_prompt_sha256": system_prompt_sha,
                "tokenizer_sha256": inputs["tokenizer"]["sha256"],
                "constant_prefix_token_cap": None,
                "max_prompt_tokens": API_PROMPT_TOKEN_LIMIT,
                "chat_overhead_reserve": CHAT_OVERHEAD_RESERVE,
                "maximum_estimated_prompt_tokens": maximum_tokens,
                "maximum_task_id": maximum_task_id,
                "all_rows_within_limit": True,
            },
            "output": file_record(f2_path),
            "expansion_partition": partition,
            "selection_seal_sha256": inputs["selection_seal"]["sha256"],
            "invariants": common_invariants,
        }

    supplemental_manifest_value = f2_manifest(
        dataset_path=output_paths["supplemental_dataset"],
        f2_path=output_paths["supplemental_f2"],
        task_ids=supplemental_ids,
        rows=SUPPLEMENTAL_ROWS,
        maximum_tokens=int(
            supplemental_max_row["prompt_preflight"][
                "estimated_prompt_tokens"
            ]
        ),
        maximum_task_id=str(supplemental_max_row["task_id"]),
        partition="supplemental_only",
    )
    atomic_write_json(
        output_paths["supplemental_f2_manifest"],
        supplemental_manifest_value,
    )
    expanded_manifest_value = f2_manifest(
        dataset_path=output_paths["expanded_dataset"],
        f2_path=output_paths["expanded_f2"],
        task_ids=expanded_ids,
        rows=EXPANDED_ROWS,
        maximum_tokens=expanded_max,
        maximum_task_id=expanded_max_task,
        partition="frozen_parent_then_supplemental",
    )
    expanded_manifest_value["source_symbol_attestation"] = {
        "used": True,
        "is_keyed": True,
        "key_id_sha256": supplemental_attestation_key,
        "partitions": [
            {
                "name": "frozen_parent",
                "rows": PARENT_ROWS,
                "manifest_sha256": inputs["parent_f2_manifest"]["sha256"],
            },
            {
                "name": "supplemental",
                "rows": SUPPLEMENTAL_ROWS,
                "attestation_file_sha256": supplemental_attestation_file,
                "binding_sha256_sequence": stable_sha256(
                    attestation_binding_sequence
                ),
            },
        ],
        "raw_names_serialized": False,
    }
    atomic_write_json(
        output_paths["expanded_f2_manifest"], expanded_manifest_value
    )

    supplemental_ineligible = list(
        supplemental_base_seal.get("execution_ineligible_task_ids") or []
    )
    parent_ineligible = list(
        parent_seal.get("execution_ineligible_task_ids") or []
    )
    if not set(supplemental_ineligible).issubset(set(supplemental_ids)):
        raise ExpansionBuildError(
            "supplemental execution-ineligible IDs are outside membership"
        )
    if not set(parent_ineligible).issubset(set(parent_ids)):
        raise ExpansionBuildError(
            "parent execution-ineligible IDs are outside membership"
        )
    expanded_ineligible = sorted(
        set(parent_ineligible) | set(supplemental_ineligible)
    )

    def training_seal(
        *,
        dataset_path: Path,
        f2_path: Path,
        f2_manifest_path: Path,
        task_ids: Sequence[str],
        rows: int,
        eligible_rows: int,
        ineligible_ids: Sequence[str],
        partition: str,
    ) -> dict[str, Any]:
        return {
            "schema": SPLIT_SEAL_SCHEMA,
            "selected_role": "fit",
            "training_allowed": True,
            "heldout_measure_only": False,
            "training_objective_scope": "sequence_imitation_all_train",
            "rows": rows,
            "task_set_sha256": stable_sha256(list(task_ids)),
            "ordered_task_ids_sha256": stable_sha256(list(task_ids)),
            "sorted_task_set_sha256": stable_sha256(sorted(task_ids)),
            "output_sha256": sha256_file(dataset_path),
            "output": file_record(dataset_path),
            "f2_output": file_record(f2_path),
            "f2_manifest": file_record(f2_manifest_path),
            "contract_sha256": inputs["frozen_contract"]["sha256"],
            "representation_schema": adapter.ADAPTER_SCHEMA,
            "frontier_f2_schema": F2_REPRESENTATION_SCHEMA,
            "adapter_contract_sha256": adapter.ADAPTER_CONTRACT_SHA256,
            "adapter_script_sha256": inputs["adapter_script"]["sha256"],
            "expansion_builder_sha256": inputs["expansion_builder"]["sha256"],
            "selection_seal_sha256": inputs["selection_seal"]["sha256"],
            "expansion_partition": partition,
            "executable_reward_eligible_rows": eligible_rows,
            "execution_ineligible_task_ids": list(ineligible_ids),
            "heldout_commitment": dict(heldout_commitment),
            "source_symbol_attestation_used": True,
            "source_symbol_attestation_is_keyed": True,
            "source_symbol_attestation_key_id_sha256": (
                supplemental_attestation_key
            ),
            "raw_source_names_serialized": False,
        }

    supplemental_seal_value = training_seal(
        dataset_path=output_paths["supplemental_dataset"],
        f2_path=output_paths["supplemental_f2"],
        f2_manifest_path=output_paths["supplemental_f2_manifest"],
        task_ids=supplemental_ids,
        rows=SUPPLEMENTAL_ROWS,
        eligible_rows=SUPPLEMENTAL_ROWS - len(supplemental_ineligible),
        ineligible_ids=supplemental_ineligible,
        partition="supplemental_only",
    )
    supplemental_seal_value["sanitation"] = {
        "seal_sha256": inputs["supplemental_base_seal"]["sha256"],
        "schema": supplemental_base_seal.get("sanitation_schema"),
        "evaluator_sha256": supplemental_base_seal.get("evaluator_sha256"),
        "completion_attestation_id": supplemental_base_seal.get(
            "completion_attestation_id"
        ),
    }
    atomic_write_json(
        output_paths["supplemental_seal"], supplemental_seal_value
    )
    expanded_seal_value = training_seal(
        dataset_path=output_paths["expanded_dataset"],
        f2_path=output_paths["expanded_f2"],
        f2_manifest_path=output_paths["expanded_f2_manifest"],
        task_ids=expanded_ids,
        rows=EXPANDED_ROWS,
        eligible_rows=EXPANDED_ROWS - len(expanded_ineligible),
        ineligible_ids=expanded_ineligible,
        partition="frozen_parent_then_supplemental",
    )
    expanded_seal_value["sanitation_partitions"] = [
        {
            "name": "frozen_parent",
            "rows": PARENT_ROWS,
            "seal_sha256": inputs["parent_seal"]["sha256"],
            "execution_ineligible_task_ids": parent_ineligible,
        },
        {
            "name": "supplemental",
            "rows": SUPPLEMENTAL_ROWS,
            "seal_sha256": inputs["supplemental_base_seal"]["sha256"],
            "execution_ineligible_task_ids": supplemental_ineligible,
        },
    ]
    expanded_seal_value["append_only_prefix"] = {
        "parent_dataset_sha256": inputs["parent_dataset"]["sha256"],
        "parent_dataset_bytes": inputs["parent_dataset"]["bytes"],
        "parent_f2_sha256": inputs["parent_f2"]["sha256"],
        "parent_f2_bytes": inputs["parent_f2"]["bytes"],
        "exact_parent_dataset_prefix": True,
        "exact_parent_f2_prefix": True,
    }
    atomic_write_json(output_paths["expanded_seal"], expanded_seal_value)

    expansion_seal_value = {
        "schema": EXPANSION_SEAL_SCHEMA,
        "counts": {
            "parent_rows": PARENT_ROWS,
            "supplemental_rows": SUPPLEMENTAL_ROWS,
            "expanded_rows": EXPANDED_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "parent_execution_ineligible_rows": len(parent_ineligible),
            "supplemental_execution_ineligible_rows": len(
                supplemental_ineligible
            ),
            "expanded_executable_reward_eligible_rows": (
                EXPANDED_ROWS - len(expanded_ineligible)
            ),
        },
        "digests": {
            "parent_ordered_task_ids_sha256": stable_sha256(parent_ids),
            "supplemental_ordered_task_ids_sha256": stable_sha256(
                supplemental_ids
            ),
            "expanded_ordered_task_ids_sha256": stable_sha256(expanded_ids),
            "heldout_ordered_task_ids_sha256": stable_sha256(heldout_ids),
        },
        "artifacts": {
            **inputs,
            "supplemental_dataset": file_record(
                output_paths["supplemental_dataset"]
            ),
            "supplemental_seal": file_record(
                output_paths["supplemental_seal"]
            ),
            "supplemental_f2": file_record(output_paths["supplemental_f2"]),
            "supplemental_f2_manifest": file_record(
                output_paths["supplemental_f2_manifest"]
            ),
            "expanded_dataset": file_record(output_paths["expanded_dataset"]),
            "expanded_seal": file_record(output_paths["expanded_seal"]),
            "expanded_f2": file_record(output_paths["expanded_f2"]),
            "expanded_f2_manifest": file_record(
                output_paths["expanded_f2_manifest"]
            ),
        },
        "heldout_commitment": dict(heldout_commitment),
        "execution_ineligible_task_ids": expanded_ineligible,
        "invariants": {
            "parent_dataset_bytes_exact_prefix": True,
            "parent_f2_bytes_exact_prefix": True,
            "parent_rows_not_reencoded": True,
            "parent_compact_ids_unchanged": True,
            "parent_prompt_text_unchanged": True,
            "frozen_contract_unchanged": True,
            "frozen_codebook_unchanged": True,
            "no_source_token_id_reassigned": True,
            "unknown_instructions_use_lossless_raw_fallback": True,
            "supplemental_zero_exclusion": True,
            "heldout_membership_unchanged": True,
            "heldout_not_present_in_fit": True,
            "all_student_sources_within_9000": True,
            "all_api_prompts_within_12000": True,
        },
        "passed": True,
    }
    atomic_write_json(output_paths["expansion_seal"], expansion_seal_value)

    report = {
        "schema": BUILD_SCHEMA,
        "counts": {
            "parent_rows": PARENT_ROWS,
            "supplemental_rows": SUPPLEMENTAL_ROWS,
            "expanded_rows": EXPANDED_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "supplemental_functions": sum(function_counts),
            "supplemental_blocks": sum(block_counts),
            "supplemental_machine_instructions": sum(instruction_counts),
            "supplemental_interfunction_transfers": sum(transfer_counts),
            "raw_fallback_instruction_occurrences": sum(
                raw_fallback_counts
            ),
            "raw_fallback_unique_instructions": len(raw_fallback_unique),
        },
        "tokens": {
            "student": {
                "limit": student_token_limit,
                "min": min(student_lengths),
                "p50": percentile(student_lengths, 0.50),
                "p95": percentile(student_lengths, 0.95),
                "p99": percentile(student_lengths, 0.99),
                "max": max(student_lengths),
            },
            "api_f2_prompt": {
                "limit": API_PROMPT_TOKEN_LIMIT,
                "min": min(api_lengths),
                "p50": percentile(api_lengths, 0.50),
                "p95": percentile(api_lengths, 0.95),
                "p99": percentile(api_lengths, 0.99),
                "max": max(api_lengths),
            },
            "graph": {
                "p95": percentile(graph_lengths, 0.95),
                "max": max(graph_lengths),
            },
            "binary_prefix": {
                "p95": percentile(prefix_lengths, 0.95),
                "max": max(prefix_lengths),
                "cap": None,
            },
        },
        "family_counts": dict(
            Counter(str(row.get("family") or "") for row in output_rows)
        ),
        "inputs": inputs,
        "expansion_seal": file_record(output_paths["expansion_seal"]),
        "outputs": {
            key: file_record(path)
            for key, path in output_paths.items()
            if key not in {"report", "expansion_seal"}
        },
        "passed": True,
    }
    atomic_write_json(output_paths["report"], report)
    print(
        "PHASE0_2776_MULTIFUNCTION_EXPANDED "
        f"parent={PARENT_ROWS} supplemental={SUPPLEMENTAL_ROWS} "
        f"expanded={EXPANDED_ROWS} raw_fallback={sum(raw_fallback_counts)} "
        f"student_max={max(student_lengths)} api_max={max(api_lengths)} "
        f"expanded_sha256={sha256_file(output_paths['expanded_dataset'])}",
        flush=True,
    )
    return report


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
