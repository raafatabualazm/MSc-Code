#!/usr/bin/env python3
"""Materialize the append-only Phase-0 fit expansion.

The historical multi-function build used 1,580 fit tasks and 175 held-out
tasks, all from the ``master`` family.  The sealed Phase-0 release contains
2,951 train tasks.  The correct fit universe is therefore the Phase-0 train
set minus the same immutable 175 held-out tasks:

    2,951 - 175 = 2,776 = 1,580 frozen parent + 1,196 supplemental.

This stage does not inspect an AOT payload and does not alter an existing
multi-function row.  It:

* proves the set arithmetic against hash-pinned Phase-0 artifacts;
* fixes the expanded order to the exact 1,580-row parent order followed by
  missing tasks in Phase-0 order;
* emits an exact 1,196-row projection of the full AOT manifest;
* reconstructs private gold/harness rows from the source-preparation receipt
  and its byte-exact canonical source row; and
* seals every output for the extraction, sanitation, and representation stages.

The existing 1,580-row representation contract remains frozen.  Supplemental
representation construction is a separate binary-only stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from hybrid_training_patch_v2_3.scripts.training import (  # noqa: E402
    hybrid_data_controls as controls,
)


SELECTION_SCHEMA = "multifunction-phase0-fit-expansion-selection-v1"
SELECTION_ROW_SCHEMA = "multifunction-phase0-fit-membership-v1"
TARGET_ROW_SCHEMA = "phase0-supplemental-target-row-v1"
REPORT_SCHEMA = "multifunction-phase0-fit-expansion-preparation-v1"
SPLIT_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
CONTRACT_SCHEMA = "direct-compact-causal-v1"

PHASE0_TRAIN_ROWS = 2_951
PHASE0_DEV_ROWS = 326
PHASE0_TOTAL_ROWS = PHASE0_TRAIN_ROWS + PHASE0_DEV_ROWS
PARENT_FIT_ROWS = 1_580
HELDOUT_ROWS = 175
SUPPLEMENTAL_ROWS = 1_196
EXPANDED_FIT_ROWS = 2_776
SUPPLEMENTAL_FAMILY_COUNTS = {
    "master": 198,
    "topup_s45": 894,
    "topup_s46": 104,
}
TARGET_FUNCTION = "fn0"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ExpansionPreparationError(ValueError):
    """The Phase-0 expansion cannot be proven from the pinned inputs."""


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise ExpansionPreparationError(f"{label} expected SHA-256 is malformed")
    record = file_record(path)
    if record["sha256"] != expected:
        raise ExpansionPreparationError(
            f"{label} hash mismatch: expected {expected}, "
            f"observed {record['sha256']}"
        )
    return record


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpansionPreparationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExpansionPreparationError(f"{label} is not a JSON object")
    return value


def load_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ExpansionPreparationError(
                    f"{label} has a blank row at line {line_number}"
                )
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExpansionPreparationError(
                    f"{label} has invalid JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ExpansionPreparationError(
                    f"{label} row {line_number} is not an object"
                )
            rows.append(value)
    if not rows:
        raise ExpansionPreparationError(f"{label} is empty")
    return rows


def load_jsonl_raw(
    path: str | Path, label: str
) -> list[tuple[dict[str, Any], bytes]]:
    rows: list[tuple[dict[str, Any], bytes]] = []
    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ExpansionPreparationError(
                    f"{label} has a blank row at line {line_number}"
                )
            payload = raw.rstrip(b"\r\n")
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExpansionPreparationError(
                    f"{label} has invalid JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ExpansionPreparationError(
                    f"{label} row {line_number} is not an object"
                )
            rows.append((value, payload))
    if not rows:
        raise ExpansionPreparationError(f"{label} is empty")
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
    payload = b"".join(
        canonical_json_bytes(dict(row)) + b"\n" for row in rows
    )
    _atomic_write_bytes(path, payload)


def _ordered_ids(
    rows: Sequence[Mapping[str, Any]], label: str
) -> list[str]:
    result = [str(row.get("task_id") or "") for row in rows]
    if any(not task_id for task_id in result):
        raise ExpansionPreparationError(f"{label} contains an empty task_id")
    if len(set(result)) != len(result):
        raise ExpansionPreparationError(f"{label} contains duplicate task IDs")
    return result


def _ordered_digest(task_ids: Sequence[str]) -> str:
    return stable_sha256(list(task_ids))


def _set_digest(task_ids: Iterable[str]) -> str:
    return stable_sha256(sorted(set(task_ids)))


def _validate_existing_split_seal(
    *,
    dataset_path: Path,
    dataset_record: Mapping[str, Any],
    seal_path: Path,
    seal_record: Mapping[str, Any],
    rows: int,
    role: str,
) -> dict[str, Any]:
    seal = load_json(seal_path, f"{role} split seal")
    if (
        seal.get("schema") != SPLIT_SEAL_SCHEMA
        or seal.get("selected_role") != role
        or int(seal.get("rows", -1)) != rows
        or seal.get("output_sha256") != dataset_record.get("sha256")
        or seal_record.get("sha256") != sha256_file(seal_path)
    ):
        raise ExpansionPreparationError(f"{role} split seal contract mismatch")
    if role == "fit":
        if seal.get("training_allowed") is not True:
            raise ExpansionPreparationError("parent fit is not training-allowed")
    else:
        if (
            seal.get("heldout_measure_only") is not True
            or seal.get("training_allowed") is not False
        ):
            raise ExpansionPreparationError(
                "historical heldout is not sealed measure-only"
            )
    observed_rows = sum(
        1 for line in dataset_path.open(encoding="utf-8") if line.strip()
    )
    if observed_rows != rows:
        raise ExpansionPreparationError(
            f"{role} dataset has {observed_rows} rows, expected {rows}"
        )
    return seal


def _validate_contract(
    contract_path: Path, contract_record: Mapping[str, Any]
) -> dict[str, Any]:
    contract = load_json(contract_path, "frozen representation contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ExpansionPreparationError("frozen contract schema mismatch")
    if contract.get("target_function") != TARGET_FUNCTION:
        raise ExpansionPreparationError(
            f"frozen contract target must be {TARGET_FUNCTION!r}"
        )
    for field in (
        "codec_sha256",
        "codebook_sha256",
        "tokenizer_json_sha256",
    ):
        if SHA256_RE.fullmatch(str(contract.get(field) or "")) is None:
            raise ExpansionPreparationError(
                f"frozen contract has no valid {field}"
            )
    if contract_record.get("sha256") != sha256_file(contract_path):
        raise ExpansionPreparationError("frozen contract record drifted")
    return contract


def _base_target_row(
    *,
    label: Mapping[str, Any],
    build_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_raw_sha256: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(label["task_id"])
    if str(build_row.get("task_id") or "") != task_id:
        raise ExpansionPreparationError(
            f"{task_id}: source-preparation build row identity mismatch"
        )
    metadata = build_row.get("compact_private_metadata")
    if not isinstance(metadata, Mapping):
        raise ExpansionPreparationError(
            f"{task_id}: source-preparation metadata missing"
        )
    if metadata.get("input_row_sha256") != source_raw_sha256:
        raise ExpansionPreparationError(
            f"{task_id}: canonical source-row digest mismatch"
        )
    if str(source_row.get("task_id") or "") != task_id:
        raise ExpansionPreparationError(
            f"{task_id}: canonical source-row task mismatch"
        )
    original_function = str(source_row.get("function") or "")
    if not original_function:
        raise ExpansionPreparationError(
            f"{task_id}: canonical source row has no target function"
        )
    tests = source_row.get("tests")
    if not isinstance(tests, str) or not tests.strip():
        raise ExpansionPreparationError(
            f"{task_id}: canonical source row has no executable harness"
        )

    # Reuse the exact lexical identifier rewriter used by the Phase-0 release.
    # The original semantic name is used only while constructing this private
    # row and is never serialized in an output artifact.
    neutral_tests = controls.replace_identifier_outside_strings(
        tests, original_function, TARGET_FUNCTION
    )
    if not neutral_tests.strip():
        raise ExpansionPreparationError(f"{task_id}: neutral harness is empty")

    prepared_source = label.get("dart_source")
    if not isinstance(prepared_source, str) or not prepared_source.strip():
        raise ExpansionPreparationError(
            f"{task_id}: prepared target source is missing"
        )
    build_source = str(build_row.get("function_source") or "")
    if prepared_source.rstrip() != build_source.rstrip():
        raise ExpansionPreparationError(
            f"{task_id}: prepared label/build target source mismatch"
        )
    neutral_source = controls.replace_identifier_outside_strings(
        prepared_source, "candidate", TARGET_FUNCTION
    )
    if controls.extract_source_signature(neutral_source, TARGET_FUNCTION) is None:
        raise ExpansionPreparationError(
            f"{task_id}: neutral target declaration is missing"
        )

    family = str(label.get("family") or "")
    if family not in SUPPLEMENTAL_FAMILY_COUNTS:
        raise ExpansionPreparationError(
            f"{task_id}: unsupported Phase-0 family {family!r}"
        )
    return {
        "schema": TARGET_ROW_SCHEMA,
        "task_id": task_id,
        "family": family,
        "lang": "Dart",
        "function": TARGET_FUNCTION,
        "dart_source": neutral_source,
        "tests": neutral_tests,
        "acceptance_tests": neutral_tests,
        "feedback_tests": "",
        # This is a target/harness staging row.  The binary-only builder
        # replaces compact_input_ids after all extraction gates pass.
        "compact_input_ids": [],
        "compact_codec_sha256": str(contract["codec_sha256"]),
        "compact_codebook_sha256": str(contract["codebook_sha256"]),
        "compact_tokenizer_sha256": str(
            contract["tokenizer_json_sha256"]
        ),
        "phase0_expansion_provenance": {
            "schema": "phase0-supplemental-target-provenance-v1",
            "phase0_split": "train",
            "phase0_split_row": int(build_row.get("split_row", -1)),
            "phase0_manifest_line": int(
                build_row.get("phase0_manifest_line", -1)
            ),
            "canonical_source_row_sha256": source_raw_sha256,
            "prepared_function_source_sha256": str(
                build_row.get("function_source_sha256") or ""
            ),
            "analysis_program_sha256": str(
                build_row.get("analysis_program_sha256") or ""
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase0-train-labels", required=True, type=Path)
    parser.add_argument("--expected-phase0-train-labels-sha256", required=True)
    parser.add_argument("--phase0-private-build-train", required=True, type=Path)
    parser.add_argument(
        "--expected-phase0-private-build-train-sha256", required=True
    )
    parser.add_argument("--phase0-aot-manifest", required=True, type=Path)
    parser.add_argument("--expected-phase0-aot-manifest-sha256", required=True)
    parser.add_argument("--phase0-source-corpus", required=True, type=Path)
    parser.add_argument("--expected-phase0-source-corpus-sha256", required=True)
    parser.add_argument("--parent-fit", required=True, type=Path)
    parser.add_argument("--expected-parent-fit-sha256", required=True)
    parser.add_argument("--parent-fit-seal", required=True, type=Path)
    parser.add_argument("--expected-parent-fit-seal-sha256", required=True)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--expected-heldout-sha256", required=True)
    parser.add_argument("--heldout-seal", required=True, type=Path)
    parser.add_argument("--expected-heldout-seal-sha256", required=True)
    parser.add_argument("--frozen-contract", required=True, type=Path)
    parser.add_argument("--expected-frozen-contract-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "phase0_train_labels": args.phase0_train_labels,
            "phase0_private_build_train": args.phase0_private_build_train,
            "phase0_aot_manifest": args.phase0_aot_manifest,
            "phase0_source_corpus": args.phase0_source_corpus,
            "parent_fit": args.parent_fit,
            "parent_fit_seal": args.parent_fit_seal,
            "heldout": args.heldout,
            "heldout_seal": args.heldout_seal,
            "frozen_contract": args.frozen_contract,
        }.items()
    }
    expected_hashes = {
        "phase0_train_labels": args.expected_phase0_train_labels_sha256,
        "phase0_private_build_train": (
            args.expected_phase0_private_build_train_sha256
        ),
        "phase0_aot_manifest": args.expected_phase0_aot_manifest_sha256,
        "phase0_source_corpus": args.expected_phase0_source_corpus_sha256,
        "parent_fit": args.expected_parent_fit_sha256,
        "parent_fit_seal": args.expected_parent_fit_seal_sha256,
        "heldout": args.expected_heldout_sha256,
        "heldout_seal": args.expected_heldout_seal_sha256,
        "frozen_contract": args.expected_frozen_contract_sha256,
    }
    inputs = {
        name: require_file_hash(paths[name], expected, name)
        for name, expected in expected_hashes.items()
    }

    contract = _validate_contract(paths["frozen_contract"], inputs["frozen_contract"])
    parent_seal = _validate_existing_split_seal(
        dataset_path=paths["parent_fit"],
        dataset_record=inputs["parent_fit"],
        seal_path=paths["parent_fit_seal"],
        seal_record=inputs["parent_fit_seal"],
        rows=PARENT_FIT_ROWS,
        role="fit",
    )
    heldout_seal = _validate_existing_split_seal(
        dataset_path=paths["heldout"],
        dataset_record=inputs["heldout"],
        seal_path=paths["heldout_seal"],
        seal_record=inputs["heldout_seal"],
        rows=HELDOUT_ROWS,
        role="measure",
    )
    if parent_seal.get("contract_sha256") != contract.get("codebook_sha256") and (
        parent_seal.get("contract_sha256")
        != inputs["frozen_contract"]["sha256"]
    ):
        # Production parent seals bind the frozen representation contract.  A
        # legacy fixture may bind its codebook directly, which remains accepted
        # only for focused tests.
        raise ExpansionPreparationError(
            "parent fit seal does not bind the frozen representation"
        )
    if heldout_seal.get("contract_sha256") != parent_seal.get(
        "contract_sha256"
    ):
        raise ExpansionPreparationError(
            "parent fit/heldout representation contracts differ"
        )

    labels = load_jsonl(paths["phase0_train_labels"], "Phase-0 train labels")
    build_rows = load_jsonl(
        paths["phase0_private_build_train"],
        "Phase-0 private build train",
    )
    aot_rows = load_jsonl(paths["phase0_aot_manifest"], "Phase-0 AOT manifest")
    parent_rows = load_jsonl(paths["parent_fit"], "parent fit")
    heldout_rows = load_jsonl(paths["heldout"], "historical heldout")
    source_rows_raw = load_jsonl_raw(
        paths["phase0_source_corpus"], "canonical Phase-0 source corpus"
    )
    if len(labels) != PHASE0_TRAIN_ROWS or len(build_rows) != PHASE0_TRAIN_ROWS:
        raise ExpansionPreparationError(
            "Phase-0 train artifacts are not exactly 2,951 rows"
        )
    if len(aot_rows) != PHASE0_TOTAL_ROWS:
        raise ExpansionPreparationError(
            f"full AOT manifest has {len(aot_rows)} rows, "
            f"expected {PHASE0_TOTAL_ROWS}"
        )
    aot_train = [row for row in aot_rows if row.get("split") == "train"]
    aot_dev = [row for row in aot_rows if row.get("split") == "dev"]
    if len(aot_train) != PHASE0_TRAIN_ROWS or len(aot_dev) != PHASE0_DEV_ROWS:
        raise ExpansionPreparationError(
            "full AOT manifest split accounting is not train=2951/dev=326"
        )

    label_ids = _ordered_ids(labels, "Phase-0 train labels")
    build_ids = _ordered_ids(build_rows, "Phase-0 private build train")
    aot_train_ids = _ordered_ids(aot_train, "Phase-0 train AOT projection")
    if label_ids != build_ids or label_ids != aot_train_ids:
        raise ExpansionPreparationError(
            "Phase-0 label/build/AOT train order differs"
        )
    for index, (label, build_row, aot_row) in enumerate(
        zip(labels, build_rows, aot_train, strict=True)
    ):
        task_id = label_ids[index]
        if (
            label.get("function") != "candidate"
            or str(build_row.get("function") or "") != "candidate"
            or build_row.get("split") != "train"
            or int(build_row.get("split_row", -1)) != index
            or aot_row.get("split") != "train"
            or int(aot_row.get("split_row", -1)) != index
        ):
            raise ExpansionPreparationError(
                f"{task_id}: Phase-0 train row contract mismatch"
            )
        aot_sha = str((aot_row.get("aot_sha256") or ""))
        if SHA256_RE.fullmatch(aot_sha) is None:
            raise ExpansionPreparationError(
                f"{task_id}: AOT manifest digest is malformed"
            )

    parent_ids = _ordered_ids(parent_rows, "parent fit")
    heldout_ids = _ordered_ids(heldout_rows, "historical heldout")
    phase0_set = set(label_ids)
    parent_set = set(parent_ids)
    heldout_set = set(heldout_ids)
    if parent_set & heldout_set:
        raise ExpansionPreparationError("parent fit overlaps historical heldout")
    if not parent_set.issubset(phase0_set) or not heldout_set.issubset(
        phase0_set
    ):
        raise ExpansionPreparationError(
            "parent fit/heldout is not a subset of Phase-0 train"
        )
    supplemental_ids = [
        task_id
        for task_id in label_ids
        if task_id not in parent_set and task_id not in heldout_set
    ]
    supplemental_set = set(supplemental_ids)
    if len(supplemental_ids) != SUPPLEMENTAL_ROWS:
        raise ExpansionPreparationError(
            f"supplemental set has {len(supplemental_ids)} rows, "
            f"expected {SUPPLEMENTAL_ROWS}"
        )
    expanded_ids = parent_ids + supplemental_ids
    if (
        len(expanded_ids) != EXPANDED_FIT_ROWS
        or len(set(expanded_ids)) != EXPANDED_FIT_ROWS
        or set(expanded_ids) != phase0_set - heldout_set
        or (set(expanded_ids) | heldout_set) != phase0_set
    ):
        raise ExpansionPreparationError(
            "expanded fit/heldout set arithmetic is not exact"
        )

    label_by_id = {str(row["task_id"]): row for row in labels}
    build_by_id = {str(row["task_id"]): row for row in build_rows}
    aot_by_id = {str(row["task_id"]): row for row in aot_train}
    family_counts = Counter(
        str(label_by_id[task_id].get("family") or "")
        for task_id in supplemental_ids
    )
    if dict(family_counts) != SUPPLEMENTAL_FAMILY_COUNTS:
        raise ExpansionPreparationError(
            "supplemental family accounting differs: "
            f"observed={dict(family_counts)!r}, "
            f"expected={SUPPLEMENTAL_FAMILY_COUNTS!r}"
        )

    source_by_line = {
        index: value
        for index, value in enumerate(source_rows_raw, 1)
    }
    supplemental_targets: list[dict[str, Any]] = []
    supplemental_manifest: list[dict[str, Any]] = []
    for supplemental_row, task_id in enumerate(supplemental_ids):
        label = label_by_id[task_id]
        build_row = build_by_id[task_id]
        metadata = build_row.get("compact_private_metadata")
        if not isinstance(metadata, Mapping):
            raise ExpansionPreparationError(
                f"{task_id}: no source-preparation metadata"
            )
        input_line = int(metadata.get("input_line", -1))
        source_item = source_by_line.get(input_line)
        if source_item is None:
            raise ExpansionPreparationError(
                f"{task_id}: canonical input line {input_line} is absent"
            )
        source_row, source_raw = source_item
        source_raw_sha = sha256_bytes(source_raw)
        target = _base_target_row(
            label=label,
            build_row=build_row,
            source_row=source_row,
            source_raw_sha256=source_raw_sha,
            contract=contract,
        )
        supplemental_targets.append(target)
        aot_row = aot_by_id[task_id]
        supplemental_manifest.append(
            {
                "schema": SELECTION_ROW_SCHEMA,
                "task_id": task_id,
                "partition": "supplemental",
                "expanded_fit_row": PARENT_FIT_ROWS + supplemental_row,
                "supplemental_row": supplemental_row,
                "phase0_train_row": int(build_row["split_row"]),
                "family": str(label["family"]),
                "aot_sha256": str(aot_row["aot_sha256"]),
                "target_sha256": sha256_text(str(target["dart_source"])),
                "tests_sha256": sha256_text(str(target["tests"])),
            }
        )

    fit_manifest: list[dict[str, Any]] = []
    for parent_row, task_id in enumerate(parent_ids):
        label = label_by_id[task_id]
        fit_manifest.append(
            {
                "schema": SELECTION_ROW_SCHEMA,
                "task_id": task_id,
                "partition": "frozen_parent",
                "expanded_fit_row": parent_row,
                "parent_fit_row": parent_row,
                "phase0_train_row": int(build_by_id[task_id]["split_row"]),
                "family": str(label["family"]),
                "aot_sha256": str(aot_by_id[task_id]["aot_sha256"]),
            }
        )
    fit_manifest.extend(supplemental_manifest)
    if [str(row["task_id"]) for row in fit_manifest] != expanded_ids:
        raise ExpansionPreparationError("expanded fit manifest order drifted")

    supplemental_aot = [aot_by_id[task_id] for task_id in supplemental_ids]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_paths = {
        "fit_task_manifest": output_dir / "fit_task_manifest_2776.jsonl",
        "supplemental_task_manifest": (
            output_dir / "supplemental_task_manifest_1196.jsonl"
        ),
        "supplemental_aot_manifest": (
            output_dir / "supplemental_aot_manifest_1196.jsonl"
        ),
        "supplemental_targets": (
            output_dir / "supplemental_targets_unsanitized_1196.jsonl"
        ),
        "supplemental_targets_seal": (
            output_dir / "supplemental_targets_unsanitized_1196.seal.json"
        ),
        "selection_seal": output_dir / "expansion_selection.seal.json",
        "report": output_dir / "preparation_report.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing expansion outputs: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_paths["fit_task_manifest"], fit_manifest)
    atomic_write_jsonl(
        output_paths["supplemental_task_manifest"], supplemental_manifest
    )
    atomic_write_jsonl(
        output_paths["supplemental_aot_manifest"], supplemental_aot
    )
    atomic_write_jsonl(
        output_paths["supplemental_targets"], supplemental_targets
    )
    atomic_write_json(
        output_paths["supplemental_targets_seal"],
        {
            "schema": SPLIT_SEAL_SCHEMA,
            "selected_role": "fit",
            "training_allowed": True,
            "training_objective_scope": "pre_sanitation_private_gold",
            "rows": SUPPLEMENTAL_ROWS,
            "output_sha256": sha256_file(output_paths["supplemental_targets"]),
            "contract_sha256": inputs["frozen_contract"]["sha256"],
            "task_set_sha256": _ordered_digest(supplemental_ids),
            "ordered_task_ids_sha256": _ordered_digest(supplemental_ids),
            "sorted_task_set_sha256": _set_digest(supplemental_ids),
            "target_function": TARGET_FUNCTION,
            "selection_manifest_sha256": sha256_file(
                output_paths["supplemental_task_manifest"]
            ),
        },
    )

    artifacts = {
        **inputs,
        "fit_task_manifest": file_record(output_paths["fit_task_manifest"]),
        "supplemental_task_manifest": file_record(
            output_paths["supplemental_task_manifest"]
        ),
        "supplemental_aot_manifest": file_record(
            output_paths["supplemental_aot_manifest"]
        ),
        "supplemental_targets": file_record(
            output_paths["supplemental_targets"]
        ),
        "supplemental_targets_seal": file_record(
            output_paths["supplemental_targets_seal"]
        ),
    }
    digests = {
        "phase0_train_ordered_task_ids_sha256": _ordered_digest(label_ids),
        "phase0_train_task_set_sha256": _set_digest(label_ids),
        "parent_fit_ordered_task_ids_sha256": _ordered_digest(parent_ids),
        "parent_fit_task_set_sha256": _set_digest(parent_ids),
        "heldout_ordered_task_ids_sha256": _ordered_digest(heldout_ids),
        "heldout_task_set_sha256": _set_digest(heldout_ids),
        "supplemental_ordered_task_ids_sha256": _ordered_digest(
            supplemental_ids
        ),
        "supplemental_task_set_sha256": _set_digest(supplemental_ids),
        "expanded_fit_ordered_task_ids_sha256": _ordered_digest(expanded_ids),
        "expanded_fit_task_set_sha256": _set_digest(expanded_ids),
    }
    counts = {
        "phase0_train_rows": PHASE0_TRAIN_ROWS,
        "phase0_dev_rows": PHASE0_DEV_ROWS,
        "parent_fit_rows": PARENT_FIT_ROWS,
        "heldout_rows": HELDOUT_ROWS,
        "supplemental_rows": SUPPLEMENTAL_ROWS,
        "expanded_fit_rows": EXPANDED_FIT_ROWS,
        "supplemental_family_counts": dict(family_counts),
    }
    selection_seal = {
        "schema": SELECTION_SCHEMA,
        "counts": counts,
        "digests": digests,
        "artifacts": artifacts,
        "heldout_commitment": {
            "dataset_sha256": inputs["heldout"]["sha256"],
            "seal_sha256": inputs["heldout_seal"]["sha256"],
            "ordered_task_ids_sha256": digests[
                "heldout_ordered_task_ids_sha256"
            ],
            "task_set_sha256": digests["heldout_task_set_sha256"],
            "rows": HELDOUT_ROWS,
            "measure_only": True,
        },
        "append_only_contract": {
            "parent_rows_are_exact_prefix": True,
            "parent_rows": PARENT_FIT_ROWS,
            "supplemental_order": "phase0_train_order",
            "frozen_representation_contract_sha256": inputs[
                "frozen_contract"
            ]["sha256"],
        },
        "invariants": {
            "phase0_train_equals_expanded_fit_union_heldout": True,
            "expanded_fit_and_heldout_disjoint": True,
            "parent_fit_and_supplemental_disjoint": True,
            "parent_fit_order_unchanged": True,
            "heldout_membership_unchanged": True,
            "supplemental_aot_is_exact_full_manifest_projection": True,
            "canonical_source_rows_byte_hash_verified": True,
            "raw_semantic_function_names_not_serialized": True,
        },
        "passed": True,
    }
    atomic_write_json(output_paths["selection_seal"], selection_seal)
    report = {
        "schema": REPORT_SCHEMA,
        "counts": counts,
        "digests": digests,
        "selection_seal": file_record(output_paths["selection_seal"]),
        "outputs": {
            key: file_record(path)
            for key, path in output_paths.items()
            if key not in {"report", "selection_seal"}
        },
        "passed": True,
    }
    atomic_write_json(output_paths["report"], report)
    print(
        "PHASE0_2776_EXPANSION_PREPARED "
        f"parent={PARENT_FIT_ROWS} supplemental={SUPPLEMENTAL_ROWS} "
        f"heldout={HELDOUT_ROWS} expanded={EXPANDED_FIT_ROWS} "
        f"selection_sha256={sha256_file(output_paths['selection_seal'])}",
        flush=True,
    )
    return report


def main() -> int:
    prepare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
