#!/usr/bin/env python3
"""Seal a paired Opus-real versus Codex-multifunction frontier cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from frontier_core import (
    atomic_write_json,
    file_record,
    load_jsonl,
    sha256_file,
    sha256_text,
    stable_sha256,
    utc_now,
)


class PairManifestError(ValueError):
    pass


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairManifestError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PairManifestError(f"{label} is not a JSON object")
    return value


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected.strip().lower():
        raise PairManifestError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return file_record(path)


def verify_seal(
    seal: Mapping[str, Any],
    *,
    dataset_sha256: str,
    rows: int,
    label: str,
) -> None:
    if (
        seal.get("schema") != "compact-public-private-join-seal-v1"
        or seal.get("selected_role") != "measure"
        or int(seal.get("rows", -1)) != rows
        or str(seal.get("output_sha256") or "") != dataset_sha256
    ):
        raise PairManifestError(f"{label} is not the expected measure-only seal")
    if "heldout_measure_only" in seal and seal["heldout_measure_only"] is not True:
        raise PairManifestError(f"{label} does not attest heldout_measure_only")
    if "training_allowed" in seal and seal["training_allowed"] is not False:
        raise PairManifestError(f"{label} permits training")


def verify_prompt_artifact(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    prompt_record: Mapping[str, Any],
    eval_rows: list[dict[str, Any]],
    label: str,
) -> tuple[str, str]:
    if manifest.get("schema") != "verified-api-readable-compact-v2":
        raise PairManifestError(f"{label} prompt manifest schema mismatch")
    if int(manifest.get("rows", -1)) != len(rows):
        raise PairManifestError(f"{label} prompt manifest row count mismatch")
    output = manifest.get("output")
    dataset = manifest.get("dataset")
    contract = manifest.get("f2_prompt_contract")
    invariants = manifest.get("invariants")
    if not all(
        isinstance(value, Mapping)
        for value in (output, dataset, contract, invariants)
    ):
        raise PairManifestError(f"{label} prompt manifest is incomplete")
    if (
        output.get("sha256") != prompt_record["sha256"]
        or int(output.get("size_bytes", output.get("bytes", -1)))
        != prompt_record["bytes"]
    ):
        raise PairManifestError(f"{label} prompt output record mismatch")
    if dataset.get("sha256") != sha256_file(Path(eval_rows[0]["_source_path"])):
        raise PairManifestError(f"{label} prompt source dataset mismatch")
    system_prompt = str(contract.get("system_prompt") or "")
    system_sha = str(contract.get("system_prompt_sha256") or "")
    if (
        not system_prompt
        or sha256_text(system_prompt) != system_sha
        or contract.get("representation_schema") != "lossless-semantic-f2"
        or contract.get("all_rows_within_limit") is not True
        or int(contract.get("max_prompt_tokens", -1)) != 12000
        or int(contract.get("chat_overhead_reserve", -1)) != 256
        or int(contract.get("maximum_estimated_prompt_tokens", 12001)) > 12000
    ):
        raise PairManifestError(f"{label} F2 prompt contract failed")
    required_invariants = (
        "all_artifact_hashes_verified",
        "all_row_contract_hashes_verified",
        "all_codec_roundtrips_verified",
        "all_student_constant_prefixes_verified",
        "all_f2_semantic_roundtrips_verified",
        "f2_system_prompt_self_contained_and_hashed",
        "all_complete_prompts_within_limit",
        "opaque_source_ids_expanded",
        "cfg_explicit",
    )
    if any(invariants.get(key) is not True for key in required_invariants):
        raise PairManifestError(f"{label} F2 invariant set failed")
    eval_ids = [str(row.get("task_id") or "") for row in eval_rows]
    prompt_ids: list[str] = []
    forbidden = {"dart_source", "tests", "acceptance_tests", "feedback_tests"}
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        text = row.get("text")
        if (
            not task_id
            or not isinstance(text, str)
            or not text
            or row.get("representation_schema") != "lossless-semantic-f2"
            or row.get("system_prompt_sha256") != system_sha
            or row.get("text_sha256") != sha256_text(text)
            or forbidden.intersection(row)
        ):
            raise PairManifestError(f"{label} invalid prompt row {index}")
        prompt_ids.append(task_id)
    if prompt_ids != eval_ids:
        raise PairManifestError(f"{label} prompt/evaluation task order differs")
    if manifest.get("task_set_sha256") != stable_sha256(prompt_ids):
        raise PairManifestError(f"{label} task-set commitment mismatch")
    return system_prompt, system_sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--opus-eval", required=True, type=Path)
    parser.add_argument("--opus-eval-sha256", required=True)
    parser.add_argument("--opus-seal", required=True, type=Path)
    parser.add_argument("--opus-seal-sha256", required=True)
    parser.add_argument("--opus-prompts", required=True, type=Path)
    parser.add_argument("--opus-prompts-sha256", required=True)
    parser.add_argument("--opus-prompt-manifest", required=True, type=Path)
    parser.add_argument("--opus-prompt-manifest-sha256", required=True)
    parser.add_argument("--codex-eval", required=True, type=Path)
    parser.add_argument("--codex-eval-sha256", required=True)
    parser.add_argument("--codex-seal", required=True, type=Path)
    parser.add_argument("--codex-seal-sha256", required=True)
    parser.add_argument("--codex-prompts", required=True, type=Path)
    parser.add_argument("--codex-prompts-sha256", required=True)
    parser.add_argument("--codex-prompt-manifest", required=True, type=Path)
    parser.add_argument("--codex-prompt-manifest-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=175)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "opus_eval": args.opus_eval.resolve(),
        "opus_seal": args.opus_seal.resolve(),
        "opus_prompts": args.opus_prompts.resolve(),
        "opus_prompt_manifest": args.opus_prompt_manifest.resolve(),
        "codex_eval": args.codex_eval.resolve(),
        "codex_seal": args.codex_seal.resolve(),
        "codex_prompts": args.codex_prompts.resolve(),
        "codex_prompt_manifest": args.codex_prompt_manifest.resolve(),
    }
    records = {
        key: require_hash(
            path, getattr(args, f"{key}_sha256"), key.replace("_", " ")
        )
        for key, path in paths.items()
    }
    opus_eval = load_jsonl(paths["opus_eval"], "Opus evaluation")
    codex_eval = load_jsonl(paths["codex_eval"], "Codex evaluation")
    if len(opus_eval) != args.expected_rows or len(codex_eval) != args.expected_rows:
        raise PairManifestError("paired evaluation row count mismatch")
    # Stash the source path outside serialized output solely so the common
    # verifier can compare each F2 manifest's dataset commitment.
    for row in opus_eval:
        row["_source_path"] = str(paths["opus_eval"])
    for row in codex_eval:
        row["_source_path"] = str(paths["codex_eval"])
    opus_seal = load_object(paths["opus_seal"], "Opus seal")
    codex_seal = load_object(paths["codex_seal"], "Codex seal")
    verify_seal(
        opus_seal,
        dataset_sha256=records["opus_eval"]["sha256"],
        rows=args.expected_rows,
        label="Opus seal",
    )
    verify_seal(
        codex_seal,
        dataset_sha256=records["codex_eval"]["sha256"],
        rows=args.expected_rows,
        label="Codex seal",
    )
    opus_prompts = load_jsonl(paths["opus_prompts"], "Opus prompts")
    codex_prompts = load_jsonl(paths["codex_prompts"], "Codex prompts")
    opus_manifest = load_object(
        paths["opus_prompt_manifest"], "Opus prompt manifest"
    )
    codex_manifest = load_object(
        paths["codex_prompt_manifest"], "Codex prompt manifest"
    )
    opus_system, opus_system_sha = verify_prompt_artifact(
        opus_prompts,
        opus_manifest,
        prompt_record=records["opus_prompts"],
        eval_rows=opus_eval,
        label="Opus",
    )
    codex_system, codex_system_sha = verify_prompt_artifact(
        codex_prompts,
        codex_manifest,
        prompt_record=records["codex_prompts"],
        eval_rows=codex_eval,
        label="Codex",
    )
    if opus_system != codex_system or opus_system_sha != codex_system_sha:
        raise PairManifestError("paired arms do not use the identical F2 grammar")

    opus_ids = [str(row["task_id"]) for row in opus_eval]
    codex_ids = [str(row["task_id"]) for row in codex_eval]
    if opus_ids != codex_ids or len(set(opus_ids)) != args.expected_rows:
        raise PairManifestError("paired arms do not use the same ordered tasks")
    opus_test_hashes = [
        sha256_text(str(row.get("acceptance_tests") or "")) for row in opus_eval
    ]
    codex_test_hashes = [
        sha256_text(str(row.get("acceptance_tests") or "")) for row in codex_eval
    ]
    if (
        opus_test_hashes != codex_test_hashes
        or any(
            not row.get("acceptance_tests")
            or row.get("acceptance_tests") != row.get("tests")
            for row in opus_eval + codex_eval
        )
    ):
        raise PairManifestError("paired arms do not use identical acceptance tests")

    # Remove local helper fields before committing the paired semantic rows.
    for row in opus_eval + codex_eval:
        row.pop("_source_path", None)
    pair = {
        "schema": "frontier-enrichment-pair-v1",
        "created_at": utc_now(),
        "arms": {
            "opus_real_fn0_cfg": {
                key.removeprefix("opus_"): records[key]
                for key in records
                if key.startswith("opus_")
            },
            "codex_multifunction_cfg": {
                key.removeprefix("codex_"): records[key]
                for key in records
                if key.startswith("codex_")
            },
        },
        "rows": args.expected_rows,
        "ordered_task_ids_sha256": stable_sha256(opus_ids),
        "ordered_acceptance_test_hashes_sha256": stable_sha256(opus_test_hashes),
        "system_prompt_sha256": opus_system_sha,
        "comparison": {
            "opus": "real binary constants + compressed fn0 assembly + explicit fn0 CFG",
            "codex": (
                "real binary constants + compressed fn0/helpers/closures assembly "
                "+ external aliases + explicit whole-user-binary CFG"
            ),
        },
        "invariants": {
            "same_ordered_175_tasks": True,
            "same_acceptance_tests": True,
            "same_f2_grammar": True,
            "same_prompt_budget_contract": True,
            "measure_only_seals_verified": True,
            "provider_prompts_exclude_gold_and_tests": True,
            "prompt_artifact_hashes_verified": True,
            "no_prompt_truncation": True,
        },
    }
    atomic_write_json(args.out, pair)
    print(
        "FRONTIER_PAIR_SEALED "
        f"rows={args.expected_rows} task_set={pair['ordered_task_ids_sha256']} "
        f"sha256={sha256_file(args.out)} out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
