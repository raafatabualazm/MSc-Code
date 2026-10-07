#!/usr/bin/env python3
"""Materialize the public, target-free typed task view for the C2 VeRPO pilot.

This CPU-only boundary step may read the gold implementation solely to derive
the already-established opaque type/arity contract.  Its output contains only
``task_id``, opaque typed F2 source, and visible TRAIN feedback tests.  The GPU
trainer consumes this output and never opens the gold-bearing source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    validate_feedback_training_boundary,
)
from scripts.training import t5gemma2_compiler_feedback_verpo as verpo
from scripts.training import t5gemma2_typed_c2_verpo_pilot150 as pilot
from scripts.training import t5gemma2_typed_contract_sft as typed_sft


VIEW_SCHEMA = "t5gemma2-typed-c2-verpo-task-view-v1"
MANIFEST_SCHEMA = "t5gemma2-typed-c2-verpo-task-view-manifest-v1"
EXPECTED_ROLLOUT_SHA256 = (
    "14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c"
)
EXPECTED_ROLLOUT_SEAL_SHA256 = (
    "045c4319b9d2b4e3e29e32eb6b3475404b1b216ced83c17c397f37b5c8fbccca"
)
EXPECTED_F2_SHA256 = (
    "c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3"
)
EXPECTED_F2_MANIFEST_SHA256 = (
    "54d0a1d1eab55a0165fd1a20b99d29dfcc9df7b4e5621d4362781d52ae2e7419"
)
EXPECTED_PUBLIC_MANIFEST_SHA256 = (
    "11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614"
)
EXPECTED_COMPACT_CONTRACT_SHA256 = (
    "f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f"
)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label} has blank row {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(value)
    return rows


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_exact_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("immutable typed task view differs")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return digest


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "rollout": Path(args.rollout_file).expanduser().resolve(),
        "rollout_seal": Path(args.rollout_seal).expanduser().resolve(),
        "f2": Path(args.f2_jsonl).expanduser().resolve(),
        "f2_manifest": Path(args.f2_manifest).expanduser().resolve(),
        "public_manifest": Path(args.feedback_public_manifest).expanduser().resolve(),
        "compact_contract": Path(args.compact_contract).expanduser().resolve(),
    }
    expected = {
        "rollout": EXPECTED_ROLLOUT_SHA256,
        "rollout_seal": EXPECTED_ROLLOUT_SEAL_SHA256,
        "f2": EXPECTED_F2_SHA256,
        "f2_manifest": EXPECTED_F2_MANIFEST_SHA256,
        "public_manifest": EXPECTED_PUBLIC_MANIFEST_SHA256,
        "compact_contract": EXPECTED_COMPACT_CONTRACT_SHA256,
    }
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise ValueError(f"sealed task-view input differs: {key}")
    boundary = validate_feedback_training_boundary(
        rollout=paths["rollout"],
        seal=paths["rollout_seal"],
        f2=paths["f2"],
        f2_manifest=paths["f2_manifest"],
        public_manifest=paths["public_manifest"],
        expected_public_manifest_sha256=EXPECTED_PUBLIC_MANIFEST_SHA256,
        contract=paths["compact_contract"],
        expected_accounting=None,
        expected_eligible_task_ids_sha256=None,
        expected_excluded_task_ids_sha256=None,
    )
    if int(boundary.get("rows", -1)) != 2386:
        raise ValueError("visible feedback boundary row count differs")
    selection = pilot.load_proxy_selection(
        args.proxy_audit_summary,
        args.proxy_audit_journal,
        args.proxy_audit_chain_head,
    )
    rollout_rows = _read_jsonl(paths["rollout"], "VeRPO rollout")
    f2_rows = _read_jsonl(paths["f2"], "VeRPO F2")
    if len(rollout_rows) != len(f2_rows):
        raise ValueError("VeRPO rollout/F2 row counts differ")
    selected = {task.task_id: task for task in selection.tasks}
    output_by_id: dict[str, dict[str, Any]] = {}
    for index, (rollout, f2) in enumerate(zip(rollout_rows, f2_rows, strict=True)):
        task_id = str(rollout.get("task_id") or "")
        if task_id != str(f2.get("task_id") or ""):
            raise ValueError(f"VeRPO rollout/F2 identity mismatch at row {index}")
        expected_task = selected.get(task_id)
        if expected_task is None:
            continue
        if task_id in output_by_id:
            raise ValueError(f"duplicate selected task {task_id}")
        tests = rollout.get("feedback_tests")
        if not isinstance(tests, str) or not tests.strip():
            raise ValueError(f"{task_id}: visible feedback tests are absent")
        verpo.split_visible_expect_harnesses(tests)
        target = verpo._target_source(rollout, task_id)  # noqa: SLF001
        source, contract = typed_sft.build_typed_encoder_source(f2, task_id, target)
        source_sha = _sha256_text(source)
        contract_sha = str(contract.get("opaque_signature_sha256") or "")
        if (
            source_sha != expected_task.source_sha256
            or contract_sha != expected_task.typed_contract_sha256
            or contract.get("function_name") != "fn0"
            or contract.get("semantic_parameter_names_exposed") is not False
        ):
            raise ValueError(f"{task_id}: rebuilt typed source differs from proxy seal")
        output_by_id[task_id] = {
            "schema": VIEW_SCHEMA,
            "position": len(output_by_id),
            "task_id": task_id,
            "source": source,
            "source_sha256": source_sha,
            "typed_contract_sha256": contract_sha,
            "feedback_tests": tests,
            "feedback_tests_sha256": _sha256_text(tests),
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "target_or_gold_present": False,
            "private_holdback_present": False,
        }
    missing = [task.task_id for task in selection.tasks if task.task_id not in output_by_id]
    if missing:
        raise ValueError(f"selected tasks are absent from feedback boundary: {missing[:5]}")
    output_rows = [output_by_id[task.task_id] for task in selection.tasks]
    # Re-number after restoring the proxy audit's canonical order.
    output_rows = [{**row, "position": index} for index, row in enumerate(output_rows)]
    output_path = Path(args.output_view).expanduser().resolve()
    output_sha = _write_exact_jsonl(output_path, output_rows)
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "rows": pilot.EXPECTED_PROXY_TASKS,
        "task_view": {
            "path": str(output_path),
            "sha256": output_sha,
            "rows": len(output_rows),
            "ordered_task_ids_sha256": canonical_sha256(
                [row["task_id"] for row in output_rows]
            ),
            "ordered_source_sha256s_sha256": canonical_sha256(
                [row["source_sha256"] for row in output_rows]
            ),
            "ordered_contract_sha256s_sha256": canonical_sha256(
                [row["typed_contract_sha256"] for row in output_rows]
            ),
        },
        "selection": {
            "proxy_summary_sha256": selection.summary_sha256,
            "proxy_journal_sha256": selection.journal_sha256,
            "proxy_chain_head_sha256": selection.chain_head_sha256,
            "proxy_contract_sha256": selection.contract_sha256,
            "ordered_task_ids_sha256": pilot.EXPECTED_PROXY_TASK_IDS_SHA256,
            "prior_candidates_actions_logprobs_rewards_reused": False,
        },
        "source_boundary": {
            "rows": int(boundary["rows"]),
            "input_sha256": expected,
            "validated": True,
        },
        "privacy": {
            "gold_used_only_for_opaque_contract_derivation": True,
            "gold_or_target_in_task_view": False,
            "acceptance_tests_in_task_view": False,
            "private_holdback_in_task_view": False,
            "visible_train_feedback_tests_in_task_view": True,
        },
        "runtime": {
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "typed_source_builder_sha256": sha256_file(Path(typed_sft.__file__).resolve()),
            "pilot_profile_sha256": sha256_file(Path(pilot.__file__).resolve()),
        },
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    require_exact_or_write(Path(args.output_manifest).expanduser().resolve(), manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--rollout_file", required=True)
    parser.add_argument("--rollout_seal", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--f2_manifest", required=True)
    parser.add_argument("--feedback_public_manifest", required=True)
    parser.add_argument("--compact_contract", required=True)
    parser.add_argument("--proxy_audit_summary", required=True)
    parser.add_argument("--proxy_audit_journal", required=True)
    parser.add_argument("--proxy_audit_chain_head", required=True)
    parser.add_argument("--output_view", required=True)
    parser.add_argument("--output_manifest", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
