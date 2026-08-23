#!/usr/bin/env python3
"""Run the explicitly separate, context-readable Anthropic F2 prompt arm.

This entry point reuses the audited Anthropic Message Batches transport and
evaluator.  It changes only the runtime prompt envelope through the
``anthropic_benign_f2_prompt_arm`` overlay.  The original sealed F2 artifacts,
task schedule, evaluator, and acceptance tests remain pinned and unchanged.

Importing this module and the default ``preflight`` action make no paid API
calls.  Paid actions remain governed by the base runner's explicit action and
cost gates.
"""
from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import anthropic_benign_f2_prompt_arm as prompt_arm
import frontier_passk as audited
import frontier_passk_anthropic_batch as batch
from frontier_core import atomic_write_json, file_record, sha256_file


_ORIGINAL_BATCH_INSTALL_HOOKS = batch.install_hooks
_ORIGINAL_BATCH_PARSE_ARGS = batch.parse_args
_ORIGINAL_BATCH_CONFIG_FOR_HASH = batch.config_for_hash
_ORIGINAL_AUDITED_PREPARE_RUN = audited.prepare_run
_ORIGINAL_BATCH_WRITE_PROGRESS = batch._write_progress_or_summary


def parse_args(argv: Sequence[str] | None = None) -> Any:
    """Require an explicit operator attestation for the contextual claim."""

    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--attest-authorized-benchmark",
        action="store_true",
        default=(
            os.environ.get(
                "ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK", ""
            )
            == "1"
        ),
        help=(
            "Attest that the supplied benchmark/compiler artifacts are "
            "authorized for this controlled research evaluation."
        ),
    )
    overlay, remaining = parser.parse_known_args(raw)
    if not overlay.attest_authorized_benchmark:
        parser.error(
            "--attest-authorized-benchmark is required because artifact "
            "manifests do not independently prove authorization"
        )
    args = _ORIGINAL_BATCH_PARSE_ARGS(remaining)
    args.operator_attests_authorized_benchmark = True
    source_label = str(getattr(args, "dataset_label", "") or "")
    if not source_label:
        parser.error("the base runner returned no dataset label")
    if prompt_arm.ARM_LABEL not in source_label:
        args.dataset_label = source_label + "__" + prompt_arm.ARM_LABEL
    args.prompt_arm_label = prompt_arm.ARM_LABEL
    return args


def config_for_hash(args: Any) -> dict[str, Any]:
    """Bind the separate prompt arm and all code that implements it."""

    config = _ORIGINAL_BATCH_CONFIG_FOR_HASH(args)
    runtime = dict(config.get("runtime_identity") or {})
    runtime.update(
        {
            "benign_prompt_arm_runner_sha256": sha256_file(
                Path(__file__).resolve()
            ),
            "benign_prompt_arm_module_sha256": sha256_file(
                Path(prompt_arm.__file__).resolve()
            ),
        }
    )
    config["runtime_identity"] = runtime
    config["anthropic_prompt_arm"] = {
        "contract": prompt_arm.arm_contract(),
        "operator_attests_authorized_benchmark": bool(
            args.operator_attests_authorized_benchmark
        ),
    }
    return config


def prepare_run(
    args: Any,
    out: Path,
) -> tuple[
    Any,
    list[dict[str, Any]],
    MutableMapping[str, dict[str, Any]],
    str,
    dict[str, Any],
]:
    """Prepare the original sealed run, then apply the explicit overlay."""

    tokenizer, plans, prompt_map, config_sha, provenance = (
        _ORIGINAL_AUDITED_PREPARE_RUN(args, out)
    )
    plans, prompt_map, arm_manifest = prompt_arm.apply_prompt_arm(
        tokenizer=tokenizer,
        plans=plans,
        prompt_map=prompt_map,
        config_sha256=config_sha,
        provenance=provenance,
        args=args,
        out=out,
    )
    provenance["artifacts"]["benign_prompt_arm_module"] = file_record(
        Path(prompt_arm.__file__).resolve()
    )
    provenance["artifacts"]["benign_prompt_arm_runner"] = file_record(
        Path(__file__).resolve()
    )
    provenance["prompt_arm"]["manifest_sha256_excluding_self"] = arm_manifest[
        "manifest_sha256_excluding_self"
    ]
    atomic_write_json(out / "provenance.json", provenance)
    # apply_prompt_arm already wrote the deterministic files.  Returning the
    # enriched object lets the base runner retain its ordinary status updates.
    return tokenizer, plans, prompt_map, config_sha, provenance


def _write_progress_or_summary(
    args: Any,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
    evaluator_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve base metrics and emit a separate refusal-specific report."""

    result = _ORIGINAL_BATCH_WRITE_PROGRESS(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
        provenance=provenance,
        evaluator_record=evaluator_record,
    )
    prompt_arm.write_refusal_report(
        out=out,
        task_ids=[str(plan["task_id"]) for plan in plans],
        k=batch.K,
        config_sha256=config_sha,
    )
    return result


def install_hooks() -> None:
    """Install process-local hooks without changing the base entry point."""

    _ORIGINAL_BATCH_INSTALL_HOOKS()
    batch.parse_args = parse_args
    audited.config_for_hash = config_for_hash
    audited.prepare_run = prepare_run
    batch._write_progress_or_summary = _write_progress_or_summary


def main(argv: Sequence[str] | None = None) -> int:
    # batch.main invokes its install hook after argument parsing.  Redirect
    # that process-local hook only in this distinct entry point.
    batch.install_hooks = install_hooks
    return batch.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
