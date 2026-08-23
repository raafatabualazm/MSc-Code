#!/usr/bin/env python3
"""Claude Opus 5 O1 gate with the separate benign-context F2 prompt arm.

The Opus schedule, Sonnet bad-result gate, evaluator, task set, K=1 policy,
and cost ceiling are inherited unchanged from ``anthropic_opus5_o1_batch``.
Only the explicitly labeled runtime prompt envelope is different.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import anthropic_benign_f2_prompt_arm as prompt_arm
import anthropic_opus5_o1_batch as opus
import frontier_passk as audited
import frontier_passk_anthropic_batch as batch
from frontier_core import atomic_write_json, file_record, sha256_file


_ORIGINAL_OPUS_CONFIGURED_ENGINE = opus.configured_engine
_ORIGINAL_OPUS_PARSE_ARGS = opus.parse_args
_ORIGINAL_OPUS_CONFIG_FOR_HASH = opus.config_for_hash
_ORIGINAL_OPUS_WRITE_PROGRESS = opus.write_progress_or_summary
_ORIGINAL_AUDITED_PREPARE_RUN = audited.prepare_run


def parse_args(argv: Sequence[str] | None = None) -> Any:
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
    )
    overlay, remaining = parser.parse_known_args(raw)
    if not overlay.attest_authorized_benchmark:
        parser.error(
            "--attest-authorized-benchmark is required because artifact "
            "manifests do not independently prove authorization"
        )
    args = _ORIGINAL_OPUS_PARSE_ARGS(remaining)
    args.operator_attests_authorized_benchmark = True
    source_label = str(getattr(args, "dataset_label", "") or "")
    if not source_label:
        parser.error("the Opus runner returned no dataset label")
    if prompt_arm.ARM_LABEL not in source_label:
        args.dataset_label = source_label + "__" + prompt_arm.ARM_LABEL
    args.prompt_arm_label = prompt_arm.ARM_LABEL
    return args


def config_for_hash(args: Any) -> dict[str, Any]:
    config = _ORIGINAL_OPUS_CONFIG_FOR_HASH(args)
    runtime = dict(config.get("runtime_identity") or {})
    runtime.update(
        {
            "benign_opus_runner_sha256": sha256_file(Path(__file__).resolve()),
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
    provenance["artifacts"]["benign_opus_runner"] = file_record(
        Path(__file__).resolve()
    )
    provenance["prompt_arm"]["manifest_sha256_excluding_self"] = arm_manifest[
        "manifest_sha256_excluding_self"
    ]
    atomic_write_json(out / "provenance.json", provenance)
    return tokenizer, plans, prompt_map, config_sha, provenance


def write_progress_or_summary(
    args: Any,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
    evaluator_record: Mapping[str, Any],
) -> dict[str, Any]:
    result = _ORIGINAL_OPUS_WRITE_PROGRESS(
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
        k=opus.K,
        config_sha256=config_sha,
    )
    return result


@contextlib.contextmanager
def configured_engine() -> Iterator[None]:
    """Layer the prompt arm over the unchanged gated Opus engine."""

    with _ORIGINAL_OPUS_CONFIGURED_ENGINE():
        prior = {
            "parse_args": batch.parse_args,
            "config_for_hash": batch.config_for_hash,
            "_write_progress_or_summary": batch._write_progress_or_summary,
            "prepare_run": audited.prepare_run,
        }
        try:
            batch.parse_args = parse_args
            batch.config_for_hash = config_for_hash
            batch._write_progress_or_summary = write_progress_or_summary
            audited.prepare_run = prepare_run
            yield
        finally:
            batch.parse_args = prior["parse_args"]
            batch.config_for_hash = prior["config_for_hash"]
            batch._write_progress_or_summary = prior[
                "_write_progress_or_summary"
            ]
            audited.prepare_run = prior["prepare_run"]


def main(argv: Sequence[str] | None = None) -> int:
    original = opus.configured_engine
    try:
        opus.configured_engine = configured_engine
        return opus.main(argv)
    finally:
        opus.configured_engine = original


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

