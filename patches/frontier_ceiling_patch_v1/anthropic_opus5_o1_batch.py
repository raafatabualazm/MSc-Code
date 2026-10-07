#!/usr/bin/env python3
"""Fail-closed Claude Opus 5 O1 follow-up for the sealed F2 ceiling pair.

This is deliberately an additive protocol wrapper around
``frontier_passk_anthropic_batch``.  It does not alter the Sonnet screen.

The wrapper will run only when both fixed-cap Sonnet summaries exist at the
predeclared location and ``min(successes_opus_arm, successes_codex_arm) <= 9``.
It then permits exactly one provider Batch creation for one complete sealed
175-task arm:

* model: ``claude-opus-5``;
* K=1 over every task (never a Sonnet-failure subset);
* adaptive thinking with high effort;
* max_tokens=8192, with length counted as failure;
* no length or transport retry batch;
* first-party token counts for every sealed prompt, capped at 16,384;
* $25.088 strict worst-case Batch list-price cap per arm.

The paired shell launcher preflights both arms before submitting either and
runs this file once for each arm.  Together, the Opus base cap is $50.176.
Combined with the predeclared Sonnet cap of $40.1408, the experiment base cap
is $90.3168, leaving $8.6832 below the $99 authorization ceiling.
"""
from __future__ import annotations

import contextlib
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import frontier_passk_anthropic_batch as batch
from frontier_core import (
    PreflightError,
    atomic_write_json,
    file_record,
    load_json,
    sha256_file,
    stable_sha256,
)

SCHEMA = "anthropic-opus5-o1-gated-batch-v1"
TRANSPORT_SCHEMA = "anthropic-message-batches-opus5-o1-v1"
MODEL_ID = "claude-opus-5"
EFFORT = "high"
K = 1
MAX_OUTPUT_TOKENS = 8192
CAP_LADDER = (MAX_OUTPUT_TOKENS,)
EXPECTED_TASKS_PER_ARM = 175
PROVIDER_PROMPT_AUDIT_CAP = 16_384

BATCH_INPUT_USD_PER_MILLION = 2.5
BATCH_OUTPUT_USD_PER_MILLION = 12.5
OPUS_ARM_BASE_CAP_USD = 25.088
OPUS_PAIR_BASE_CAP_USD = 50.176
SONNET_PAIR_BASE_CAP_USD = 40.1408
OVERALL_EXPERIMENT_BASE_CAP_USD = 99.0
COMBINED_BASE_CAP_USD = SONNET_PAIR_BASE_CAP_USD + OPUS_PAIR_BASE_CAP_USD
BASE_CAP_HEADROOM_USD = OVERALL_EXPERIMENT_BASE_CAP_USD - COMBINED_BASE_CAP_USD

BAD_GATE_MAX_SUCCESSES = 9
SONNET_RUN_ROOT = Path(
    "/workspace/artifacts/frontier_ceiling_two_enrichments/runs/"
    "anthropic_sonnet5_batch_screen_k2_warm_v1"
)
SONNET_ARMS = ("opus", "codex")
OPUS_RUN_ROOT = Path(
    "/workspace/artifacts/frontier_ceiling_two_enrichments/runs/"
    "anthropic_opus5_o1_k1_warm_v1"
)
PAIR_MANIFEST_SHA256 = (
    "35f4cfcaf0732928312bed3f2f27c3f3e347525c0076921caeab7ee6539c132e"
)
EVALUATOR_SHA256 = "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
SEALED_ARM_IDENTITIES = {
    "opus": {
        "pair_arm_key": "opus_real_fn0_cfg",
        "prompt_jsonl_sha256": (
            "4aae71997aa98b4a273fdedca17d1df2266f18dd5a03fe164b9cf81e342648cd"
        ),
        "prompt_manifest_sha256": (
            "35e25fa9d7a2bd813b6aec55a1149304d4dd160c82b27b691f27c4cb0bd6068b"
        ),
        "eval_jsonl_sha256": (
            "a4ed1cf185d52c3d212e2d7348fdb2a1dffd0035f4c395e2e897fd072fa70001"
        ),
        "eval_seal_sha256": (
            "2909d279d7c87279b5b0e59cdcd7598742b25a2bd111382f6c8216103f906799"
        ),
    },
    "codex": {
        "pair_arm_key": "codex_multifunction_cfg",
        "prompt_jsonl_sha256": (
            "6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab"
        ),
        "prompt_manifest_sha256": (
            "777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44"
        ),
        "eval_jsonl_sha256": (
            "abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7"
        ),
        "eval_seal_sha256": (
            "5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a"
        ),
    },
}

_SONNET_PARSE_ARGS = batch.parse_args
_SONNET_FIXED_SLOT_POLICY = batch.fixed_slot_policy
_SONNET_CONFIG_FOR_HASH = batch.config_for_hash
_SONNET_SUBMIT = batch._submit
_SONNET_WRITE_PROGRESS_OR_SUMMARY = batch._write_progress_or_summary
_SONNET_OUTCOME_PAYLOAD = batch._outcome_payload

_GATE_ATTESTATION: dict[str, Any] | None = None


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreflightError(f"{field} must be an integer")
    return value


def _validate_sonnet_primary_summary(
    path: Path,
    *,
    arm: str,
) -> tuple[dict[str, Any], int, list[str]]:
    if not path.is_file():
        raise PreflightError(f"missing Sonnet primary summary: {path}")
    summary = load_json(path, f"Sonnet {arm} primary summary")
    exact = {
        "status": "primary_fixed_cap_complete",
        "metric": "primary_fixed_cap_8192",
        "capacity_adaptive": False,
        "length_is_failure": True,
        "model": "claude-sonnet-5",
        "effort": "high",
        "thinking": {"type": "adaptive"},
        "k": 2,
        "fixed_max_output_tokens": 8192,
        "tasks": EXPECTED_TASKS_PER_ARM,
        "logical_slots": EXPECTED_TASKS_PER_ARM * 2,
        "bad_gate_threshold_successes": BAD_GATE_MAX_SUCCESSES,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise PreflightError(
                f"Sonnet {arm} summary field {key!r} is "
                f"{summary.get(key)!r}, expected {expected!r}"
            )
    config_sha = str(summary.get("config_sha256") or "")
    if len(config_sha) != 64:
        raise PreflightError(f"Sonnet {arm} config_sha256 is malformed")
    metric = summary.get("pass_at_2_fixed_8192")
    if not isinstance(metric, Mapping):
        raise PreflightError(f"Sonnet {arm} pass@2 metric is missing")
    successes = _require_int(
        metric.get("successes"),
        field=f"Sonnet {arm} pass@2 successes",
    )
    total = _require_int(metric.get("total"), field=f"Sonnet {arm} total")
    if total != EXPECTED_TASKS_PER_ARM or not 0 <= successes <= total:
        raise PreflightError(f"Sonnet {arm} pass@2 counts are invalid")
    expected_bad = successes <= BAD_GATE_MAX_SUCCESSES
    if summary.get("bad_gate_triggered") is not expected_bad:
        raise PreflightError(
            f"Sonnet {arm} bad_gate_triggered disagrees with successes"
        )
    task_results = summary.get("task_results")
    if not isinstance(task_results, list) or len(task_results) != total:
        raise PreflightError(f"Sonnet {arm} task_results is not complete")
    task_ids: list[str] = []
    for index, row in enumerate(task_results):
        if not isinstance(row, Mapping):
            raise PreflightError(f"Sonnet {arm} task result {index} is malformed")
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PreflightError(f"Sonnet {arm} task result {index} has no task_id")
        task_ids.append(task_id)
    if len(set(task_ids)) != total:
        raise PreflightError(f"Sonnet {arm} task IDs are not unique")
    return summary, successes, task_ids


def load_sonnet_bad_gate(
    root: Path | None = None,
) -> dict[str, Any]:
    """Read and seal the two Sonnet summaries; reject unless BAD is true."""

    if root is None:
        root = SONNET_RUN_ROOT
    arms: dict[str, dict[str, Any]] = {}
    task_order: list[str] | None = None
    for arm in SONNET_ARMS:
        path = root / arm / "primary_8192_summary.json"
        summary, successes, current_order = _validate_sonnet_primary_summary(
            path,
            arm=arm,
        )
        if task_order is None:
            task_order = current_order
        elif current_order != task_order:
            raise PreflightError(
                "the two Sonnet summaries do not have the same sealed task order"
            )
        arms[arm] = {
            "summary": file_record(path),
            "summary_config_sha256": summary["config_sha256"],
            "successes": successes,
            "total": EXPECTED_TASKS_PER_ARM,
            "bad_individually": successes <= BAD_GATE_MAX_SUCCESSES,
            "task_order_sha256": stable_sha256(current_order),
        }
    minimum = min(int(arms[arm]["successes"]) for arm in SONNET_ARMS)
    bad = minimum <= BAD_GATE_MAX_SUCCESSES
    if not bad:
        raise PreflightError(
            "Opus O1 gate is closed: both Sonnet arms exceed 9/175 "
            f"(opus={arms['opus']['successes']}, "
            f"codex={arms['codex']['successes']})"
        )
    attestation = {
        "schema": "anthropic-opus5-o1-bad-gate-attestation-v1",
        "sonnet_run_root": str(root),
        "definition": "min(opus_successes,codex_successes)<=9",
        "threshold_successes": BAD_GATE_MAX_SUCCESSES,
        "minimum_successes": minimum,
        "bad_gate_satisfied": True,
        "arms": arms,
        "paired_task_order_sha256": arms["opus"]["task_order_sha256"],
        "budget": {
            "sonnet_pair_base_cap_usd": SONNET_PAIR_BASE_CAP_USD,
            "opus_pair_base_cap_usd": OPUS_PAIR_BASE_CAP_USD,
            "combined_base_cap_usd": COMBINED_BASE_CAP_USD,
            "overall_experiment_base_cap_usd": (OVERALL_EXPERIMENT_BASE_CAP_USD),
            "base_cap_headroom_usd": BASE_CAP_HEADROOM_USD,
            "within_overall_base_cap": (
                COMBINED_BASE_CAP_USD <= OVERALL_EXPERIMENT_BASE_CAP_USD
            ),
        },
    }
    if not attestation["budget"]["within_overall_base_cap"]:
        raise PreflightError("combined Sonnet+Opus base cap exceeds $99")
    return attestation


def parse_args(argv: Sequence[str] | None = None) -> Any:
    raw = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(raw):
        if value == "--action" and index + 1 < len(raw) and raw[index + 1] == "auto":
            raise PreflightError(
                "--action auto is disabled for O1; use one explicit submit, "
                "then status/harvest"
            )
        if value == "--action=auto":
            raise PreflightError(
                "--action auto is disabled for O1; use one explicit submit, "
                "then status/harvest"
            )
    args = _SONNET_PARSE_ARGS(argv)
    if args.action == "auto":
        raise PreflightError(
            "--action auto is disabled for O1; use one explicit submit, "
            "then status/harvest"
        )
    if args.model != MODEL_ID:
        raise PreflightError(f"--model must be exactly {MODEL_ID!r}")
    if args.k != K:
        raise PreflightError("--k must be exactly 1")
    if args.max_output_tokens != MAX_OUTPUT_TOKENS:
        raise PreflightError("--max-output-tokens must be exactly 8192")
    if args.expected_task_count != EXPECTED_TASKS_PER_ARM:
        raise PreflightError("--expected-task-count must be exactly 175")
    if args.screen_cost_cap_usd > OPUS_ARM_BASE_CAP_USD + 1e-12:
        raise PreflightError(f"per-arm cost cap exceeds ${OPUS_ARM_BASE_CAP_USD:.3f}")
    return args


def fixed_slot_policy(args: Any) -> dict[str, Any]:
    policy = _SONNET_FIXED_SLOT_POLICY(args)
    policy.update(
        {
            "schema": "anthropic-opus5-o1-exact-slot-v1",
            "transport_schema": TRANSPORT_SCHEMA,
            "k": K,
            "initial_max_output_tokens": MAX_OUTPUT_TOKENS,
            "length_retry_cap_ladder": [MAX_OUTPUT_TOKENS],
            "length_retry_enabled": False,
            "transport_retry_batch_enabled": False,
            "one_paid_batch_create_per_arm": True,
            "length_is_failure": True,
            "sonnet_bad_gate_attestation": _require_gate_attestation(),
            "pair_execution_scope": (
                "all 175 tasks in each of both sealed arms; never Sonnet failures"
            ),
        }
    )
    return policy


def config_for_hash(args: Any) -> dict[str, Any]:
    config = _SONNET_CONFIG_FOR_HASH(args)
    config.pop("anthropic_batch_screen", None)
    runtime = dict(config.get("runtime_identity") or {})
    runtime["opus_o1_gate_runner_sha256"] = sha256_file(Path(__file__).resolve())
    config["runtime_identity"] = runtime
    config["anthropic_opus5_o1_gate"] = {
        "schema": SCHEMA,
        "transport_schema": TRANSPORT_SCHEMA,
        "model": MODEL_ID,
        "k": K,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "length_is_failure": True,
        "length_retries": False,
        "transport_retry_batches": False,
        "one_paid_batch_create_per_arm": True,
        "provider_prompt_audit_cap": PROVIDER_PROMPT_AUDIT_CAP,
        "batch_input_usd_per_million": BATCH_INPUT_USD_PER_MILLION,
        "batch_output_usd_per_million": BATCH_OUTPUT_USD_PER_MILLION,
        "opus_arm_base_cap_usd": OPUS_ARM_BASE_CAP_USD,
        "opus_pair_base_cap_usd": OPUS_PAIR_BASE_CAP_USD,
        "sonnet_pair_base_cap_usd": SONNET_PAIR_BASE_CAP_USD,
        "combined_base_cap_usd": COMBINED_BASE_CAP_USD,
        "overall_experiment_base_cap_usd": OVERALL_EXPERIMENT_BASE_CAP_USD,
        "base_cap_headroom_usd": BASE_CAP_HEADROOM_USD,
        "sonnet_bad_gate_attestation": _require_gate_attestation(),
    }
    return config


def _require_gate_attestation() -> dict[str, Any]:
    if _GATE_ATTESTATION is None:
        raise PreflightError("Sonnet BAD gate was not attested")
    return _GATE_ATTESTATION


def _assert_no_prior_paid_create(out: Path, config_sha: str) -> None:
    events = batch._batch_events(out, config_sha)
    submissions = [row for row in events if row.get("event_type") == "batch_submitted"]
    if submissions:
        raise batch.audited.RunFailure(
            "O1 permits exactly one paid Batch creation per arm; "
            "a submission is already recorded"
        )


def _assert_both_exact_arms_preflighted(out: Path) -> None:
    resolved_out = out.resolve()
    expected_root = OPUS_RUN_ROOT.resolve()
    if resolved_out.parent != expected_root or resolved_out.name not in SONNET_ARMS:
        raise batch.audited.RunFailure(
            f"O1 output must be exactly {expected_root}/{{opus,codex}}"
        )
    gate = _require_gate_attestation()
    for arm, expected in SEALED_ARM_IDENTITIES.items():
        provenance_path = expected_root / arm / "provenance.json"
        if not provenance_path.is_file():
            raise batch.audited.RunFailure(
                "both exact O1 arms must complete offline preflight before "
                f"either submission; missing {provenance_path}"
            )
        provenance = load_json(provenance_path, f"Opus O1 {arm} provenance")
        if provenance.get("status") not in {
            "preflight_complete",
            "preflight_only_complete",
        }:
            raise batch.audited.RunFailure(
                f"Opus O1 {arm} does not have a clean preflight state"
            )
        if provenance.get("tasks_selected") != EXPECTED_TASKS_PER_ARM:
            raise batch.audited.RunFailure(
                f"Opus O1 {arm} preflight is not the full 175 tasks"
            )
        config = provenance.get("config")
        if not isinstance(config, Mapping):
            raise batch.audited.RunFailure(
                f"Opus O1 {arm} provenance has no sealed config"
            )
        if (
            config.get("model_requested") != MODEL_ID
            or config.get("k") != K
            or config.get("max_output_tokens") != MAX_OUTPUT_TOKENS
            or config.get("expected_evaluator_sha256") != EVALUATOR_SHA256
            or config.get("pair_arm_key") != expected["pair_arm_key"]
        ):
            raise batch.audited.RunFailure(
                f"Opus O1 {arm} preflight contract identity is wrong"
            )
        sealed = config.get("sealed_inputs")
        if not isinstance(sealed, Mapping):
            raise batch.audited.RunFailure(
                f"Opus O1 {arm} sealed input identity is missing"
            )
        exact_sealed = {
            "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
            **expected,
        }
        for field, value in exact_sealed.items():
            if sealed.get(field) != value:
                raise batch.audited.RunFailure(
                    f"Opus O1 {arm} sealed field {field} is wrong"
                )
        protocol = config.get("anthropic_opus5_o1_gate")
        if not isinstance(protocol, Mapping):
            raise batch.audited.RunFailure(f"Opus O1 {arm} protocol seal is missing")
        if protocol.get("sonnet_bad_gate_attestation") != gate:
            raise batch.audited.RunFailure(
                f"Opus O1 {arm} was preflighted against a different Sonnet gate"
            )


def submit_once(
    args: Any,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
) -> dict[str, Any]:
    _assert_both_exact_arms_preflighted(out)
    _assert_no_prior_paid_create(out, config_sha)
    if len(plans) != EXPECTED_TASKS_PER_ARM:
        raise batch.audited.RunFailure("O1 must submit all 175 sealed tasks")
    specs = batch.pending_request_specs(plans, [], [])
    if len(specs) != EXPECTED_TASKS_PER_ARM:
        raise batch.audited.RunFailure("O1 must create exactly 175 requests")
    if any(
        int(spec["sample_index"]) != 0
        or int(spec["cap"]) != MAX_OUTPUT_TOKENS
        or int(spec["cap_attempt_index"]) != 0
        for spec in specs
    ):
        raise batch.audited.RunFailure("O1 initial request set is malformed")
    return _SONNET_SUBMIT(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
    )


def outcome_payload(**kwargs: Any) -> dict[str, Any]:
    kwargs["metric_role"] = "primary_fixed_cap_8192"
    return _SONNET_OUTCOME_PAYLOAD(**kwargs)


def _rewrite_metric_artifacts(out: Path) -> dict[str, Any] | None:
    primary_path = out / "primary_8192_summary.json"
    if primary_path.is_file():
        primary = load_json(primary_path, "Opus primary summary")
        pass_metric = primary.pop("pass_at_2_fixed_8192", None)
        compile_metric = primary.pop("compile_at_2_fixed_8192", None)
        if pass_metric is None or compile_metric is None:
            raise batch.audited.RunFailure(
                "Opus primary summary lacks the inherited metric fields"
            )
        primary.pop("bad_gate_threshold_successes", None)
        primary.pop("bad_gate_triggered", None)
        primary.update(
            {
                "schema": SCHEMA,
                "metric": "primary_fixed_cap_8192",
                "model": MODEL_ID,
                "k": K,
                "pass_at_1_fixed_8192": pass_metric,
                "compile_at_1_fixed_8192": compile_metric,
                "source_sonnet_bad_gate": _require_gate_attestation(),
            }
        )
        atomic_write_json(primary_path, primary)

    progress_path = out / "progress.json"
    if not progress_path.is_file():
        return None
    progress = load_json(progress_path, "Opus progress")
    progress.update(
        {
            "schema": SCHEMA,
            "model": MODEL_ID,
            "k": K,
            "opus_o1_no_retry_batches": True,
            "source_sonnet_bad_gate": _require_gate_attestation(),
        }
    )
    primary_progress = progress.get("primary_fixed_cap_8192")
    if isinstance(primary_progress, dict):
        metric = primary_progress.pop("pass_at_2_fixed_8192", None)
        primary_progress.pop("bad_gate_triggered", None)
        if metric is not None:
            primary_progress["pass_at_1_fixed_8192"] = metric
        if primary_path.is_file():
            primary_progress["summary"] = file_record(primary_path)
    atomic_write_json(progress_path, progress)

    summary_path = out / "summary.json"
    if not summary_path.is_file():
        return progress
    summary = load_json(summary_path, "Opus summary")
    pass_metric = summary.pop("capacity_adaptive_pass_at_2", None)
    compile_metric = summary.pop("capacity_adaptive_compile_at_2", None)
    if pass_metric is None or compile_metric is None:
        raise batch.audited.RunFailure(
            "Opus final summary lacks inherited metric fields"
        )
    summary.update(
        {
            "schema": SCHEMA,
            "transport_schema": TRANSPORT_SCHEMA,
            "requested_model": MODEL_ID,
            "k": K,
            "metric": "primary_fixed_cap_8192",
            "fixed_max_output_tokens": MAX_OUTPUT_TOKENS,
            "length_is_failure": True,
            "length_retries": False,
            "transport_retry_batches": False,
            "one_paid_batch_create_per_arm": True,
            "pass_at_1_fixed_8192": pass_metric,
            "compile_at_1_fixed_8192": compile_metric,
            "source_sonnet_bad_gate": _require_gate_attestation(),
            "budget": _require_gate_attestation()["budget"],
        }
    )
    artifact_names = (
        "tasks.jsonl",
        "prompts.jsonl",
        "batch_events.jsonl",
        "batch_slot_attempts.jsonl",
        "anthropic_input_token_counts.jsonl",
        "anthropic_input_token_audit.json",
        "primary_8192_outcomes.jsonl",
        "primary_8192_summary.json",
        "terminal_slots.jsonl",
        "outcomes.jsonl",
        "progress.json",
    )
    summary["artifacts"] = {
        name: file_record(out / name)
        for name in artifact_names
        if (out / name).is_file()
    }
    atomic_write_json(summary_path, summary)

    provenance_path = out / "provenance.json"
    provenance = load_json(provenance_path, "Opus provenance")
    provenance["status"] = "complete"
    provenance["completed_at"] = summary["completed_at"]
    provenance["summary_sha256"] = sha256_file(summary_path)
    provenance["sonnet_bad_gate_attestation"] = _require_gate_attestation()
    atomic_write_json(provenance_path, provenance)

    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path, "Opus manifest")
        names = list((manifest.get("files") or {}).keys())
        manifest["schema"] = SCHEMA
        manifest["files"] = {
            name: file_record(out / name) for name in names if (out / name).is_file()
        }
        atomic_write_json(manifest_path, manifest)
    return summary


def write_progress_or_summary(
    args: Any,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
    evaluator_record: Mapping[str, Any],
) -> dict[str, Any]:
    result = _SONNET_WRITE_PROGRESS_OR_SUMMARY(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
        provenance=provenance,
        evaluator_record=evaluator_record,
    )
    rewritten = _rewrite_metric_artifacts(out)
    return result if rewritten is None else rewritten


@contextlib.contextmanager
def configured_engine() -> Iterator[None]:
    """Temporarily specialize the shared tested transport for Opus O1."""

    names = {
        "SCHEMA": SCHEMA,
        "TRANSPORT_SCHEMA": TRANSPORT_SCHEMA,
        "MODEL_ID": MODEL_ID,
        "EFFORT": EFFORT,
        "K": K,
        "CAP_LADDER": CAP_LADDER,
        "PROVIDER_PROMPT_AUDIT_CAP": PROVIDER_PROMPT_AUDIT_CAP,
        "BATCH_INPUT_USD_PER_MILLION": BATCH_INPUT_USD_PER_MILLION,
        "BATCH_OUTPUT_USD_PER_MILLION": BATCH_OUTPUT_USD_PER_MILLION,
        "DEFAULT_ARM_COST_CAP_USD": OPUS_ARM_BASE_CAP_USD,
        "MAX_RETRIES_PER_CAP": 1,
        "parse_args": parse_args,
        "fixed_slot_policy": fixed_slot_policy,
        "config_for_hash": config_for_hash,
        "_submit": submit_once,
        "_outcome_payload": outcome_payload,
        "_write_progress_or_summary": write_progress_or_summary,
    }
    prior = {name: getattr(batch, name) for name in names}
    audited_prior = {
        "resolve_api_configuration": batch.audited.resolve_api_configuration,
        "fixed_slot_policy": batch.audited.fixed_slot_policy,
        "config_for_hash": batch.audited.config_for_hash,
    }
    try:
        for name, value in names.items():
            setattr(batch, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(batch, name, value)
        for name, value in audited_prior.items():
            setattr(batch.audited, name, value)


def main(argv: Sequence[str] | None = None) -> int:
    global _GATE_ATTESTATION
    try:
        _GATE_ATTESTATION = load_sonnet_bad_gate()
    except Exception as exc:
        print(
            "ANTHROPIC_OPUS5_O1_GATE_CLOSED " f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        with configured_engine():
            return batch.main(argv)
    except Exception as exc:
        # The underlying runner normally catches and seals run failures.  This
        # guard covers failures that occur before it selects an output folder.
        print(
            "ANTHROPIC_OPUS5_O1_FAILED_CLOSED " f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 2
    finally:
        _GATE_ATTESTATION = None


if __name__ == "__main__":
    raise SystemExit(main())
