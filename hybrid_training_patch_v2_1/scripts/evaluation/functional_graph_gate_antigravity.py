#!/usr/bin/env python3
"""Functional, pass@k-based graph causality and RS-SFT acceptance gate.

The script separates two questions that the old NLL margin conflated:

* deployment performance is measured with the real configured prompt;
* graph causality is measured in a graph-only arm where correct, shape-matched
  permuted, and null prefixes share the identical target prompt and tests.

No teacher-forced loss is an acceptance criterion.  A stage passes only when
permuting the graph destroys held-out functional performance by the configured
amount, and (when supplied) the new checkpoint improves deployment pass@k over
its frozen baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.hybrid_data_controls import (  # noqa: E402
    candidate_fact_match,
    infer_function_name,
    parse_facts_comment,
    read_jsonl,
    task_key,
)


def load_evaluator():
    try:
        from scripts.evaluation.graph_compile_at_k_antigravity import (  # type: ignore
            evaluate_dart_jit_tests_detail,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("project-aligned pass@k evaluator is unavailable") from exc
    return evaluate_dart_jit_tests_detail


def observed_pass_at_k(successes: list[bool], k: int) -> float:
    n = len(successes)
    if n == 0:
        return 0.0
    c = sum(bool(value) for value in successes)
    k = max(1, min(k, n))
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Linearly interpolate a percentile from an already sorted sample."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0,1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def exact_one_sided_sign_p_value(wins: int, losses: int) -> float:
    """Exact P[X >= wins] for X~Binomial(wins+losses, 0.5).

    Ties are conditioned away. For binary task-level pass@k this is the exact
    McNemar test; for fractional per-task pass@k it is the exact paired sign
    test. The pre-registered one-sided alternative is left > right.
    """
    if wins < 0 or losses < 0:
        raise ValueError("wins/losses must be non-negative")
    n = wins + losses
    if n == 0:
        return 1.0
    numerator = sum(math.comb(n, value) for value in range(wins, n + 1))
    return float(numerator / (2 ** n))


def paired_bootstrap_mean_difference(
    left: list[float],
    right: list[float],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Paired task bootstrap for mean(left-right), in percentage points.

    Tasks, not generated candidates, are the resampling unit. Candidate
    stochasticity is reflected in each task's observed pass@k, although final
    paper estimates should still repeat generation under multiple seeds.
    """
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires aligned non-empty samples")
    if iterations < 100:
        raise ValueError("paired bootstrap requires at least 100 iterations")
    if not 0.5 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0.5,1)")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    n = len(differences)
    point = sum(differences) / n
    rng = random.Random(seed)
    samples: list[float] = []
    choose = rng.randrange
    for _ in range(iterations):
        samples.append(sum(differences[choose(n)] for _ in range(n)) / n)
    samples.sort()
    alpha = 1.0 - confidence
    return {
        "unit": "task",
        "rows": n,
        "iterations": iterations,
        "confidence": confidence,
        "seed": seed,
        "point_estimate_pp": 100.0 * point,
        "one_sided_lower_pp": 100.0 * _percentile(samples, alpha),
        "two_sided_ci_pp": [
            100.0 * _percentile(samples, alpha / 2.0),
            100.0 * _percentile(samples, 1.0 - alpha / 2.0),
        ],
        "bootstrap_probability_positive": sum(value > 0.0 for value in samples) / len(samples),
    }


def paired_task_comparison(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    metric: str,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Return direction counts, exact sign test, and paired task bootstrap."""
    left_by_key = {str(row["task_key"]): row for row in left_rows}
    right_by_key = {str(row["task_key"]): row for row in right_rows}
    if set(left_by_key) != set(right_by_key):
        raise RuntimeError(f"paired task keys do not align for {metric}")
    keys = sorted(left_by_key)
    left = [float(left_by_key[key][metric]) for key in keys]
    right = [float(right_by_key[key][metric]) for key in keys]
    wins = losses = ties = 0
    for a, b in zip(left, right):
        if a > b:
            wins += 1
        elif a < b:
            losses += 1
        else:
            ties += 1
    task_metric_binary = all(value in {0.0, 1.0} for value in left + right)
    return {
        "metric": metric,
        "rows": len(keys),
        "left_above_right_tasks": wins,
        "equal_tasks": ties,
        "right_above_left_tasks": losses,
        "effective_non_ties": wins + losses,
        "exact_one_sided_sign_p_value": exact_one_sided_sign_p_value(wins, losses),
        "task_metric_binary": task_metric_binary,
        "exact_test": (
            "one_sided_exact_mcnemar"
            if task_metric_binary
            else "one_sided_exact_paired_sign"
        ),
        "bootstrap": paired_bootstrap_mean_difference(
            left,
            right,
            iterations=iterations,
            confidence=confidence,
            seed=seed,
        ),
    }


def fixed_aggregate_baseline_comparison(
    current_rows: list[dict[str, Any]],
    *,
    metric: str,
    baseline: float,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Task bootstrap versus a fixed aggregate baseline.

    This is weaker than a checkpoint-paired comparison because uncertainty in
    the externally supplied baseline is unavailable. The curriculum defaults
    to a matched checkpoint control.
    """
    current = [float(row[metric]) for row in current_rows]
    comparison = paired_bootstrap_mean_difference(
        current,
        [float(baseline)] * len(current),
        iterations=iterations,
        confidence=confidence,
        seed=seed,
    )
    return {
        "metric": metric,
        "comparison_type": "fixed_aggregate_baseline",
        "baseline": float(baseline),
        "baseline_uncertainty_ignored": True,
        "bootstrap": comparison,
    }


def _paired_statistical_failures(
    label: str,
    comparison: dict[str, Any],
    *,
    minimum_effective_pairs: int,
    maximum_p_value: float,
    minimum_lower_bound_pp: float,
) -> list[str]:
    failures: list[str] = []
    effective = int(comparison["effective_non_ties"])
    p_value = float(comparison["exact_one_sided_sign_p_value"])
    lower = float(comparison["bootstrap"]["one_sided_lower_pp"])
    if effective < minimum_effective_pairs:
        failures.append(
            f"{label} has only {effective} discordant/effective task pairs; "
            f"minimum is {minimum_effective_pairs}"
        )
    if p_value > maximum_p_value:
        failures.append(
            f"{label} exact one-sided sign-test p={p_value:.6g} > {maximum_p_value:.6g}"
        )
    # Strictly positive when the configured floor is zero.
    if lower <= minimum_lower_bound_pp:
        failures.append(
            f"{label} paired-bootstrap one-sided lower bound {lower:.3f} pp "
            f"<= {minimum_lower_bound_pp:.3f} pp"
        )
    return failures


def evaluate_predictions(
    dataset: list[dict[str, Any]],
    prediction_path: Path,
    *,
    k_values: list[int],
    workers: int,
    timeout: int,
    expected_candidates: int,
) -> dict[str, Any]:
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"{prediction_path} is not a prediction list")
    evaluator = load_evaluator()
    if len(predictions) != len(dataset):
        raise ValueError(
            f"{prediction_path} contains {len(predictions)} rows for a {len(dataset)}-row gate dataset"
        )
    jobs: list[tuple[int, int, dict[str, Any], str]] = []
    seen_source_lines: set[int] = set()
    for pred_index, prediction in enumerate(predictions):
        source_line = int(prediction.get("source_line") or pred_index + 1) - 1
        if not 0 <= source_line < len(dataset):
            raise IndexError(f"prediction row {pred_index} source_line out of range")
        if source_line in seen_source_lines:
            raise ValueError(f"{prediction_path} duplicates source_line={source_line + 1}")
        seen_source_lines.add(source_line)
        candidates = prediction.get("predictions") or []
        if len(candidates) != expected_candidates:
            raise ValueError(
                f"{prediction_path} source_line={source_line + 1} has {len(candidates)} "
                f"candidates; expected exactly {expected_candidates}"
            )
        row = dataset[source_line]
        for candidate_index, candidate in enumerate(candidates):
            jobs.append((source_line, candidate_index, row, str(candidate or "")))
    expected_lines = set(range(len(dataset)))
    if seen_source_lines != expected_lines:
        missing = sorted(expected_lines - seen_source_lines)
        raise ValueError(f"{prediction_path} is missing source lines: {missing[:12]}")

    outcomes: dict[int, dict[int, dict[str, bool]]] = {}

    def run(job):
        source_line, candidate_index, row, candidate = job
        compiled, passed, _diagnostic, _source = evaluator(
            candidate,
            str(row.get("tests") or ""),
            f"functional_gate_{source_line}_{candidate_index}",
            timeout=timeout,
        )
        claims = parse_facts_comment(candidate)
        facts_ok, _reasons = candidate_fact_match(
            row,
            candidate,
            teacher_claim=claims,
            mode="conservative",
            require_claims=True,
        )
        return source_line, candidate_index, {
            "compiled": bool(compiled),
            "passed": bool(passed),
            "facts_ok": bool(facts_ok),
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            source_line, candidate_index, result = future.result()
            outcomes.setdefault(source_line, {})[candidate_index] = result
            if done % 100 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] evaluated candidates from {prediction_path.name}")

    pass_at: dict[str, float] = {}
    compile_at: dict[str, float] = {}
    facts_at: dict[str, float] = {}
    per_task: list[dict[str, Any]] = []
    for source_line in range(len(dataset)):
        ordered = [
            outcomes.get(source_line, {}).get(
                index, {"compiled": False, "passed": False, "facts_ok": False}
            )
            for index in range(expected_candidates)
        ]
        passes = [item["passed"] for item in ordered]
        compiles = [item["compiled"] for item in ordered]
        facts = [item["facts_ok"] for item in ordered]
        row_metrics = {"task_key": task_key(dataset[source_line], source_line), "n": len(ordered)}
        for k in k_values:
            row_metrics[f"pass@{k}"] = observed_pass_at_k(passes, k)
            row_metrics[f"compile@{k}"] = observed_pass_at_k(compiles, k)
            row_metrics[f"facts@{k}"] = observed_pass_at_k(facts, k)
        per_task.append(row_metrics)
    for k in k_values:
        pass_at[str(k)] = sum(row[f"pass@{k}"] for row in per_task) / max(1, len(per_task))
        compile_at[str(k)] = sum(row[f"compile@{k}"] for row in per_task) / max(1, len(per_task))
        facts_at[str(k)] = sum(row[f"facts@{k}"] for row in per_task) / max(1, len(per_task))
    return {
        "rows": len(dataset),
        "candidates": len(jobs),
        "expected_candidates_per_task": expected_candidates,
        "coverage_validated": True,
        "pass_at_k": pass_at,
        "compile_at_k": compile_at,
        "facts_at_k": facts_at,
        "per_task": per_task,
        "prediction_file": str(prediction_path.resolve()),
    }


def run_inference(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    output: Path,
    prompt_mode: str,
    graph_ablation: str,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.project_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["GRAPH_PROMPT_ASSEMBLY_MODE"] = prompt_mode
    env["GRAPH_PROMPT_BINARY_EVIDENCE"] = "1" if (prompt_mode == "full" and args.deployment_binary_evidence) else "0"
    env["GRAPH_REQUIRE_NEUTRAL_CONTRACT"] = "1"
    env["GRAPH_NEUTRAL_FUNCTION_NAME"] = "fn0"
    env["GRAPH_LOAD_4BIT"] = str(int(args.load_4bit))
    # The causal gate uses an explicit zero-context ablation. Never inherit a
    # manual prefix-gate override from the calling shell.
    env.pop("GRAPH_QWEN_PREFIX_GATE_OVERRIDE", None)
    command = [
        args.python,
        str(args.project_root / "scripts/evaluation/graph_inference_antigravity.py"),
        "--dataset",
        str(args.dataset),
        "--decoder_model",
        args.decoder_model,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--num_samples",
        str(args.num_samples),
        "--generation_batch_size",
        str(args.generation_batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--decoder_prompt_max_length",
        str(args.decoder_prompt_max_length),
        "--decoder_revision",
        args.decoder_revision,
        "--encoder_revision",
        args.encoder_revision,
        "--seed",
        str(args.seed),
        "--graph_input_ablation",
        graph_ablation,
        "--graph_ablation_seed",
        str(args.seed),
    ]
    if args.limit:
        command += ["--limit", str(args.limit)]
    print("\n" + " ".join(command))
    subprocess.run(command, cwd=args.project_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--project_root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decoder_model", default=os.environ.get("GRAPH_DECODER_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--decoder_revision", default=os.environ.get("GRAPH_DECODER_REVISION", ""))
    parser.add_argument("--encoder_revision", default=os.environ.get("GRAPH_ENCODER_REVISION", ""))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--decoder_prompt_max_length", type=int, default=768)
    parser.add_argument(
        "--deployment_prompt_mode", "--performance_prompt_mode",
        dest="deployment_prompt_mode",
        choices=["full", "graph_only", "none"],
        default="full",
        help="Prompt used for the performance/improvement arm.",
    )
    parser.add_argument(
        "--causality_prompt_mode",
        choices=["graph_only"],
        default="graph_only",
        help="The causal graph-use arm is deliberately graph-only to remove the text oracle.",
    )
    parser.add_argument(
        "--run_deployment_arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the real-prompt performance arm used for checkpoint-improvement gates.",
    )
    parser.add_argument(
        "--run_causality_arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run correct/permuted/null graph-only generations. Disable for text-primary performance gates.",
    )
    parser.add_argument("--deployment_binary_evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_4bit", type=int, choices=[0, 1], default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--min_rows", type=int, default=96,
        help="Minimum held-out tasks required before any statistical gate is evaluated.",
    )
    parser.add_argument(
        "--bootstrap_iterations", type=int, default=10000,
        help="Paired task-bootstrap replicates (deterministic under --bootstrap_seed).",
    )
    parser.add_argument(
        "--bootstrap_seed", type=int, default=-1,
        help="Bootstrap RNG seed; negative reuses --seed.",
    )
    parser.add_argument("--statistical_confidence", type=float, default=0.95)
    parser.add_argument("--max_sign_test_p_value", type=float, default=0.05)
    parser.add_argument("--min_causal_effective_pairs", type=int, default=8)
    parser.add_argument("--min_deployment_effective_pairs", type=int, default=8)
    parser.add_argument("--min_causal_permutation_ci_lower_pp", type=float, default=0.0)
    parser.add_argument("--min_causal_null_ci_lower_pp", type=float, default=0.0)
    parser.add_argument("--min_facts_permutation_ci_lower_pp", type=float, default=0.0)
    parser.add_argument("--min_deployment_ci_lower_pp", type=float, default=0.0)
    parser.add_argument(
        "--require_facts_statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also require sign-test/bootstrap evidence for FACTS@k degradation.",
    )
    parser.add_argument(
        "--min_causal_permutation_drop_pp", "--min_permutation_drop_pp",
        dest="min_causal_permutation_drop_pp", type=float, default=0.0,
        help=("Optional pre-registered practical-effect floor. Default 0 avoids inventing "
              "a noise threshold; set this from an external repeatability study."),
    )
    parser.add_argument("--min_causal_null_drop_pp", type=float, default=0.0)
    parser.add_argument("--min_facts_permutation_drop_pp", type=float, default=0.0)
    parser.add_argument(
        "--min_causal_task_losses", type=int, default=0,
        help=(
            "Deprecated extra floor on correct>permuted tasks. The load-bearing "
            "controls are effective pairs, exact sign p-value, and bootstrap CI."
        ),
    )
    parser.add_argument(
        "--min_deployment_improvement_pp", "--min_improvement_pp",
        dest="min_deployment_improvement_pp", type=float, default=0.0,
        help="Minimum observed deployment effect; CI/sign evidence is required separately.",
    )
    parser.add_argument(
        "--baseline_pass_at_k",
        type=float,
        default=-1.0,
        help="Optional frozen absolute pass@k baseline in [0,1]; used when no baseline checkpoint is supplied.",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num_samples must be positive")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.k > args.num_samples:
        parser.error("--k cannot exceed --num_samples; pass@k would otherwise be silently clamped")
    if args.generation_batch_size <= 0:
        parser.error("--generation_batch_size must be positive")
    if args.workers <= 0 or args.timeout <= 0:
        parser.error("--workers and --timeout must be positive")
    if args.min_rows <= 1:
        parser.error("--min_rows must be greater than 1")
    if args.bootstrap_iterations < 100:
        parser.error("--bootstrap_iterations must be at least 100")
    if not 0.5 < args.statistical_confidence < 1.0:
        parser.error("--statistical_confidence must be in (0.5,1)")
    if not 0.0 < args.max_sign_test_p_value < 1.0:
        parser.error("--max_sign_test_p_value must be in (0,1)")
    if args.min_causal_effective_pairs <= 0 or args.min_deployment_effective_pairs <= 0:
        parser.error("minimum effective-pair counts must be positive")
    if args.min_causal_task_losses < 0:
        parser.error("--min_causal_task_losses must be non-negative")
    if any(
        value < 0.0
        for value in (
            args.min_causal_permutation_drop_pp,
            args.min_causal_null_drop_pp,
            args.min_facts_permutation_drop_pp,
            args.min_deployment_improvement_pp,
            args.min_causal_permutation_ci_lower_pp,
            args.min_causal_null_ci_lower_pp,
            args.min_facts_permutation_ci_lower_pp,
            args.min_deployment_ci_lower_pp,
        )
    ):
        parser.error("all minimum gate thresholds must be non-negative")
    if args.baseline_pass_at_k > 1.0:
        parser.error("--baseline_pass_at_k must be <=1 or negative to disable")

    args.project_root = args.project_root.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.baseline_checkpoint:
        args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
        if not args.baseline_checkpoint.is_file():
            raise SystemExit(f"baseline checkpoint not found: {args.baseline_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("functional gate dataset is empty")
    if len(rows) < args.min_rows:
        raise SystemExit(
            f"functional gate has only {len(rows)} held-out tasks; minimum is {args.min_rows}"
        )
    bootstrap_seed = args.seed if args.bootstrap_seed < 0 else args.bootstrap_seed
    for index, row in enumerate(rows):
        metadata = row.get("hybrid_metadata") or {}
        if metadata.get("evaluation_only") is not True:
            raise SystemExit(f"gate row {index} is not marked evaluation_only")
        if infer_function_name(row) != "fn0":
            raise SystemExit(f"gate row {index} does not use neutral fn0 contract")

    if not args.run_deployment_arm and not args.run_causality_arm:
        raise SystemExit("at least one of --run_deployment_arm or --run_causality_arm is required")

    files: dict[str, Path] = {}
    if args.run_deployment_arm:
        files["deployment"] = args.output_dir / "deployment_current.json"
        run_inference(
            args,
            checkpoint=args.checkpoint,
            output=files["deployment"],
            prompt_mode=args.deployment_prompt_mode,
            graph_ablation="none",
        )
    if args.run_causality_arm:
        files.update(
            {
                "causal_correct": args.output_dir / "causal_correct.json",
                "causal_permuted": args.output_dir / "causal_matched_permutation.json",
                "causal_null": args.output_dir / "causal_null.json",
            }
        )
        run_inference(
            args, checkpoint=args.checkpoint, output=files["causal_correct"],
            prompt_mode=args.causality_prompt_mode, graph_ablation="none",
        )
        run_inference(
            args, checkpoint=args.checkpoint, output=files["causal_permuted"],
            prompt_mode=args.causality_prompt_mode, graph_ablation="matched_permutation",
        )
        run_inference(
            args,
            checkpoint=args.checkpoint,
            output=files["causal_null"],
            prompt_mode=args.causality_prompt_mode,
            graph_ablation="null",
        )
    if args.baseline_checkpoint and not args.run_deployment_arm:
        raise SystemExit("--baseline_checkpoint requires --run_deployment_arm")
    if args.baseline_pass_at_k >= 0.0 and not args.run_deployment_arm:
        raise SystemExit("--baseline_pass_at_k requires --run_deployment_arm")
    if args.baseline_checkpoint:
        files["deployment_baseline"] = args.output_dir / "deployment_baseline.json"
        run_inference(
            args,
            checkpoint=args.baseline_checkpoint,
            output=files["deployment_baseline"],
            prompt_mode=args.deployment_prompt_mode,
            graph_ablation="none",
        )

    permutation_provenance = None
    causal_provenance: dict[str, dict[str, Any]] = {}
    if args.run_causality_arm:
        for name in ("causal_correct", "causal_permuted", "causal_null"):
            provenance_path = Path(str(files[name]) + ".provenance.json")
            if not provenance_path.is_file():
                raise RuntimeError(f"{name} inference did not emit provenance")
            causal_provenance[name] = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )

        for name, provenance in causal_provenance.items():
            load_contract = provenance.get("checkpoint_load") or {}
            if load_contract.get("status") != "passed":
                raise RuntimeError(
                    f"{name} did not validate its checkpoint architecture: "
                    + json.dumps(load_contract, sort_keys=True)
                )
        checkpoint_hashes = {
            name: str((value.get("checkpoint") or {}).get("sha256") or "")
            for name, value in causal_provenance.items()
        }
        if any(not value for value in checkpoint_hashes.values()) or len(set(checkpoint_hashes.values())) != 1:
            raise RuntimeError(
                "causal arms did not use the identical checkpoint: "
                + json.dumps(checkpoint_hashes, sort_keys=True)
            )

        null_ablation = causal_provenance["causal_null"].get("graph_input_ablation") or {}
        if null_ablation.get("mode") != "null" or null_ablation.get("final_context_zeroed") is not True:
            raise RuntimeError(
                "causal null arm did not zero the final graph context: "
                + json.dumps(null_ablation, sort_keys=True)
            )
        for name in ("causal_correct", "causal_permuted", "causal_null"):
            gate = causal_provenance[name].get("graph_prefix_gate") or {}
            if gate.get("override_requested") is not None:
                raise RuntimeError(f"{name} unexpectedly used a prefix-gate override")

        permutation_provenance = causal_provenance["causal_permuted"]
        ablation_meta = permutation_provenance.get("graph_input_ablation") or {}
        if ablation_meta.get("mode") != "matched_permutation":
            raise RuntimeError("causal permutation provenance has the wrong ablation mode")
        if int(ablation_meta.get("self_mapped_rows", -1)) != 0:
            raise RuntimeError("matched permutation contains self-mapped rows")

        # The causal comparison is valid only when target-side text is byte-for-
        # byte identical. The inference sidecar hashes every rendered prompt; fail
        # closed if a graph donor accidentally changes the text prompt or row set.
        prompt_hashes = {
            name: str(value.get("prompt_stream_sha256") or "")
            for name, value in causal_provenance.items()
        }
        if any(not value for value in prompt_hashes.values()):
            raise RuntimeError("causal inference provenance lacks prompt_stream_sha256")
        if len(set(prompt_hashes.values())) != 1:
            raise RuntimeError(
                "causal correct/permuted/null arms rendered different text prompts: "
                + json.dumps(prompt_hashes, sort_keys=True)
            )
        row_counts = {
            name: int(value.get("row_count", -1))
            for name, value in causal_provenance.items()
        }
        if len(set(row_counts.values())) != 1 or next(iter(row_counts.values())) != len(rows):
            raise RuntimeError(
                "causal correct/permuted/null arms used different row sets: "
                + json.dumps(row_counts, sort_keys=True)
            )
        generation_fingerprints = {
            name: json.dumps(value.get("generation") or {}, sort_keys=True)
            for name, value in causal_provenance.items()
        }
        if len(set(generation_fingerprints.values())) != 1:
            raise RuntimeError("causal arms used different generation settings")

    deployment_provenance: dict[str, dict[str, Any]] = {}
    if args.run_deployment_arm:
        deployment_names = [
            name for name in ("deployment", "deployment_baseline") if name in files
        ]
        for name in deployment_names:
            provenance_path = Path(str(files[name]) + ".provenance.json")
            if not provenance_path.is_file():
                raise RuntimeError(f"{name} inference did not emit provenance")
            deployment_provenance[name] = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            load_contract = deployment_provenance[name].get("checkpoint_load") or {}
            if load_contract.get("status") != "passed":
                raise RuntimeError(
                    f"{name} did not validate its checkpoint architecture: "
                    + json.dumps(load_contract, sort_keys=True)
                )
        if "deployment_baseline" in deployment_provenance:
            current = deployment_provenance["deployment"]
            baseline = deployment_provenance["deployment_baseline"]
            for field in ("prompt_stream_sha256", "row_count"):
                if current.get(field) != baseline.get(field):
                    raise RuntimeError(
                        f"deployment current/baseline mismatch for {field}: "
                        f"{current.get(field)!r} != {baseline.get(field)!r}"
                    )
            if (current.get("generation") or {}) != (baseline.get("generation") or {}):
                raise RuntimeError(
                    "deployment current/baseline used different generation settings"
                )

    metrics = {
        name: evaluate_predictions(
            rows,
            path,
            k_values=[args.k],
            workers=args.workers,
            timeout=args.timeout,
            expected_candidates=args.num_samples,
        )
        for name, path in files.items()
    }
    k = str(args.k)
    deployment = (
        100.0 * metrics["deployment"]["pass_at_k"][k]
        if "deployment" in metrics
        else None
    )
    causal_correct = causal_permuted = causal_null = None
    facts_correct = facts_permuted = None
    permutation_drop = null_drop = facts_drop = None
    if args.run_causality_arm:
        causal_correct = 100.0 * metrics["causal_correct"]["pass_at_k"][k]
        causal_permuted = 100.0 * metrics["causal_permuted"]["pass_at_k"][k]
        causal_null = 100.0 * metrics["causal_null"]["pass_at_k"][k]
        facts_correct = 100.0 * metrics["causal_correct"]["facts_at_k"][k]
        facts_permuted = 100.0 * metrics["causal_permuted"]["facts_at_k"][k]
        permutation_drop = causal_correct - causal_permuted
        null_drop = causal_correct - causal_null
        facts_drop = facts_correct - facts_permuted

    statistical_tests: dict[str, Any] = {}
    paired_causality = None
    if args.run_causality_arm:
        correct_rows = metrics["causal_correct"]["per_task"]
        permuted_rows = metrics["causal_permuted"]["per_task"]
        null_rows = metrics["causal_null"]["per_task"]
        permutation_comparison = paired_task_comparison(
            correct_rows, permuted_rows, metric=f"pass@{args.k}",
            iterations=args.bootstrap_iterations, confidence=args.statistical_confidence,
            seed=bootstrap_seed,
        )
        null_comparison = paired_task_comparison(
            correct_rows, null_rows, metric=f"pass@{args.k}",
            iterations=args.bootstrap_iterations, confidence=args.statistical_confidence,
            seed=bootstrap_seed + 1,
        )
        facts_comparison = paired_task_comparison(
            correct_rows, permuted_rows, metric=f"facts@{args.k}",
            iterations=args.bootstrap_iterations, confidence=args.statistical_confidence,
            seed=bootstrap_seed + 2,
        )
        statistical_tests.update({
            "causal_correct_vs_matched_permutation": permutation_comparison,
            "causal_correct_vs_null": null_comparison,
            "facts_correct_vs_matched_permutation": facts_comparison,
        })
        correct_by_key = {row["task_key"]: row for row in correct_rows}
        permuted_by_key = {row["task_key"]: row for row in permuted_rows}
        correct_only = sum(
            float(correct_by_key[key][f"pass@{args.k}"]) > 0.0
            and float(permuted_by_key[key][f"pass@{args.k}"]) == 0.0
            for key in correct_by_key
        )
        paired_causality = {
            "correct_above_permuted_tasks": permutation_comparison["left_above_right_tasks"],
            "equal_tasks": permutation_comparison["equal_tasks"],
            "permuted_above_correct_tasks": permutation_comparison["right_above_left_tasks"],
            "effective_non_ties": permutation_comparison["effective_non_ties"],
            "exact_one_sided_sign_p_value": permutation_comparison["exact_one_sided_sign_p_value"],
            "correct_nonzero_permuted_zero_tasks": int(correct_only),
            "rows": permutation_comparison["rows"],
        }

    improvement = None
    baseline_source = None
    deployment_comparison = None
    if "deployment_baseline" in metrics:
        assert deployment is not None
        deployment_comparison = paired_task_comparison(
            metrics["deployment"]["per_task"], metrics["deployment_baseline"]["per_task"],
            metric=f"pass@{args.k}", iterations=args.bootstrap_iterations,
            confidence=args.statistical_confidence, seed=bootstrap_seed + 3,
        )
        improvement = float(deployment_comparison["bootstrap"]["point_estimate_pp"])
        baseline_source = "checkpoint"
        statistical_tests["deployment_current_vs_baseline"] = deployment_comparison
    elif args.baseline_pass_at_k >= 0.0:
        if not 0.0 <= args.baseline_pass_at_k <= 1.0:
            raise SystemExit("--baseline_pass_at_k must be in [0,1] or negative to disable")
        assert deployment is not None
        deployment_comparison = fixed_aggregate_baseline_comparison(
            metrics["deployment"]["per_task"], metric=f"pass@{args.k}",
            baseline=args.baseline_pass_at_k, iterations=args.bootstrap_iterations,
            confidence=args.statistical_confidence, seed=bootstrap_seed + 3,
        )
        improvement = float(deployment_comparison["bootstrap"]["point_estimate_pp"])
        baseline_source = "absolute"
        statistical_tests["deployment_current_vs_fixed_aggregate"] = deployment_comparison

    failures: list[str] = []
    if args.run_causality_arm:
        assert permutation_drop is not None and null_drop is not None and facts_drop is not None
        if permutation_drop < args.min_causal_permutation_drop_pp:
            failures.append(
                f"causal permutation drop {permutation_drop:.3f} pp < "
                f"{args.min_causal_permutation_drop_pp:.3f} pp practical-effect floor"
            )
        if null_drop < args.min_causal_null_drop_pp:
            failures.append(
                f"causal null drop {null_drop:.3f} pp < {args.min_causal_null_drop_pp:.3f} pp"
            )
        if facts_drop < args.min_facts_permutation_drop_pp:
            failures.append(
                f"FACTS permutation drop {facts_drop:.3f} pp < {args.min_facts_permutation_drop_pp:.3f} pp"
            )
        failures.extend(_paired_statistical_failures(
            "causal correct-vs-permuted pass@k",
            statistical_tests["causal_correct_vs_matched_permutation"],
            minimum_effective_pairs=args.min_causal_effective_pairs,
            maximum_p_value=args.max_sign_test_p_value,
            minimum_lower_bound_pp=args.min_causal_permutation_ci_lower_pp,
        ))
        failures.extend(_paired_statistical_failures(
            "causal correct-vs-null pass@k",
            statistical_tests["causal_correct_vs_null"],
            minimum_effective_pairs=args.min_causal_effective_pairs,
            maximum_p_value=args.max_sign_test_p_value,
            minimum_lower_bound_pp=args.min_causal_null_ci_lower_pp,
        ))
        if args.require_facts_statistics:
            failures.extend(_paired_statistical_failures(
                "FACTS correct-vs-permuted",
                statistical_tests["facts_correct_vs_matched_permutation"],
                minimum_effective_pairs=args.min_causal_effective_pairs,
                maximum_p_value=args.max_sign_test_p_value,
                minimum_lower_bound_pp=args.min_facts_permutation_ci_lower_pp,
            ))
        assert paired_causality is not None
        if (args.min_causal_task_losses > 0 and
                paired_causality["correct_above_permuted_tasks"] < args.min_causal_task_losses):
            failures.append(
                f"only {paired_causality['correct_above_permuted_tasks']} tasks lost pass@k "
                f"under matched graph permutation; deprecated extra minimum is "
                f"{args.min_causal_task_losses}"
            )

    if improvement is not None:
        if improvement < args.min_deployment_improvement_pp:
            failures.append(
                f"deployment improvement {improvement:.3f} pp < "
                f"{args.min_deployment_improvement_pp:.3f} pp practical-effect floor"
            )
        assert deployment_comparison is not None
        if baseline_source == "checkpoint":
            failures.extend(_paired_statistical_failures(
                "deployment current-vs-matched-baseline pass@k", deployment_comparison,
                minimum_effective_pairs=args.min_deployment_effective_pairs,
                maximum_p_value=args.max_sign_test_p_value,
                minimum_lower_bound_pp=args.min_deployment_ci_lower_pp,
            ))
        else:
            lower = float(deployment_comparison["bootstrap"]["one_sided_lower_pp"])
            if lower <= args.min_deployment_ci_lower_pp:
                failures.append(
                    f"deployment-vs-fixed-baseline bootstrap lower bound {lower:.3f} pp "
                    f"<= {args.min_deployment_ci_lower_pp:.3f} pp; baseline uncertainty is unavailable"
                )

    report = {
        "schema_version": 3,
        "status": "passed" if not failures else "failed",
        "dataset": str(args.dataset),
        "held_out_rows": len(rows),
        "checkpoint": str(args.checkpoint),
        "baseline_checkpoint": str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        "baseline_pass_at_k": args.baseline_pass_at_k if args.baseline_pass_at_k >= 0.0 else None,
        "baseline_source": baseline_source,
        "k": args.k,
        "deployment_prompt_mode": args.deployment_prompt_mode if args.run_deployment_arm else None,
        "deployment_arm_ran": args.run_deployment_arm,
        "causality_arm_ran": args.run_causality_arm,
        "causal_prompt_mode": args.causality_prompt_mode if args.run_causality_arm else None,
        "matched_permutation_provenance": permutation_provenance,
        "deployment_provenance": deployment_provenance,
        "paired_causality": paired_causality,
        "causal_prompt_stream_sha256": (
            causal_provenance.get("causal_correct", {}).get("prompt_stream_sha256")
            if args.run_causality_arm else None
        ),
        "metrics": metrics,
        "statistical_tests": statistical_tests,
        "statistical_configuration": {
            "task_is_resampling_unit": True,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "confidence": args.statistical_confidence,
            "one_sided_sign_test_alpha": args.max_sign_test_p_value,
            "minimum_causal_effective_pairs": args.min_causal_effective_pairs,
            "minimum_deployment_effective_pairs": args.min_deployment_effective_pairs,
            "facts_statistics_required": bool(args.require_facts_statistics),
            "note": (
                "The task bootstrap addresses finite held-out-task uncertainty. "
                "Multiple generation seeds are still recommended for final estimates."
            ),
        },
        "gates_percentage_points": {
            "causal_permutation_drop": permutation_drop,
            "causal_null_drop": null_drop,
            "facts_permutation_drop": facts_drop,
            "deployment_improvement": improvement,
        },
        "thresholds_percentage_points": {
            "causal_permutation_observed_drop": args.min_causal_permutation_drop_pp,
            "causal_permutation_bootstrap_lower": args.min_causal_permutation_ci_lower_pp,
            "causal_null_observed_drop": args.min_causal_null_drop_pp,
            "causal_null_bootstrap_lower": args.min_causal_null_ci_lower_pp,
            "facts_permutation_observed_drop": args.min_facts_permutation_drop_pp,
            "facts_permutation_bootstrap_lower": args.min_facts_permutation_ci_lower_pp,
            "deprecated_causal_task_losses": args.min_causal_task_losses,
            "deployment_observed_improvement": (
                args.min_deployment_improvement_pp if improvement is not None else None
            ),
            "deployment_bootstrap_lower": (
                args.min_deployment_ci_lower_pp if improvement is not None else None
            ),
        },
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
