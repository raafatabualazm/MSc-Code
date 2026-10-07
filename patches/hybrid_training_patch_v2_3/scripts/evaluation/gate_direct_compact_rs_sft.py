#!/usr/bin/env python3
"""Report paired held-out RS-SFT results without steering later stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rs_report", required=True)
    parser.add_argument("--control_report", required=True)
    parser.add_argument("--rs_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--build_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_improvement_pp", type=float, default=6.0)
    parser.add_argument("--max_one_sided_pvalue", type=float, default=0.05)
    parser.add_argument(
        "--report_only",
        action="store_true",
        help=(
            "Persist the predeclared comparison and return success regardless "
            "of performance. Required when a later stage must not be selected "
            "or stopped using held-out outcomes."
        ),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binomial_upper_tail(successes: int, trials: int) -> float:
    """P[X >= successes] for X~Binomial(trials, .5), exact."""
    if trials <= 0:
        return 1.0
    return math.fsum(
        math.comb(trials, value) for value in range(successes, trials + 1)
    ) / (2**trials)


def task_map(report: Mapping[str, Any]) -> dict[str, bool]:
    rows = report.get("task_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("score report has no task_results")
    result: dict[str, bool] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError(f"score report has missing/duplicate task {task_id!r}")
        result[task_id] = bool(row.get("pass_at_k"))
    return result


def main() -> None:
    args = parse_args()
    if args.min_improvement_pp < 0:
        raise ValueError("min improvement must be non-negative")
    if not 0 < args.max_one_sided_pvalue <= 1:
        raise ValueError("max p-value must lie in (0, 1]")

    paths = {
        "rs_report": Path(args.rs_report).expanduser().resolve(),
        "control_report": Path(args.control_report).expanduser().resolve(),
        "rs_checkpoint": Path(args.rs_checkpoint).expanduser().resolve(),
        "control_checkpoint": Path(args.control_checkpoint).expanduser().resolve(),
        "build_report": Path(args.build_report).expanduser().resolve(),
        "output": Path(args.output).expanduser().resolve(),
    }
    for key, path in paths.items():
        if key == "output":
            continue
        if key.endswith("checkpoint"):
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    if paths["output"].exists():
        raise ValueError(f"refusing to overwrite gate report: {paths['output']}")
    if paths["rs_checkpoint"] == paths["control_checkpoint"]:
        raise ValueError("RS-SFT and control checkpoint paths are identical")

    rs = load(paths["rs_report"])
    control = load(paths["control_report"])
    build = load(paths["build_report"])
    rs_prov = load(paths["rs_checkpoint"] / "run_provenance.json")
    control_prov = load(paths["control_checkpoint"] / "run_provenance.json")
    if rs.get("schema") != "direct-compact-attested-passk-v1":
        raise ValueError("RS report has wrong schema")
    if control.get("schema") != "direct-compact-attested-passk-v1":
        raise ValueError("control report has wrong schema")
    if build.get("schema") != "direct-compact-rs-sft-matched-build-v1":
        raise ValueError("matched dataset build report has wrong schema")
    if not build.get("arms", {}).get("source_sequence_exactly_matched"):
        raise ValueError("dataset builder did not attest matched compact inputs")
    if build.get("low_coverage_smoke_override"):
        raise ValueError("a low-coverage smoke dataset cannot pass a production gate")
    if int(build.get("unique_recertified_tasks", 0)) < int(
        build.get("unique_repair_floor", 1)
    ):
        raise ValueError("repair corpus is below its sealed production floor")

    for field in ("k", "tasks", "timeout", "stability_runs"):
        if rs.get(field) != control.get(field):
            raise ValueError(f"score reports differ on {field}")
    if rs.get("evaluation", {}).get("sha256") != control.get(
        "evaluation", {}
    ).get("sha256"):
        raise ValueError("score reports use different evaluation datasets")

    rs_prediction_prov = load(
        Path(str(rs["predictions"]["path"]) + ".provenance.json")
    )
    control_prediction_prov = load(
        Path(str(control["predictions"]["path"]) + ".provenance.json")
    )
    generation_fields = (
        "dataset_sha256",
        "alignment_sha256",
        "selected_role",
        "contract_sha256",
        "decoder_model",
        "decoder_revision",
        "model_config_sha256",
        "attn_implementation",
        "num_rows",
        "num_samples",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "batch_size",
        "limit",
        "seed",
    )
    mismatched_generation = [
        field
        for field in generation_fields
        if rs_prediction_prov.get(field) != control_prediction_prov.get(field)
    ]
    if mismatched_generation:
        raise ValueError(
            "inference settings differ: " + ", ".join(mismatched_generation)
        )
    if (
        rs_prediction_prov.get("decoder_adapter_sha256")
        == control_prediction_prov.get("decoder_adapter_sha256")
    ):
        raise ValueError("inference reports resolve to the same decoder adapter")

    for arm, score, prediction_prov, checkpoint_prov in (
        ("rs", rs, rs_prediction_prov, rs_prov),
        ("control", control, control_prediction_prov, control_prov),
    ):
        prediction_path = Path(score["predictions"]["path"]).expanduser().resolve()
        prediction_provenance_path = Path(
            str(prediction_path) + ".provenance.json"
        )
        if sha256_file(prediction_path) != score["predictions"].get("sha256"):
            raise ValueError(f"{arm} score report prediction hash is stale")
        if sha256_file(prediction_provenance_path) != score["predictions"].get(
            "provenance_sha256"
        ):
            raise ValueError(f"{arm} score report provenance hash is stale")
        if prediction_prov.get("output_sha256") != score["predictions"].get(
            "sha256"
        ):
            raise ValueError(f"{arm} inference provenance is not bound to predictions")
        if prediction_prov.get(
            "decoder_adapter_sha256"
        ) != checkpoint_prov.get("decoder_adapter_sha256"):
            raise ValueError(f"{arm} inference used a different decoder adapter")
        if prediction_prov.get(
            "source_overlay_sha256"
        ) != checkpoint_prov.get("source_overlay_sha256"):
            raise ValueError(f"{arm} inference used a different compact overlay")

    provenance_equal_fields = (
        "architecture",
        "decoder_model",
        "decoder_revision",
        "model_config_sha256",
        "contract_sha256",
        "codebook_sha256",
        "codec_sha256",
        "training_schedule",
    )
    mismatched_training = [
        field
        for field in provenance_equal_fields
        if rs_prov.get(field) != control_prov.get(field)
    ]
    if mismatched_training:
        raise ValueError(
            "training arms differ outside the target dataset: "
            + ", ".join(mismatched_training)
        )
    if rs_prov.get("warmstart_checkpoint") != control_prov.get(
        "warmstart_checkpoint"
    ):
        raise ValueError("training arms do not share one exact warm-start checkpoint")
    if not rs_prov.get("warmstart_checkpoint"):
        raise ValueError("matched RS-SFT arms were not warm-started")

    expected_rs_train = build["outputs"]["intervention"]["sha256"]
    expected_control_train = build["outputs"]["control"]["sha256"]
    if rs_prov.get("train_file_sha256") != expected_rs_train:
        raise ValueError("RS checkpoint is not bound to the matched intervention")
    if control_prov.get("train_file_sha256") != expected_control_train:
        raise ValueError("control checkpoint is not bound to matched gold data")

    rs_tasks = task_map(rs)
    control_tasks = task_map(control)
    if set(rs_tasks) != set(control_tasks):
        raise ValueError("paired score reports have different task IDs")
    new_wins = sorted(
        task_id
        for task_id in rs_tasks
        if rs_tasks[task_id] and not control_tasks[task_id]
    )
    regressions = sorted(
        task_id
        for task_id in rs_tasks
        if control_tasks[task_id] and not rs_tasks[task_id]
    )
    discordant = len(new_wins) + len(regressions)
    pvalue = binomial_upper_tail(len(new_wins), discordant)
    tasks = len(rs_tasks)
    rs_passes = sum(rs_tasks.values())
    control_passes = sum(control_tasks.values())
    improvement_pp = 100.0 * (rs_passes - control_passes) / tasks
    passed = (
        improvement_pp >= args.min_improvement_pp
        and pvalue <= args.max_one_sided_pvalue
    )

    report = {
        "schema": "direct-compact-rs-sft-functional-gate-v1",
        "passed": passed,
        "integrity_passed": True,
        "performance_passed": passed,
        "report_only": bool(args.report_only),
        "used_for_stage_selection_or_launch": False,
        "tasks": tasks,
        "k": rs["k"],
        "rs_passes": rs_passes,
        "control_passes": control_passes,
        "improvement_pp": improvement_pp,
        "minimum_improvement_pp": args.min_improvement_pp,
        "new_wins": len(new_wins),
        "regressions": len(regressions),
        "new_win_task_ids": new_wins,
        "regression_task_ids": regressions,
        "mcnemar_exact_one_sided_pvalue": pvalue,
        "maximum_one_sided_pvalue": args.max_one_sided_pvalue,
        "matched": {
            "warmstart": rs_prov["warmstart_checkpoint"],
            "training_schedule": rs_prov["training_schedule"],
            "evaluation_sha256": rs["evaluation"]["sha256"],
            "generation": {
                field: rs_prediction_prov.get(field)
                for field in generation_fields
            },
        },
        "artifacts": {
            key: {
                "path": str(path),
                "sha256": (
                    sha256_file(path)
                    if path.is_file()
                    else sha256_file(path / "run_provenance.json")
                ),
            }
            for key, path in paths.items()
            if key != "output"
        },
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
