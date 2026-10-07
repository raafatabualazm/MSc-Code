#!/usr/bin/env python3
"""Frozen representation probe for local, graph, projection, and prefix stages.

The probe runs before another decoder-training job. It asks whether mechanically
recoverable binary facts survive each representation stage, and whether the final
prefix remains task-specific after the graph-to-decoder projection.  Closed-form
ridge probes are evaluated on held-out rows against a permuted-label control.
Projected-graph to prefix retrieval uses a train-fitted ridge map rather than raw
cosine between two differently transformed spaces.

This is a representation-availability probe, not evidence of causal free-running
use. The authoritative use gate is ``functional_graph_gate_antigravity.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Model-dependent imports are intentionally deferred until after CLI environment
# variables are installed in main(). Importing the training module earlier would
# silently construct a model with stale GRAPH_* defaults.
from scripts.training.hybrid_data_controls import mechanical_facts, read_jsonl  # noqa: E402
from scripts.training.checkpoint_contract import validate_trainable_checkpoint_load  # noqa: E402

COUNT_TARGETS = (
    "log_instruction_count",
    "log_block_count",
    "log_edge_count",
    "log_call_count",
    "log_branch_count",
    "log_conditional_branches",
    "log_return_count",
    "log_comparisons",
    "log_memory_ops",
    "arity",
)
ARCHITECTURE_BUCKETS = ("arm64", "x86_64", "unknown")
RETURN_BUCKETS = (
    "void",
    "bool",
    "int",
    "float",
    "string",
    "collection",
    "future",
    "dynamic",
    "other",
)
HASH_BUCKETS = {
    "numeric_constant": 16,
    "string_literal": 8,
    "direct_callee": 8,
}


def masked_mean(states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if states.ndim == 2:
        states = states.unsqueeze(0)
    if mask is None:
        return states.mean(dim=1)
    weights = mask.to(states.dtype).unsqueeze(-1)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Dual ridge prediction with train-only normalisation."""
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True) + 1e-6
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = train_y.std(axis=0, keepdims=True) + 1e-6
    x = (train_x - x_mean) / x_std
    xt = (test_x - x_mean) / x_std
    y = (train_y - y_mean) / y_std
    kernel = x @ x.T
    regularised = kernel + alpha * np.eye(kernel.shape[0], dtype=np.float64)
    try:
        dual = np.linalg.solve(regularised, y)
    except np.linalg.LinAlgError:
        dual = np.linalg.pinv(regularised) @ y
    weights = x.T @ dual
    prediction = xt @ weights
    return prediction * y_std + y_mean


def r2_scores(target: np.ndarray, prediction: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    for column in range(target.shape[1]):
        actual = target[:, column]
        pred = prediction[:, column]
        denom = float(((actual - actual.mean()) ** 2).sum())
        if denom <= 1e-12:
            values.append(None)
        else:
            values.append(1.0 - float(((actual - pred) ** 2).sum()) / denom)
    return values


def retrieval_accuracy(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or not len(left):
        return 0.0
    left = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-8)
    right = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-8)
    similarity = left @ right.T
    return float((similarity.argmax(axis=1) == np.arange(len(left))).mean())


def _stable_bucket(value: Any, buckets: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _return_bucket(return_type: Any) -> str:
    value = str(return_type or "dynamic").strip().lower().replace("?", "")
    if value == "void":
        return "void"
    if value == "bool":
        return "bool"
    if value == "int":
        return "int"
    if value in {"double", "num", "float"}:
        return "float"
    if "string" in value:
        return "string"
    if any(fragment in value for fragment in ("list", "map", "set", "iterable", "record")):
        return "collection"
    if "future" in value:
        return "future"
    if value in {"dynamic", "object", "object?", ""}:
        return "dynamic"
    return "other"


def target_schema() -> tuple[list[str], dict[str, list[int]]]:
    names = list(COUNT_TARGETS)
    groups: dict[str, list[int]] = {"counts_and_contract": list(range(len(names)))}

    start = len(names)
    names.extend(f"architecture:{value}" for value in ARCHITECTURE_BUCKETS)
    groups["architecture"] = list(range(start, len(names)))

    start = len(names)
    names.extend(f"return:{value}" for value in RETURN_BUCKETS)
    groups["return_type"] = list(range(start, len(names)))

    for group, buckets in HASH_BUCKETS.items():
        start = len(names)
        names.extend(f"{group}:hash_{index}" for index in range(buckets))
        groups[group] = list(range(start, len(names)))
    return names, groups


def fact_vector(facts: dict[str, Any]) -> list[float]:
    def log_count(name: str) -> float:
        value = facts.get(name)
        try:
            return math.log1p(max(0.0, float(value or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    values = [
        log_count("instruction_count"),
        log_count("block_count"),
        log_count("edge_count"),
        log_count("call_count"),
        log_count("branch_count"),
        log_count("conditional_branches"),
        log_count("return_count"),
        log_count("comparisons"),
        log_count("memory_ops"),
        float(facts.get("arity")) if facts.get("arity") is not None else -1.0,
    ]

    architecture = str(facts.get("architecture") or "unknown").lower()
    architecture = architecture if architecture in ARCHITECTURE_BUCKETS else "unknown"
    values.extend(float(architecture == item) for item in ARCHITECTURE_BUCKETS)

    return_bucket = _return_bucket(facts.get("return_type"))
    values.extend(float(return_bucket == item) for item in RETURN_BUCKETS)

    hashed_sources: dict[str, Iterable[Any]] = {
        "numeric_constant": facts.get("salient_numeric_constants") or [],
        "string_literal": facts.get("string_literals") or [],
        "direct_callee": facts.get("direct_callees") or [],
    }
    for group, buckets in HASH_BUCKETS.items():
        bag = [0.0] * buckets
        for item in hashed_sources[group]:
            bag[_stable_bucket(item, buckets)] = 1.0
        values.extend(bag)
    return values


def _active_columns(train_y: np.ndarray, test_y: np.ndarray) -> list[int]:
    return [
        index
        for index in range(train_y.shape[1])
        if float(np.var(train_y[:, index])) > 1e-12
        and float(np.var(test_y[:, index])) > 1e-12
    ]


def _mean_scores(scores: list[float | None], indices: Iterable[int]) -> float | None:
    values = [scores[index] for index in indices if scores[index] is not None]
    return float(np.mean(values)) if values else None


def _load_checkpoint_state(path: Path, device: str) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = loaded.get(key)
            if isinstance(nested, dict) and nested and all(isinstance(k, str) for k in nested):
                loaded = nested
                break
    if not isinstance(loaded, dict):
        raise ValueError(f"unsupported checkpoint payload: {type(loaded)}")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decoder_model", default=os.environ.get("GRAPH_DECODER_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--encoder_model", default=os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base"))
    parser.add_argument("--decoder_revision", default=os.environ.get("GRAPH_DECODER_REVISION", ""))
    parser.add_argument("--encoder_revision", default=os.environ.get("GRAPH_ENCODER_REVISION", ""))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=256)
    parser.add_argument("--min_rows", type=int, default=32)
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--min_prefix_semantic_r2", type=float, default=-1.0)
    parser.add_argument("--min_prefix_mean_lift", type=float, default=0.02)
    parser.add_argument("--min_retrieval_above_chance", type=float, default=0.0)
    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test_fraction must be in (0,1)")
    if args.ridge_alpha <= 0.0:
        parser.error("--ridge_alpha must be positive")
    args.dataset = args.dataset.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    rows = read_jsonl(args.dataset)
    rows = [
        row
        for row in rows
        if (row.get("hybrid_metadata") or {}).get("phase0_approved") is True
    ]
    random.Random(args.seed).shuffle(rows)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if len(rows) < args.min_rows:
        raise SystemExit(f"probe rows={len(rows)} < {args.min_rows}")

    os.environ["GRAPH_DECODER_MODEL"] = args.decoder_model
    os.environ["GRAPH_ENCODER_MODEL"] = args.encoder_model
    os.environ["GRAPH_DECODER_REVISION"] = args.decoder_revision
    os.environ["GRAPH_ENCODER_REVISION"] = args.encoder_revision

    # Import optional heavy dependencies only after argument parsing and after
    # the requested architecture is installed in the process environment.
    # This keeps ``--help`` usable on CPU/preflight hosts without transformers,
    # and prevents project modules from capturing stale GRAPH_* defaults.
    from transformers import AutoTokenizer, set_seed

    # Several project modules read GRAPH_* at import time.
    from scripts.evaluation.graph_inference_antigravity import build_blocks
    from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
    from scripts.data.dfg_extractor import LightweightDFGExtractor
    from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
        GraphCodeBERTT5Seq2Seq,
        maybe_override_qwen_prefix_gate,
    )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_model,
        revision=args.encoder_revision or None,
        trust_remote_code=True,
    )
    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=512)
    dfg_extractor = LightweightDFGExtractor()
    model = GraphCodeBERTT5Seq2Seq().to(device)
    state = _load_checkpoint_state(args.checkpoint, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    checkpoint_load_report = validate_trainable_checkpoint_load(
        model,
        state,
        missing_keys=missing,
        unexpected_keys=unexpected,
        context="representation-probe checkpoint",
    )
    model.eval()
    maybe_override_qwen_prefix_gate(model)
    print(
        "Loaded probe checkpoint under a validated architecture contract: "
        f"recognised={checkpoint_load_report['recognised_checkpoint_tensor_count']} "
        f"missing_frozen={checkpoint_load_report['missing_frozen_tensor_count']} "
        f"unexpected={checkpoint_load_report['unexpected_tensor_count']}"
    )

    representations: dict[str, list[np.ndarray]] = {
        "local": [],
        "post_gnn": [],
        "projected_graph": [],
        "final_prefix": [],
    }
    labels: list[list[float]] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            block_tensors, graph_data = build_blocks(row, tensor_builder, dfg_extractor)
            block_vectors: list[torch.Tensor] = []
            for block in block_tensors:
                output = model.local_encoder(
                    input_ids=block["input_ids"].to(device),
                    attention_mask=block["attention_mask"].to(device),
                    position_ids=block["position_ids"].to(device),
                    token_type_ids=block["token_type_ids"].to(device),
                )
                block_vectors.append(output.squeeze(0))
            if not block_vectors:
                raise RuntimeError(f"probe row {index} produced no block vectors")
            # cls pooling -> [blocks, hidden]; multi-query -> [blocks, K, hidden].
            node_states = torch.stack(block_vectors)
            local_summary = node_states.reshape(-1, node_states.shape[-1]).mean(
                dim=0, keepdim=True
            )
            edge_index = getattr(graph_data, "edge_index", None)
            edge_attr = getattr(graph_data, "edge_attr", None)
            region_id = getattr(graph_data, "region_id", None)
            post_gnn, graph_mask = model.graph_encoder(
                node_states,
                edge_index=edge_index.to(device) if edge_index is not None else None,
                edge_attr=edge_attr.to(device) if edge_attr is not None else None,
                list_of_B_i=None,
                region_ids=region_id.to(device) if region_id is not None else None,
            )
            post_summary = masked_mean(post_gnn, graph_mask)
            projected = model.projection(post_gnn)
            projected_summary = masked_mean(projected, graph_mask)
            prefix, prefix_mask = model.prepare_decoder_context(projected, graph_mask)
            prefix_summary = masked_mean(prefix, prefix_mask)

            representations["local"].append(
                local_summary.squeeze(0).float().cpu().numpy()
            )
            representations["post_gnn"].append(
                post_summary.squeeze(0).float().cpu().numpy()
            )
            representations["projected_graph"].append(
                projected_summary.squeeze(0).float().cpu().numpy()
            )
            representations["final_prefix"].append(
                prefix_summary.squeeze(0).float().cpu().numpy()
            )
            facts = row.get("binary_facts") or mechanical_facts(row)
            labels.append(fact_vector(facts))
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"[{index + 1}/{len(rows)}] extracted frozen representations")

    target_names, target_groups = target_schema()
    y = np.asarray(labels, dtype=np.float64)
    if y.shape[1] != len(target_names):
        raise RuntimeError(
            f"target schema mismatch: vector={y.shape[1]} names={len(target_names)}"
        )
    n = len(rows)
    indices = list(range(n))
    random.Random(args.seed + 1).shuffle(indices)
    test_count = max(2, round(n * args.test_fraction))
    test_count = min(test_count, n - 2)
    test_indices = np.asarray(indices[:test_count])
    train_indices = np.asarray(indices[test_count:])
    if len(train_indices) < 2 or len(test_indices) < 2:
        raise SystemExit("probe split needs at least two train and two test rows")

    active = _active_columns(y[train_indices], y[test_indices])
    if not active:
        raise SystemExit("probe has no non-constant held-out semantic targets")
    stage_reports: dict[str, Any] = {}
    rng = np.random.default_rng(args.seed + 2)
    permuted_labels = y[train_indices][rng.permutation(len(train_indices))]
    for name, values in representations.items():
        x = np.asarray(values, dtype=np.float64)
        prediction = ridge_predict(
            x[train_indices], y[train_indices], x[test_indices], alpha=args.ridge_alpha
        )
        control_prediction = ridge_predict(
            x[train_indices], permuted_labels, x[test_indices], alpha=args.ridge_alpha
        )
        scores = r2_scores(y[test_indices], prediction)
        controls = r2_scores(y[test_indices], control_prediction)
        mean_r2 = _mean_scores(scores, active)
        mean_control = _mean_scores(controls, active)
        group_reports = {
            group: {
                "active_targets": sum(index in active for index in columns),
                "mean_r2": _mean_scores(scores, [index for index in columns if index in active]),
                "control_mean_r2": _mean_scores(
                    controls, [index for index in columns if index in active]
                ),
            }
            for group, columns in target_groups.items()
        }
        stage_reports[name] = {
            "dimension": int(x.shape[1]),
            "r2": dict(zip(target_names, scores)),
            "control_permuted_label_r2": dict(zip(target_names, controls)),
            "active_targets": len(active),
            "mean_r2": mean_r2,
            "mean_control_r2": mean_control,
            "mean_lift": (
                float(mean_r2 - mean_control)
                if mean_r2 is not None and mean_control is not None
                else None
            ),
            "groups": group_reports,
        }

    projected = np.asarray(representations["projected_graph"], dtype=np.float64)
    prefix = np.asarray(representations["final_prefix"], dtype=np.float64)
    direct_retrieval = retrieval_accuracy(
        projected[test_indices], prefix[test_indices]
    )
    mapped_prefix = ridge_predict(
        projected[train_indices],
        prefix[train_indices],
        projected[test_indices],
        alpha=args.ridge_alpha,
    )
    mapped_retrieval = retrieval_accuracy(mapped_prefix, prefix[test_indices])
    chance = 1.0 / len(test_indices)

    failures: list[str] = []
    prefix_report = stage_reports["final_prefix"]
    prefix_mean = prefix_report["mean_r2"]
    prefix_lift = prefix_report["mean_lift"]
    if prefix_mean is None or prefix_mean < args.min_prefix_semantic_r2:
        failures.append(
            f"prefix mean semantic R2 {prefix_mean!r} < {args.min_prefix_semantic_r2:.4f}"
        )
    if prefix_lift is None or prefix_lift < args.min_prefix_mean_lift:
        failures.append(
            f"prefix semantic lift {prefix_lift!r} < {args.min_prefix_mean_lift:.4f}"
        )
    if mapped_retrieval - chance < args.min_retrieval_above_chance:
        failures.append(
            f"mapped retrieval above chance {mapped_retrieval - chance:.4f} "
            f"< {args.min_retrieval_above_chance:.4f}"
        )

    report = {
        "schema_version": 2,
        "status": "passed" if not failures else "failed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_load_contract": checkpoint_load_report,
        "rows": n,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "targets": target_names,
        "target_groups": target_groups,
        "active_target_count": len(active),
        "active_targets": [target_names[index] for index in active],
        "stages": stage_reports,
        "projected_to_prefix_retrieval": {
            "method": "train-fitted ridge map, held-out top-1 cosine retrieval",
            "mapped_top1_accuracy": mapped_retrieval,
            "direct_unmapped_top1_accuracy_diagnostic": direct_retrieval,
            "chance": chance,
            "mapped_above_chance": mapped_retrieval - chance,
        },
        "thresholds": {
            "min_prefix_semantic_r2": args.min_prefix_semantic_r2,
            "min_prefix_mean_lift": args.min_prefix_mean_lift,
            "min_retrieval_above_chance": args.min_retrieval_above_chance,
        },
        "failures": failures,
        "note": (
            "This probe tests held-out representational availability. Only the "
            "free-running correct/permuted/null gate tests causal decoder use."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
