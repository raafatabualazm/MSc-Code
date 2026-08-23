"""Fail-closed verification of the frozen Regions16 comparator pool.

The comparator was generated before the v3 inference provenance additions.  Its
schema binds the complete prediction JSON, dataset, checkpoint, prompt stream,
model revisions, and historical prompt renderer, but does not record the
checkpoint-load key counts or graph-prefix gate.  Those two omissions are
reported as unavailable rather than silently presented as verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fixed-scrub-v3-comparator-verification-v1"
EXPECTED_PREDICTIONS_SHA256 = (
    "474dac0836c3810fa6fdad176c60021e08e0f3d722365672927c1f36484e0118"
)
EXPECTED_PROVENANCE_SHA256 = (
    "6ea5b7be98b1f8ba13c04f6cba05a4a578cfa071b8559531ad22bb07dd3cb93e"
)
EXPECTED_DATASET_SHA256 = (
    "8453876a40d2279684a190a5bf1430a62897c84e063a78e25c57198287bc6928"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e8e872608f22ae8e1c5607d6179feeb5f133401fb2a7d1fc40fe8894d8c347fc"
)
EXPECTED_PROMPT_STREAM_SHA256 = (
    "55adb80e8a24df956c82c2eed260523a2f6c1b6e00a566bfb7b269c7eab75d0d"
)
EXPECTED_DECODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_ENCODER_REVISION = "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"
EXPECTED_RENDERER_SHA256 = (
    "6234f82bd4c64c29a561160374888d1bc6916af8d7d7368e30ba97cb5f237e13"
)
EXPECTED_INFERENCE_SOURCE_SHA256 = (
    "cb5969372363e7d738611946105a96759f59f65067ce1fe84671a492d8706a51"
)
EXPECTED_PROMPT_SCHEMA = "antigravity-v2-no-test-hints"

EXPECTED_GRAPH_ENV = {
    "GRAPH_ADD_REVERSE_EDGES": "1",
    "GRAPH_BLOCK_POOLING": "cls",
    "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
    "GRAPH_BLOCK_VECTORS_PER_BLOCK": "4",
    "GRAPH_CAUSAL_POSITION_IDS": "cumsum",
    "GRAPH_DFG_MODE": "edges",
    "GRAPH_EDGE_ABLATION": "full",
    "GRAPH_GNN_ABLATION": "identity",
    "GRAPH_MAX_BLOCK_INSTRS": "20",
    "GRAPH_MAX_DATAFLOW_EDGES": "0",
    "GRAPH_POSITION_SCHEME": "roberta",
    "GRAPH_PROMPT_ASSEMBLY_MODE": "none",
    "GRAPH_PROMPT_CLEAN_ASM": "0",
    "GRAPH_PROMPT_FIT_ASSEMBLY": "0",
    "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
    "GRAPH_QWEN_PREFIX_GATE_INIT": "0.2",
    "GRAPH_QWEN_PREFIX_GATE_MODE": "token",
    "GRAPH_QWEN_PREFIX_MIN_TOKENS": "4",
    "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
    "GRAPH_QWEN_PREFIX_TOKENS": "64",
    "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": "4",
    "GRAPH_REGION_COMPRESSION": "linear_residual",
    "GRAPH_REGION_MAX_BLOCKS": "16",
    "GRAPH_SEED": "42",
    "GRAPH_STRICT_GRAPH": "1",
    "GRAPH_USE_REASONING": "0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VERIFY FAILED: {message}")


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"dataset line {line_number} is not an object")
        rows.append(value)
    return rows


def candidate_digest(rows: list[dict[str, Any]]) -> str:
    """Match the digest used by the v3 hidden-label join."""

    digest = hashlib.sha256()
    for row in rows:
        payload = [str(row.get("id", "")), row.get("predictions", [])]
        digest.update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def source_record(
    provenance: dict[str, Any], suffix: str, expected_sha256: str
) -> dict[str, Any]:
    records = provenance.get("source_files")
    require(isinstance(records, list), "source_files is missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and str(record.get("path", "")).replace("\\", "/").endswith(suffix)
    ]
    require(len(matches) == 1, f"expected one source record ending in {suffix!r}")
    record = matches[0]
    require(record.get("sha256") == expected_sha256, f"source hash mismatch: {suffix}")
    require(
        isinstance(record.get("size_bytes"), int) and record["size_bytes"] > 0,
        f"source size missing: {suffix}",
    )
    return record


def verify_runtime_observations(provenance: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    load = provenance.get("checkpoint_load")
    if load is None:
        result["checkpoint_load"] = {
            "status": "not_recorded_in_legacy_provenance",
            "verified": False,
        }
    else:
        require(isinstance(load, dict), "checkpoint_load is not an object")
        require(load.get("strict") is False, "checkpoint strict-load mode changed")
        require(load.get("missing_keys_count") == 602, "checkpoint missing-key count changed")
        require(load.get("unexpected_keys_count") == 0, "checkpoint unexpected-key count changed")
        result["checkpoint_load"] = {"status": "verified", "verified": True, **load}

    prefix = provenance.get("graph_prefix_gate")
    if prefix is None:
        result["graph_prefix_gate"] = {
            "status": "not_recorded_in_legacy_provenance",
            "verified": False,
        }
    else:
        require(isinstance(prefix, dict), "graph_prefix_gate is not an object")
        gate = prefix.get("mean_sigmoid")
        require(isinstance(gate, (int, float)), "prefix gate mean is missing")
        require(0.15 <= float(gate) <= 0.25, f"prefix gate out of range: {gate}")
        result["graph_prefix_gate"] = {
            "status": "verified",
            "verified": True,
            "mean_sigmoid": float(gate),
        }
    return result


def verify(args: argparse.Namespace) -> dict[str, Any]:
    predictions_hash = sha256(args.predictions)
    provenance_hash = sha256(args.provenance)
    dataset_hash = sha256(args.dataset)
    require(
        predictions_hash == args.expected_predictions_sha256,
        f"frozen predictions SHA-256 mismatch: {predictions_hash}",
    )
    require(
        provenance_hash == args.expected_provenance_sha256,
        f"frozen provenance SHA-256 mismatch: {provenance_hash}",
    )
    require(
        dataset_hash == args.expected_dataset_sha256,
        f"comparator dataset SHA-256 mismatch: {dataset_hash}",
    )

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    require(isinstance(rows, list), "predictions must be a JSON array")
    require(len(rows) == args.expected_rows, f"prediction rows={len(rows)}")
    require(all(isinstance(row, dict) for row in rows), "prediction row is not an object")
    dataset_rows = load_jsonl(args.dataset)
    require(len(dataset_rows) == args.expected_rows, f"dataset rows={len(dataset_rows)}")

    seen_ids: set[str] = set()
    for index, (row, source) in enumerate(zip(rows, dataset_rows, strict=True), 1):
        task_id = str(row.get("id", ""))
        source_id = str(source.get("task_id", source.get("id", "")))
        require(task_id != "", f"row {index}: missing ID")
        require(task_id not in seen_ids, f"row {index}: duplicate ID {task_id!r}")
        seen_ids.add(task_id)
        require(task_id == source_id, f"row {index}: prediction/dataset ID mismatch")
        require(row.get("source_line") == index, f"row {index}: bad source_line")
        require(
            str(row.get("filename", "")) == str(source.get("filename", "")),
            f"row {index}: prediction/dataset filename mismatch",
        )
        candidates = row.get("predictions")
        require(
            isinstance(candidates, list) and len(candidates) == args.expected_samples,
            f"row {index}: expected {args.expected_samples} candidates",
        )
        require(all(isinstance(item, str) for item in candidates), f"row {index}: non-string candidate")
        require(row.get("reference") == source.get("dart_source"), f"row {index}: reference mismatch")
        require(row.get("tests") == source.get("tests"), f"row {index}: tests mismatch")
        ablation = row.get("graph_input_ablation")
        require(isinstance(ablation, dict), f"row {index}: graph ablation missing")
        require(ablation.get("mode") == "none", f"row {index}: graph ablation is not none")
        require(
            str(ablation.get("target_id")) == task_id
            and str(ablation.get("donor_id")) == task_id,
            f"row {index}: graph input is not self-mapped",
        )

    provenance = load_object(args.provenance)
    require(provenance.get("schema_version") == 1, "legacy provenance schema mismatch")
    require(provenance.get("prompt_schema_version") == EXPECTED_PROMPT_SCHEMA, "prompt schema mismatch")
    require(
        provenance.get("prompt_stream_sha256") == args.expected_prompt_stream_sha256,
        "prompt stream SHA-256 mismatch",
    )
    require(provenance.get("scoring_tests_visible_to_policy") is False, "policy could see scoring tests")
    require(provenance.get("row_count") == args.expected_rows, "provenance row count mismatch")
    require(provenance.get("seed") == args.seed, "generation seed mismatch")

    generation = provenance.get("generation")
    require(isinstance(generation, dict), "generation record missing")
    require(generation.get("num_samples") == args.expected_samples, "num_samples mismatch")
    require(generation.get("generation_batch_size") == args.expected_samples, "batch size mismatch")
    require(generation.get("max_new_tokens") == 768, "generation token budget mismatch")
    require(generation.get("decoder_prompt_max_length") == 2048, "prompt budget mismatch")
    require(generation.get("use_cache") is True, "decoder cache disabled")
    require(generation.get("decoder_gradient_checkpointing") is False, "generation checkpointing enabled")

    output = provenance.get("output")
    require(isinstance(output, dict), "output record missing")
    require(output.get("sha256") == predictions_hash, "provenance does not bind prediction output")
    require(output.get("size_bytes") == args.predictions.stat().st_size, "output size mismatch")
    dataset = provenance.get("dataset")
    require(isinstance(dataset, dict), "dataset record missing")
    require(dataset.get("sha256") == dataset_hash, "provenance does not bind comparator dataset")
    require(dataset.get("size_bytes") == args.dataset.stat().st_size, "dataset size mismatch")
    checkpoint = provenance.get("checkpoint")
    require(isinstance(checkpoint, dict), "checkpoint record missing")
    require(checkpoint.get("sha256") == args.checkpoint_sha256, "checkpoint SHA-256 mismatch")
    require(
        isinstance(checkpoint.get("size_bytes"), int) and checkpoint["size_bytes"] > 0,
        "checkpoint size is missing",
    )

    models = provenance.get("models")
    require(isinstance(models, dict), "model records missing")
    for label, expected_id, expected_revision in (
        ("decoder", "Qwen/Qwen3-8B", args.decoder_revision),
        ("encoder", "microsoft/graphcodebert-base", args.encoder_revision),
    ):
        model = models.get(label)
        require(isinstance(model, dict), f"{label} model record missing")
        require(model.get("requested_id") == expected_id, f"{label} model ID mismatch")
        require(model.get("requested_revision") == expected_revision, f"{label} revision mismatch")
        require(model.get("resolved_commit") == expected_revision, f"{label} resolved commit mismatch")

    graph_env = provenance.get("graph_environment")
    require(isinstance(graph_env, dict), "graph environment missing")
    for key, value in EXPECTED_GRAPH_ENV.items():
        require(str(graph_env.get(key)) == value, f"graph environment mismatch: {key}")
    require(graph_env.get("GRAPH_DECODER_REVISION") == args.decoder_revision, "decoder env revision mismatch")
    require(graph_env.get("GRAPH_ENCODER_REVISION") == args.encoder_revision, "encoder env revision mismatch")

    graph_ablation = provenance.get("graph_input_ablation")
    require(isinstance(graph_ablation, dict), "graph ablation provenance missing")
    require(graph_ablation.get("mode") == "none", "provenance graph ablation changed")
    require(graph_ablation.get("seed") == args.seed, "graph ablation seed mismatch")
    require(graph_ablation.get("self_mapped_rows") == args.expected_rows, "self-map count mismatch")

    renderer = source_record(
        provenance,
        "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
        args.renderer_sha256,
    )
    inference_source = source_record(
        provenance,
        "scripts/evaluation/graph_inference_antigravity.py",
        args.inference_source_sha256,
    )
    runtime_observations = verify_runtime_observations(provenance)

    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "rows": len(rows),
        "samples_per_row": args.expected_samples,
        "candidate_count": len(rows) * args.expected_samples,
        "candidate_stream_sha256": candidate_digest(rows),
        "bindings": {
            "predictions_sha256": predictions_hash,
            "provenance_sha256": provenance_hash,
            "dataset_sha256": dataset_hash,
            "checkpoint_sha256": checkpoint.get("sha256"),
            "prompt_stream_sha256": provenance.get("prompt_stream_sha256"),
            "renderer_sha256": renderer.get("sha256"),
            "inference_source_sha256": inference_source.get("sha256"),
        },
        "prompt_schema_version": provenance.get("prompt_schema_version"),
        "seed": provenance.get("seed"),
        "model_revisions": {
            "decoder": args.decoder_revision,
            "encoder": args.encoder_revision,
        },
        "scoring_tests_visible_to_policy": False,
        "runtime_observations": runtime_observations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected_rows", type=int, default=154)
    parser.add_argument("--expected_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_predictions_sha256", default=EXPECTED_PREDICTIONS_SHA256)
    parser.add_argument("--expected_provenance_sha256", default=EXPECTED_PROVENANCE_SHA256)
    parser.add_argument("--expected_dataset_sha256", default=EXPECTED_DATASET_SHA256)
    parser.add_argument("--checkpoint_sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument(
        "--expected_prompt_stream_sha256", default=EXPECTED_PROMPT_STREAM_SHA256
    )
    parser.add_argument("--decoder_revision", default=EXPECTED_DECODER_REVISION)
    parser.add_argument("--encoder_revision", default=EXPECTED_ENCODER_REVISION)
    parser.add_argument("--renderer_sha256", default=EXPECTED_RENDERER_SHA256)
    parser.add_argument(
        "--inference_source_sha256", default=EXPECTED_INFERENCE_SOURCE_SHA256
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
