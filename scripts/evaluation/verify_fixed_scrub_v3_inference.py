"""Fail-closed verification for a fixed-scrub-v3 raw inference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_GRAPH_ENV = {
    "GRAPH_PROMPT_ASSEMBLY_MODE": "none",
    "GRAPH_PROMPT_CLEAN_ASM": "0",
    "GRAPH_PROMPT_FIT_ASSEMBLY": "0",
    "GRAPH_QWEN_PREFIX_TOKENS": "64",
    "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
    "GRAPH_QWEN_PREFIX_MIN_TOKENS": "4",
    "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": "4",
    "GRAPH_QWEN_PREFIX_GATE_MODE": "token",
    "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
    "GRAPH_REGION_COMPRESSION": "linear_residual",
    "GRAPH_REGION_MAX_BLOCKS": "16",
    "GRAPH_DFG_MODE": "edges",
    "GRAPH_EDGE_ABLATION": "full",
    "GRAPH_GNN_ABLATION": "identity",
    "GRAPH_ADD_REVERSE_EDGES": "1",
    "GRAPH_STRICT_GRAPH": "1",
    "GRAPH_USE_REASONING": "0",
    "GRAPH_SEED": "42",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VERIFY FAILED: {message}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    public = read_jsonl(args.public_dataset)
    raw = json.loads(args.raw_predictions.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))

    require(len(public) == args.expected_rows, f"public rows={len(public)}")
    require(len(raw) == args.expected_rows, f"prediction rows={len(raw)}")
    public_ids = [str(row.get("task_id")) for row in public]
    raw_ids = [str(row.get("id")) for row in raw]
    require(raw_ids == public_ids, "raw prediction IDs/order differ from public input")

    for index, (source, output) in enumerate(zip(public, raw, strict=True), start=1):
        require(output.get("source_line") == index, f"row {index}: bad source_line")
        predictions = output.get("predictions")
        require(
            isinstance(predictions, list) and len(predictions) == args.expected_samples,
            f"row {index}: predictions={len(predictions) if isinstance(predictions, list) else 'invalid'}",
        )
        require(all(isinstance(item, str) for item in predictions), f"row {index}: non-string candidate")
        require(not str(output.get("tests") or "").strip(), f"row {index}: tests leaked into raw output")
        require(not str(output.get("reference") or "").strip(), f"row {index}: reference leaked into raw output")
        ablation = output.get("graph_input_ablation") or {}
        require(ablation.get("mode") == "none", f"row {index}: graph ablation is not none")
        require(
            str(ablation.get("target_id")) == str(source.get("task_id"))
            and str(ablation.get("donor_id")) == str(source.get("task_id")),
            f"row {index}: graph donor mismatch",
        )

    require(provenance.get("prompt_schema_version") == args.prompt_schema, "prompt schema mismatch")
    require(provenance.get("scoring_tests_visible_to_policy") is False, "policy could see scoring tests")
    require(provenance.get("row_count") == args.expected_rows, "provenance row count mismatch")
    require(provenance.get("seed") == args.seed, "generation seed mismatch")
    generation = provenance.get("generation") or {}
    require(generation.get("num_samples") == args.expected_samples, "num_samples mismatch")
    require(generation.get("generation_batch_size") == args.expected_samples, "batch size mismatch")
    require(generation.get("max_new_tokens") == 768, "generation token budget mismatch")
    require(generation.get("decoder_prompt_max_length") == 2048, "prompt budget mismatch")
    require(generation.get("use_cache") is True, "decoder cache disabled")
    require(generation.get("decoder_gradient_checkpointing") is False, "generation checkpointing enabled")

    dataset_record = provenance.get("dataset") or {}
    require(dataset_record.get("sha256") == sha256(args.public_dataset), "public dataset hash mismatch")
    output_record = provenance.get("output") or {}
    require(output_record.get("sha256") == sha256(args.raw_predictions), "raw output hash mismatch")
    checkpoint_record = provenance.get("checkpoint") or {}
    require(checkpoint_record.get("sha256") == args.checkpoint_sha256, "checkpoint hash mismatch")

    load = provenance.get("checkpoint_load") or {}
    require(load.get("strict") is False, "checkpoint strict-load mode changed")
    require(load.get("missing_keys_count") == args.missing_keys, "checkpoint missing-key signature changed")
    require(load.get("unexpected_keys_count") == args.unexpected_keys, "checkpoint unexpected-key signature changed")
    gate = (provenance.get("graph_prefix_gate") or {}).get("mean_sigmoid")
    require(isinstance(gate, (int, float)), "prefix gate not recorded")
    require(args.gate_min <= float(gate) <= args.gate_max, f"prefix gate out of range: {gate}")

    graph_env = provenance.get("graph_environment") or {}
    for key, value in EXPECTED_GRAPH_ENV.items():
        require(str(graph_env.get(key)) == value, f"graph environment mismatch: {key}")
    require(provenance.get("graph_input_ablation", {}).get("mode") == "none", "provenance graph ablation changed")
    require(
        provenance.get("graph_input_ablation", {}).get("self_mapped_rows") == args.expected_rows,
        "graph ablation self-map count mismatch",
    )

    models = provenance.get("models") or {}
    require(
        (models.get("decoder") or {}).get("requested_revision") == args.decoder_revision,
        "decoder revision mismatch",
    )
    require(
        (models.get("encoder") or {}).get("requested_revision") == args.encoder_revision,
        "encoder revision mismatch",
    )
    prompt_digest = str(provenance.get("prompt_stream_sha256") or "")
    require(len(prompt_digest) == 64, "prompt digest missing")
    require(
        prompt_digest == args.expected_prompt_stream_sha256,
        f"prompt stream digest mismatch: {prompt_digest}",
    )

    result = {
        "verified": True,
        "rows": len(raw),
        "samples_per_row": args.expected_samples,
        "dataset_sha256": dataset_record.get("sha256"),
        "output_sha256": output_record.get("sha256"),
        "checkpoint_sha256": checkpoint_record.get("sha256"),
        "prompt_schema_version": provenance.get("prompt_schema_version"),
        "prompt_stream_sha256": prompt_digest,
        "checkpoint_load": load,
        "graph_prefix_gate": gate,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public_dataset", required=True, type=Path)
    parser.add_argument("--raw_predictions", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--expected_prompt_stream_sha256", required=True)
    parser.add_argument("--prompt_schema", default="antigravity-v3-matched-function-contract")
    parser.add_argument("--expected_rows", type=int, default=154)
    parser.add_argument("--expected_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_keys", type=int, default=602)
    parser.add_argument("--unexpected_keys", type=int, default=0)
    parser.add_argument("--gate_min", type=float, default=0.15)
    parser.add_argument("--gate_max", type=float, default=0.25)
    parser.add_argument("--decoder_revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--encoder_revision", default="2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d")
    parser.add_argument("--output", type=Path, default=None)
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
