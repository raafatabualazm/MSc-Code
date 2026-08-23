from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "data" / "harmonize_scrubbed_humaneval_dfg.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dfg_harmonizer_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frozen_extractor_and_actual_harmonized_pair_are_exact():
    h = load_module()
    extractor = ROOT / "scrubbed_master_v2_release" / "extractors" / "dfg_extractor.py"
    codec = ROOT / "scripts" / "data" / "build_compact_qwen_v1.py"
    assert h.sha256_file(extractor) == h.EXPECTED_DFG_SHA256
    canonicalize = h.load_symbol(codec, "test_compact_codec", "canonicalize")
    build_dfg = h.load_symbol(extractor, "test_frozen_dfg", "build_cross_block_dfg")

    source_public = read_jsonl(ROOT / "data" / "testing" / "grpo_data_graphv2_sigscrub_v2_nameonly_public.jsonl")
    source_private = read_jsonl(ROOT / "data" / "testing" / "grpo_data_graphv2_sigscrub_v2_nameonly_private.jsonl")
    out = ROOT / "data" / "testing" / "humaneval_v2_nameonly_dfg_harmonized"
    output_public_path = out / "humaneval_v2_nameonly_dfg_harmonized_public.jsonl"
    output_private_path = out / "humaneval_v2_nameonly_dfg_harmonized_private.jsonl"
    output_public = read_jsonl(output_public_path)
    output_private = read_jsonl(output_private_path)
    assert len(source_public) == len(source_private) == len(output_public) == len(output_private) == 154
    assert output_public_path.read_bytes() == h.encode_jsonl(output_public)
    assert output_private_path.read_bytes() == h.encode_jsonl(output_private)
    h.validate_public_privacy(output_public)

    source_public_by_id = {row["task_id"]: row for row in source_public}
    source_private_by_id = {row["task_id"]: row for row in source_private}
    output_public_by_id = {row["task_id"]: row for row in output_public}
    output_private_by_id = {row["task_id"]: row for row in output_private}
    assert set(source_public_by_id) == set(source_private_by_id) == set(output_public_by_id) == set(output_private_by_id)
    for task_id in source_public_by_id:
        source_pub = source_public_by_id[task_id]
        source_priv = source_private_by_id[task_id]
        out_pub = output_public_by_id[task_id]
        out_priv = output_private_by_id[task_id]
        assert h.protected_projection(source_pub) == h.protected_projection(out_pub)
        assert h.protected_projection(source_priv) == h.protected_projection(out_priv)
        assert out_pub["assembly"] == out_priv["assembly"]
        assert out_pub["cfg"] == out_priv["cfg"]
        assert h.non_dataflow_edges(out_pub) == h.non_dataflow_edges(out_priv)
        canonical = canonicalize(out_pub, "runtime_aware")
        expected_dfg = build_dfg(canonical["blocks"], canonical["cfg_edges"], max_edges=100000)
        assert canonical["dfg_edges"] == sorted(
            expected_dfg,
            key=lambda edge: (edge["source"], edge["target"], edge["edge_type"]),
        )
        assert h.dataflow_edges(out_pub) == h.dataflow_edges(out_priv)


def test_compact_measure_is_lossless_and_sealed_to_training_codebook():
    expected_codebook = "d44f9be95debe6e7d8766bf434cf9aeabd89a3d6ca5b09a06e3c50272543e76c"
    out = ROOT / "data" / "testing" / "direct_compact_humaneval_v2_nameonly_harmonized"
    report = json.loads((out / "preflight_report.json").read_text(encoding="utf-8"))
    alignment = read_jsonl(out / "alignment_private.jsonl")
    measures = [row for row in alignment if row["role"] == "measure"]
    assert report["passed"] is True
    assert report["failures_count"] == 0
    assert report["quarantined"] == 0
    assert report["rows_by_role"]["measure"] == 154
    assert len(measures) == len({row["task_id"] for row in measures}) == 154
    assert max(row["source_tokens"] for row in measures) == 7766
    assert all(row["source_tokens"] <= 9000 for row in measures)
    invariants = report["lossless_invariants"]
    assert invariants["unknown_tokens"] == 0
    assert invariants["truncated_rows"] == 0
    assert invariants["raw_fallback_is_reversible"] is True
    assert invariants["dfg_extractor_sha256"] == "beb237cf2ad8e3d65a536e8d30b698e14486ade36a019c247d580c372b858000"
    assert report["contract"]["compact_codebook_sha256"] == expected_codebook
    assert load_module().sha256_file(out / "codebook.json") == expected_codebook


def test_public_privacy_gate_rejects_nested_private_key():
    h = load_module()
    row = {"task_id": "sigless_test", "metadata": {"tests": "must not be public"}}
    try:
        h.validate_public_privacy([row])
    except ValueError as exc:
        assert "private keys" in str(exc)
    else:
        raise AssertionError("privacy gate accepted a private key")
