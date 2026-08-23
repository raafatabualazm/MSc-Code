from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import v3_lossless_serializer as v3


def _small_tokenizer() -> tuple[Tokenizer, tuple[str, ...]]:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=384,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        [
            "push rbp mov rbp,rsp cmp rax,0x1 je @B2 call @U0 ret",
            "conditional true false call pool string use sites",
        ],
        trainer=trainer,
    )
    symbols = tuple(chr(value) for value in range(0x4E00, 0x4E80))
    tokenizer.add_tokens(list(symbols))
    return tokenizer, symbols


def _canonical_with_call_and_pool() -> dict:
    return {
        "architecture": "x86_64",
        "dfg_route": "current_combined_v2",
        "entry_blocks": [0],
        "blocks": [
            {
                "id": 0,
                "instructions": [
                    "push rbp",
                    "mov rbp,rsp",
                    "cmp rax,0x1",
                    "je @B2",
                ],
            },
            {"id": 1, "instructions": ["call @U0"]},
            {"id": 2, "instructions": ["ret"]},
        ],
        "cfg_edges": [
            {"source": 0, "target": 2, "edge_type": "conditional_true"},
            {"source": 0, "target": 1, "edge_type": "conditional_false"},
            {"source": 1, "target": 2, "edge_type": "call"},
            {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
        ],
        "dfg_edges": [
            {
                "source": 0,
                "target": 1,
                "edge_type": "dataflow",
                "locations": ["rax"],
                "dependency_count": 1,
            }
        ],
        "binary_pool": {
            "schema": "dart-aot-target-literal-pool-v1",
            "target_function": "candidate",
            "uses": [
                {
                    "pp_offset": 15,
                    "kind": "string",
                    "payload": {"code_units": [65, 0xD800, 66]},
                    "use_sites": [{"block": 1, "instruction": 0}],
                },
                {
                    "pp_offset": 23,
                    "kind": "composite",
                    "payload": {
                        "composite_type": "array_storage",
                        "elements": [
                            {
                                "index": 0,
                                "kind": "int",
                                "payload": {"value": -7},
                            },
                            {
                                "index": 0,
                                "kind": "null",
                                "payload": {},
                            },
                        ],
                        "omitted_edge_count": 2,
                    },
                    "use_sites": [
                        {"block": 0, "instruction": 2},
                        {"block": 1, "instruction": 0},
                    ],
                },
            ],
        },
    }


def _tokenizer_path() -> Path | None:
    candidates = []
    configured = os.environ.get("QWEN3_8B_TOKENIZER_JSON")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Qwen--Qwen3-8B"
            / "snapshots"
            / "b968826d9c46dd6066d109eabc6255188de91218"
            / "tokenizer.json",
            ROOT
            / ".hf_home"
            / "hub"
            / "models--Qwen--Qwen3-8B"
            / "snapshots"
            / "b968826d9c46dd6066d109eabc6255188de91218"
            / "tokenizer.json",
        ]
    )
    for path in candidates:
        if path.is_file() and v3.sha256_file(path) == str(
            v3.KNOWN_OPUS_V3_175_SEALS["tokenizer_sha256"]
        ):
            return path
    return None


def _real_paths() -> v3.V3ArtifactPaths | None:
    tokenizer = _tokenizer_path()
    clean = ROOT / "pod_sync_20260723" / "artifacts" / "v3_clean"
    release = (
        ROOT
        / "scrubbed_master_v2_release"
        / "direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release"
    )
    required = [
        clean / "v3eval_public.jsonl",
        clean / "v3eval_align.jsonl",
        clean / "v3eval_tests.jsonl",
        release / "binary_build" / "prepared" / "train_codec_private.jsonl",
        release / "compact" / "pool_reconciliation_private.jsonl",
        release / "release_manifest.json",
        release / "compact" / "compact_contract.json",
        release / "compact" / "codebook.json",
        ROOT / "scripts" / "data" / "build_compact_qwen_v3_cfgtypes_v2.py",
        ROOT / "scrubbed_master_v2_release" / "extractors" / "cfg_extractor.py",
        ROOT / "scrubbed_master_v2_release" / "extractors" / "dfg_extractor.py",
        ROOT / "scripts" / "data" / "cfg_extractor.py",
        ROOT / "scripts" / "data" / "dfg_extractor.py",
    ]
    if tokenizer is None or not all(path.is_file() for path in required):
        return None
    return v3.V3ArtifactPaths(
        public=required[0],
        alignment=required[1],
        tests=required[2],
        private=required[3],
        pool_reconciliation=required[4],
        release_manifest=required[5],
        contract=required[6],
        codebook=required[7],
        codec=required[8],
        tokenizer=tokenizer,
        legacy_cfg_extractor=required[9],
        legacy_dfg_extractor=required[10],
        current_cfg_extractor=required[11],
        current_dfg_extractor=required[12],
    )


def test_v3_f2_roundtrip_preserves_calls_pool_values_and_use_sites():
    tokenizer, symbols = _small_tokenizer()
    canonical = _canonical_with_call_and_pool()

    text = v3.serialize_v3_api_readable(
        canonical,
        tokenizer=tokenizer,
        visible_symbols=symbols,
    )
    decoded = v3.decode_v3_api_readable(text)

    assert decoded == {
        key: canonical[key] for key in v3.PROJECTION_FIELDS
    }
    assert decoded["cfg_edges"][2] == {
        "source": 1,
        "target": 2,
        "edge_type": "call",
    }
    assert decoded["binary_pool"]["uses"][0]["payload"]["code_units"] == [
        65,
        0xD800,
        66,
    ]
    assert decoded["binary_pool"]["uses"][1]["use_sites"] == [
        {"block": 0, "instruction": 2},
        {"block": 1, "instruction": 0},
    ]


def test_call_edge_ordinals_fail_closed():
    non_calls = [
        {"source": 0, "target": 1, "edge_type": "linear_fallthrough"}
    ]
    with pytest.raises(v3.V3SealError, match="duplicate_call_ordinal"):
        v3._merge_call_edges(
            non_calls,
            [
                {"ordinal": 0, "source": 0, "target": 1},
                {"ordinal": 0, "source": 0, "target": 1},
            ],
        )
    with pytest.raises(v3.V3SealError, match="call_ordinal_out_of_range"):
        v3._merge_call_edges(
            non_calls,
            [{"ordinal": 2, "source": 0, "target": 1}],
        )


@pytest.fixture(scope="module")
def real_preflight():
    paths = _real_paths()
    if paths is None:
        pytest.skip("sealed Opus-v3 artifacts/tokenizer are not present")
    serializer = v3.V3ArtifactSerializer(paths)
    return serializer.prepare_all()


def test_known_opus_v3_175_full_lossless_preflight(real_preflight):
    rows, manifest = real_preflight

    assert len(rows) == 175
    assert manifest["rows"] == 175
    assert manifest["task_sequence_sha256"] == (
        "9e81225a3ea89c5f9763366b18c8bc5b777bd4e48741276f0bc155c117f25d8f"
    )
    assert manifest["cohort_totals"]["cfg_edge_types"]["call"] == 3
    assert manifest["cohort_totals"]["binary_pool_records"] == 792
    assert manifest["cohort_totals"]["binary_pool_use_sites"] == 1256
    assert manifest["prompt_tokens"]["max"] == 11687
    assert manifest["prompt_tokens"]["rows_over_limit"] == 0
    assert all(
        row["invariants"]["public_ids_retokenized_exactly"]
        and row["invariants"]["api_semantic_roundtrip_exact"]
        and row["invariants"]["ordered_cfg_including_call_preserved"]
        and row["invariants"]["binary_pool_values_and_use_sites_preserved"]
        for row in rows
    )


def test_known_opus_v3_public_tamper_fails_before_decode(tmp_path: Path):
    paths = _real_paths()
    if paths is None:
        pytest.skip("sealed Opus-v3 artifacts/tokenizer are not present")
    tampered = tmp_path / "v3eval_public.jsonl"
    tampered.write_bytes(paths.public.read_bytes() + b" ")

    with pytest.raises(v3.V3SealError, match="public_sha256_mismatch"):
        v3.V3ArtifactSerializer(replace(paths, public=tampered))
