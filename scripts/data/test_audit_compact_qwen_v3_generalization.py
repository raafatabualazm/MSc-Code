import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from hybrid_training_patch_v2_3.models.direct_compact_causal import (
    tokenizer_fingerprint,
)
from scripts.data import audit_compact_qwen_v3_generalization as audit
from scripts.data import build_compact_qwen_v3 as codec
from scripts.data import build_compact_qwen_v3_release as release


def _source_blind_row(task_id: str, split: str, family: str, source_pool=None):
    uses = [
        {
            "pp_offset": 0x17,
            "kind": "string",
            "payload": {"code_units": [120]},
            "use_sites": [{"block": 0, "instruction": 0}],
        }
    ]
    receipt = {
        "schema": "dart-aot-reconciled-pool-receipts-v1",
        "static": {
            "schema": "static-test",
            "source_blind": True,
            "target_function": "candidate",
            "entries": [
                {
                    "pp_offset": 0x17,
                    "category": "literal",
                    "literal": {"type": "string", "code_units": [120]},
                    "uses": [{"function_id": "candidate", "pc": "0x4"}],
                }
            ],
        },
        "runtime": {
            "schema": "runtime-test",
            "source_blind": True,
            "target_function": "candidate",
            "target_scope": [{"function_id": "candidate", "aot_address": "0x1000"}],
            "entries": [
                {
                    "pp_offset": 0x17,
                    "category": "literal",
                    "literal": {"type": "string", "code_units": [120]},
                    "uses": [{"function_id": "candidate", "pc": "0x1004"}],
                }
            ],
        },
        "projection_sha256": release.canonical_sha256(uses),
    }
    return {
        "task_id": task_id,
        "split": split,
        "split_row": 0,
        "family": family,
        "compact_private_metadata": {"source_pool": source_pool},
        "function": "candidate",
        "graph_v2": {
            "extractor_sha256": codec.ROUTE_SPECS[
                codec.ROUTE_CURRENT
            ].combined_sha256
        },
        "cfg": [
            {
                "id": 0,
                "start_address": "0x0",
                "instructions": ["mov rax,QWORD PTR [r15+0x17]", "ret"],
            }
        ],
        "edges": [],
        "integrity": {"entry_blocks": [0]},
        "binary_pool_uses": uses,
        "pool_projection_accounting": {
            "scope": "canonical_graph_retained_fixed_r15_xrefs",
            "target_exact_xrefs": 1,
            "graph_retained_xrefs": 1,
            "excluded_non_graph_xrefs": [],
            "excluded_non_graph_xref_count": 0,
            "all_target_xrefs_accounted": True,
        },
        "binary_pool_private_receipt": receipt,
        "aot": {"sha256": "a" * 64},
    }


def _json_line(value):
    return release.canonical_bytes(value).decode("ascii") + "\n"


def _build_bundle(tmp_path: Path):
    fit = _source_blind_row("fit-unit", "train", "master")
    dev = _source_blind_row("dev-unit", "dev", "topup_s45", "topup_s45")
    fit_path = tmp_path / "train_codec_private.jsonl"
    dev_path = tmp_path / "dev_codec_private.jsonl"
    fit_path.write_text(_json_line(fit), encoding="utf-8")
    dev_path.write_text(_json_line(dev), encoding="utf-8")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        [
            codec.canonical_pool_json(codec.pool_envelope(fit["binary_pool_uses"])),
            "mov rax QWORD PTR r15 ret compact graph literal pool block other",
        ],
        trainer=BpeTrainer(
            vocab_size=320,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<unk>"],
        ),
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        release.canonical_bytes({"vocab_size": tokenizer.get_vocab_size()}).decode(
            "ascii"
        ),
        encoding="utf-8",
    )
    pool_manifest = tmp_path / "pool.json"
    pool_manifest.write_text(
        release.canonical_bytes(
            {"schema": "combined-pool-extractor-test-v1"}
        ).decode("ascii"),
        encoding="utf-8",
    )
    toolchain_manifest = tmp_path / "toolchain.json"
    toolchain_manifest.write_text(
        release.canonical_bytes({"schema": "dart-toolchain-test-v1"}).decode(
            "ascii"
        ),
        encoding="utf-8",
    )
    binary_build_manifest = tmp_path / "binary_build_manifest.json"
    binary_build_manifest.write_text(
        release.canonical_bytes(
            {
                "schema": release.FINAL_BINARY_BUILD_SCHEMA,
                "splits": {
                    split: {
                        "rows": 1,
                        "codec_private": {"sha256": release.sha256_file(dataset)},
                    }
                    for split, dataset in (("train", fit_path), ("dev", dev_path))
                },
                "artifacts": {
                    "pool_extractor_manifest": {
                        "sha256": release.sha256_file(pool_manifest)
                    },
                    "dart_toolchain_manifest": {
                        "sha256": release.sha256_file(toolchain_manifest)
                    },
                },
                "gates": {
                    "all_3277_rows_present": True,
                    "all_aots_present_and_hash_valid": True,
                },
            }
        ).decode("ascii"),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    status = release.main(
        [
            "--fit",
            str(fit_path),
            "--measure",
            str(dev_path),
            "--output-dir",
            str(bundle),
            "--tokenizer-json",
            str(tokenizer_path),
            "--model-config",
            str(config_path),
            "--combined-pool-extractor-manifest",
            str(pool_manifest),
            "--aot-manifest",
            str(binary_build_manifest),
            "--dart-toolchain-manifest",
            str(toolchain_manifest),
            "--codebook-size",
            "1",
            "--max-blocks",
            "8",
            "--tokenizer-fingerprint-sha256",
            tokenizer_fingerprint(tokenizer),
            "--decoder-model",
            "local-qwen-test",
            "--decoder-revision",
            "immutable-test-revision",
        ]
    )
    assert status == 0
    return bundle, tokenizer_path, fit


def _humaneval_row(base, index):
    row = copy.deepcopy(base)
    row.update(
        {
            "task_id": f"sigless-unit-{index:03d}",
            "camel_case_function_name": "candidate",
            "prompt_signature_mode": "name_only",
            "dart_function_signature": "",
            "benchmark_protocol": {
                "neutral_target_name": "candidate",
                "prompt_withholds": [
                    "return_type",
                    "parameter_types",
                    "parameter_names",
                ],
            },
        }
    )
    return row


def _write_humaneval(tmp_path, base):
    path = tmp_path / "humaneval.jsonl"
    rows = [_humaneval_row(base, index) for index in range(154)]
    path.write_text("".join(_json_line(row) for row in rows), encoding="utf-8")
    return path, rows


def _verify(bundle, tokenizer_path):
    return audit.verify_bundle(
        bundle,
        tokenizer_path,
        codec_path=Path(codec.__file__).resolve(),
        graph_codec_path=Path(audit.graph_codec.__file__).resolve(),
        release_builder_path=Path(release.__file__).resolve(),
        legacy_cfg_extractor=(
            audit.ROOT / "scrubbed_master_v2_release/extractors/cfg_extractor.py"
        ),
        legacy_dfg_extractor=(
            audit.ROOT / "scrubbed_master_v2_release/extractors/dfg_extractor.py"
        ),
        current_cfg_extractor=audit.ROOT / "scripts/data/cfg_extractor.py",
        current_dfg_extractor=audit.ROOT / "scripts/data/dfg_extractor.py",
    )


def test_v3_generalization_cli_emits_transactional_sealed_reports(tmp_path):
    bundle, tokenizer_path, base = _build_bundle(tmp_path)
    humaneval, _ = _write_humaneval(tmp_path, base)
    output = tmp_path / "audits"
    status = audit.main(
        [
            "--bundle",
            str(bundle),
            "--tokenizer-json",
            str(tokenizer_path),
            "--humaneval",
            str(humaneval),
            "--output-dir",
            str(output),
        ],
        expected_humaneval_sha256=audit.sha256_file(humaneval),
    )
    assert status == 0
    assert {item.name for item in output.iterdir()} == audit.OUTPUT_NAMES
    checksums = audit._parse_sha256s(output / "SHA256SUMS.txt")
    assert set(checksums) == audit.OUTPUT_NAMES - {"SHA256SUMS.txt"}
    assert all(audit.sha256_file(output / name) == digest for name, digest in checksums.items())
    humaneval_report = json.loads(
        (output / "scrubbed_humaneval_instruction_codebook_audit.json").read_text()
    )
    assert humaneval_report["scope"] == "instruction_codebook_coverage_only"
    assert humaneval_report["full_v3_source_token_measurement"] is False
    assert humaneval_report["exact_canonical_and_dfg_roundtrip_rows"] == 154
    assert humaneval_report["non_claims"]["full_v3_source_token_count_was_measured"] is False
    assert humaneval_report["non_claims"]["human_eval_9000_token_gate_was_evaluated"] is False
    assert humaneval_report["fallback_representation"]["reversible"] is True
    topup = json.loads(
        (output / "topup_family_source_pool_fallback_audit.json").read_text()
    )
    assert topup["coverage_by_family"]["topup_s45"]["rows"] == 1
    assert topup["coverage_by_family_and_source_pool"]["topup_s45:topup_s45"]["rows"] == 1


def test_bundle_checksum_tamper_is_rejected_before_interpretation(tmp_path):
    bundle, tokenizer_path, _ = _build_bundle(tmp_path)
    with (bundle / "codebook.json").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="bundle_checksum_mismatch:codebook.json"):
        _verify(bundle, tokenizer_path)


def test_alignment_counter_drift_is_rejected(tmp_path):
    bundle, tokenizer_path, _ = _build_bundle(tmp_path)
    state = _verify(bundle, tokenizer_path)
    bad_preflight = copy.deepcopy(state.preflight)
    bad_preflight["fallback_by_role"]["measure"]["fallback"] += 1
    bad_state = audit.dataclasses.replace(state, preflight=bad_preflight)
    with pytest.raises(ValueError, match="preflight_role_measure_fallback_mismatch"):
        audit.aggregate_and_crosscheck_alignment(bad_state)


def test_humaneval_identity_name_only_and_current_extractor_are_fail_closed(tmp_path):
    bundle, tokenizer_path, base = _build_bundle(tmp_path)
    state = _verify(bundle, tokenizer_path)
    humaneval, rows = _write_humaneval(tmp_path, base)
    expected_sha = audit.sha256_file(humaneval)
    report = audit.audit_humaneval_instruction_coverage(
        state, humaneval, expected_sha256=expected_sha
    )
    assert report["rows"] == 154
    assert report["gates"]["all_dfgs_regenerated_edge_for_edge"] is True

    rows[0]["prompt_signature_mode"] = "exact"
    humaneval.write_text("".join(_json_line(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="is_not_name_only"):
        audit.audit_humaneval_instruction_coverage(
            state,
            humaneval,
            expected_sha256=audit.sha256_file(humaneval),
        )

    rows[0]["prompt_signature_mode"] = "name_only"
    rows[0]["graph_v2"]["extractor_sha256"] = codec.ROUTE_SPECS[
        codec.ROUTE_LEGACY
    ].combined_sha256
    humaneval.write_text("".join(_json_line(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="is_not_current_extractor"):
        audit.audit_humaneval_instruction_coverage(
            state,
            humaneval,
            expected_sha256=audit.sha256_file(humaneval),
        )


def test_checked_in_humaneval_canonical_identity_is_frozen():
    assert audit.sha256_file(audit.CANONICAL_HUMANEVAL) == (
        audit.CANONICAL_HUMANEVAL_SHA256
    )
    rows = audit.read_jsonl(audit.CANONICAL_HUMANEVAL, "humaneval")
    assert len(rows) == audit.CANONICAL_HUMANEVAL_ROWS
    assert len({row["task_id"] for row in rows}) == audit.CANONICAL_HUMANEVAL_ROWS
    assert {row["function"] for row in rows} == {"candidate"}
    assert {row["prompt_signature_mode"] for row in rows} == {"name_only"}
    assert {row["graph_v2"]["extractor_sha256"] for row in rows} == {
        codec.ROUTE_SPECS[codec.ROUTE_CURRENT].combined_sha256
    }


def test_direct_script_help_bootstraps_repo_imports():
    result = subprocess.run(
        [
            sys.executable,
            str(audit.ROOT / "scripts/data/audit_compact_qwen_v3_generalization.py"),
            "--help",
        ],
        cwd=audit.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--bundle" in result.stdout
