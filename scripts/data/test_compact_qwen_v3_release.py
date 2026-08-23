import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from scripts.data import build_compact_qwen_v3 as codec
from scripts.data import build_compact_qwen_v3_release as release
from hybrid_training_patch_v2_3.models.direct_compact_causal import (
    DirectCompactContract,
    tokenizer_fingerprint,
)


def source_blind_row():
    uses = [
        {
            "pp_offset": 0x17,
            "kind": "string",
            "payload": {"code_units": [120]},
            "use_sites": [{"block": 0, "instruction": 0}],
        }
    ]
    static_entry = {
        "pp_offset": 0x17,
        "category": "literal",
        "literal": {"type": "string", "code_units": [120]},
        "uses": [{"function_id": "candidate", "pc": "0x4"}],
    }
    runtime_entry = {
        "pp_offset": 0x17,
        "category": "literal",
        "literal": {"type": "string", "code_units": [120]},
        "uses": [{"function_id": "candidate", "pc": "0x1004"}],
    }
    receipt = {
        "schema": "dart-aot-reconciled-pool-receipts-v1",
        "static": {
            "schema": "static-test",
            "source_blind": True,
            "target_function": "candidate",
            "entries": [static_entry],
        },
        "runtime": {
            "schema": "runtime-test",
            "source_blind": True,
            "target_function": "candidate",
            "target_scope": [
                {"function_id": "candidate", "aot_address": "0x1000"}
            ],
            "entries": [runtime_entry],
        },
        "projection_sha256": release.canonical_sha256(uses),
    }
    return {
        "task_id": "release-unit",
        "split": "train",
        "split_row": 0,
        "family": "master",
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


def nested_nonliteral_composite_row():
    row = source_blind_row()
    descriptor = {
        "kind": "nonliteral",
        "payload": {
            "nonliteral_kind": "runtime_object",
            "profile_type": "Instance",
        },
    }
    row["binary_pool_uses"] = [
        {
            "pp_offset": 0x17,
            "kind": "composite",
            "payload": {
                "complete": True,
                "composite_type": "array_storage",
                "elements": [{"index": 0, "value": descriptor}],
                "omitted_edge_counts": {},
            },
            "use_sites": [{"block": 0, "instruction": 0}],
        }
    ]
    static_entry = row["binary_pool_private_receipt"]["static"]["entries"][0]
    static_entry.pop("literal")
    static_entry.update(
        {
            "category": "composite",
            "complete": True,
            "composite_type": "array_storage",
            "elements": [
                {
                    "index": 0,
                    "value": {
                        "category": "nonliteral",
                        "nonliteral_kind": "runtime_object",
                        "profile_type": "Instance",
                    },
                }
            ],
            "omitted_edge_counts": {},
        }
    )
    runtime_entry = row["binary_pool_private_receipt"]["runtime"]["entries"][0]
    runtime_entry.pop("literal")
    runtime_entry.update(
        {"category": "nonliteral", "nonliteral_kind": "runtime_heap_object"}
    )
    row["binary_pool_private_receipt"]["projection_sha256"] = (
        release.canonical_sha256(row["binary_pool_uses"])
    )
    return row


def test_pool_alignment_is_exact_six_field_source_free_projection():
    row = source_blind_row()
    metadata = release.pool_alignment_metadata(row)
    assert set(metadata) == release.POOL_METADATA_FIELDS
    assert metadata["schema"] == "dart-aot-target-pool-alignment-v1"
    assert metadata["use_count"] == 1
    assert metadata["source_blind"] is True
    assert "payload" not in metadata
    release.assert_alignment_source_free({"pool_metadata": metadata})


def test_raw_xref_reconciliation_proves_included_literal():
    row = source_blind_row()
    canonical = codec.canonicalize(row)
    metadata = release.pool_alignment_metadata(row)
    reconciliation = release.reconcile_raw_pool_xrefs(row, canonical, metadata)
    assert reconciliation["counts"] == {
        "raw_target_xrefs": 1,
        "included_graph_retained_literal": 1,
        "graph_retained_nonliteral": 0,
        "excluded_non_graph_xref": 0,
        "encoded_pool_records": 1,
        "encoded_pool_use_sites": 1,
    }
    assert reconciliation["xrefs"][0]["classification"] == (
        "included_graph_retained_literal"
    )
    assert reconciliation["xrefs"][0]["use_site"] == {
        "block": 0,
        "instruction": 0,
    }
    assert "payload" not in set(release._walk_keys(reconciliation))


def test_raw_xref_reconciliation_rejects_omitted_graph_literal():
    row = source_blind_row()
    canonical = codec.canonicalize(row)
    canonical["binary_pool"]["uses"] = []
    with pytest.raises(ValueError, match="omitted_graph_retained_literal"):
        release.reconcile_raw_pool_xrefs(
            row, canonical, release.pool_alignment_metadata(row)
        )


def test_raw_xref_reconciliation_accounts_pruned_instruction():
    row = source_blind_row()
    row["binary_pool_uses"] = []
    row["binary_pool_private_receipt"]["projection_sha256"] = (
        release.canonical_sha256([])
    )
    row["pool_projection_accounting"].update(
        {
            "graph_retained_xrefs": 0,
            "excluded_non_graph_xrefs": [
                {
                    "pp_offset": 0x17,
                    "function_offset": 4,
                    "reason": "deterministically_pruned_non_graph_instruction",
                    "static_category": "literal",
                    "runtime_category": "literal",
                }
            ],
            "excluded_non_graph_xref_count": 1,
        }
    )
    canonical = codec.canonicalize(row)
    reconciliation = release.reconcile_raw_pool_xrefs(
        row, canonical, release.pool_alignment_metadata(row)
    )
    assert reconciliation["counts"]["excluded_non_graph_xref"] == 1
    assert reconciliation["xrefs"][0]["classification"] == (
        "excluded_non_graph_xref"
    )


def test_raw_xref_reconciliation_classifies_retained_nonliteral():
    row = source_blind_row()
    row["binary_pool_uses"] = []
    row["binary_pool_private_receipt"]["projection_sha256"] = (
        release.canonical_sha256([])
    )
    for receipt_side in ("static", "runtime"):
        entry = row["binary_pool_private_receipt"][receipt_side]["entries"][0]
        entry.pop("literal")
        entry["category"] = "nonliteral"
        entry["nonliteral_kind"] = "code_or_vm_object"
    canonical = codec.canonicalize(row)
    reconciliation = release.reconcile_raw_pool_xrefs(
        row, canonical, release.pool_alignment_metadata(row)
    )
    assert reconciliation["counts"]["graph_retained_nonliteral"] == 1
    assert reconciliation["xrefs"][0]["classification"] == (
        "graph_retained_nonliteral"
    )


def test_nested_nonliteral_descriptor_is_preserved_inside_composite():
    row = nested_nonliteral_composite_row()
    canonical = codec.canonicalize(row)
    reconciliation = release.reconcile_raw_pool_xrefs(
        row, canonical, release.pool_alignment_metadata(row)
    )
    encoded = canonical["binary_pool"]["uses"][0]["payload"]["elements"][0][
        "value"
    ]
    assert encoded == {
        "kind": "nonliteral",
        "payload": {
            "nonliteral_kind": "runtime_object",
            "profile_type": "Instance",
        },
    }
    stats = release.pool_statistics(canonical)
    assert stats["all_node_kinds"] == {"composite": 1, "nonliteral": 1}
    assert stats["nested_nonliteral_descriptor_pairs"] == {
        "Instance->runtime_object": 1
    }
    assert reconciliation["counts"]["included_graph_retained_literal"] == 1


def test_alignment_payload_guard_is_recursive():
    with pytest.raises(ValueError, match="alignment_contains_private_payload"):
        release.assert_alignment_source_free({"nested": [{"payload": {}}]})


def test_release_cli_smoke_emits_sealed_four_field_rows(tmp_path):
    fit = source_blind_row()
    dev = source_blind_row()
    dev.update(
        {
            "task_id": "release-unit-dev",
            "split": "dev",
            "split_row": 0,
            "family": "topup_s45",
            "compact_private_metadata": {"source_pool": "topup_s45"},
        }
    )
    fit_path = tmp_path / "train_codec_private.jsonl"
    dev_path = tmp_path / "dev_codec_private.jsonl"
    fit_path.write_text(release.canonical_bytes(fit).decode("ascii") + "\n")
    dev_path.write_text(release.canonical_bytes(dev).decode("ascii") + "\n")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        [
            codec.canonical_pool_json(codec.pool_envelope(fit["binary_pool_uses"])),
            "mov rax QWORD PTR r15 ret compact graph literal pool block",
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
        )
    )
    pool_manifest = tmp_path / "pool.json"
    pool_manifest.write_text(
        release.canonical_bytes(
            {"schema": "combined-pool-extractor-test-v1"}
        ).decode("ascii")
    )
    toolchain_manifest = tmp_path / "toolchain.json"
    toolchain_manifest.write_text(
        release.canonical_bytes({"schema": "dart-toolchain-test-v1"}).decode(
            "ascii"
        )
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
        ).decode("ascii")
    )

    output = tmp_path / "release"
    status = release.main(
        [
            "--fit",
            str(fit_path),
            "--measure",
            str(dev_path),
            "--output-dir",
            str(output),
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
            "8",
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
    model_rows = [
        __import__("json").loads(line)
        for line in (output / "compact_model_inputs.jsonl").read_text().splitlines()
    ]
    assert len(model_rows) == 2
    assert all(set(row) == release.PUBLIC_FIELDS for row in model_rows)
    alignment_rows = [
        __import__("json").loads(line)
        for line in (output / "alignment_private.jsonl").read_text().splitlines()
    ]
    assert all(
        set(row["pool_metadata"]) == release.POOL_METADATA_FIELDS
        for row in alignment_rows
    )
    assert alignment_rows[0]["source_pool"] is None
    assert alignment_rows[1]["source_pool"] == "topup_s45"
    assert all("compact_text" not in row for row in alignment_rows)
    reconciliation = (output / "pool_reconciliation_private.jsonl").read_text()
    assert reconciliation.count("\n") == 2
    assert not (output / "aot_manifest_bundle.json").exists()
    contract = DirectCompactContract.load(output / "compact_contract.json")
    assert contract.schema == codec.CONTRACT_SCHEMA
    assert contract.pool_reconciliation_manifest_sha256 == release.sha256_file(
        output / "pool_reconciliation_private.jsonl"
    )
    contract.validate_artifacts(
        tokenizer=tokenizer,
        tokenizer_json_path=tokenizer_path,
        codec_path=codec.__file__,
        codebook_path=output / "codebook.json",
    )
    report = __import__("json").loads(
        (output / "preflight_report.json").read_text()
    )
    assert report["fallback_by_role_and_family"]["fit"]["master"]["rows"] == 1
    assert (
        report["fallback_by_role_and_family"]["measure"]["topup_s45"]["rows"]
        == 1
    )
    checksums = (output / "SHA256SUMS.txt").read_text()
    for path in output.iterdir():
        if path.name != "SHA256SUMS.txt":
            assert f"  {path.name}\n" in checksums


def test_exact_dataset_binding_rejects_wrong_sha_count_or_split():
    inputs = [
        {"split": "train", "rows": 2, "sha256": "a" * 64},
        {"split": "dev", "rows": 1, "sha256": "b" * 64},
    ]
    release._require_exact_dataset_bindings(inputs, [dict(item) for item in inputs])
    for field, wrong in (
        ("sha256", "c" * 64),
        ("rows", 3),
        ("split", "other"),
    ):
        sealed = [dict(item) for item in inputs]
        sealed[1][field] = wrong
        with pytest.raises(ValueError, match="aot_manifest_dataset_binding_mismatch"):
            release._require_exact_dataset_bindings(inputs, sealed)


def test_sealed_manifest_requires_final_aot_hash_gate(tmp_path):
    toolchain_sha = "d" * 64
    manifest = {
        "schema": release.FINAL_BINARY_BUILD_SCHEMA,
        "splits": {
            "train": {
                "rows": 2,
                "codec_private": {"sha256": "a" * 64},
            }
        },
        "artifacts": {
            "pool_extractor_manifest": {"sha256": "c" * 64},
            "dart_toolchain_manifest": {"sha256": toolchain_sha},
        },
        "gates": {
            "all_3277_rows_present": True,
            "all_aots_present_and_hash_valid": True,
        },
    }
    path = tmp_path / "train.json"
    path.write_text(release.canonical_bytes(manifest).decode("ascii"))
    bindings, pool_links, toolchain_links = release._sealed_aot_dataset_bindings(
        [path]
    )
    assert bindings == [
        {
            "manifest": "train.json",
            "split": "train",
            "rows": 2,
            "sha256": "a" * 64,
        }
    ]
    assert pool_links == {"c" * 64}
    assert toolchain_links == {toolchain_sha}

    manifest["gates"].pop("all_aots_present_and_hash_valid")
    path.write_text(release.canonical_bytes(manifest).decode("ascii"))
    with pytest.raises(ValueError, match="lacks_aot_hash_gate"):
        release._sealed_aot_dataset_bindings([path])


def test_split_manifest_mode_is_rejected(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        release.canonical_bytes(
            {
                "schema": "phase0-s44-binary-pool-aot-manifest-v1",
                "gates": {"all_selected_rows_built": True},
            }
        ).decode("ascii")
    )
    with pytest.raises(
        ValueError, match="must_be_finalized_binary_build_seal"
    ):
        release._sealed_aot_dataset_bindings([path])
