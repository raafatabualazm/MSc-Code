import json
from pathlib import Path

import pytest

from scrubbed_master_v2_release import finalize_phase0_compact_qwen_v3 as release


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def checksum(root: Path, name: str, names) -> None:
    release._write_checksum(root, name, names)


def tiny_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    splits = {"train": 2, "dev": 1}
    roles = {"fit": 2, "measure": 1}
    statuses = {
        "included-train": 2,
        "included-dev": 1,
        "quarantined": 1,
        "excluded": 1,
    }
    monkeypatch.setattr(release, "EXPECTED_SPLITS", splits)
    monkeypatch.setattr(release, "EXPECTED_ROLES", roles)
    monkeypatch.setattr(release, "EXPECTED_MODEL_ROWS", 3)
    monkeypatch.setattr(release, "EXPECTED_CANONICAL_ROWS", 5)
    monkeypatch.setattr(release, "EXPECTED_STATUS_COUNTS", statuses)

    roots = {
        name: tmp_path / name
        for name in ("v2", "source", "binary", "compact", "supervised", "audits")
    }
    for root in roots.values():
        root.mkdir()

    # Canonical v2 membership inherited by the source preparation.
    v2_prepared = roots["v2"] / "prepared"
    reconciliation = [
        {"input_line": 1, "task_id": "t0", "status": "included-train"},
        {"input_line": 2, "task_id": "d0", "status": "included-dev"},
        {"input_line": 3, "task_id": "q0", "status": "quarantined"},
        {"input_line": 4, "task_id": "t1", "status": "included-train"},
        {"input_line": 5, "task_id": "x0", "status": "excluded"},
    ]
    write_jsonl(v2_prepared / "reconciliation.jsonl", reconciliation)
    write_json(
        v2_prepared / "preparation_manifest.json",
        {
            "schema": "phase0-compact-qwen-v2-preparation",
            "counts": {"input": 5, "statuses": statuses},
            "gates": {"passed": True},
        },
    )
    for name in release.V2_PREP_FILES - {
        "reconciliation.jsonl",
        "preparation_manifest.json",
    }:
        write_jsonl(v2_prepared / name, [])
    checksum(v2_prepared, "SHA256SUMS.txt", release.V2_PREP_FILES)
    write_json(
        roots["v2"] / "release_manifest.json",
        {
            "schema": "compact-qwen-phase0-s44-v2-release-seal-v1",
            "gates": {"passed": True},
        },
    )
    checksum(
        roots["v2"],
        "SHA256SUMS.txt",
        {
            "release_manifest.json",
            "prepared/SHA256SUMS.txt",
            "prepared/preparation_manifest.json",
            "prepared/reconciliation.jsonl",
        },
    )

    split_ids = {"train": ["t0", "t1"], "dev": ["d0"]}
    split_rows = {}
    for split, ids in split_ids.items():
        build = [{"task_id": task_id} for task_id in ids]
        labels = [
            {
                "task_id": task_id,
                "function": "candidate",
                "lang": "dart",
                "dart_source": f"int candidate() => {index};\n",
            }
            for index, task_id in enumerate(ids)
        ]
        split_rows[split] = (build, labels)
        write_jsonl(roots["source"] / f"private_build_inputs/{split}.jsonl", build)
        write_jsonl(roots["source"] / f"prepared/{split}_private_labels.jsonl", labels)
    source_manifest = {
        "schema": "phase0-s44-binary-pool-v3-source-preparation-v1",
        "source_release_manifest_sha256": release.sha256_file(
            roots["v2"] / "release_manifest.json"
        ),
        "gates": {"all_rows_preserved": True},
        "splits": [],
    }
    for split, expected in splits.items():
        source_manifest["splits"].append(
            {
                "split": split,
                "rows": expected,
                "private_build_inputs": {
                    "sha256": release.sha256_file(
                        roots["source"] / f"private_build_inputs/{split}.jsonl"
                    )
                },
                "private_labels": {
                    "sha256": release.sha256_file(
                        roots["source"] / f"prepared/{split}_private_labels.jsonl"
                    )
                },
            }
        )
    write_json(roots["source"] / "source_preparation_manifest.json", source_manifest)
    checksum(roots["source"], "SOURCE_SHA256SUMS.txt", release.SOURCE_FILES)

    # Finalized binary-build metadata.  There are no AOT payload files here;
    # the seal says they were previously verified and the packager binds it.
    ordered_ids = split_ids["train"] + split_ids["dev"]
    for split, ids in split_ids.items():
        codec = [
            {
                "task_id": task_id,
                "split": split,
                "split_row": index,
                "function": "candidate",
            }
            for index, task_id in enumerate(ids)
        ]
        write_jsonl(roots["binary"] / f"prepared/{split}_codec_private.jsonl", codec)
        write_jsonl(roots["binary"] / f"quarantine/{split}.jsonl", [])
        write_json(
            roots["binary"] / f"manifests/{split}.json",
            {
                "counts": {"built_or_resumed": len(ids), "failed": 0},
                "gates": {"passed": True},
            },
        )
    write_jsonl(
        roots["binary"] / "aot_manifest.jsonl",
        [
            {
                "task_id": task_id,
                "aot_path": f"aot/{task_id}.aot",
                "aot_sha256": f"{index + 1:064x}",
                "aot_size_bytes": index + 1,
            }
            for index, task_id in enumerate(ordered_ids)
        ],
    )
    write_jsonl(
        roots["binary"] / "pool_reconciliation_private.jsonl",
        [{"task_id": task_id} for task_id in ordered_ids],
    )
    write_json(
        roots["binary"] / "pool_extractor_manifest.json",
        {
            "target_function": "candidate",
            "source_blind_after_aot": True,
        },
    )
    write_json(roots["binary"] / "dart_toolchain_manifest.json", {"dart": "pinned"})
    binary_manifest = {
        "schema": "phase0-s44-binary-pool-build-seal-v1",
        "rows": 3,
        "splits": {split: {"rows": count} for split, count in splits.items()},
        "artifacts": {},
        "gates": {
            "all_aots_present_and_hash_valid": True,
            "all_rows_present": True,
        },
    }
    for field, name in {
        "aot_manifest": "aot_manifest.jsonl",
        "pool_reconciliation_private": "pool_reconciliation_private.jsonl",
        "pool_extractor_manifest": "pool_extractor_manifest.json",
        "dart_toolchain_manifest": "dart_toolchain_manifest.json",
    }.items():
        binary_manifest["artifacts"][field] = {
            "sha256": release.sha256_file(roots["binary"] / name)
        }
    binary_manifest["artifacts"]["source_preparation_manifest"] = {
        "sha256": release.sha256_file(
            roots["source"] / "source_preparation_manifest.json"
        )
    }
    write_json(roots["binary"] / "binary_build_manifest.json", binary_manifest)
    checksum(
        roots["binary"], "BINARY_BUILD_SHA256SUMS.txt", release.BINARY_FILES
    )

    # Compact four-field stream and private alignment.
    codebook = {
        "schema": "compact-qwen-v3-codebook",
        "fit_scope": "train_only",
        "measure_excluded_from_fit": True,
        "fit_retained": 2,
    }
    write_json(roots["compact"] / "codebook.json", codebook)
    contract = {
        "schema": "direct-compact-causal-v3",
        "target_function": "candidate",
        "target_language": "Dart",
        "target_architecture": "x86_64",
        "max_source_tokens": 9000,
        "codec_sha256": "a" * 64,
        "tokenizer_json_sha256": "b" * 64,
        "codebook_sha256": release.sha256_file(roots["compact"] / "codebook.json"),
        "pool_extractor_sha256": release.sha256_file(
            roots["binary"] / "pool_extractor_manifest.json"
        ),
        "dart_toolchain_manifest_sha256": release.sha256_file(
            roots["binary"] / "dart_toolchain_manifest.json"
        ),
        "aot_manifest_sha256": release.sha256_file(
            roots["binary"] / "binary_build_manifest.json"
        ),
        "lossless_domain": "test-lossless-domain",
    }
    public_rows = [
        {
            "compact_input_ids": [100 + index],
            "compact_codec_sha256": contract["codec_sha256"],
            "compact_codebook_sha256": contract["codebook_sha256"],
            "compact_tokenizer_sha256": contract["tokenizer_json_sha256"],
        }
        for index in range(3)
    ]
    alignment = []
    compact_recon = []
    for index, (task_id, role) in enumerate(
        zip(ordered_ids, ["fit", "fit", "measure"], strict=True)
    ):
        alignment.append(
            {
                "model_row": index,
                "model_row_sha256": release.canonical_sha256(public_rows[index]),
                "role": role,
                "task_id": task_id,
                "pool_metadata": {
                    "schema": "dart-aot-target-pool-alignment-v1",
                    "source_blind": True,
                    "target_function": "candidate",
                },
            }
        )
        compact_recon.append(
            {
                "status": "included",
                "model_row": index,
                "task_id": task_id,
                "gates": {"passed": True},
            }
        )
    write_jsonl(roots["compact"] / "compact_model_inputs.jsonl", public_rows)
    write_jsonl(roots["compact"] / "alignment_private.jsonl", alignment)
    write_jsonl(
        roots["compact"] / "pool_reconciliation_private.jsonl", compact_recon
    )
    contract["pool_reconciliation_manifest_sha256"] = release.sha256_file(
        roots["compact"] / "pool_reconciliation_private.jsonl"
    )
    write_json(roots["compact"] / "compact_contract.json", contract)
    write_jsonl(roots["compact"] / "quarantine.jsonl", [])
    write_jsonl(roots["compact"] / "failures.jsonl", [])
    write_json(
        roots["compact"] / "preflight_report.json",
        {
            "passed": True,
            "rows_retained": 3,
            "rows_by_role": roles,
            "quarantined": 0,
            "failures_count": 0,
            "tokens": {"limit": 9000, "rows_over_limit": 0, "max": 1},
            "lossless_invariants": {
                "exact_graph_and_pool_roundtrip_rows": 3,
                "compact_id_stream_roundtrip_rows": 3,
                "dfg_regenerated_and_matched_rows": 3,
                "source_blind_pool_rows": 3,
                "public_private_rows_bijective_and_hash_bound": 3,
                "unknown_tokens": 0,
                "truncated_rows": 0,
                "call_edges_encoded_explicitly": True,
            },
            "fallback_by_role": {
                "fit": {"instructions": 2, "fallback": 0},
                "measure": {"instructions": 1, "fallback": 0},
            },
        },
    )
    checksum(
        roots["compact"],
        "SHA256SUMS.txt",
        release.COMPACT_REQUIRED,
    )

    # Strict supervised joins.
    by_role = {
        "fit": list(zip(public_rows[:2], alignment[:2], strict=True)),
        "measure": list(zip(public_rows[2:], alignment[2:], strict=True)),
    }
    for split, role in {"train": "fit", "dev": "measure"}.items():
        labels = split_rows[split][1]
        selected = by_role[role]
        rows = []
        for label, (public, _sidecar) in zip(labels, selected, strict=True):
            rows.append(
                {
                    "lang": "Dart",
                    "function": "candidate",
                    "dart_source": label["dart_source"].strip(),
                    **public,
                }
            )
        output = roots["supervised"] / f"{split}.jsonl"
        write_jsonl(output, rows)
        mapping = [
            {
                "public_line": sidecar["model_row"],
                "alignment_line": sidecar["model_row"],
                "private_line": index,
                "identity_sha256": release.sha256_bytes(
                    sidecar["task_id"].encode("utf-8")
                ),
            }
            for index, (_public, sidecar) in enumerate(selected)
        ]
        seal = {
            "schema": "compact-public-private-join-seal-v2",
            "contract_schema": "direct-compact-causal-v3",
            "rows": len(rows),
            "source_rows": 3,
            "selected_role": role,
            "skipped_rows": 3 - len(rows),
            "public_sha256": release.sha256_file(
                roots["compact"] / "compact_model_inputs.jsonl"
            ),
            "alignment_sha256": release.sha256_file(
                roots["compact"] / "alignment_private.jsonl"
            ),
            "contract_sha256": release.sha256_file(
                roots["compact"] / "compact_contract.json"
            ),
            "private_sha256": release.sha256_file(
                roots["source"] / f"prepared/{split}_private_labels.jsonl"
            ),
            "output_sha256": release.sha256_file(output),
            "output_size_bytes": output.stat().st_size,
            "private_bijection": {
                "required": True,
                "verified": True,
                "private_rows": len(rows),
                "unused_private_rows": 0,
            },
            "model_visible_fields": sorted(release.SUPERVISED_FIELDS),
            "pool_metadata": {
                "schema": "dart-aot-target-pool-alignment-v1",
                "rows": len(rows),
                "source_blind_rows": len(rows),
                "target_function": "candidate",
            },
            "mapping": mapping,
            "mapping_sha256": release.sha256_bytes(
                json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
        }
        write_json(roots["supervised"] / f"{split}.join_seal.json", seal)

    # Three generalization reports plus their hash-binding umbrella.
    audit_common = {
        "schema": "compact-qwen-v3-generalization-audit-v1",
        "passed": True,
        "bundle_contract_sha256": release.sha256_file(
            roots["compact"] / "compact_contract.json"
        ),
        "codebook_sha256": release.sha256_file(
            roots["compact"] / "codebook.json"
        ),
    }
    audit_values = {
        "dev_fallback_audit.json": {
            "audit": "sealed_v3_dev_instruction_codebook_fallback",
            "scope": "full-v3-release-alignment-derived-instruction-fallback",
            "coverage": {"rows": 1, "instructions": 1, "fallback": 0},
            "cross_checks": {"alignment_matches_preflight": True},
        },
        "scrubbed_humaneval_instruction_codebook_audit.json": {
            "audit": "scrubbed_humaneval_instruction_codebook_coverage",
            "scope": "instruction_codebook_coverage_only",
            "full_v3_source_token_measurement": False,
            "rows": 154,
            "exact_canonical_and_dfg_roundtrip_rows": 154,
            "fallback_representation": {"reversible": True},
            "non_claims": {
                "full_v3_binary_pool_stream_was_available": False,
                "full_v3_source_token_count_was_measured": False,
                "human_eval_9000_token_gate_was_evaluated": False,
                "human_eval_truncation_count": None,
            },
        },
        "topup_family_source_pool_fallback_audit.json": {
            "audit": "sealed_v3_topup_family_and_source_pool_instruction_fallback",
            "scope": "full-v3-release-alignment-derived-instruction-fallback",
            "coverage": {"rows": 2},
            "coverage_by_family": {"topup_s45": {"rows": 2}},
            "cross_checks": {"family_matches_preflight": True},
        },
    }
    for name, value in audit_values.items():
        write_json(
            roots["audits"] / name,
            {**audit_common, **value},
        )
    write_json(
        roots["audits"] / "generalization_audit_seal.json",
        {
            "schema": "compact-qwen-v3-generalization-audit-seal-v1",
            "all_passed": True,
            "reports": {
                name: {
                    "passed": True,
                    "sha256": release.sha256_file(roots["audits"] / name),
                }
                for name in release.AUDIT_REPORTS
            },
        },
    )
    checksum(roots["audits"], "SHA256SUMS.txt", release.AUDIT_FILES)
    return roots


def package(roots, output: Path):
    return release.package_release(
        source_prep_root=roots["source"],
        binary_build_root=roots["binary"],
        compact_root=roots["compact"],
        supervised_root=roots["supervised"],
        audits_root=roots["audits"],
        v2_release=roots["v2"],
        output_dir=output,
    )


def reseal_audits(audits: Path) -> None:
    seal_path = audits / "generalization_audit_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name in release.AUDIT_REPORTS:
        seal["reports"][name]["sha256"] = release.sha256_file(audits / name)
    write_json(seal_path, seal)
    checksum(audits, "SHA256SUMS.txt", release.AUDIT_FILES)


def test_packager_materializes_only_sealed_metadata(tmp_path, monkeypatch) -> None:
    roots = tiny_fixture(tmp_path, monkeypatch)
    output = tmp_path / "release-v3"
    manifest = package(roots, output)
    assert manifest["counts"]["model_rows"] == 3
    assert manifest["external_aot_payload"] == {
        "shipped": False,
        "policy": "external-hash-bound-only",
        "rows": 3,
        "size_bytes": 6,
        "aot_manifest": {
            "path": "binary_build/aot_manifest.jsonl",
            "sha256": release.sha256_file(output / "binary_build/aot_manifest.jsonl"),
        },
        "binary_build_seal": {
            "path": "binary_build/binary_build_manifest.json",
            "sha256": release.sha256_file(
                output / "binary_build/binary_build_manifest.json"
            ),
        },
    }
    assert not list(output.rglob("*.aot"))
    assert len(release.read_jsonl(output / "canonical_membership/reconciliation.jsonl")) == 5
    release.verify_checksum(
        output,
        "SHA256SUMS.txt",
        {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
            and path.relative_to(output).as_posix() != "SHA256SUMS.txt"
        },
    )


def test_packager_rejects_join_output_drift(tmp_path, monkeypatch) -> None:
    roots = tiny_fixture(tmp_path, monkeypatch)
    with (roots["supervised"] / "train.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(release.ReleaseValidationError, match="count drift"):
        package(roots, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_instruction_only_humaneval_may_not_claim_full_tokens(
    tmp_path, monkeypatch
) -> None:
    roots = tiny_fixture(tmp_path, monkeypatch)
    human_path = (
        roots["audits"] / "scrubbed_humaneval_instruction_codebook_audit.json"
    )
    human = json.loads(human_path.read_text(encoding="utf-8"))
    human["tokens"] = {"limit": 9000, "rows_over_limit": 0, "max": 1}
    write_json(human_path, human)
    reseal_audits(roots["audits"])
    with pytest.raises(
        release.ReleaseValidationError,
        match="forbidden full-token/9K claim",
    ):
        package(roots, tmp_path / "false-human-token-claim")
    assert not (tmp_path / "false-human-token-claim").exists()


def test_future_full_v3_humaneval_requires_and_accepts_real_token_gate(
    tmp_path, monkeypatch
) -> None:
    roots = tiny_fixture(tmp_path, monkeypatch)
    human_path = (
        roots["audits"] / "scrubbed_humaneval_instruction_codebook_audit.json"
    )
    human = json.loads(human_path.read_text(encoding="utf-8"))
    human["scope"] = "full_v3_source_token_measurement"
    human["full_v3_source_token_measurement"] = True
    human["tokens"] = {"limit": 9000, "rows_over_limit": 0, "max": 8999}
    human.pop("non_claims")
    write_json(human_path, human)
    reseal_audits(roots["audits"])
    manifest = package(roots, tmp_path / "full-human-measurement")
    assert manifest["gates"]["generalization_audits_passed"] is True


def test_checksum_requires_exact_coverage(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    checksum(tmp_path, "SUMS", {"a"})
    with pytest.raises(release.ReleaseValidationError, match="coverage drift"):
        release.verify_checksum(tmp_path, "SUMS", {"a", "b"})
