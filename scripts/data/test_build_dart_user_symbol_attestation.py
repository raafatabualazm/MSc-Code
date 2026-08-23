from __future__ import annotations

import json

import pytest

from scripts.data.build_dart_user_symbol_attestation import (
    SymbolAttestationError,
    build_attestation_row,
    canonical_sha256,
    key_id_sha256,
    ordered_commitment,
    symbol_digest,
)


KEY = bytes(range(32))


def source_row() -> dict:
    function_source = "class SecretType {}\nint secretHelper() => 1;\nint candidate() => secretHelper();\n"
    analysis_program = function_source + "\nvoid main() {}\n"
    import hashlib

    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return {
        "schema": "dart-source-only-aot-build-input-v1",
        "task_id": "sigless_attest",
        "split": "train",
        "split_row": 7,
        "analysis_program": analysis_program,
        "analysis_program_sha256": digest(analysis_program),
        "function_source": function_source,
        "function_source_sha256": digest(function_source),
        "source_symbols": {
            "functions": ["secretHelper", "candidate"],
            "types": ["SecretType"],
        },
        "transform_metadata": {
            "source_symbols": {
                "functions": ["secretHelper", "candidate"],
                "types": ["SecretType"],
            }
        },
    }


def aot_row(build: dict) -> dict:
    return {
        "schema": "phase0-s44-source-only-aot-row-v1",
        "task_id": build["task_id"],
        "split": build["split"],
        "split_row": build["split_row"],
        "analysis_program_sha256": build["analysis_program_sha256"],
        "function_source_sha256": build["function_source_sha256"],
        "producer": {"script_sha256": "a" * 64},
    }


def test_attestation_is_ordered_keyed_and_name_free() -> None:
    build = source_row()
    row = build_attestation_row(
        build_row=build, aot_row=aot_row(build), key=KEY
    )
    assert row["key_id_sha256"] == key_id_sha256(KEY)
    assert [entry["alias"] for entry in row["function_symbols"]] == [
        "AF0",
        "AF1",
    ]
    assert [entry["alias"] for entry in row["type_symbols"]] == ["T0"]
    assert row["completeness"]["complete_source_symbols_projection"] is True
    assert row["completeness"]["only_dart_scheme_imports"] is True
    serialized = json.dumps(row, sort_keys=True)
    assert "secretHelper" not in serialized
    assert "SecretType" not in serialized
    assert "candidate" not in serialized
    assert "class " not in serialized
    assert "void main" not in serialized

    expected = symbol_digest(
        KEY,
        task_id=row["task_id"],
        salt_hex=row["salt_hex"],
        kind="type",
        index=0,
        symbol="SecretType",
    )
    assert row["type_symbols"][0]["digest"] == expected
    wrong_index = symbol_digest(
        KEY,
        task_id=row["task_id"],
        salt_hex=row["salt_hex"],
        kind="type",
        index=1,
        symbol="SecretType",
    )
    assert wrong_index != expected
    assert row["completeness"]["ordered_commitment"] == ordered_commitment(
        KEY,
        task_id=row["task_id"],
        salt_hex=row["salt_hex"],
        function_digests=[
            entry["digest"] for entry in row["function_symbols"]
        ],
        type_digests=[entry["digest"] for entry in row["type_symbols"]],
    )


def test_attestation_fails_if_source_symbol_projection_is_incomplete() -> None:
    build = source_row()
    build["transform_metadata"]["source_symbols"]["functions"] = ["candidate"]
    with pytest.raises(
        SymbolAttestationError, match="source_symbol_transform_mismatch"
    ):
        build_attestation_row(
            build_row=build, aot_row=aot_row(build), key=KEY
        )


def test_attestation_fails_if_source_hash_or_producer_binding_drifts() -> None:
    build = source_row()
    bad_aot = aot_row(build)
    bad_aot["analysis_program_sha256"] = "b" * 64
    with pytest.raises(SymbolAttestationError, match="source_aot_hash_mismatch"):
        build_attestation_row(build_row=build, aot_row=bad_aot, key=KEY)

    bad_aot = aot_row(build)
    bad_aot["producer"]["script_sha256"] = "not-a-hash"
    with pytest.raises(SymbolAttestationError, match="invalid_producer_hash"):
        build_attestation_row(build_row=build, aot_row=bad_aot, key=KEY)


def test_attestation_rejects_package_file_and_part_directives() -> None:
    import hashlib

    for directive in (
        "import 'package:private/helpers.dart';\n",
        "import 'file:///private/helpers.dart';\n",
        "part 'helpers.dart';\n",
        "export 'package:private/helpers.dart';\n",
    ):
        build = source_row()
        build["analysis_program"] = directive + build["analysis_program"]
        build["analysis_program_sha256"] = hashlib.sha256(
            build["analysis_program"].encode()
        ).hexdigest()
        with pytest.raises(
            SymbolAttestationError, match="non_dart_library_directive"
        ):
            build_attestation_row(
                build_row=build,
                aot_row=aot_row(build),
                key=KEY,
            )
