from __future__ import annotations

import importlib.util
import math
import struct
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "extract_attested_binary_pool_constants.py"
)
SPEC = importlib.util.spec_from_file_location("pool_constants", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pool_constants = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool_constants)


def _bundle(*texts: str) -> dict:
    return {
        "task_id": "sigless_test",
        "functions": [
            {
                "instructions": [
                    {"text": text, "offset": index}
                    for index, text in enumerate(texts)
                ]
            }
        ],
        "accounting": {
            "emitted_function_count": 1,
            "emitted_instruction_count": len(texts),
        },
    }


def _raw_object(class_id: int, payload: bytes, length: int = 0) -> str:
    word0 = class_id << 12
    word1 = length << 1
    return (
        word0.to_bytes(8, "little")
        + word1.to_bytes(8, "little")
        + payload
    ).hex()


def _raw_number(class_id: int, payload: bytes) -> str:
    return ((class_id << 12).to_bytes(8, "little") + payload).hex()


def test_collects_every_function_pool_offset_without_symbol_lookup() -> None:
    bundle = _bundle(
        "mov rax,QWORD PTR [r15+0x7a7]",
        "mov rbx,QWORD PTR [r15-0x10]",
        "mov rcx,QWORD PTR [r15]",
        "ret",
    )
    assert pool_constants.collect_pool_offsets(bundle) == [-16, 0, 1959]
    assert pool_constants.collect_pool_accesses(bundle) == {
        -16: ["tagged_word"],
        0: ["tagged_word"],
        1959: ["tagged_word"],
    }


def test_rejects_unaccountable_dynamic_r15_operand() -> None:
    with pytest.raises(
        pool_constants.ConstantExtractionError,
        match="unsupported r15 operand",
    ):
        pool_constants.collect_pool_offsets(
            _bundle("mov rax,QWORD PTR [r15+rcx*8]")
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (math.inf, "double.infinity"),
        (-math.inf, "double.negativeInfinity"),
        (math.nan, "double.nan"),
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (1.25, "1.25"),
    ],
)
def test_double_format_is_total_and_type_preserving(
    value: float, expected: str
) -> None:
    assert pool_constants.format_double(value) == expected


def test_access_widths_come_from_exact_assembly_mnemonic() -> None:
    bundle = _bundle(
        "movsd xmm0,QWORD PTR [r15+0x827]",
        "movss xmm1,DWORD PTR [r15+0x82f]",
        "movups xmm2,XMMWORD PTR [r15+0x837]",
    )
    assert pool_constants.collect_pool_accesses(bundle) == {
        0x827: ["inline_float64"],
        0x82F: ["inline_float32"],
        0x837: ["inline_float32x4"],
    }


def test_decode_inline_float_widths() -> None:
    entries = [
        {"inline_float64_raw": struct.pack("<d", 3.14).hex()},
        {"inline_float32_raw": struct.pack("<f", 3.14).hex()},
        {
            "inline_float32x4_raw": struct.pack(
                "<4f", 1.0, 2.0, 3.0, 4.0
            ).hex()
        },
    ]
    _, numbers, counters = pool_constants.decode_entries(entries)
    assert "3.14" in numbers
    assert "3.1400001" in numbers
    assert "Float32x4(1.0, 2.0, 3.0, 4.0)" in numbers
    assert counters["inline_float64_entries"] == 1
    assert counters["inline_float32_entries"] == 1
    assert counters["inline_float32x4_entries"] == 1


def test_decode_preserves_nonfinite_boxed_doubles() -> None:
    entries = [
        {
                "raw": _raw_number(
                    pool_constants.BOXED_DOUBLE_CLASS_ID,
                    struct.pack("<d", math.inf),
                )
        },
        {
                "raw": _raw_number(
                    pool_constants.BOXED_DOUBLE_CLASS_ID,
                    struct.pack("<d", math.nan),
                )
        },
    ]
    strings, numbers, counters = pool_constants.decode_entries(entries)
    assert strings == []
    assert numbers == ["double.nan", "double.infinity"]
    assert counters["supported_number_objects"] == 2


def test_decode_keeps_exact_long_string_without_old_80_char_cap() -> None:
    value = "A <semantic> string " + ("x" * 200)
    entries = [
        {
            "raw": _raw_object(
                94,
                value.encode("latin-1"),
                length=len(value),
            )
        }
    ]
    strings, numbers, counters = pool_constants.decode_entries(entries)
    assert strings == [value]
    assert numbers == []
    assert counters["supported_string_objects"] == 1


def test_decode_accounts_empty_string_without_failing_row() -> None:
    entries = [{"raw": _raw_object(94, b"", length=0)}]
    strings, numbers, counters = pool_constants.decode_entries(entries)
    assert strings == []
    assert numbers == []
    assert counters["metadata_strings_rejected"] == 1


def test_decode_classifies_known_non_object_tagged_sentinel() -> None:
    entries = [
        {
            "offset": 0x5E7,
            "word": 0x8000000000000001,
            "raw": None,
            "word_class": "non_object_tagged_sentinel",
        }
    ]
    strings, numbers, counters = pool_constants.decode_entries(entries)
    assert strings == []
    assert numbers == []
    assert counters["tagged_sentinel_entries"] == 1
    assert counters["unreadable_entries"] == 0


def test_decode_classifies_noncanonical_zap_word_as_non_object() -> None:
    entries = [
        {
            "offset": 0x7E7,
            "word": 0x4EC4EC4EC4EC4EC5,
            "raw": None,
            "word_class": "non_object_noncanonical_word",
        }
    ]
    _, _, counters = pool_constants.decode_entries(entries)
    assert counters["tagged_sentinel_entries"] == 1
    assert counters["unreadable_entries"] == 0


def test_decode_fails_on_unclassified_unreadable_tagged_word() -> None:
    entries = [
        {
            "offset": 0x123,
            "word": 0xDEADBEEF0001,
            "raw": None,
            "read_error": "memory",
        }
    ]
    with pytest.raises(
        pool_constants.ConstantExtractionError,
        match="unreadable tagged object-pool word",
    ):
        pool_constants.decode_entries(entries)


def test_decode_mint_preserves_signed_value() -> None:
    value = -(2**62) + 17
    entries = [
        {
            "raw": _raw_number(
                pool_constants.BOXED_MINT_CLASS_ID,
                value.to_bytes(8, "little", signed=True),
            )
        }
    ]
    _, numbers, counters = pool_constants.decode_entries(entries)
    assert numbers == [str(value)]
    assert counters["supported_number_objects"] == 1


def test_path_metadata_is_rejected_without_mutating_program_strings() -> None:
    assert pool_constants.keep_string("file:///tmp/source.dart")[0] is None
    assert pool_constants.keep_string("A <user> value") == (
        "A <user> value",
        None,
    )
