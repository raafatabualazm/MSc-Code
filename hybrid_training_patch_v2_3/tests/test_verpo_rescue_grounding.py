from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import verpo_rescue_grounding as grounding


# Two blocks, one conditional branch encoded with true and false CFG edges.
# ``frontier_f2.decode_f2`` expands the operandless jne to ``jne @B1``.
VALID_F2 = (
    "F2\n"
    "C0\n"
    "\n"
    "Ax86_64\n"
    "Ea\n"
    "D\n"
    "S\n"
    "B\n"
    "a{jne}|tb\n"
    "b{ret}|\n"
    "X\n"
)

CANDIDATES = (
    "int f(int x) {\r\n"
    "  if (x == 0) return 1;\r\n"
    "  return 0;\r\n"
    "}\r\n",
    "int f(int x) {\n"
    "  return x;\n"
    "}\n",
)
DIAGNOSTICS = ("expected 2, got 1", "visible test failed")


def make_catalog() -> grounding.GroundingCatalog:
    return grounding.build_grounding_catalog(
        VALID_F2,
        CANDIDATES,
        diagnostics=DIAGNOSTICS,
    )


def valid_evidence() -> list[dict[str, str]]:
    return [
        {
            "kind": "f2_edge",
            "ref": "F2E000",
            "claim": "The branch has a true successor.",
        },
        {
            "kind": "candidate_line",
            "ref": "C000:L0002",
            "claim": "This is the candidate branch.",
        },
        {
            "kind": "diagnostic",
            "ref": "C000:DIAGNOSTIC",
            "claim": "The visible result is wrong.",
        },
    ]


def test_catalog_uses_exact_decoder_and_stable_compact_refs() -> None:
    catalog = make_catalog()

    assert catalog.f2_schema == grounding.EXPECTED_F2_SCHEMA
    assert catalog.architecture == "x86_64"
    assert catalog.entry_block_refs == ("F2B000",)
    assert [block.ref for block in catalog.blocks] == [
        "F2B000",
        "F2B001",
    ]
    assert [instruction.ref for instruction in catalog.instructions] == [
        "F2B000:I000",
        "F2B001:I000",
    ]
    assert [instruction.text for instruction in catalog.instructions] == [
        "jne @B1",
        "ret",
    ]
    assert [
        (
            edge.ref,
            edge.source_ref,
            edge.target_ref,
            edge.edge_type,
        )
        for edge in catalog.edges
    ] == [
        (
            "F2E000",
            "F2B000",
            "F2B001",
            "conditional_true",
        ),
        (
            "F2E001",
            "F2B000",
            "F2B001",
            "conditional_false",
        ),
    ]


def test_prompt_catalog_is_compact_canonical_and_hash_verified() -> None:
    catalog = make_catalog()
    prompt = catalog.to_prompt_dict()

    assert prompt["catalog_sha256"] == grounding.canonical_payload_sha256(
        prompt
    )
    assert catalog.catalog_sha256 == prompt["catalog_sha256"]
    assert prompt["frontier_f2_sha256"] == hashlib.sha256(
        grounding.default_frontier_f2_path().read_bytes()
    ).hexdigest()
    assert prompt["f2_source_sha256"] == hashlib.sha256(
        VALID_F2.encode("utf-8")
    ).hexdigest()
    assert prompt["blocks"][0]["instruction_refs"] == ["F2B000:I000"]
    assert prompt["edges"][0] == {
        "ref": "F2E000",
        "source_ref": "F2B000",
        "target_ref": "F2B001",
        "edge_type": "conditional_true",
    }

    # Do not undo F2 compression by copying expanded assembly into the prompt.
    serialized = json.dumps(prompt, ensure_ascii=False)
    assert "jne @B1" not in serialized
    assert "C000:L9999" not in prompt["allowed_refs"]["candidate_line"]

    # Returning a deep copy prevents a caller from mutating catalog identity.
    prompt["blocks"][0]["ref"] = "corrupt"
    assert catalog.to_prompt_dict()["blocks"][0]["ref"] == "F2B000"


def test_candidate_lines_bound_exact_text_and_boundaries() -> None:
    catalog = make_catalog()
    first = catalog.candidate(0)

    assert first.bof_ref == "C000:BOF"
    assert first.eof_ref == "C000:EOF"
    assert first.diagnostic_ref == "C000:DIAGNOSTIC"
    assert [(line.ref, line.text) for line in first.lines] == [
        ("C000:L0001", "int f(int x) {"),
        ("C000:L0002", "  if (x == 0) return 1;"),
        ("C000:L0003", "  return 0;"),
        ("C000:L0004", "}"),
    ]
    assert "C000:BOF" in catalog.candidate_anchor_refs
    assert "C000:EOF" in catalog.candidate_anchor_refs
    assert "C000:BOF" not in catalog.candidate_line_refs


def test_valid_insert_and_evidence_are_normalized() -> None:
    catalog = make_catalog()
    item = {
        "fault_class": "missing_branch",
        "evidence": valid_evidence(),
        "edit_location": {
            "operation": "insert_after",
            "anchor_ref": "C000:L0002",
            "anchor_text": "  if (x == 0) return 1;",
        },
    }

    result = grounding.validate_diagnosis_item(
        item,
        catalog,
        expected_candidate_index=0,
    )

    assert result.valid is True
    assert result.candidate_index == 0
    assert result.issues == ()
    assert len(result.normalized_evidence) == 3
    assert result.normalized_edit_location == item["edit_location"]


@pytest.mark.parametrize(
    "operation,anchor_ref",
    [
        ("insert_before", "C000:BOF"),
        ("insert_after", "C000:EOF"),
    ],
)
def test_bof_and_eof_are_exact_empty_text_insertion_anchors(
    operation: str,
    anchor_ref: str,
) -> None:
    result = grounding.validate_diagnosis_item(
        {
            "evidence": valid_evidence(),
            "edit_location": {
                "operation": operation,
                "anchor_ref": anchor_ref,
                "anchor_text": "",
            },
        },
        make_catalog(),
        expected_candidate_index=0,
    )

    assert result.valid is True


def test_range_edit_requires_ordered_refs_and_exact_joined_source() -> None:
    catalog = make_catalog()
    base = {
        "evidence": valid_evidence(),
        "edit_location": {
            "operation": "replace_range",
            "start_ref": "C000:L0002",
            "end_ref": "C000:L0003",
            "anchor_text": (
                "  if (x == 0) return 1;\n"
                "  return 0;"
            ),
        },
    }
    assert grounding.validate_diagnosis_item(
        base,
        catalog,
        expected_candidate_index=0,
    ).valid

    reversed_item = {
        **base,
        "edit_location": {
            **base["edit_location"],
            "start_ref": "C000:L0003",
            "end_ref": "C000:L0002",
            "anchor_text": "irrelevant",
        },
    }
    reversed_result = grounding.validate_diagnosis_item(
        reversed_item,
        catalog,
        expected_candidate_index=0,
    )
    assert not reversed_result.valid
    assert "edit_range_order" in reversed_result.rejection_causes

    mismatch_item = {
        **base,
        "edit_location": {
            **base["edit_location"],
            "anchor_text": "same-looking but not exact",
        },
    }
    mismatch_result = grounding.validate_diagnosis_item(
        mismatch_item,
        catalog,
        expected_candidate_index=0,
    )
    assert not mismatch_result.valid
    assert "edit_anchor_text_mismatch" in mismatch_result.rejection_causes


def test_unknown_location_is_valid_without_fabricated_anchor() -> None:
    catalog = make_catalog()
    valid = grounding.validate_diagnosis_item(
        {
            "evidence": valid_evidence(),
            "edit_location": {"operation": "unknown"},
        },
        catalog,
        expected_candidate_index=0,
    )
    assert valid.valid
    assert valid.normalized_edit_location == {"operation": "unknown"}

    fabricated = grounding.validate_diagnosis_item(
        {
            "evidence": valid_evidence(),
            "edit_location": {
                "operation": "unknown",
                "anchor_ref": "C000:L0002",
            },
        },
        catalog,
        expected_candidate_index=0,
    )
    assert not fabricated.valid
    assert "edit_unknown_has_anchor" in fabricated.rejection_causes


def test_refs_are_kind_checked_and_candidate_local() -> None:
    catalog = make_catalog()
    item = {
        "candidate_index": 0,
        "evidence": [
            {
                "kind": "f2_edge",
                "ref": "F2B000",
                "claim": "Wrong reference kind.",
            },
            {
                "kind": "candidate_line",
                "ref": "C001:L0002",
                "claim": "Sibling candidate.",
            },
            {
                "kind": "diagnostic",
                "ref": "C001:DIAGNOSTIC",
                "claim": "Sibling diagnostic.",
            },
        ],
        "edit_location": {
            "operation": "insert_before",
            "anchor_ref": "C001:L0002",
            "anchor_text": "  return x;",
        },
    }

    result = grounding.validate_diagnosis_item(
        item,
        catalog,
        expected_candidate_index=0,
    )

    assert not result.valid
    assert result.rejection_causes == (
        "evidence_ref_unknown",
        "evidence_candidate_mismatch",
        "edit_candidate_mismatch",
    )


def test_one_bad_item_does_not_invalidate_a_valid_sibling() -> None:
    catalog = make_catalog()
    bad = grounding.validate_diagnosis_item(
        {
            "evidence": [
                {
                    "kind": "f2_edge",
                    "ref": "F2E999",
                    "claim": "Invented edge.",
                }
            ],
            "edit_location": {
                "operation": "insert_after",
                "anchor_ref": "C000:L9999",
                "anchor_text": "",
            },
        },
        catalog,
        expected_candidate_index=0,
    )
    good = grounding.validate_diagnosis_item(
        {
            "evidence": [
                {
                    "kind": "f2_instruction",
                    "ref": "F2B001:I000",
                    "claim": "The target returns.",
                }
            ],
            "edit_location": {"operation": "unknown"},
        },
        catalog,
        expected_candidate_index=1,
    )

    assert not bad.valid
    assert "evidence_ref_unknown" in bad.rejection_causes
    assert "edit_anchor_ref_unknown" in bad.rejection_causes
    assert good.valid


def test_malformed_model_shapes_fail_closed_without_raising() -> None:
    catalog = make_catalog()

    not_an_item = grounding.validate_diagnosis_item(
        ["not", "an", "object"],
        catalog,
        expected_candidate_index=0,
    )
    assert not not_an_item.valid
    assert not_an_item.rejection_causes == ("item_not_object",)

    malformed = grounding.validate_diagnosis_item(
        {
            "evidence": [
                {
                    "kind": "f2_block",
                    "ref": "F2B000",
                    "claim": "valid",
                    "unexpected": True,
                }
            ],
            "edit_location": {
                "operation": "insert_after",
                "anchor_ref": "C000:L0001",
                "anchor_text": "int f(int x) {",
                "unexpected": True,
            },
        },
        catalog,
        expected_candidate_index=0,
    )
    assert not malformed.valid
    assert "evidence_shape" in malformed.rejection_causes
    assert "edit_shape" in malformed.rejection_causes


def test_f2_and_decoder_identity_fail_closed() -> None:
    with pytest.raises(grounding.GroundingError, match="exact frontier_f2"):
        grounding.build_grounding_catalog(
            VALID_F2.replace("\nX\n", "\n"),
            ("candidate",),
        )

    with pytest.raises(grounding.GroundingError, match="pinned digest"):
        grounding.build_grounding_catalog(
            VALID_F2,
            ("candidate",),
            expected_frontier_f2_sha256="0" * 64,
        )


def test_diagnostic_refs_exist_only_for_nonempty_supplied_diagnostics() -> None:
    catalog = grounding.build_grounding_catalog(
        VALID_F2,
        ("candidate zero", "candidate one"),
        diagnostics=(None, ""),
    )
    assert catalog.diagnostic_refs == frozenset()

    result = grounding.validate_diagnosis_item(
        {
            "evidence": [
                {
                    "kind": "diagnostic",
                    "ref": "C000:DIAGNOSTIC",
                    "claim": "This ref was never declared.",
                }
            ],
            "edit_location": {"operation": "unknown"},
        },
        catalog,
        expected_candidate_index=0,
    )
    assert not result.valid
    assert result.rejection_causes == ("evidence_ref_unknown",)
