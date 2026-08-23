import json
from pathlib import Path

import pytest

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation import t5gemma2_f2_passk_verpo_compat as compat


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, dict, set[str]]:
    targets = {"model.encoder.q_proj", "model.decoder.v_proj"}
    warm = tmp_path / "warm"
    checkpoint = tmp_path / "verpo"
    warm_contract = {
        "schema": compat.SFT_RUN_SCHEMA,
        "architecture": "native_encoder_decoder",
        "status": "training",
        "base_model": {
            "name": inference.MODEL_NAME,
            "requested_revision": inference.MODEL_REVISION,
            "resolved_commit": inference.MODEL_REVISION,
            "is_encoder_decoder": True,
            "config_sha256": "config-hash",
        },
        "lora": {"targets": sorted(targets)},
    }
    _write_json(warm / "run_contract.json", warm_contract)
    (warm / "tokenizer").mkdir()
    (warm / "tokenizer" / "tokenizer.json").write_text("same", encoding="utf-8")

    code = {
        "trainer": {
            "relative_path": "scripts/training/trainer.py",
            "sha256": "abc",
        }
    }
    contract = {
        "schema": compat.VERPO_RUN_SCHEMA,
        "architecture": "native_t5gemma2_encoder_decoder",
        "objective": "on_policy_visible_execution_verpo_plus_local_compiler_repair",
        "no_frontier_api": True,
        "llm_judge": False,
        "acceptance_tests_exposed": False,
        "private_holdback_exposed": False,
        "feedback_boundary": {
            "schema": "verpo-train-feedback-view-v1",
            "acceptance_tests_exposed": False,
            "reward_holdback_exposed": False,
            "heldout_bytes_opened_during_validation": False,
            "parent_or_private_bytes_opened_during_validation": False,
        },
        "runtime_provenance": {
            "schema": compat.VERPO_RUNTIME_SCHEMA,
            "code": code,
            "code_bundle_sha256": inference.canonical_sha256(code),
        },
        "warmstart": {
            "path": str(warm.resolve()),
            "stage_schema": compat.SFT_RUN_SCHEMA,
            "run_contract_sha256": inference.canonical_sha256(warm_contract),
            "production_floor_eligible": True,
        },
    }
    _write_json(checkpoint / "run_contract.json", contract)
    _write_json(checkpoint / "adapter" / "adapter_config.json", {"r": 64})
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "tokenizer" / "tokenizer.json").write_text(
        "same", encoding="utf-8"
    )
    return checkpoint, contract, targets


def test_verpo_loader_inherits_pinned_base_and_lora_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, contract, targets = _fixture(tmp_path)
    monkeypatch.setattr(
        inference, "_adapter_weight_target_modules", lambda _checkpoint: targets
    )

    observed, record = compat._verpo_checkpoint_record(checkpoint, "sft")

    assert observed == contract
    assert record["training_stage_schema"] == compat.VERPO_RUN_SCHEMA
    assert record["revision"] == inference.MODEL_REVISION
    assert record["adapter"]["target_modules"] == len(targets)
    assert (
        record["adapter"]["run_contract_sha256"]
        == inference.canonical_sha256(contract)
    )
    assert (
        record["tokenizer_sha256"]
        == record["source_warmstart"]["tokenizer_sha256"]
    )


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("acceptance_tests_exposed", True),
        ("private_holdback_exposed", True),
        ("no_frontier_api", False),
    ],
)
def test_verpo_loader_fails_closed_on_privacy_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    unsafe_value: bool,
) -> None:
    checkpoint, contract, targets = _fixture(tmp_path)
    contract[field] = unsafe_value
    _write_json(checkpoint / "run_contract.json", contract)
    monkeypatch.setattr(
        inference, "_adapter_weight_target_modules", lambda _checkpoint: targets
    )

    with pytest.raises(ValueError, match="privacy"):
        compat._verpo_checkpoint_record(checkpoint, "sft")


def test_verpo_loader_rejects_tokenizer_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _contract, targets = _fixture(tmp_path)
    (checkpoint / "tokenizer" / "tokenizer.json").write_text(
        "changed", encoding="utf-8"
    )
    monkeypatch.setattr(
        inference, "_adapter_weight_target_modules", lambda _checkpoint: targets
    )

    with pytest.raises(ValueError, match="tokenizer"):
        compat._verpo_checkpoint_record(checkpoint, "sft")
