from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import t5gemma2_enriched_sft as trainer


def _f2_row(task_id: str, text: str) -> dict[str, object]:
    return {
        "schema": trainer.F2_ROW_SCHEMA,
        "representation_schema": trainer.REPRESENTATION_SCHEMA,
        "task_id": task_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "verified": dict(trainer._REQUIRED_F2_ATTESTATIONS),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def test_dataset_join_and_encoder_source_are_content_sealed(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "train.jsonl"
    f2_path = tmp_path / "f2.jsonl"
    dataset_rows = [
        {
            "task_id": "t0",
            "dart_source": "int fn0(int x) => x + 7;",
            "acceptance_tests": "private and never serialized",
        },
        {
            "task_id": "t1",
            "dart_source": "String fn0(String x) => x.toUpperCase();",
            "holdback_tests": "also private",
        },
    ]
    f2_rows = [
        _f2_row("t0", "CONST 7\nCFG B0 -> RET"),
        _f2_row("t1", "CALL toUpperCase\nCFG B0 -> RET"),
    ]
    # Expanded multi-function rows may add further attestations. They are
    # accepted only after the exact base attestation values pass.
    f2_rows[1]["verified"]["all_user_functions_retained"] = True  # type: ignore[index]
    _write_jsonl(dataset_path, dataset_rows)
    _write_jsonl(f2_path, f2_rows)

    pairs, manifest = trainer.load_text_pairs(
        dataset_path,
        f2_path,
        expected_dataset_sha256=trainer.sha256_file(dataset_path),
        expected_f2_sha256=trainer.sha256_file(f2_path),
        expected_rows=2,
    )

    assert [pair.task_id for pair in pairs] == ["t0", "t1"]
    assert "CONST 7" in pairs[0].source
    assert "private and never serialized" not in pairs[0].source
    assert "int fn0" not in pairs[0].source
    assert pairs[0].target == "int fn0(int x) => x + 7;"
    assert manifest["model_visible_fields"] == ["F2.text"]
    assert "path" not in manifest["dataset"]
    assert "path" not in manifest["f2"]

    missing_digest = copy.deepcopy(f2_rows[0])
    missing_digest.pop("text_sha256")
    with pytest.raises(ValueError, match="text digest mismatch"):
        trainer.build_encoder_source(missing_digest, "t0")

    wrong_absence_attestation = copy.deepcopy(f2_rows[0])
    wrong_absence_attestation["verified"]["opaque_custom_ids_in_text"] = True
    with pytest.raises(ValueError, match="verification contract failed"):
        trainer.build_encoder_source(wrong_absence_attestation, "t0")

    leaked_target = copy.deepcopy(f2_rows[0])
    leaked_target["dart_source"] = "int fn0() => 7;"
    with pytest.raises(ValueError, match="forbidden field"):
        trainer.build_encoder_source(leaked_target, "t0")

    wrong_producer = copy.deepcopy(f2_rows[0])
    wrong_producer["schema"] = "unsealed"
    with pytest.raises(ValueError, match="producer schema"):
        trainer.build_encoder_source(wrong_producer, "t0")


class _RecordingTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        padding: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        assert truncation is False
        assert padding is False
        assert return_attention_mask is False
        self.calls.append(add_special_tokens)
        body = [10 + index for index, _ in enumerate(text)]
        if add_special_tokens:
            body = [self.bos_token_id, *body]
        return {"input_ids": body}


def _pair(source: str = "abc", target: str = "xy") -> trainer.TextPair:
    return trainer.TextPair(
        task_id="t0",
        source=source,
        target=target,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        target_sha256=hashlib.sha256(target.encode()).hexdigest(),
    )


def test_tokenization_never_truncates_and_labels_exactly_one_eos() -> None:
    tokenizer = _RecordingTokenizer()
    rows, report = trainer.tokenize_pairs(
        tokenizer,
        [_pair()],
        max_source_tokens=4,
        max_target_tokens=3,
    )

    assert tokenizer.calls == [True, False]
    assert rows[0].input_ids == (2, 10, 11, 12)
    assert rows[0].labels == (10, 11, 1)
    assert rows[0].labels.count(tokenizer.eos_token_id) == 1
    assert rows[0].labels[-1] == tokenizer.eos_token_id
    assert report["truncated_rows"] == 0

    with pytest.raises(ValueError, match="source truncation is forbidden"):
        trainer.tokenize_pairs(
            _RecordingTokenizer(),
            [_pair()],
            max_source_tokens=3,
            max_target_tokens=3,
        )
    with pytest.raises(ValueError, match="target truncation is forbidden"):
        trainer.tokenize_pairs(
            _RecordingTokenizer(),
            [_pair()],
            max_source_tokens=4,
            max_target_tokens=2,
        )


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2)
        self.k_proj = torch.nn.Linear(2, 2)
        self.v_proj = torch.nn.Linear(2, 2)
        self.o_proj = torch.nn.Linear(2, 2)


class _Mlp(torch.nn.Module):
    def __init__(self, *, include_down: bool = True) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 2)
        self.up_proj = torch.nn.Linear(2, 2)
        if include_down:
            self.down_proj = torch.nn.Linear(2, 2)


class _Layer(torch.nn.Module):
    def __init__(self, *, include_down: bool = True) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp(include_down=include_down)


class _DummyT5Gemma(torch.nn.Module):
    def __init__(self, *, decoder_down: bool = True) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.encoder = torch.nn.Module()
        self.model.encoder.text_model = torch.nn.Module()
        self.model.encoder.text_model.layers = torch.nn.ModuleList([_Layer()])
        self.model.encoder.vision_model = torch.nn.Module()
        self.model.encoder.vision_model.layers = torch.nn.ModuleList([_Layer()])
        self.model.decoder = torch.nn.Module()
        self.model.decoder.layers = torch.nn.ModuleList(
            [_Layer(include_down=decoder_down)]
        )


def _lora_args(targets: str) -> SimpleNamespace:
    return SimpleNamespace(lora_target_modules=targets)


def test_lora_targets_are_exactly_text_encoder_and_decoder() -> None:
    requested = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    targets = trainer._resolve_lora_targets(
        _DummyT5Gemma(),
        _lora_args(requested),
    )

    assert len(targets) == 14
    assert all("vision" not in name for name in targets)
    assert sum(".encoder.text_model.layers." in f".{name}" for name in targets) == 7
    assert sum(".decoder.layers." in f".{name}" for name in targets) == 7
    assert not any(name.endswith("lm_head") for name in targets)

    with pytest.raises(ValueError, match="unsupported LoRA target"):
        trainer._resolve_lora_targets(
            _DummyT5Gemma(),
            _lora_args("q_proj,lm_head"),
        )
    with pytest.raises(ValueError, match="missing_by_side"):
        trainer._resolve_lora_targets(
            _DummyT5Gemma(decoder_down=False),
            _lora_args(requested),
        )


def test_t5gemma_capacity_and_schedule_are_exact_and_deterministic() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="t5gemma2",
            is_encoder_decoder=True,
            encoder=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=131_072)
            ),
            decoder=SimpleNamespace(max_position_embeddings=32_768),
        )
    )
    assert trainer._config_position_capacities(model) == (131_072, 32_768)
    model.config.model_type = "gemma3"
    with pytest.raises(ValueError, match="not a native T5Gemma 2"):
        trainer._config_position_capacities(model)

    schedule = trainer.calculate_training_schedule(
        rows=10,
        epochs=2,
        batch_size=3,
        gradient_accumulation=3,
        max_updates=3,
        warmup_ratio=0.25,
    )
    assert schedule == {
        "microbatches_per_epoch": 4,
        "updates_per_epoch": 2,
        "available_updates": 4,
        "planned_updates": 3,
        "warmup_updates": 1,
    }
    assert trainer.cosine_schedule_multiplier(
        0,
        warmup_updates=1,
        total_updates=3,
    ) == pytest.approx(1e-8)
    assert trainer.cosine_schedule_multiplier(
        1,
        warmup_updates=1,
        total_updates=3,
    ) == pytest.approx(1.0)
    assert trainer.cosine_schedule_multiplier(
        3,
        warmup_updates=1,
        total_updates=3,
    ) == pytest.approx(0.1)

    rows = [
        trainer.TokenizedPair(
            task_id=f"t{index}",
            input_ids=(index + 3,),
            labels=(index + 4, 1),
        )
        for index in range(12)
    ]
    first = trainer.deterministic_epoch_order(rows, seed=42, epoch=0)
    assert first == trainer.deterministic_epoch_order(rows, seed=42, epoch=0)
    assert sorted(first) == list(range(12))
    assert first != trainer.deterministic_epoch_order(
        rows,
        seed=42,
        epoch=1,
    )


def test_hf_loaders_pass_only_supported_explicit_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[str, dict[str, object]]] = {}

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(
            cls,
            name: str,
            **kwargs: object,
        ) -> _FakeTokenizer:
            calls["tokenizer"] = (name, kwargs)
            return _FakeTokenizer()

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="t5gemma2",
            is_encoder_decoder=True,
            encoder=SimpleNamespace(text_config=SimpleNamespace()),
            decoder=SimpleNamespace(),
        )
    )

    class _AutoModel:
        @classmethod
        def from_pretrained(
            cls,
            name: str,
            **kwargs: object,
        ) -> object:
            calls["model"] = (name, kwargs)
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_AutoTokenizer,
            AutoModelForSeq2SeqLM=_AutoModel,
        ),
    )
    loaded_tokenizer = trainer._load_tokenizer(
        "google/t5gemma-2-4b-4b",
        "a" * 40,
        None,
    )
    loaded_model = trainer._load_base_model(
        SimpleNamespace(
            model="google/t5gemma-2-4b-4b",
            model_revision="a" * 40,
            bf16=True,
            attn_implementation="sdpa",
        ),
        None,
    )

    assert loaded_tokenizer is not None
    assert loaded_model is model
    tokenizer_kwargs = calls["tokenizer"][1]
    model_kwargs = calls["model"][1]
    assert "token" not in tokenizer_kwargs
    assert "token" not in model_kwargs
    assert "low_cpu_mem_usage" not in model_kwargs
    assert tokenizer_kwargs == {
        "trust_remote_code": False,
        "use_fast": True,
        "revision": "a" * 40,
    }
    assert model_kwargs["dtype"] is torch.bfloat16
    assert model_kwargs["attn_implementation"] == "sdpa"
    assert model_kwargs["trust_remote_code"] is False
    assert model_kwargs["use_safetensors"] is True
    assert model_kwargs["revision"] == "a" * 40


def _retention_contract(*, foreign: bool = False) -> dict[str, object]:
    contract: dict[str, object] = {
        "schema": trainer.RUN_SCHEMA,
        "status": "training",
        "lora": {"targets": ["model.encoder.text_model.layers.0.q_proj"]},
    }
    if foreign:
        contract["foreign_run"] = True
    return contract


def _write_retention_checkpoint(
    root: Path,
    *,
    update: int,
    contract: dict[str, object],
    complete: bool = True,
) -> Path:
    checkpoint = root / trainer._checkpoint_name(update)
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    trainer._atomic_json(checkpoint / "run_contract.json", contract)
    trainer._atomic_json(
        checkpoint / "adapter" / "adapter_config.json",
        {
            "task_type": "SEQ_2_SEQ_LM",
            "target_modules": contract["lora"]["targets"],  # type: ignore[index]
        },
    )
    trainer._atomic_json(
        checkpoint / "tokenizer" / "tokenizer_config.json",
        {"eos_token": "</s>"},
    )
    if complete:
        from safetensors.torch import save_file

        adapter_tensors = {}
        for target in contract["lora"]["targets"]:  # type: ignore[index]
            adapter_tensors[f"base_model.model.{target}.lora_A.weight"] = torch.zeros(
                1, 1
            )
            adapter_tensors[f"base_model.model.{target}.lora_B.weight"] = torch.zeros(
                1, 1
            )
        save_file(
            adapter_tensors,
            checkpoint / "adapter" / "adapter_model.safetensors",
        )
    torch.save(
        {
            "schema": trainer.CHECKPOINT_SCHEMA,
            "update": update,
            "epoch": 0,
            "next_row": 0,
            "optimizer": {},
            "scheduler": {},
            "rng": {},
            "run_contract_sha256": trainer.canonical_sha256(contract),
        },
        checkpoint / "training_state.pt",
    )
    return checkpoint


def test_resume_audits_weight_keys_when_peft_minimizes_target_modules(
    tmp_path: Path,
) -> None:
    contract = _retention_contract()
    checkpoint = _write_retention_checkpoint(
        tmp_path,
        update=1,
        contract=contract,
    )
    adapter_config_path = checkpoint / "adapter" / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["target_modules"] = ["q_proj"]
    trainer._atomic_json(adapter_config_path, adapter_config)

    saved_contract, state = trainer._load_resume_artifacts(
        checkpoint,
        exact_targets=contract["lora"]["targets"],  # type: ignore[arg-type,index]
        weights_only=True,
    )

    assert saved_contract == contract
    assert state["update"] == 1


def test_resume_contract_allows_only_an_explicit_trainer_source_migration() -> None:
    old_hash = "1" * 64
    new_hash = "2" * 64
    saved = {
        "schema": trainer.RUN_SCHEMA,
        "status": "training",
        "runtime": {"trainer_sha256": old_hash, "torch": "x"},
        "dataset": {"sha256": "3" * 64},
    }
    current = copy.deepcopy(saved)
    current["runtime"]["trainer_sha256"] = new_hash

    migrated = trainer._bind_resume_contract(
        current,
        saved,
        expected_legacy_trainer_sha256=old_hash,
    )

    assert migrated["resume_migration"]["accepted_contract_sha256"] == (
        trainer.canonical_sha256(saved)
    )
    assert migrated["resume_migration"]["from_trainer_sha256"] == old_hash
    assert migrated["resume_migration"]["to_trainer_sha256"] == new_hash

    with pytest.raises(ValueError, match="exact expected legacy hash"):
        trainer._bind_resume_contract(
            current,
            saved,
            expected_legacy_trainer_sha256="4" * 64,
        )
    changed_data = copy.deepcopy(current)
    changed_data["dataset"]["sha256"] = "5" * 64
    with pytest.raises(ValueError, match="more than"):
        trainer._bind_resume_contract(
            changed_data,
            saved,
            expected_legacy_trainer_sha256=old_hash,
        )

    resumed_again = trainer._bind_resume_contract(
        migrated,
        migrated,
        expected_legacy_trainer_sha256=old_hash,
    )
    assert resumed_again == migrated


@pytest.mark.parametrize("bad_kind", ["foreign", "incomplete", "symlink"])
def test_checkpoint_pruning_fails_before_deleting_any_valid_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_kind: str,
) -> None:
    root_contract = _retention_contract()
    trainer._atomic_json(tmp_path / "run_contract.json", root_contract)
    first = _write_retention_checkpoint(
        tmp_path,
        update=1,
        contract=root_contract,
    )
    second = _write_retention_checkpoint(
        tmp_path,
        update=2,
        contract=root_contract,
    )
    bad_contract = (
        _retention_contract(foreign=True) if bad_kind == "foreign" else root_contract
    )
    bad = _write_retention_checkpoint(
        tmp_path,
        update=3,
        contract=bad_contract,
        complete=bad_kind != "incomplete",
    )
    if bad_kind == "symlink":
        real_is_link_like = trainer._is_link_like

        def report_bad_as_link(path: Path) -> bool:
            return path == bad or real_is_link_like(path)

        monkeypatch.setattr(trainer, "_is_link_like", report_bad_as_link)

    with pytest.raises((ValueError, FileNotFoundError)):
        trainer.prune_checkpoints(
            output_dir=tmp_path,
            keep_last=1,
            run_contract=root_contract,
        )

    assert first.is_dir()
    assert second.is_dir()
    assert bad.is_dir()


def test_checkpoint_pruning_keeps_newest_exact_run_checkpoints(
    tmp_path: Path,
) -> None:
    contract = _retention_contract()
    trainer._atomic_json(tmp_path / "run_contract.json", contract)
    checkpoints = [
        _write_retention_checkpoint(
            tmp_path,
            update=update,
            contract=contract,
        )
        for update in (1, 2, 3)
    ]

    removed = trainer.prune_checkpoints(
        output_dir=tmp_path,
        keep_last=2,
        run_contract=contract,
    )

    assert removed == [checkpoints[0]]
    assert not checkpoints[0].exists()
    assert checkpoints[1].is_dir()
    assert checkpoints[2].is_dir()
