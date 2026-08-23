from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import frontier_core as core
import frontier_passk as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _build_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        [
            core.COMPACT_F2_SYSTEM_PROMPT,
            "ret x86_64 constants control flow",
        ],
        trainer=trainer,
    )
    tokenizer.add_tokens(
        [chr(value) for value in range(0x4E00, 0x4E80)]
    )
    tokenizer.save(str(path))
    return tokenizer


def _fixture(
    root: Path,
    *,
    rows: int = 1,
    rich_seal: bool = False,
    pair_arm_key: str = "opus_real_fn0_cfg",
    leak_private_source: bool = False,
    mutate_prompt_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_eval_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
    mutate_seal: Callable[[dict[str, Any]], None] | None = None,
    mutate_pair: Callable[[dict[str, Any]], None] | None = None,
) -> SimpleNamespace:
    tokenizer_path = root / "tokenizer.json"
    tokenizer = _build_tokenizer(tokenizer_path)
    system_sha = core.sha256_text(core.COMPACT_F2_SYSTEM_PROMPT)

    prompt_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for index in range(rows):
        task_id = f"heldout_{index}"
        canonical = {
            "architecture": "x86_64",
            "entry_blocks": [0],
            "blocks": [
                {
                    "id": 0,
                    "instructions": ["fn @SELF", "ret"],
                }
            ],
            "cfg_edges": [],
        }
        prefix = f"// numbers: {index}\n"
        if leak_private_source:
            prefix += "void fn0() {}\n"
        text = core.serialize_compact_graph(
            prefix,
            canonical,
            {},
            tokenizer=tokenizer,
            visible_symbols=core.visible_one_token_symbols(tokenizer),
        )
        count = core.count_prompt_tokens(
            [
                {
                    "role": "system",
                    "content": core.COMPACT_F2_SYSTEM_PROMPT,
                },
                {"role": "user", "content": text},
            ],
            tokenizer,
            chat_overhead_reserve=256,
        )
        prompt_row = {
            "schema": core.SCHEMA_VERSION,
            "representation_schema": core.F2_SCHEMA,
            "system_prompt_sha256": system_sha,
            "task_id": task_id,
            "text": text,
            "text_sha256": core.sha256_text(text),
            "compact_ids_sha256": core.stable_sha256([index, 1]),
            "compact_text_sha256": core.sha256_text(f"compact-{index}"),
            "canonical_sha256": core.stable_sha256(canonical),
            "constants_record_sha256": core.stable_sha256(
                {"task_id": task_id}
            ),
            "constants_extraction_error": None,
            "constant_prefix_tokens": 1,
            "graph_tokens": 2,
            "prompt_preflight": count,
            "verified": dict(runner.REQUIRED_F2_ROW_VERIFICATION),
        }
        if pair_arm_key == "codex_multifunction_cfg":
            binding = {
                "schema": "dart-user-symbol-attestation-v1",
                "complete": True,
                "raw_names_present": False,
                "attestation_file_sha256": "a" * 64,
                "attestation_row_sha256": core.stable_sha256(
                    {"task_id": task_id}
                ),
                "key_id_sha256": "b" * 64,
                "function_symbol_count": 1,
                "type_symbol_count": 0,
            }
            prompt_row.update(
                {
                    "source_symbol_attestation_used": True,
                    "source_symbol_attestation_is_keyed": True,
                    "source_symbol_attestation_binding": binding,
                    "source_symbol_attestation_binding_sha256": (
                        core.stable_sha256(binding)
                    ),
                }
            )
            prompt_row["verified"].update(
                {
                    "all_user_functions_retained": True,
                    "all_external_symbols_retained": True,
                    "transfer_table_redundancy_proven": True,
                    "keyed_source_symbol_attestation_bound": True,
                    "raw_source_names_not_serialized": True,
                }
            )
        prompt_rows.append(prompt_row)
        tests = "void main() { fn0(); }\n"
        eval_rows.append(
            {
                "task_id": task_id,
                "function": "fn0",
                "lang": "Dart",
                "dart_source": "void fn0() {}\n",
                "tests": tests,
                "acceptance_tests": tests,
            }
        )

    if mutate_prompt_rows is not None:
        mutate_prompt_rows(prompt_rows)
    if mutate_eval_rows is not None:
        mutate_eval_rows(eval_rows)

    prompt_path = root / "prompts.jsonl"
    eval_path = root / "eval.jsonl"
    _write_jsonl(prompt_path, prompt_rows)
    _write_jsonl(eval_path, eval_rows)
    task_ids = [str(row["task_id"]) for row in prompt_rows]
    maximum = max(
        prompt_rows,
        key=lambda row: row["prompt_preflight"][
            "estimated_prompt_tokens"
        ],
    )
    artifacts = {
        "tokenizer": _file_binding(tokenizer_path),
    }
    if rich_seal:
        artifacts["frontier_f2"] = _file_binding(
            PATCH / "frontier_f2.py"
        )
    manifest = {
        "schema": runner.F2_MANIFEST_SCHEMA,
        "rows": len(prompt_rows),
        "task_set_sha256": core.stable_sha256(task_ids),
        "dataset": _file_binding(eval_path),
        "output": _file_binding(prompt_path),
        "artifacts": artifacts,
        "binary_constant_extraction_errors": {
            "count": 0,
            "task_ids": [],
        },
        "f2_prompt_contract": {
            "representation_schema": core.F2_SCHEMA,
            "system_prompt": core.COMPACT_F2_SYSTEM_PROMPT,
            "system_prompt_sha256": system_sha,
            "tokenizer_sha256": _sha(tokenizer_path),
            "max_prompt_tokens": 12000,
            "chat_overhead_reserve": 256,
            "maximum_estimated_prompt_tokens": maximum[
                "prompt_preflight"
            ]["estimated_prompt_tokens"],
            "maximum_task_id": maximum["task_id"],
            "all_rows_within_limit": True,
        },
        "invariants": {
            name: True
            for name in runner.REQUIRED_F2_MANIFEST_INVARIANTS
        },
    }
    if pair_arm_key == "codex_multifunction_cfg":
        manifest["invariants"].update(
            {
                "all_user_functions_retained": True,
                "all_external_symbols_retained": True,
                "transfer_table_redundancy_proven": True,
                "keyed_private_source_symbol_attestation_used": True,
                "raw_source_names_not_serialized": True,
            }
        )
        manifest["source_symbol_attestation"] = {
            "used": True,
            "is_keyed": True,
            "raw_names_serialized": False,
        }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_path = root / "prompts.jsonl.manifest.json"
    _write_json(manifest_path, manifest)

    seal: dict[str, Any] = {
        "schema": runner.EVAL_SEAL_SCHEMA,
        "selected_role": "measure",
        "rows": len(eval_rows),
        "output_sha256": _sha(eval_path),
        "contract_sha256": "a" * 64,
    }
    if rich_seal:
        seal.update(
            {
                "training_allowed": False,
                "heldout_measure_only": True,
                "task_set_sha256": core.stable_sha256(
                    [str(row["task_id"]) for row in eval_rows]
                ),
                "output": _file_binding(eval_path),
                "f2_output": _file_binding(prompt_path),
                "f2_manifest": _file_binding(manifest_path),
                "frontier_f2_schema": core.F2_SCHEMA,
                "completion_attestation_id": (
                    runner.REQUIRED_ATTESTATION_ID
                ),
            }
        )
    if mutate_seal is not None:
        mutate_seal(seal)
    seal_path = root / "eval.seal.json"
    _write_json(seal_path, seal)

    acceptance_hashes = [
        core.sha256_text(str(row["acceptance_tests"]))
        for row in eval_rows
    ]
    arm_record = {
        "eval": _file_binding(eval_path),
        "seal": _file_binding(seal_path),
        "prompts": _file_binding(prompt_path),
        "prompt_manifest": _file_binding(manifest_path),
    }
    pair_manifest = {
        "schema": runner.PAIR_MANIFEST_SCHEMA,
        "rows": rows,
        "ordered_task_ids_sha256": core.stable_sha256(task_ids),
        "ordered_acceptance_test_hashes_sha256": core.stable_sha256(
            acceptance_hashes
        ),
        "system_prompt_sha256": system_sha,
        "arms": {
            "opus_real_fn0_cfg": dict(arm_record),
            "codex_multifunction_cfg": dict(arm_record),
        },
        "comparison": {
            "opus": "synthetic Opus fixture",
            "codex": "synthetic Codex fixture",
        },
        "invariants": {
            name: True for name in runner.REQUIRED_PAIR_INVARIANTS
        },
    }
    if mutate_pair is not None:
        mutate_pair(pair_manifest)
    pair_manifest_path = root / "pair_manifest.json"
    _write_json(pair_manifest_path, pair_manifest)

    return SimpleNamespace(
        prompt_jsonl=prompt_path,
        prompt_manifest=manifest_path,
        eval_jsonl=eval_path,
        eval_seal=seal_path,
        pair_manifest=pair_manifest_path,
        pair_arm_key=pair_arm_key,
        expected_prompt_jsonl_sha256=_sha(prompt_path),
        expected_prompt_manifest_sha256=_sha(manifest_path),
        expected_eval_jsonl_sha256=_sha(eval_path),
        expected_eval_seal_sha256=_sha(seal_path),
        expected_pair_manifest_sha256=_sha(pair_manifest_path),
        expected_task_count=rows,
        tokenizer_json=tokenizer_path,
        max_prompt_tokens=12000,
        chat_overhead_reserve=256,
    )


def _complete_runner_args(value: SimpleNamespace) -> SimpleNamespace:
    value.provider = "deepseek"
    value.model = "deepseek-v4-pro"
    value.arm = "compact"
    value.input_mode = "prematerialized_f2"
    value.k = 10
    value.workers = 2
    value.limit = 0
    value.max_output_tokens = 12000
    value.budget = 0
    value.temperature = 0.8
    value.top_p = 0.95
    value.timeout_seconds = 600
    value.max_attempts_per_sample = 6
    value.retry_base_seconds = 2.0
    value.retry_max_seconds = 30.0
    value.eval_timeout_seconds = 30
    value.eval_stability_runs = 2
    value.dataset_label = "fixture"
    value.extra_body = {}
    value.evaluator_module = PATCH / "frontier_passk.py"
    value.expected_evaluator_sha256 = "a" * 64
    value.dart = PATCH / "frontier_passk.py"
    value.expected_dart_sha256 = "b" * 64
    value.api_key = ""
    value.base_url = "https://api.deepseek.com"
    value.deepseek_env_file = PATCH / "does-not-exist.env"
    value.resume = True
    return value


@pytest.mark.parametrize("rich_seal", [False, True])
def test_valid_prematerialized_f2_pair_passes(
    tmp_path: Path,
    rich_seal: bool,
) -> None:
    args = _fixture(tmp_path, rows=2, rich_seal=rich_seal)
    value = runner.validate_prematerialized_f2_inputs(args)
    assert value["task_ids"] == ["heldout_0", "heldout_1"]
    assert value["task_set_sha256"] == core.stable_sha256(
        value["task_ids"]
    )
    assert len(value["eval_rows"]) == 2
    assert value["eval_rows"][0]["acceptance_tests"] == value[
        "eval_rows"
    ][0]["tests"]


def test_valid_codex_pair_requires_keyed_multifunction_proofs(
    tmp_path: Path,
) -> None:
    args = _fixture(
        tmp_path,
        rich_seal=True,
        pair_arm_key="codex_multifunction_cfg",
    )
    value = runner.validate_prematerialized_f2_inputs(args)
    assert value["pair_arm_key"] == "codex_multifunction_cfg"
    assert value["prompt_rows"][0][
        "source_symbol_attestation_is_keyed"
    ] is True


def test_paid_config_fingerprint_binds_pair_and_runtime_identity(
    tmp_path: Path,
) -> None:
    args = _complete_runner_args(_fixture(tmp_path))
    config = runner.config_for_hash(args)
    assert config["sealed_inputs"]["pair_manifest_sha256"] == (
        args.expected_pair_manifest_sha256
    )
    assert config["sealed_inputs"]["pair_arm_key"] == "opus_real_fn0_cfg"
    assert config["runtime_identity"]["runner_sha256"] == _sha(
        Path(runner.__file__)
    )
    assert config["runtime_identity"]["core_sha256"] == _sha(
        Path(core.__file__)
    )
    assert config["runtime_identity"]["frontier_f2_sha256"] == _sha(
        PATCH / "frontier_f2.py"
    )


def test_per_row_text_hash_tamper_fails_closed(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["text"] += "\n"

    args = _fixture(tmp_path, mutate_prompt_rows=mutate)
    with pytest.raises(core.PreflightError, match="text SHA-256"):
        runner.validate_prematerialized_f2_inputs(args)


def test_ordered_prompt_eval_task_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows.reverse()

    args = _fixture(tmp_path, rows=2, mutate_eval_rows=mutate)
    with pytest.raises(core.PreflightError, match="ordered prompt task IDs"):
        runner.validate_prematerialized_f2_inputs(args)


def test_embedded_token_count_tamper_fails_closed(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["prompt_preflight"]["user_tokens"] += 1

    args = _fixture(tmp_path, mutate_prompt_rows=mutate)
    with pytest.raises(core.PreflightError, match="token count user_tokens"):
        runner.validate_prematerialized_f2_inputs(args)


def test_false_manifest_invariant_fails_closed(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["invariants"]["cfg_explicit"] = False

    args = _fixture(tmp_path, mutate_manifest=mutate)
    with pytest.raises(core.PreflightError, match="cfg_explicit"):
        runner.validate_prematerialized_f2_inputs(args)


def test_training_split_seal_fails_closed(tmp_path: Path) -> None:
    def mutate(seal: dict[str, Any]) -> None:
        seal["selected_role"] = "fit"

    args = _fixture(tmp_path, mutate_seal=mutate)
    with pytest.raises(core.PreflightError, match="selected_role"):
        runner.validate_prematerialized_f2_inputs(args)


def test_private_fields_in_prompt_rows_fail_closed(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["tests"] = "private"

    args = _fixture(tmp_path, mutate_prompt_rows=mutate)
    with pytest.raises(core.PreflightError, match="private field"):
        runner.validate_prematerialized_f2_inputs(args)


def test_missing_rich_seal_f2_binding_is_rejected_when_declared(
    tmp_path: Path,
) -> None:
    def mutate(seal: dict[str, Any]) -> None:
        seal["f2_output"]["sha256"] = "0" * 64

    args = _fixture(tmp_path, rich_seal=True, mutate_seal=mutate)
    with pytest.raises(core.PreflightError, match="seal.f2_output"):
        runner.validate_prematerialized_f2_inputs(args)


def test_pair_manifest_selected_arm_binding_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    def mutate(pair: dict[str, Any]) -> None:
        pair["arms"]["opus_real_fn0_cfg"]["prompts"]["sha256"] = "0" * 64

    args = _fixture(tmp_path, mutate_pair=mutate)
    with pytest.raises(core.PreflightError, match="pair manifest arm"):
        runner.validate_prematerialized_f2_inputs(args)


def test_pair_acceptance_sequence_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    def mutate(pair: dict[str, Any]) -> None:
        pair["ordered_acceptance_test_hashes_sha256"] = "0" * 64

    args = _fixture(tmp_path, mutate_pair=mutate)
    with pytest.raises(
        core.PreflightError,
        match="acceptance-test sequence",
    ):
        runner.validate_prematerialized_f2_inputs(args)


def test_exact_private_source_embedded_in_f2_fails_closed(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path, leak_private_source=True)
    with pytest.raises(core.PreflightError, match="exact private dart_source"):
        runner.validate_prematerialized_f2_inputs(args)


def test_codex_missing_keyed_attestation_fails_closed(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["source_symbol_attestation_is_keyed"] = False

    args = _fixture(
        tmp_path,
        rich_seal=True,
        pair_arm_key="codex_multifunction_cfg",
        mutate_prompt_rows=mutate,
    )
    with pytest.raises(core.PreflightError, match="is not keyed"):
        runner.validate_prematerialized_f2_inputs(args)


def test_no_resume_rejects_orphan_attempt_journal(tmp_path: Path) -> None:
    (tmp_path / "attempts.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.RunFailure, match="existing run state"):
        runner.enforce_output_state_policy(
            SimpleNamespace(resume=False),
            tmp_path,
        )


def test_resume_policy_allows_existing_state_when_explicit(
    tmp_path: Path,
) -> None:
    (tmp_path / "attempts.jsonl").write_text("{}\n", encoding="utf-8")
    runner.enforce_output_state_policy(
        SimpleNamespace(resume=True),
        tmp_path,
    )
