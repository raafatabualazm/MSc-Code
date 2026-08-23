from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import openai56_batch_fasttrack as runner
from frontier_f2 import F2_SYSTEM_PROMPT, serialize_f2


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding([ord(value) for value in text])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in ids)


def _f2_payload() -> str:
    canonical = {
        "architecture": "x86_64",
        "entry_blocks": [0],
        "blocks": [
            {"id": 0, "instructions": ["mov rax,0x1", "jne @B2"]},
            {"id": 1, "instructions": ["xor rax,rax", "ret"]},
            {"id": 2, "instructions": ["ret"]},
        ],
        "cfg_edges": [
            {"source": 0, "target": 2, "edge_type": "conditional_true"},
            {"source": 0, "target": 1, "edge_type": "conditional_false"},
        ],
    }
    return serialize_f2(
        "// constants: [1]\n",
        canonical,
        tokenizer=_CharacterTokenizer(),
        visible_symbols=[chr(value) for value in range(0x4E00, 0x4E80)],
    )


def test_request_contract_is_exact_k2_32k_max_without_sampling() -> None:
    plans = [{"task_id": f"task-{index:03d}"} for index in range(175)]
    specs = runner.request_specs(plans)

    assert len(specs) == 350
    assert len({row["custom_id"] for row in specs}) == 350
    assert specs[0] == {
        "task_id": "task-000",
        "task_index": 0,
        "sample_index": 0,
        "custom_id": "oai56_s00_t000",
    }
    assert specs[174]["custom_id"] == "oai56_s00_t174"
    assert specs[175]["custom_id"] == "oai56_s01_t000"
    assert specs[-1]["custom_id"] == "oai56_s01_t174"

    body = runner.response_body(
        SimpleNamespace(model="gpt-5.6-sol"),
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "payload"},
            ]
        },
    )
    assert body == {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "system"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "payload"}],
            },
        ],
        "max_output_tokens": 32768,
        "reasoning": {"effort": "max"},
        "store": False,
        "truncation": "disabled",
    }
    assert not ({"temperature", "top_p", "seed"} & set(body))


def test_benign_overlay_preserves_f2_bytes_and_decoded_semantics() -> None:
    payload = _f2_payload()
    transformed, proof = runner.build_overlay_messages(
        [
            {"role": "system", "content": F2_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
    )

    assert transformed[0]["content"] == runner.RUNTIME_SYSTEM_PROMPT
    assert transformed[1]["content"] == runner.USER_HEADER + payload
    assert transformed[1]["content"][len(runner.USER_HEADER) :].encode("utf-8") == (
        payload.encode("utf-8")
    )
    assert proof["f2_payload_utf8_bytes_identical"] is True
    assert proof["decoded_f2_semantics_identical"] is True


@pytest.mark.parametrize(
    (
        "model",
        "expected_input",
        "expected_cache_write_input",
        "expected_output",
        "expected_gate_total",
        "expected_cache_write_total",
    ),
    [
        (
            "gpt-5.6-sol",
            "3.999300",
            "4.999125",
            "344.064000",
            "348.063300",
            "349.063125",
        ),
        (
            "gpt-5.6-terra",
            "1.999650",
            "2.499562",
            "172.032000",
            "174.031650",
            "174.531562",
        ),
    ],
)
def test_known_paired_k2_batch_cost_bound(
    model: str,
    expected_input: str,
    expected_cache_write_input: str,
    expected_output: str,
    expected_gate_total: str,
    expected_cache_write_total: str,
) -> None:
    projection = runner.cost_projection(
        model=model,
        exact_input_tokens=1_599_720,
        requests=700,
    )
    assert projection["maximum_output_tokens_all_requests"] == 22_937_600
    assert projection["exact_input_usd"] == expected_input
    assert projection["all_cache_write_input_usd"] == expected_cache_write_input
    assert projection["maximum_output_usd"] == expected_output
    assert projection["exact_worst_case_total_usd"] == expected_gate_total
    assert projection["all_cache_write_total_usd"] == expected_cache_write_total
    assert projection["cost_gate_uses_published_batch_input_rate"] is True


def test_deterministic_shards_preserve_fixed_slot_order_without_loss() -> None:
    specs = [
        {"task_id": "a", "custom_id": "a0"},
        {"task_id": "b", "custom_id": "b0"},
        {"task_id": "a", "custom_id": "a1"},
        {"task_id": "b", "custom_id": "b1"},
    ]
    shards = runner.deterministic_shards(
        specs,
        [
            {"task_id": "a", "input_tokens": 6},
            {"task_id": "b", "input_tokens": 4},
        ],
        input_token_cap=10,
    )

    assert [row["input_tokens"] for row in shards] == [10, 10]
    assert [row["request_count"] for row in shards] == [2, 2]
    assert [
        spec["custom_id"] for shard in shards for spec in shard["request_specs"]
    ] == ["a0", "b0", "a1", "b1"]
    assert all(row["shard_count"] == 2 for row in shards)


def test_native_refusal_is_explicit_and_not_misreported_as_stop() -> None:
    normalized = runner.normalize_responses_body(
        {
            "id": "resp_refusal",
            "model": "gpt-5.6-sol",
            "created_at": 1,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "I cannot assist."}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
            },
        }
    )

    assert normalized["_native_responses_metadata"]["native_refusal_present"] is True
    assert normalized["choices"][0]["finish_reason"] == "content_filter"
    assert normalized["choices"][0]["message"]["refusal"] == "I cannot assist."


def test_cli_rejects_attempts_to_override_fixed_generation_contract() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--attest-authorized-benchmark", "--k", "1"])
    with pytest.raises(SystemExit):
        runner.parse_args(
            ["--attest-authorized-benchmark", "--max-output-tokens", "8192"]
        )


def test_launcher_uses_only_supported_fasttrack_controls() -> None:
    launcher = (PATCH / "run_openai56_batch_ceiling.sh").read_text(encoding="utf-8")
    for forbidden in (
        "--operator-authorization-attestation",
        "--count-workers",
        "--k ",
        "--max-output-tokens",
    ):
        assert forbidden not in launcher
    assert "--attest-authorized-benchmark" in launcher
    assert "--input-token-workers" in launcher
    assert "--shard-input-token-cap" in launcher
    assert 'RUNNER="${RUNNER:-${PATCH_DIR}/openai56_batch_fasttrack.py}"' in launcher
    assert "--max-prompt-tokens 12000" in launcher
    assert 'MODEL="${MODEL:-terra}"' in launcher
    assert 'SHARD_INPUT_TOKEN_CAP="${SHARD_INPUT_TOKEN_CAP:-700000}"' in launcher
    assert '[[ "${ACTION}" == "submit" || "${ACTION}" == "auto" ]]' in launcher


def test_watcher_pair_preflights_then_loops_auto_until_all_summaries() -> None:
    watcher = (PATCH / "watch_openai56_batch_ceiling.sh").read_text(encoding="utf-8")
    preflight = watcher.index("ACTION=preflight")
    loop = watcher.index("while true")
    auto = watcher.index("ACTION=auto")
    assert preflight < loop < auto
    assert "OPENAI56_AUTHORIZE_PAID_BATCH:-0" in watcher
    assert 'if [[ -s "${summary}" ]]' in watcher
    assert "all_shards_submitted_and_harvested" in watcher
    assert "invalid_incomplete_provider_slot_coverage" in watcher
    assert "OPENAI56_BATCH_WATCH_COMPLETE" in watcher
