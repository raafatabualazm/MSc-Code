from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import seq2seq_verpo_core as core


class RecordingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_kwargs,
    ) -> SimpleNamespace:
        self.calls += 1
        hidden = (
            input_ids.float() * attention_mask.to(dtype=torch.float32) * self.scale
        ).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class RecordingSeq2SeqLM(torch.nn.Module):
    """A tiny conditioned seq2seq policy with no Transformers dependency."""

    def __init__(self, vocab_size: int = 12, max_steps: int = 8) -> None:
        super().__init__()
        # Deliberately keep BOS distinct from PAD. T5Gemma's composite config
        # exposes these values under config.decoder.
        self.config = SimpleNamespace(
            decoder=SimpleNamespace(
                bos_token_id=2,
                pad_token_id=0,
                eos_token_id=1,
            )
        )
        self.encoder = RecordingEncoder()
        table = torch.arange(
            max_steps * vocab_size,
            dtype=torch.float32,
        ).reshape(max_steps, vocab_size)
        self.step_logits = torch.nn.Parameter(table / 37.0)
        self.register_buffer(
            "context_direction",
            torch.linspace(-0.2, 0.2, vocab_size),
        )
        self.last_decoder_input_ids: torch.Tensor | None = None
        self.last_decoder_attention_mask: torch.Tensor | None = None
        self.last_encoder_outputs: object | None = None
        self.last_logits: torch.Tensor | None = None
        self.last_use_cache: bool | None = None

    def get_encoder(self) -> RecordingEncoder:
        return self.encoder

    def prepare_decoder_input_ids_from_labels(
        self,
        *,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shifted = labels.new_full(labels.shape, 0)
        shifted[:, 0] = 2
        shifted[:, 1:] = labels[:, :-1]
        shifted.masked_fill_(shifted == -100, 0)
        return shifted

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        encoder_outputs: object | None = None,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        use_cache: bool,
        **_kwargs,
    ) -> SimpleNamespace:
        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("missing encoder input")
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden = encoder_outputs.last_hidden_state
        denominator = attention_mask.sum(dim=1).clamp(min=1).float()
        context = (hidden.squeeze(-1) * attention_mask.float()).sum(dim=1) / denominator
        length = decoder_input_ids.size(1)
        logits = (
            self.step_logits[:length]
            .unsqueeze(0)
            .expand(
                decoder_input_ids.size(0),
                -1,
                -1,
            )
        )
        logits = logits + (
            context[:, None, None] * self.context_direction[None, None, :]
        )
        # Make a shift error observable in the returned distribution as well
        # as in the recorded call.
        previous_token_bias = torch.nn.functional.one_hot(
            (decoder_input_ids + 1) % logits.size(-1),
            num_classes=logits.size(-1),
        ).float()
        logits = logits + 0.3 * previous_token_bias

        self.last_decoder_input_ids = decoder_input_ids.detach().clone()
        self.last_decoder_attention_mask = decoder_attention_mask.detach().clone()
        self.last_encoder_outputs = encoder_outputs
        self.last_logits = logits.detach().clone()
        self.last_use_cache = use_cache
        return SimpleNamespace(logits=logits)


def detail(
    passes: list[bool],
    *,
    compiled: bool,
    full_pass: bool = False,
    diagnostic: str = "",
) -> dict[str, object]:
    return {
        "test_passes": passes,
        "compiled": compiled,
        "full_pass": full_pass,
        "diagnostic": diagnostic,
    }


def test_generated_prefix_is_not_an_action_and_real_eos_is_retained() -> None:
    assert core.normalize_generated_seq2seq_ids(
        [2, 7, 8, 1, 0, 0],
        decoder_prefix_ids=[2],
        eos_token_ids=1,
        pad_token_id=0,
    ) == [7, 8, 1]
    # Max-length output without EOS remains visibly unterminated. The helper
    # must not manufacture EOS merely to make batch shapes convenient.
    assert core.normalize_generated_seq2seq_ids(
        [2, 7, 8],
        decoder_prefix_ids=[2],
        eos_token_ids=1,
        pad_token_id=0,
    ) == [7, 8]
    # T5-family models can use PAD as their decoder prefix. Remove only that
    # exact leading conditioning token.
    assert core.normalize_generated_seq2seq_ids(
        [0, 7, 1, 0],
        decoder_prefix_ids=[0],
        eos_token_ids=1,
        pad_token_id=0,
    ) == [7, 1]
    with pytest.raises(ValueError, match="exact decoder prefix"):
        core.normalize_generated_seq2seq_ids(
            [0, 7, 1],
            decoder_prefix_ids=[2],
            eos_token_ids=1,
            pad_token_id=0,
        )
    with pytest.raises(RuntimeError, match="sampled PAD before EOS"):
        core.normalize_generated_seq2seq_ids(
            [2, 7, 0, 8, 1],
            decoder_prefix_ids=[2],
            eos_token_ids=1,
            pad_token_id=0,
        )


def test_action_mask_includes_first_eos_and_excludes_pad_or_tail() -> None:
    targets = torch.tensor(
        [
            [7, 1, 9, 0],
            [7, 8, 0, 0],
            [7, 8, 9, 10],
        ]
    )
    actual = core.decoder_action_mask(
        targets,
        eos_token_ids=1,
        pad_token_id=0,
    )
    assert actual.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [True, True, True, True],
    ]
    same_pad_and_eos = core.decoder_action_mask(
        torch.tensor([7, 0, 0]),
        eos_token_ids=0,
        pad_token_id=0,
    )
    assert same_pad_and_eos.tolist() == [True, True, False]


def test_logprobs_are_teacher_shifted_temperature_scaled_and_include_eos() -> None:
    model = RecordingSeq2SeqLM().eval()
    completion = torch.tensor([7, 8, 1, 0])
    values = core.seq2seq_completion_token_logprobs(
        model,
        completion,
        encoder_input_ids=torch.tensor([4, 5, 999]),
        encoder_attention_mask=torch.tensor([1, 1, 0]),
        temperature=0.5,
        pad_token_id=0,
        eos_token_ids=1,
        with_grad=False,
    )

    assert model.last_decoder_input_ids.tolist() == [[2, 7, 8, 1]]
    assert model.last_decoder_attention_mask.tolist() == [[1, 1, 1, 0]]
    assert model.last_use_cache is False
    assert values.shape == (3,)
    expected = (
        torch.log_softmax(model.last_logits.float() / 0.5, dim=-1)
        .gather(-1, completion.view(1, -1, 1))
        .squeeze(-1)[0, :3]
    )
    torch.testing.assert_close(values, expected)

    model.zero_grad(set_to_none=True)
    differentiable = core.seq2seq_completion_token_logprobs(
        model,
        completion,
        encoder_input_ids=torch.tensor([4, 5, 999]),
        encoder_attention_mask=torch.tensor([1, 1, 0]),
        temperature=0.5,
        pad_token_id=0,
        eos_token_ids=1,
        with_grad=True,
    )
    (-differentiable.mean()).backward()
    assert model.step_logits.grad is not None
    # If the trainer declares the encoder trainable, update-side encoder
    # outputs must not be silently detached.
    assert model.encoder.scale.grad is not None


def test_suppressed_pad_is_removed_from_recomputed_policy_support() -> None:
    model = RecordingSeq2SeqLM().eval()
    completion = torch.tensor([7, 8, 1])
    values = core.seq2seq_completion_token_logprobs(
        model,
        completion,
        encoder_input_ids=torch.tensor([4, 5]),
        encoder_attention_mask=torch.tensor([1, 1]),
        temperature=0.8,
        pad_token_id=0,
        eos_token_ids=1,
        suppressed_token_ids=[0],
        with_grad=False,
    )
    expected_logits = model.last_logits.float() / 0.8
    expected_logits[..., 0] = float("-inf")
    expected = (
        torch.log_softmax(expected_logits, dim=-1)
        .gather(-1, completion.view(1, -1, 1))
        .squeeze(-1)[0]
    )
    torch.testing.assert_close(values, expected)

    with pytest.raises(RuntimeError, match="removed from sampling support"):
        core.seq2seq_completion_token_logprobs(
            model,
            [7, 0],
            encoder_input_ids=torch.tensor([4, 5]),
            encoder_attention_mask=torch.tensor([1, 1]),
            temperature=0.8,
            pad_token_id=11,
            eos_token_ids=1,
            suppressed_token_ids=[0],
            with_grad=False,
        )
    with pytest.raises(ValueError, match="EOS cannot"):
        core.seq2seq_completion_token_logprobs(
            model,
            completion,
            encoder_input_ids=torch.tensor([4, 5]),
            encoder_attention_mask=torch.tensor([1, 1]),
            temperature=0.8,
            pad_token_id=0,
            eos_token_ids=1,
            suppressed_token_ids=[1],
            with_grad=False,
        )


def test_fixed_encoder_output_is_reused_and_conditioning_changes_policy() -> None:
    model = RecordingSeq2SeqLM().eval()
    mask = torch.tensor([[1, 1, 0]])
    source_a = torch.tensor([[3, 4, 999]])
    source_b = torch.tensor([[8, 9, 999]])
    with torch.no_grad():
        encoded_a = model.get_encoder()(
            input_ids=source_a,
            attention_mask=mask,
        )
    assert model.encoder.calls == 1

    first = core.seq2seq_completion_token_logprobs(
        model,
        [7, 1],
        encoder_outputs=encoded_a,
        encoder_attention_mask=mask,
        temperature=1.0,
        pad_token_id=0,
        eos_token_ids=1,
        with_grad=False,
    )
    second = core.seq2seq_completion_token_logprobs(
        model,
        [7, 1],
        encoder_outputs=encoded_a,
        encoder_attention_mask=mask,
        temperature=1.0,
        pad_token_id=0,
        eos_token_ids=1,
        with_grad=False,
    )
    torch.testing.assert_close(first, second)
    assert model.encoder.calls == 1
    assert model.last_encoder_outputs is encoded_a

    with torch.no_grad():
        encoded_b = model.get_encoder()(
            input_ids=source_b,
            attention_mask=mask,
        )
    conditioned_b = core.seq2seq_completion_token_logprobs(
        model,
        [7, 1],
        encoder_outputs=encoded_b,
        encoder_attention_mask=mask,
        temperature=1.0,
        pad_token_id=0,
        eos_token_ids=1,
        with_grad=False,
    )
    assert model.encoder.calls == 2
    assert not torch.allclose(first, conditioned_b)


def test_compiler_sanitizer_keeps_actionable_semantics_not_oracles() -> None:
    raw = (
        "\x1b[31mC:\\tmp\\task\\test.dart:12:7: Error: "
        "The operator '==' isn't defined for the type 'Widget'.\x1b[0m\n"
        "Expected: <42>\n"
        "Actual: <7>\n"
        "Input: [1, 2, 3]\n"
        "2 != 1\n"
        "\x00"
    )
    clean = core.sanitize_compiler_diagnostic(raw)
    assert "<path>/test.dart:12:7" in clean
    assert "operator '=='" in clean
    assert "isn't defined" in clean
    assert "type 'Widget'" in clean
    assert "test-oracle values redacted" in clean
    for forbidden in ("Expected:", "Actual:", "Input:", "<42>", "<7>", "2 != 1"):
        assert forbidden not in clean
    assert "\x1b" not in clean
    assert "\x00" not in clean

    long = "\n".join(f"Error: useful compiler line {index}" for index in range(50))
    first = core.sanitize_compiler_diagnostic(long, max_chars=120)
    second = core.sanitize_compiler_diagnostic(long, max_chars=120)
    assert first == second
    assert len(first) <= 120
    assert "middle omitted" in first


def test_repair_context_is_canonical_hash_bound_and_compiler_only() -> None:
    candidate = (
        'int fn0(int x) => x + ; // "},' '"acceptance_tests":"injected text only"'
    )
    kwargs = {
        "task_id": "task-7",
        "source_sha256": "a" * 64,
        "candidate": candidate,
        "diagnostic": (
            "C:\\tmp\\test.dart:1:22: Error: Expected an expression.\n"
            "Expected: <99>\nActual: <0>"
        ),
        "compiled": False,
    }
    first = core.build_compiler_repair_context(**kwargs)
    second = core.build_compiler_repair_context(**kwargs)
    assert first == second
    assert first["text_sha256"] == core.sha256_text(first["text"])
    marker, serialized = first["text"].split("\n", 1)
    assert marker == core.COMPILER_REPAIR_MARKER
    decoded = json.loads(serialized)
    assert decoded == first["payload"]
    assert decoded["candidate"] == candidate
    assert decoded["candidate_sha256"] == core.sha256_text(candidate)
    assert decoded["feedback_kind"] == "compiler_only"
    assert "99" not in decoded["compiler_feedback"]
    assert "judge_feedback" not in decoded
    assert "feedback_tests" not in decoded
    assert "reference" not in decoded
    # Injection-looking candidate text remains a string value and cannot add
    # a key to the canonical object.
    assert "acceptance_tests" not in decoded.keys()

    with pytest.raises(ValueError, match="non-compiling"):
        core.build_compiler_repair_context(**{**kwargs, "compiled": True})


def test_compile_advantage_breaks_only_verifiable_ties() -> None:
    dead = [
        detail([False, False], compiled=False, diagnostic="one"),
        detail([False, False], compiled=False, diagnostic="two"),
    ]
    dead_result = core.verpo_execution_compile_advantages(
        dead,
        alpha=2.0,
        local_weight=1.0,
        compile_weight=0.25,
    )
    assert dead_result["unified_advantages"] == [0.0, 0.0]

    mixed = [
        detail([False, False], compiled=False, diagnostic="syntax A"),
        detail([False, False], compiled=True, diagnostic="runtime B"),
        detail([True, False], compiled=True, diagnostic="partial C"),
    ]
    result = core.verpo_execution_compile_advantages(
        mixed,
        alpha=2.0,
        local_weight=1.0,
        compile_weight=0.25,
    )
    advantages = result["unified_advantages"]
    assert advantages[2] > advantages[1] > advantages[0]
    assert sum(advantages) == pytest.approx(0.0)
    assert sum(result["compile_advantages"]) == pytest.approx(0.0)

    changed_words = [dict(row, diagnostic="completely different") for row in mixed]
    assert (
        core.verpo_execution_compile_advantages(
            changed_words,
            alpha=2.0,
            local_weight=1.0,
            compile_weight=0.25,
        )
        == result
    )


def test_combined_failure_can_coexist_with_all_isolated_cases_passing() -> None:
    stateful = [
        {
            "compiled": True,
            "full_pass": False,
            "test_passes": [True, True],
        },
        {
            "compiled": True,
            "full_pass": False,
            "test_passes": [True, False],
        },
    ]
    result = core.verpo_execution_compile_advantages(
        stateful,
        alpha=2.0,
        local_weight=1.0,
        compile_weight=0.25,
    )
    assert result["global_rewards"] == [0.0, 0.0]
    assert result["local_rewards"][0] > result["local_rewards"][1]
    assert result["unified_advantages"][0] > 0.0
    assert result["unified_advantages"][1] < 0.0

    for impossible in (
        {
            "compiled": False,
            "full_pass": True,
            "test_passes": [True, True],
        },
        {
            "compiled": True,
            "full_pass": True,
            "test_passes": [True, False],
        },
    ):
        with pytest.raises(
            ValueError,
            match="full-pass flag disagrees",
        ):
            core.verpo_execution_compile_advantages(
                [impossible],
                alpha=2.0,
                local_weight=1.0,
                compile_weight=0.25,
            )


def test_diverse_selection_is_content_based_and_permutation_stable() -> None:
    candidates = [
        "int f() { return 0; }",
        "int f() { return 1; }",
        "int f() { while (true) { break; } return 9; }",
    ]
    selected = core.max_min_diverse_indices(candidates, 2)
    permuted = [candidates[2], candidates[0], candidates[1]]
    selected_permuted = core.max_min_diverse_indices(permuted, 2)
    assert {candidates[index] for index in selected} == {
        permuted[index] for index in selected_permuted
    }
    assert core.code_distance(candidates[0], candidates[0]) == 0.0
    assert 0.0 < core.code_distance(candidates[0], candidates[2]) <= 1.0


def test_on_policy_drift_ignores_masked_tail_and_binds_conditioning() -> None:
    saved = torch.tensor([[-1.0, -2.0, -100.0]])
    current = torch.tensor([[-1.0, -2.0, 100.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    assert (
        core.validate_on_policy_logprob_drift(
            current,
            saved,
            tolerance=1e-4,
            action_mask=mask,
            rollout_conditioning_sha256="a" * 64,
            current_conditioning_sha256="a" * 64,
            rollout_temperature=0.8,
            current_temperature=0.8,
        )
        == 0.0
    )

    changed = current.detach().clone()
    changed[0, 1] += 0.01
    with pytest.raises(RuntimeError, match="max drift"):
        core.validate_on_policy_logprob_drift(
            changed,
            saved,
            tolerance=1e-4,
            action_mask=mask,
        )
    with pytest.raises(RuntimeError, match="conditioning differs"):
        core.validate_on_policy_logprob_drift(
            current,
            saved,
            tolerance=1e-4,
            action_mask=mask,
            rollout_conditioning_sha256="a" * 64,
            current_conditioning_sha256="b" * 64,
        )
    with pytest.raises(RuntimeError, match="temperature differs"):
        core.validate_on_policy_logprob_drift(
            current,
            saved,
            tolerance=1e-4,
            action_mask=mask,
            rollout_temperature=0.8,
            current_temperature=1.0,
        )
