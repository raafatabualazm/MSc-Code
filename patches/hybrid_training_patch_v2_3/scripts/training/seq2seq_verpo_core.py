"""Pure helpers for native encoder-decoder VeRPO.

This module deliberately contains no model loading, API calls, Dart execution,
or checkpoint I/O.  The trainer owns those concerns.  These helpers seal the
parts of a seq2seq policy update that are easy to get subtly wrong:

* generated decoder-prefix removal and action masking;
* teacher-shifted decoder log-probabilities, including the sampled EOS;
* execution, partial-test, and compile-relative advantages;
* deterministic code-diversity selection for flat groups;
* compiler-only feedback sanitization and canonical repair conditioning; and
* masked, context-bound on-policy drift validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch


COMPILER_REPAIR_SCHEMA = "seq2seq-verpo-compiler-repair-v1"
COMPILER_REPAIR_ARTIFACT_SCHEMA = "seq2seq-verpo-compiler-repair-artifact-v1"
COMPILER_REPAIR_MARKER = "COMPILER_REPAIR_CONTEXT_JSON"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
_PATH_RE = re.compile(
    r"(?:(?:[A-Za-z]:)?[\\/])" r"(?:[^\\/\s:\r\n]+[\\/])+" r"([^\\/\s:\r\n]+)"
)
_ORACLE_LABEL_RE = re.compile(
    r"\b(expected|actual|input|output|received|wanted|got|" r"value|values)\b\s*[:=]",
    flags=re.IGNORECASE,
)
_ASSERTION_LINE_RE = re.compile(
    r"\b(expect\s*\(|assert\s*\(|matcher|comparison failed)\b",
    flags=re.IGNORECASE,
)
_BARE_COMPARISON_RE = re.compile(
    r"^(?:unhandled exception:\s*)?"
    r"(?:[-+]?\d+(?:\.\d+)?|true|false|null|"
    r"<[^>\r\n]{0,200}>|\[[^\]\r\n]{0,200}\]|"
    r"\{[^}\r\n]{0,200}\}|['\"][^'\"\r\n]{0,200}['\"]|"
    r"[A-Za-z_]\w*)"
    r"\s*(?:!=|==)\s*"
    r"(?:[-+]?\d+(?:\.\d+)?|true|false|null|"
    r"<[^>\r\n]{0,200}>|\[[^\]\r\n]{0,200}\]|"
    r"\{[^}\r\n]{0,200}\}|['\"][^'\"\r\n]{0,200}['\"]|"
    r"[A-Za-z_]\w*)$",
    flags=re.IGNORECASE,
)
_CODE_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?|"
    r"===|!==|==|!=|<=|>=|=>|&&|\|\||\+\+|--|<<|>>|"
    r"[^\s]",
    re.UNICODE,
)


def canonical_json(value: Any) -> str:
    """Return a stable, UTF-8-preserving JSON serialization."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plain_token_id(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer token ID")
    return value


def _token_id_list(value: Any, label: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        values = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise ValueError(f"{label} must be a token-ID sequence")
    return [
        _plain_token_id(token, f"{label}[{index}]")
        for index, token in enumerate(values)
    ]


def _eos_token_ids(value: int | Sequence[int]) -> tuple[int, ...]:
    raw = [value] if type(value) is int else _token_id_list(value, "EOS IDs")
    ids = tuple(_plain_token_id(item, "EOS ID") for item in raw)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("EOS IDs must be a non-empty unique set")
    return ids


def normalize_generated_seq2seq_ids(
    generated_ids: Sequence[int] | torch.Tensor,
    *,
    decoder_prefix_ids: Sequence[int] | torch.Tensor,
    eos_token_ids: int | Sequence[int],
    pad_token_id: int,
) -> list[int]:
    """Remove the exact decoder prefix and retain only sampled actions.

    Hugging Face encoder-decoder ``generate`` returns the supplied decoder
    prefix in ``sequences``.  Those prefix tokens are conditioning, not policy
    actions.  The first sampled EOS is retained because its probability is a
    policy action; synthetic PAD and everything after EOS/PAD are excluded.
    No EOS is fabricated for a max-length generation.
    """

    generated = _token_id_list(generated_ids, "generated IDs")
    prefix = _token_id_list(decoder_prefix_ids, "decoder prefix IDs")
    eos_ids = set(_eos_token_ids(eos_token_ids))
    pad_id = _plain_token_id(pad_token_id, "PAD ID")
    if not prefix:
        raise ValueError("decoder prefix must be non-empty")
    if generated[: len(prefix)] != prefix:
        raise ValueError(
            "generated sequence does not begin with the exact decoder prefix"
        )

    actions: list[int] = []
    for token_id in generated[len(prefix) :]:
        if token_id in eos_ids:
            actions.append(token_id)
            break
        if token_id == pad_id:
            raise RuntimeError(
                "generated sequence contains sampled PAD before EOS; "
                "refusing an unscored policy action"
            )
        actions.append(token_id)
    if not actions:
        raise ValueError("seq2seq rollout produced no sampled actions")
    return actions


def decoder_action_mask(
    target_ids: torch.Tensor,
    *,
    eos_token_ids: int | Sequence[int],
    pad_token_id: int,
) -> torch.Tensor:
    """Mask sampled targets through the first EOS, excluding synthetic PAD."""

    if not isinstance(target_ids, torch.Tensor) or target_ids.ndim not in {
        1,
        2,
    }:
        raise ValueError("target IDs must be a one- or two-dimensional tensor")
    if target_ids.dtype == torch.bool or target_ids.is_floating_point():
        raise ValueError("target IDs must use an integer tensor dtype")
    if bool((target_ids < 0).any()):
        raise ValueError("target IDs must be non-negative")

    eos_ids = _eos_token_ids(eos_token_ids)
    pad_id = _plain_token_id(pad_token_id, "PAD ID")
    eos = torch.zeros_like(target_ids, dtype=torch.bool)
    for eos_id in eos_ids:
        eos |= target_ids == eos_id
    pad = target_ids == pad_id
    terminal = eos | pad
    terminal_count = terminal.to(torch.int64).cumsum(dim=-1)
    # The first EOS is an action.  PAD is never an action.  When PAD and EOS
    # are the same token, treating the first occurrence as EOS is the only
    # faithful interpretation of a generated terminal token.
    valid = (terminal_count == 0) | (eos & (terminal_count == 1))
    return valid


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encoder_output_device(encoder_outputs: Any) -> torch.device | None:
    hidden = None
    if isinstance(encoder_outputs, Mapping):
        hidden = encoder_outputs.get("last_hidden_state")
    else:
        hidden = getattr(encoder_outputs, "last_hidden_state", None)
        if hidden is None and isinstance(encoder_outputs, tuple):
            hidden = encoder_outputs[0] if encoder_outputs else None
    return hidden.device if isinstance(hidden, torch.Tensor) else None


def _as_single_batch(
    value: Sequence[int] | torch.Tensor,
    *,
    label: str,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise ValueError(f"{label} must use an integer tensor dtype")
        tensor = tensor.to(dtype=torch.long)
    else:
        tensor = torch.tensor(
            [_token_id_list(value, label)],
            dtype=torch.long,
            device=device,
        )
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.size(0) != 1 or tensor.size(1) == 0:
        raise ValueError(f"{label} must have shape [1, nonzero_length]")
    if bool((tensor < 0).any()):
        raise ValueError(f"{label} must be non-negative")
    return tensor


def _resolve_decoder_start_token_id(
    model: torch.nn.Module,
    explicit: int | None,
) -> int:
    if explicit is not None:
        return _plain_token_id(explicit, "decoder start token ID")
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "decoder_start_token_id", None),
        getattr(getattr(config, "decoder", None), "bos_token_id", None),
        getattr(config, "bos_token_id", None),
    ]
    for candidate in candidates:
        if type(candidate) is int and candidate >= 0:
            return candidate
    raise ValueError(
        "model has no explicit decoder start/BOS token; refusing PAD fallback"
    )


def prepare_shifted_decoder_input_ids(
    model: torch.nn.Module,
    target_ids: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int | None = None,
) -> torch.Tensor:
    """Teacher-shift action targets using the model's native helper if present."""

    if (
        target_ids.ndim != 2
        or action_mask.shape != target_ids.shape
        or action_mask.dtype != torch.bool
    ):
        raise ValueError("target IDs and boolean action mask must have equal 2D shape")
    pad_id = _plain_token_id(pad_token_id, "PAD ID")
    labels = target_ids.clone()
    labels.masked_fill_(~action_mask, -100)
    prepare = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    if callable(prepare):
        shifted = prepare(labels=labels)
    else:
        start_id = _resolve_decoder_start_token_id(model, decoder_start_token_id)
        shifted = target_ids.new_full(target_ids.shape, pad_id)
        shifted[:, 0] = start_id
        shifted[:, 1:] = labels[:, :-1]
        shifted.masked_fill_(shifted == -100, pad_id)
    if not isinstance(shifted, torch.Tensor) or shifted.shape != target_ids.shape:
        raise ValueError("model returned invalid shifted decoder input IDs")
    return shifted.to(device=target_ids.device, dtype=torch.long)


def seq2seq_completion_token_logprobs(
    model: torch.nn.Module,
    completion_ids: Sequence[int] | torch.Tensor,
    *,
    encoder_attention_mask: Sequence[int] | torch.Tensor,
    temperature: float,
    pad_token_id: int,
    eos_token_ids: int | Sequence[int],
    encoder_input_ids: Sequence[int] | torch.Tensor | None = None,
    encoder_outputs: Any | None = None,
    decoder_start_token_id: int | None = None,
    suppressed_token_ids: Sequence[int] | None = None,
    with_grad: bool,
) -> torch.Tensor:
    """Score sampled decoder actions under the exact untruncated policy.

    Exactly one of ``encoder_input_ids`` and ``encoder_outputs`` must be
    supplied.  The returned vector contains only real sampled actions and
    includes the first EOS.
    """

    if (encoder_input_ids is None) == (encoder_outputs is None):
        raise ValueError("provide exactly one of encoder_input_ids or encoder_outputs")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    suppressed_ids = (
        tuple()
        if suppressed_token_ids is None
        else tuple(_token_id_list(suppressed_token_ids, "suppressed token IDs"))
    )
    if len(set(suppressed_ids)) != len(suppressed_ids):
        raise ValueError("suppressed token IDs must be unique")
    if set(suppressed_ids) & set(_eos_token_ids(eos_token_ids)):
        raise ValueError("EOS cannot be removed from sampling support")

    device = (
        _encoder_output_device(encoder_outputs) if encoder_outputs is not None else None
    ) or _model_device(model)
    targets = _as_single_batch(
        completion_ids,
        label="completion IDs",
        device=device,
    )
    encoder_mask = _as_single_batch(
        encoder_attention_mask,
        label="encoder attention mask",
        device=device,
    )
    if bool(((encoder_mask != 0) & (encoder_mask != 1)).any()):
        raise ValueError("encoder attention mask must be binary")
    if encoder_input_ids is not None:
        encoder_ids = _as_single_batch(
            encoder_input_ids,
            label="encoder input IDs",
            device=device,
        )
        if encoder_ids.shape != encoder_mask.shape:
            raise ValueError("encoder input IDs/mask shapes differ")
    else:
        encoder_ids = None

    action_mask = decoder_action_mask(
        targets,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
    )
    if not bool(action_mask.any()):
        raise ValueError("completion contains no scoreable decoder action")
    scoreable_targets = targets[action_mask]
    if suppressed_ids and any(
        bool((scoreable_targets == token_id).any()) for token_id in suppressed_ids
    ):
        raise RuntimeError("completion contains a token removed from sampling support")
    decoder_input_ids = prepare_shifted_decoder_input_ids(
        model,
        targets,
        action_mask,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )
    decoder_attention_mask = action_mask.to(dtype=torch.long)

    kwargs: dict[str, Any] = {
        "attention_mask": encoder_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "use_cache": False,
    }
    if encoder_outputs is not None:
        kwargs["encoder_outputs"] = encoder_outputs
    else:
        kwargs["input_ids"] = encoder_ids

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        outputs = model(**kwargs)
        logits = (
            outputs.get("logits")
            if isinstance(outputs, Mapping)
            else getattr(outputs, "logits", None)
        )
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != targets.shape
        ):
            raise ValueError("seq2seq model returned misaligned decoder logits")
        if int(targets.max()) >= logits.size(-1):
            raise ValueError("completion token lies outside decoder vocabulary")
        if any(token_id >= logits.size(-1) for token_id in suppressed_ids):
            raise ValueError("suppressed token lies outside decoder vocabulary")
        scaled_logits = logits.float() / temperature
        if suppressed_ids:
            support_mask = torch.zeros(
                logits.size(-1),
                dtype=torch.bool,
                device=logits.device,
            )
            support_mask[list(suppressed_ids)] = True
            scaled_logits = scaled_logits.masked_fill(
                support_mask.view(1, 1, -1),
                float("-inf"),
            )
        selected = (
            torch.log_softmax(scaled_logits, dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )
        values = selected[action_mask]
    if values.ndim != 1 or values.numel() != int(action_mask.sum()):
        raise RuntimeError("seq2seq action log-probability alignment failed")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("seq2seq action log-probabilities are non-finite")
    return values


def mean_centered_advantages(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("cannot center an empty reward group")
    floats = [float(value) for value in values]
    if any(not math.isfinite(value) for value in floats):
        raise ValueError("rewards must be finite")
    mean = sum(floats) / len(floats)
    return [value - mean for value in floats]


def verpo_local_rewards(
    group_details: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    density_norm: bool = True,
    epsilon: float = 1e-8,
) -> list[float]:
    """Compute VeRPO density-calibrated partial-test rewards."""

    if not group_details:
        raise ValueError("VeRPO needs a non-empty group")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("VeRPO alpha must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("VeRPO epsilon must be finite and positive")
    n_tests = len(group_details[0].get("test_passes") or [])
    if n_tests <= 0:
        raise ValueError("VeRPO group has no per-test evidence")

    matrix: list[list[bool]] = []
    for detail in group_details:
        raw_passes = detail.get("test_passes")
        if (
            not isinstance(raw_passes, list)
            or len(raw_passes) != n_tests
            or any(type(value) is not bool for value in raw_passes)
        ):
            raise ValueError("VeRPO group has inconsistent per-test vectors")
        compiled = detail.get("compiled")
        full_pass = detail.get("full_pass")
        if type(compiled) is not bool or type(full_pass) is not bool:
            raise ValueError("VeRPO compile/full-pass flags must be booleans")
        # The combined harness and the isolated cases are separate execution
        # observations.  A stateful candidate can fail when all cases run in
        # one process while every case passes in a fresh process.  Conversely,
        # a combined full pass proves that every isolated case passed because
        # the completion attestation is emitted only after the whole harness.
        if full_pass and (not compiled or not all(raw_passes)):
            raise ValueError("VeRPO full-pass flag disagrees with test evidence")
        matrix.append(list(raw_passes))

    group_size = len(matrix)
    rho = [
        sum(matrix[row][test] for row in range(group_size)) / group_size
        for test in range(n_tests)
    ]
    weights = [math.exp(-alpha * value) for value in rho]
    if density_norm:
        mean = sum(rho) / n_tests
        variance = sum((value - mean) ** 2 for value in rho) / n_tests
        sigma = math.sqrt(variance) / 2.0
        if sigma <= epsilon:
            densities = [float(n_tests)] * n_tests
        else:
            denominator = 2.0 * sigma * sigma
            densities = [
                sum(
                    math.exp(-((rho[left] - rho[right]) ** 2) / denominator)
                    for right in range(n_tests)
                )
                for left in range(n_tests)
            ]
        weights = [
            weight / (density + epsilon)
            for weight, density in zip(weights, densities, strict=True)
        ]
    return [
        sum(weights[test] for test in range(n_tests) if matrix[row][test])
        for row in range(group_size)
    ]


def verpo_execution_compile_advantages(
    group_details: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    local_weight: float,
    compile_weight: float,
    density_norm: bool = True,
) -> dict[str, list[float]]:
    """Fuse separately centered full-pass, local-test, and compile signals.

    Compiler output text never enters the scalar reward.  The compile term is
    a verifiable boolean tie-breaker, independently centered so it cannot
    change the group's zero-mean baseline.
    """

    if (
        not math.isfinite(local_weight)
        or not math.isfinite(compile_weight)
        or local_weight < 0.0
        or compile_weight < 0.0
    ):
        raise ValueError("local and compile weights must be finite/non-negative")
    local_rewards = verpo_local_rewards(
        group_details,
        alpha=alpha,
        density_norm=density_norm,
    )
    global_rewards = [float(bool(detail["full_pass"])) for detail in group_details]
    compile_rewards = [float(bool(detail["compiled"])) for detail in group_details]
    global_advantages = mean_centered_advantages(global_rewards)
    local_advantages = mean_centered_advantages(local_rewards)
    compile_advantages = mean_centered_advantages(compile_rewards)
    unified = [
        global_value + local_weight * local_value + compile_weight * compile_value
        for global_value, local_value, compile_value in zip(
            global_advantages,
            local_advantages,
            compile_advantages,
            strict=True,
        )
    ]
    return {
        "global_rewards": global_rewards,
        "local_rewards": local_rewards,
        "compile_rewards": compile_rewards,
        "global_advantages": global_advantages,
        "local_advantages": local_advantages,
        "compile_advantages": compile_advantages,
        "unified_advantages": unified,
    }


def _normalized_code_tokens(code: str) -> tuple[str, ...]:
    if not isinstance(code, str):
        raise ValueError("candidate code must be text")
    without_block = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    without_line = re.sub(r"//[^\r\n]*", " ", without_block)
    return tuple(token.lower() for token in _CODE_TOKEN_RE.findall(without_line))


def _token_ngrams(code: str, width: int = 3) -> frozenset[tuple[str, ...]]:
    if type(width) is not int or width <= 0:
        raise ValueError("n-gram width must be a positive integer")
    tokens = _normalized_code_tokens(code)
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset((token,) for token in tokens)
    return frozenset(
        tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)
    )


def code_distance(left: str, right: str) -> float:
    """Return normalized token-trigram Jaccard distance in ``[0, 1]``."""

    first = _token_ngrams(left)
    second = _token_ngrams(right)
    if not first and not second:
        return 0.0
    return 1.0 - len(first & second) / len(first | second)


def max_min_diverse_indices(
    candidates: Sequence[str],
    k: int,
) -> list[int]:
    """Select a deterministic maximum-minimum-diversity candidate subset."""

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("candidates must be a sequence")
    if type(k) is not int or k <= 0:
        raise ValueError("diversity k must be a positive integer")
    values = list(candidates)
    if len(values) < k:
        raise ValueError("cannot select more candidates than available")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("candidate code must be text")

    hashes = [sha256_text(value) for value in values]
    cache: dict[tuple[int, int], float] = {}

    def distance(left: int, right: int) -> float:
        key = (min(left, right), max(left, right))
        if key not in cache:
            cache[key] = code_distance(values[left], values[right])
        return cache[key]

    means = [
        sum(distance(index, other) for other in range(len(values)) if other != index)
        / max(1, len(values) - 1)
        for index in range(len(values))
    ]
    if k == 1:
        return [
            min(
                range(len(values)),
                key=lambda index: (-means[index], hashes[index], index),
            )
        ]

    pairs = [
        (left, right)
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    left, right = min(
        pairs,
        key=lambda pair: (
            -distance(*pair),
            tuple(sorted((hashes[pair[0]], hashes[pair[1]]))),
            pair,
        ),
    )
    selected = sorted((left, right), key=lambda index: (hashes[index], index))
    while len(selected) < k:
        remaining = [index for index in range(len(values)) if index not in selected]
        selected.append(
            min(
                remaining,
                key=lambda index: (
                    -min(distance(index, member) for member in selected),
                    -means[index],
                    hashes[index],
                    index,
                ),
            )
        )
    return selected


def sanitize_compiler_diagnostic(
    text: str,
    *,
    max_chars: int = 4000,
) -> str:
    """Remove paths/test-oracle material while preserving compiler semantics.

    Unlike the general verifier sanitizer, this compiler-only path does not
    blanket-redact quoted text.  Dart uses quotes for useful operators, types,
    identifiers, and punctuation, and an apostrophe in ``isn't`` is not a
    string delimiter.
    """

    if not isinstance(text, str):
        raise ValueError("compiler diagnostic must be text")
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    normalized = text.replace("\x00", "")
    normalized = _ANSI_ESCAPE_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _PATH_RE.sub(r"<path>/\1", normalized)

    output: list[str] = []
    redacted_oracle = False
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if (
            _ORACLE_LABEL_RE.search(line)
            or _ASSERTION_LINE_RE.search(line)
            or _BARE_COMPARISON_RE.fullmatch(line)
        ):
            redacted_oracle = True
            continue
        if "candidate(" in line or re.fullmatch(r"[\s^~|+-]+", line):
            redacted_oracle = True
            continue
        # Runtime stack frames do not explain a static compiler failure and
        # can contain machine-local paths.
        if re.match(r"^#\d+\s+", line):
            continue
        output.append(line)
    if redacted_oracle:
        output.append("[test-oracle values redacted]")
    compact = "\n".join(output).strip()
    if not compact:
        compact = "compiler_failed_without_safe_diagnostic"
    if len(compact) <= max_chars:
        return compact

    marker = "\n... <compiler diagnostic middle omitted> ...\n"
    if len(marker) >= max_chars:
        return compact[:max_chars]
    available = max_chars - len(marker)
    head = available * 3 // 5
    tail = available - head
    return compact[:head] + marker + compact[-tail:]


def build_compiler_repair_context(
    *,
    task_id: str,
    source_sha256: str,
    candidate: str,
    diagnostic: str,
    compiled: bool,
    max_diagnostic_chars: int = 4000,
) -> dict[str, Any]:
    """Build a canonical, hash-bound compiler-only repair observation."""

    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("repair task_id must be non-empty text")
    if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
        raise ValueError("repair source SHA-256 is invalid")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("repair candidate must be non-empty text")
    if type(compiled) is not bool:
        raise ValueError("compiled must be a boolean")
    if compiled:
        raise ValueError(
            "compiler-only repair context requires a non-compiling candidate"
        )
    safe_diagnostic = sanitize_compiler_diagnostic(
        diagnostic,
        max_chars=max_diagnostic_chars,
    )
    payload = {
        "schema": COMPILER_REPAIR_SCHEMA,
        "feedback_kind": "compiler_only",
        "task_id": task_id,
        "source_sha256": source_sha256,
        "candidate": candidate,
        "candidate_sha256": sha256_text(candidate),
        "compiler_failed": True,
        "compiler_feedback": safe_diagnostic,
        "compiler_feedback_sha256": sha256_text(safe_diagnostic),
        "instruction": (
            "Return only a complete corrected Dart compilation unit. "
            "Do not return prose or Markdown fences."
        ),
    }
    text = f"{COMPILER_REPAIR_MARKER}\n{canonical_json(payload)}\n"
    return {
        "schema": COMPILER_REPAIR_ARTIFACT_SCHEMA,
        "payload": payload,
        "text": text,
        "text_sha256": sha256_text(text),
    }


def validate_on_policy_logprob_drift(
    current_logprobs: torch.Tensor,
    saved_logprobs: torch.Tensor,
    *,
    tolerance: float,
    action_mask: torch.Tensor | None = None,
    rollout_conditioning_sha256: str | None = None,
    current_conditioning_sha256: str | None = None,
    rollout_temperature: float | None = None,
    current_temperature: float | None = None,
) -> float:
    """Fail closed when an update is not scored under its rollout policy."""

    if (
        not isinstance(current_logprobs, torch.Tensor)
        or not isinstance(saved_logprobs, torch.Tensor)
        or current_logprobs.shape != saved_logprobs.shape
        or current_logprobs.ndim not in {1, 2}
        or current_logprobs.numel() == 0
    ):
        raise ValueError("saved/current log-probability shapes differ or are empty")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("on-policy drift tolerance must be finite and positive")

    if action_mask is None:
        mask = torch.ones_like(current_logprobs, dtype=torch.bool)
    else:
        if (
            not isinstance(action_mask, torch.Tensor)
            or action_mask.shape != current_logprobs.shape
        ):
            raise ValueError("action mask/log-probability shapes differ")
        mask = action_mask.to(
            device=current_logprobs.device,
            dtype=torch.bool,
        )
    if not bool(mask.any()):
        raise ValueError("on-policy drift mask contains no actions")

    saved = saved_logprobs.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    selected_current = current_logprobs[mask]
    selected_saved = saved[mask]
    if not bool(
        torch.isfinite(selected_current).all() and torch.isfinite(selected_saved).all()
    ):
        raise ValueError("on-policy log-probabilities must be finite")

    hash_values = (
        rollout_conditioning_sha256,
        current_conditioning_sha256,
    )
    if any(value is not None for value in hash_values):
        if not all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value)
            for value in hash_values
        ):
            raise ValueError("both conditioning SHA-256 values are required")
        if rollout_conditioning_sha256 != current_conditioning_sha256:
            raise RuntimeError("current conditioning differs from rollout conditioning")

    temperature_values = (rollout_temperature, current_temperature)
    if any(value is not None for value in temperature_values):
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in temperature_values
        ):
            raise ValueError("both positive rollout/current temperatures are required")
        if float(rollout_temperature) != float(current_temperature):
            raise RuntimeError("current temperature differs from rollout temperature")

    drift = float(
        (selected_current.detach() - selected_saved.detach()).abs().max().cpu()
    )
    if drift > tolerance:
        raise RuntimeError(
            "current policy differs from saved rollout log-probabilities "
            f"(max drift {drift:.6g})"
        )
    return drift


__all__ = [
    "COMPILER_REPAIR_ARTIFACT_SCHEMA",
    "COMPILER_REPAIR_MARKER",
    "COMPILER_REPAIR_SCHEMA",
    "build_compiler_repair_context",
    "canonical_json",
    "code_distance",
    "decoder_action_mask",
    "max_min_diverse_indices",
    "mean_centered_advantages",
    "normalize_generated_seq2seq_ids",
    "prepare_shifted_decoder_input_ids",
    "sanitize_compiler_diagnostic",
    "seq2seq_completion_token_logprobs",
    "sha256_text",
    "validate_on_policy_logprob_drift",
    "verpo_execution_compile_advantages",
    "verpo_local_rewards",
]
