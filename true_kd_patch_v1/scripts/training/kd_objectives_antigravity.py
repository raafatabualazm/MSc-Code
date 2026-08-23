"""Numerically honest objectives for dense and sparse knowledge distillation.

``dense_forward_kl`` is the usual full-vocabulary forward KL

    KL(p_teacher(. | x) || p_student(. | x))

at every supervised next-token position.  ``sparse_topk_tail_forward_kl`` is a
different, explicitly coarsened objective.  It preserves the teacher's observed
top-k probabilities and treats every unobserved vocabulary item as one aggregate
tail event.  It never renormalizes top-k probabilities as if the tail had zero
mass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _positive_temperature(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("temperature must be finite and positive")
    return result


def dense_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return full-vocabulary ``KL(teacher || student)`` with ``T**2`` scaling.

    The leading dimensions may be arbitrary but the complete final vocabulary
    dimension must be present.  No top-k or sampled approximation is made.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim < 2 or student_logits.size(-1) <= 1:
        raise ValueError("logits must include a non-trivial vocabulary dimension")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    temperature = _positive_temperature(temperature)

    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    teacher_log_probs = F.log_softmax(
        teacher_logits.float() / temperature, dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    per_position = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    per_position = per_position * (temperature * temperature)
    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()
    if per_position.numel() == 0:
        raise ValueError("cannot average an empty set of supervised positions")
    return per_position.mean()


def sparse_topk_tail_forward_kl(
    student_logits: torch.Tensor,
    teacher_top_token_ids: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    teacher_tail_mass: torch.Tensor,
    *,
    teacher_top_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    reduction: str = "mean",
    probability_tolerance: float = 1e-6,
) -> torch.Tensor:
    """KL on the partition ``{top token 1, ..., top token k, all other tokens}``.

    This is a rigorous lower bound on full-vocabulary forward KL, not full KD.
    Sparse top-k data cannot be re-tempered exactly because the distribution
    inside the aggregate tail is unknown, so this objective deliberately only
    accepts ``temperature == 1``.
    """

    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [positions, vocabulary]")
    if teacher_top_token_ids.ndim != 2:
        raise ValueError("teacher_top_token_ids must have shape [positions, k]")
    if teacher_top_logprobs.shape != teacher_top_token_ids.shape:
        raise ValueError("teacher top IDs and logprobs must have identical shapes")
    if teacher_top_token_ids.size(0) != student_logits.size(0):
        raise ValueError("teacher and student position counts do not match")
    if teacher_tail_mass.shape != (student_logits.size(0),):
        raise ValueError("teacher_tail_mass must have shape [positions]")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    if probability_tolerance <= 0 or not math.isfinite(probability_tolerance):
        raise ValueError("probability_tolerance must be finite and positive")
    temperature = _positive_temperature(temperature)
    if abs(temperature - 1.0) > 1e-12:
        raise ValueError(
            "sparse_topk_tail_kl requires temperature=1: aggregate tail mass "
            "cannot be temperature-scaled without the hidden tail distribution"
        )

    if teacher_top_mask is None:
        teacher_top_mask = torch.ones_like(
            teacher_top_token_ids, dtype=torch.bool
        )
    if teacher_top_mask.shape != teacher_top_token_ids.shape:
        raise ValueError("teacher_top_mask must match the top-k tensor shape")
    teacher_top_mask = teacher_top_mask.to(dtype=torch.bool)
    if torch.any(teacher_top_mask.sum(dim=-1) == 0):
        raise ValueError("every position must contain at least one top-k token")

    vocab_size = int(student_logits.size(-1))
    valid_ids = teacher_top_token_ids[teacher_top_mask]
    if valid_ids.numel() and (
        torch.any(valid_ids < 0) or torch.any(valid_ids >= vocab_size)
    ):
        raise ValueError("teacher top-k token ID lies outside the student vocabulary")
    if not torch.all(torch.isfinite(teacher_top_logprobs[teacher_top_mask])):
        raise ValueError("teacher top-k logprobs must be finite")
    if not torch.all(torch.isfinite(teacher_tail_mass)):
        raise ValueError("teacher tail masses must be finite")
    if torch.any(teacher_tail_mass < 0) or torch.any(teacher_tail_mass > 1):
        raise ValueError("teacher tail masses must lie in [0, 1]")

    # Duplicate top IDs would double-count one event and corrupt both tail masses.
    for row_ids, row_mask in zip(teacher_top_token_ids, teacher_top_mask):
        selected = row_ids[row_mask]
        if selected.unique().numel() != selected.numel():
            raise ValueError("teacher top-k token IDs must be unique per position")

    safe_ids = teacher_top_token_ids.masked_fill(~teacher_top_mask, 0)
    teacher_top_probs = teacher_top_logprobs.float().exp().masked_fill(
        ~teacher_top_mask, 0.0
    )
    teacher_total = teacher_top_probs.sum(dim=-1) + teacher_tail_mass.float()
    if torch.any((teacher_total - 1.0).abs() > probability_tolerance):
        maximum_error = float((teacher_total - 1.0).abs().max().detach().cpu())
        raise ValueError(
            "teacher top-k probabilities plus tail mass do not sum to one "
            f"(maximum error {maximum_error:.3g})"
        )

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_top_log_probs = student_log_probs.gather(dim=-1, index=safe_ids)
    student_top_probs = student_top_log_probs.exp().masked_fill(
        ~teacher_top_mask, 0.0
    )
    student_tail_mass = 1.0 - student_top_probs.sum(dim=-1)
    # Softmax gives strictly positive tail mass whenever k < vocab size.  Clamp
    # only protects the logarithm from last-bit cancellation for tiny toy vocabs.
    tiny = torch.finfo(student_log_probs.dtype).tiny
    student_tail_log_probs = student_tail_mass.clamp_min(tiny).log()

    teacher_top_log_probs_safe = teacher_top_logprobs.float().masked_fill(
        ~teacher_top_mask, 0.0
    )
    top_terms = teacher_top_probs * (
        teacher_top_log_probs_safe - student_top_log_probs
    )
    top_terms = top_terms.masked_fill(~teacher_top_mask, 0.0).sum(dim=-1)

    teacher_tail = teacher_tail_mass.float()
    teacher_tail_log = torch.where(
        teacher_tail > 0,
        teacher_tail.clamp_min(tiny).log(),
        torch.zeros_like(teacher_tail),
    )
    tail_terms = torch.where(
        teacher_tail > 0,
        teacher_tail * (teacher_tail_log - student_tail_log_probs),
        torch.zeros_like(teacher_tail),
    )
    per_position = top_terms + tail_terms
    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()
    if per_position.numel() == 0:
        raise ValueError("cannot average an empty set of supervised positions")
    return per_position.mean()


def accumulation_window_size(
    batch_index: int, *, total_batches: int, gradient_accumulation_steps: int
) -> int:
    """Return the true divisor for an accumulation window, including a remainder."""

    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if batch_index < 0 or batch_index >= total_batches:
        raise ValueError("batch_index is outside the epoch")
    window_start = (
        batch_index // gradient_accumulation_steps
    ) * gradient_accumulation_steps
    return min(
        gradient_accumulation_steps,
        total_batches - window_start,
    )


def is_accumulation_boundary(
    batch_index: int, *, total_batches: int, gradient_accumulation_steps: int
) -> bool:
    """Whether this batch must perform an optimizer step."""

    accumulation_window_size(
        batch_index,
        total_batches=total_batches,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return (
        (batch_index + 1) % gradient_accumulation_steps == 0
        or batch_index + 1 == total_batches
    )
