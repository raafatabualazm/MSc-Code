"""Fail-closed sparse top-k+tail auxiliary for direct-compact sequence SFT.

At an exactly aligned shared-tokenizer prefix, the teacher top-k token events
and the aggregate complement define a categorical partition. The forward KL on
that coarsened partition is mathematically exact for the partition and is a
data-processing lower bound on a hypothetical full-vocabulary KL. It is never
dense/full-vocabulary KD.

The API artifact does not contain a teacher distribution for the terminal EOS.
Consequently this module applies the auxiliary only to logged content-token
positions. The primary sequence NLL remains responsible for content and EOS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from models.direct_compact_causal import (
    DirectCompactBatchCollator,
    DirectCompactCausalLM,
    sha256_file,
)


SPARSE_FIELD = "sparse_topk_tail_auxiliary"
SPARSE_ROW_SCHEMA = "direct-compact-sparse-topk-tail-row-v1"
SPARSE_MANIFEST_SCHEMA = "direct-compact-sparse-topk-tail-manifest-v1"
PROBABILITY_TOLERANCE = 1e-8


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_sparse_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path,
    *,
    contract_path: str | Path,
    tokenizer_json_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("sparse auxiliary manifest must be an object")
    if manifest.get("schema") != SPARSE_MANIFEST_SCHEMA:
        raise ValueError("sparse auxiliary manifest schema mismatch")
    exact = {
        "dataset_sha256": sha256_file(dataset_path),
        "contract_sha256": sha256_file(contract_path),
        "student_tokenizer_json_sha256": sha256_file(tokenizer_json_path),
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise ValueError(f"sparse auxiliary manifest {field} mismatch")
    if manifest.get("objective") != "coarsened_topk_plus_tail_forward_kl":
        raise ValueError("sparse auxiliary objective declaration mismatch")
    if manifest.get("sequence_monte_carlo_forward_kl_nll_primary") is not True:
        raise ValueError("sequence Monte Carlo NLL must remain primary")
    if manifest.get("dense_full_vocabulary_kl") is not False:
        raise ValueError("sparse auxiliary manifest makes a dense-KL claim")
    if manifest.get("full_vocabulary_kd") is not False:
        raise ValueError("sparse auxiliary manifest makes a full-KD claim")
    if manifest.get("global_provider_tokenizer_identity_claimed") is not False:
        raise ValueError(
            "sparse auxiliary manifest makes a global tokenizer-identity claim"
        )
    if manifest.get("target_transform") != (
        "trim_trailing_outer_whitespace_on_provider_token_boundaries"
    ):
        raise ValueError("sparse auxiliary target-alignment declaration mismatch")
    eos = manifest.get("eos_policy")
    if not isinstance(eos, Mapping):
        raise ValueError("sparse auxiliary manifest lacks EOS policy")
    required_eos = {
        "teacher_eos_distribution_available": False,
        "sparse_auxiliary_applied_to_eos": False,
        "student_eos_supervised_by_primary_sequence_nll": True,
    }
    if any(eos.get(key) is not value for key, value in required_eos.items()):
        raise ValueError("sparse auxiliary EOS policy is not fail-closed")
    rows = manifest.get("rows")
    eligible = manifest.get("rows_with_sparse_auxiliary")
    positions = manifest.get("sparse_positions")
    for value, label in (
        (rows, "rows"),
        (eligible, "rows_with_sparse_auxiliary"),
        (positions, "sparse_positions"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"sparse auxiliary manifest {label} is invalid")
    if rows <= 0 or eligible <= 0 or eligible > rows or positions <= 0:
        raise ValueError("sparse auxiliary manifest has no usable coverage")
    return manifest


def attach_sparse_metadata(
    dataset: Any,
    dataset_path: str | Path,
    *,
    tokenizer: Any,
    eos_token_id: int,
    output_vocab_size: int,
    expected_rows_with_auxiliary: int,
    expected_sparse_positions: int,
) -> dict[str, int]:
    """Attach and revalidate sparse positions against exact trainer token IDs."""

    raw_rows: list[dict[str, Any]] = []
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"{dataset_path}:{line_number}: blank rows are forbidden"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{dataset_path}:{line_number}: row is not an object")
            raw_rows.append(value)
    if len(raw_rows) != len(dataset.rows):
        raise ValueError("sparse metadata row count differs from direct dataset")

    rows_with_auxiliary = 0
    sparse_positions = 0
    for row_index, (raw, prepared) in enumerate(
        zip(raw_rows, dataset.rows, strict=True)
    ):
        sparse = raw.get(SPARSE_FIELD)
        if sparse is None:
            prepared[SPARSE_FIELD] = None
            continue
        if not isinstance(sparse, Mapping) or sparse.get("schema") != SPARSE_ROW_SCHEMA:
            raise ValueError(f"row {row_index}: invalid sparse auxiliary schema")
        alignment = sparse.get("target_alignment")
        if (
            not isinstance(alignment, Mapping)
            or alignment.get("transform") != "trim_trailing_outer_whitespace"
            or alignment.get("trim_on_provider_token_boundaries") is not True
            or alignment.get("leading_provider_tokens_omitted") != 0
            or isinstance(
                alignment.get("trailing_provider_tokens_omitted"), bool
            )
            or not isinstance(
                alignment.get("trailing_provider_tokens_omitted"), int
            )
            or alignment.get("trailing_provider_tokens_omitted") < 0
        ):
            raise ValueError(
                f"row {row_index}: sparse teacher/student prefix alignment "
                "is not exact"
            )
        target_ids = sparse.get("target_token_ids")
        positions = sparse.get("teacher_positions")
        if not isinstance(target_ids, list) or not target_ids:
            raise ValueError(f"row {row_index}: sparse target IDs are empty")
        if not isinstance(positions, list) or len(positions) != len(target_ids):
            raise ValueError(
                f"row {row_index}: sparse positions do not align with target IDs"
            )
        prepared_target = list(prepared["target_input_ids"])
        if (
            not prepared_target
            or prepared_target[-1] != int(eos_token_id)
            or prepared_target[:-1] != target_ids
        ):
            raise ValueError(
                f"row {row_index}: sparse target IDs differ from exact trainer "
                "tokenization or appended EOS"
            )
        if int(eos_token_id) in target_ids:
            raise ValueError(
                f"row {row_index}: logged content unexpectedly contains EOS"
            )
        canonical_positions: list[dict[str, Any]] = []
        for position_index, (target_id, position) in enumerate(
            zip(target_ids, positions, strict=True)
        ):
            if not isinstance(position, Mapping):
                raise ValueError(
                    f"row {row_index}:{position_index}: position is not an object"
                )
            if position.get("observed_token_id") != target_id:
                raise ValueError(
                    f"row {row_index}:{position_index}: observed/target ID mismatch"
                )
            top_ids = position.get("top_token_ids")
            top_logprobs = position.get("top_logprobs")
            if (
                not isinstance(top_ids, list)
                or not top_ids
                or not isinstance(top_logprobs, list)
                or len(top_ids) != len(top_logprobs)
            ):
                raise ValueError(
                    f"row {row_index}:{position_index}: invalid top-k arrays"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= int(output_vocab_size)
                for value in top_ids
            ):
                raise ValueError(
                    f"row {row_index}:{position_index}: invalid top token ID"
                )
            if len(set(top_ids)) != len(top_ids):
                raise ValueError(
                    f"row {row_index}:{position_index}: duplicate top token IDs"
                )
            logprobs = [
                _finite(
                    value,
                    f"row {row_index}:{position_index}:top_logprobs",
                )
                for value in top_logprobs
            ]
            tail = _finite(
                position.get("tail_probability_mass"),
                f"row {row_index}:{position_index}:tail_probability_mass",
            )
            if tail < 0.0 or tail > 1.0:
                raise ValueError(
                    f"row {row_index}:{position_index}: tail mass is outside [0,1]"
                )
            total = math.fsum(math.exp(value) for value in logprobs) + tail
            if abs(total - 1.0) > PROBABILITY_TOLERANCE:
                raise ValueError(
                    f"row {row_index}:{position_index}: top-k plus tail sums "
                    f"to {total:.17g}, not one"
                )
            canonical_positions.append(
                {
                    "observed_token_id": int(target_id),
                    "top_token_ids": [int(value) for value in top_ids],
                    "top_logprobs": logprobs,
                    "tail_probability_mass": tail,
                }
            )
        prepared[SPARSE_FIELD] = {
            "target_token_ids": [int(value) for value in target_ids],
            "teacher_positions": canonical_positions,
        }
        rows_with_auxiliary += 1
        sparse_positions += len(canonical_positions)

    if rows_with_auxiliary != int(expected_rows_with_auxiliary):
        raise ValueError("sparse auxiliary eligible-row count differs from manifest")
    if sparse_positions != int(expected_sparse_positions):
        raise ValueError("sparse auxiliary position count differs from manifest")
    return {
        "rows_with_sparse_auxiliary": rows_with_auxiliary,
        "sparse_positions": sparse_positions,
    }


class SparseTopKTailCollator:
    """Wrap the standard direct collator and pad only loss-side sparse metadata."""

    def __init__(self, base_collator: DirectCompactBatchCollator) -> None:
        self.base_collator = base_collator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base_collator(features)
        sparse_values = [feature.get(SPARSE_FIELD) for feature in features]
        max_positions = max(
            (
                len(value["teacher_positions"])
                for value in sparse_values
                if isinstance(value, Mapping)
            ),
            default=0,
        )
        max_topk = max(
            (
                len(position["top_token_ids"])
                for value in sparse_values
                if isinstance(value, Mapping)
                for position in value["teacher_positions"]
            ),
            default=1,
        )
        shape = (len(features), max_positions, max_topk)
        top_ids = torch.full(shape, -1, dtype=torch.long)
        top_logprobs = torch.zeros(shape, dtype=torch.float64)
        top_mask = torch.zeros(shape, dtype=torch.bool)
        tails = torch.zeros((len(features), max_positions), dtype=torch.float64)
        observed = torch.full(
            (len(features), max_positions), -1, dtype=torch.long
        )
        position_mask = torch.zeros(
            (len(features), max_positions), dtype=torch.bool
        )
        for row_index, sparse in enumerate(sparse_values):
            if not isinstance(sparse, Mapping):
                continue
            for position_index, position in enumerate(sparse["teacher_positions"]):
                count = len(position["top_token_ids"])
                top_ids[row_index, position_index, :count] = torch.tensor(
                    position["top_token_ids"], dtype=torch.long
                )
                top_logprobs[row_index, position_index, :count] = torch.tensor(
                    position["top_logprobs"], dtype=torch.float64
                )
                top_mask[row_index, position_index, :count] = True
                tails[row_index, position_index] = float(
                    position["tail_probability_mass"]
                )
                observed[row_index, position_index] = int(
                    position["observed_token_id"]
                )
                position_mask[row_index, position_index] = True
        batch.update(
            {
                "sparse_teacher_top_token_ids": top_ids,
                "sparse_teacher_top_logprobs": top_logprobs,
                "sparse_teacher_top_mask": top_mask,
                "sparse_teacher_tail_mass": tails,
                "sparse_teacher_observed_ids": observed,
                "sparse_teacher_position_mask": position_mask,
            }
        )
        return batch


def _log_one_minus_exp(log_value: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(x)) for x <= 0 without probability clamping."""

    if torch.any(log_value > 1e-7):
        raise ValueError("top-k student probability exceeds one")
    if torch.any(log_value > 0):
        # Positive last-bit noise is not silently clamped into a distribution.
        raise FloatingPointError("student top-k log mass is numerically positive")
    split = -math.log(2.0)
    return torch.where(
        log_value < split,
        torch.log1p(-torch.exp(log_value)),
        torch.log(-torch.expm1(log_value)),
    )


def coarsened_topk_tail_forward_kl(
    student_logits: torch.Tensor,
    teacher_top_token_ids: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    teacher_tail_mass: torch.Tensor,
    *,
    teacher_top_mask: torch.Tensor | None = None,
    reduction: str = "mean",
    probability_tolerance: float = PROBABILITY_TOLERANCE,
) -> torch.Tensor:
    """Exact forward KL on top-k singleton categories plus one tail category."""

    if student_logits.ndim != 2 or student_logits.size(-1) <= 1:
        raise ValueError("student logits must have shape [positions,vocabulary]")
    if teacher_top_token_ids.ndim != 2:
        raise ValueError("teacher top IDs must have shape [positions,k]")
    if teacher_top_logprobs.shape != teacher_top_token_ids.shape:
        raise ValueError("teacher top IDs/logprobs shape mismatch")
    if teacher_top_token_ids.size(0) != student_logits.size(0):
        raise ValueError("teacher/student position count mismatch")
    if teacher_tail_mass.shape != (student_logits.size(0),):
        raise ValueError("teacher tail shape mismatch")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("invalid reduction")
    if teacher_top_mask is None:
        teacher_top_mask = torch.ones_like(
            teacher_top_token_ids, dtype=torch.bool
        )
    if teacher_top_mask.shape != teacher_top_token_ids.shape:
        raise ValueError("teacher top mask shape mismatch")
    mask = teacher_top_mask.bool()
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every sparse position needs a top-k category")
    ids = teacher_top_token_ids
    valid_ids = ids[mask]
    vocab_size = student_logits.size(-1)
    if torch.any(valid_ids < 0) or torch.any(valid_ids >= vocab_size):
        raise ValueError("teacher top ID outside student vocabulary")
    for row_ids, row_mask in zip(ids, mask):
        selected = row_ids[row_mask]
        if selected.unique().numel() != selected.numel():
            raise ValueError("teacher top IDs must be unique per position")
    valid_logprobs = teacher_top_logprobs[mask]
    if not torch.all(torch.isfinite(valid_logprobs)):
        raise ValueError("teacher top logprobs must be finite")
    if torch.any(valid_logprobs > 0):
        raise ValueError("teacher log probability cannot be positive")
    if not torch.all(torch.isfinite(teacher_tail_mass)):
        raise ValueError("teacher tail mass must be finite")
    if torch.any(teacher_tail_mass < 0) or torch.any(teacher_tail_mass > 1):
        raise ValueError("teacher tail mass outside [0,1]")

    safe_ids = ids.masked_fill(~mask, 0)
    teacher_logp = teacher_top_logprobs.double()
    teacher_top_probs = teacher_logp.exp().masked_fill(~mask, 0.0)
    totals = teacher_top_probs.sum(dim=-1) + teacher_tail_mass.double()
    if torch.any((totals - 1.0).abs() > probability_tolerance):
        raise ValueError("teacher top-k probabilities plus tail do not sum to one")

    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    student_top_logp = student_logp.gather(-1, safe_ids)
    student_top_logp = student_top_logp.masked_fill(~mask, -torch.inf)
    student_top_log_mass = torch.logsumexp(student_top_logp, dim=-1)
    student_tail_logp = _log_one_minus_exp(student_top_log_mass)

    teacher_top_logp_float = teacher_top_logprobs.float().masked_fill(~mask, 0.0)
    top_terms = teacher_top_probs.float() * (
        teacher_top_logp_float
        - student_top_logp.masked_fill(~mask, 0.0)
    )
    top_terms = top_terms.masked_fill(~mask, 0.0).sum(dim=-1)
    teacher_tail = teacher_tail_mass.float()
    teacher_tail_logp = torch.where(
        teacher_tail > 0,
        teacher_tail.log(),
        torch.zeros_like(teacher_tail),
    )
    tail_terms = torch.where(
        teacher_tail > 0,
        teacher_tail * (teacher_tail_logp - student_tail_logp),
        torch.zeros_like(teacher_tail),
    )
    values = top_terms + tail_terms
    if not torch.all(torch.isfinite(values)):
        raise FloatingPointError("coarsened sparse KL became non-finite")
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if values.numel() == 0:
        raise ValueError("cannot average zero sparse positions")
    return values.mean()


class DirectCompactSparseTopKTailCausalLM(DirectCompactCausalLM):
    """Direct-compact model with primary sequence NLL plus sparse auxiliary KL."""

    def __init__(
        self,
        causal_lm: torch.nn.Module,
        *,
        auxiliary_weight: float,
        position_chunk_size: int = 32,
        sequence_sum_nll: bool = False,
    ) -> None:
        super().__init__(
            causal_lm,
            sequence_sum_nll=sequence_sum_nll,
        )
        weight = float(auxiliary_weight)
        if not math.isfinite(weight) or not 0.0 < weight < 1.0:
            raise ValueError(
                "sparse auxiliary weight must be in (0,1) so sequence NLL "
                "remains primary"
            )
        self.auxiliary_weight = weight
        if isinstance(position_chunk_size, bool) or int(position_chunk_size) <= 0:
            raise ValueError("sparse position chunk size must be positive")
        self.position_chunk_size = int(position_chunk_size)
        self.last_sparse_auxiliary_loss: float | None = None
        self.last_primary_sequence_nll: float | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sparse_teacher_top_token_ids: torch.Tensor | None = None,
        sparse_teacher_top_logprobs: torch.Tensor | None = None,
        sparse_teacher_top_mask: torch.Tensor | None = None,
        sparse_teacher_tail_mass: torch.Tensor | None = None,
        sparse_teacher_observed_ids: torch.Tensor | None = None,
        sparse_teacher_position_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        primary = (
            outputs["loss"] if isinstance(outputs, Mapping) else outputs.loss
        )
        if labels is None or primary is None:
            raise ValueError("sparse direct training requires primary sequence labels")
        self.last_primary_sequence_nll = float(primary.detach().cpu())
        if (
            sparse_teacher_position_mask is None
            or not torch.any(sparse_teacher_position_mask)
        ):
            self.last_sparse_auxiliary_loss = None
            return outputs
        required = {
            "top IDs": sparse_teacher_top_token_ids,
            "top logprobs": sparse_teacher_top_logprobs,
            "top mask": sparse_teacher_top_mask,
            "tail": sparse_teacher_tail_mass,
            "observed IDs": sparse_teacher_observed_ids,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("sparse batch is missing " + ", ".join(missing))

        position_mask = sparse_teacher_position_mask.bool()
        batch_indices, target_indices = torch.where(position_mask)
        prediction_indices: list[int] = []
        observed_labels: list[int] = []
        for batch_index, target_index in zip(
            batch_indices.tolist(), target_indices.tolist(), strict=True
        ):
            label_positions = torch.where(labels[batch_index].ne(-100))[0]
            # label_positions includes the appended EOS; sparse positions do not.
            if target_index >= label_positions.numel() - 1:
                raise ValueError("sparse auxiliary attempted to supervise EOS")
            label_position = int(label_positions[target_index])
            if label_position <= 0:
                raise ValueError("causal sparse position has no prediction prefix")
            prediction_indices.append(label_position - 1)
            observed_labels.append(int(labels[batch_index, label_position]))
        observed_tensor = torch.tensor(
            observed_labels, device=labels.device, dtype=torch.long
        )
        sealed_observed = sparse_teacher_observed_ids[position_mask]
        if not torch.equal(observed_tensor, sealed_observed):
            raise ValueError("sparse observed IDs differ from causal target labels")

        prediction_tensor = torch.tensor(
            prediction_indices, device=labels.device, dtype=torch.long
        )
        logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
        selected_top_ids = sparse_teacher_top_token_ids[position_mask]
        selected_top_logprobs = sparse_teacher_top_logprobs[position_mask]
        selected_tail = sparse_teacher_tail_mass[position_mask]
        selected_top_mask = sparse_teacher_top_mask[position_mask]
        auxiliary_sum = logits.new_zeros((), dtype=torch.float32)
        position_count = int(batch_indices.numel())
        # Bound the temporary [positions,vocabulary] fp32 log-softmax. Without
        # this chunking, a 3K-token target can allocate gigabytes.
        for start in range(0, position_count, self.position_chunk_size):
            end = min(position_count, start + self.position_chunk_size)
            chunk_logits = logits[
                batch_indices[start:end], prediction_tensor[start:end]
            ]
            auxiliary_sum = auxiliary_sum + coarsened_topk_tail_forward_kl(
                chunk_logits,
                selected_top_ids[start:end],
                selected_top_logprobs[start:end],
                selected_tail[start:end],
                teacher_top_mask=selected_top_mask[start:end],
                reduction="sum",
            )
        auxiliary = auxiliary_sum / position_count
        total = primary + self.auxiliary_weight * auxiliary
        self.last_sparse_auxiliary_loss = float(auxiliary.detach().cpu())
        if isinstance(outputs, Mapping):
            outputs["loss"] = total
            outputs["primary_sequence_nll"] = primary.detach()
            outputs["sparse_topk_tail_forward_kl"] = auxiliary.detach()
        else:
            outputs.loss = total
            setattr(outputs, "primary_sequence_nll", primary.detach())
            setattr(outputs, "sparse_topk_tail_forward_kl", auxiliary.detach())
        return outputs
