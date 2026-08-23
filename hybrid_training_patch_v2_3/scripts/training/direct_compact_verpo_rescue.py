#!/usr/bin/env python3
"""Auditable asynchronous rescue experiment for direct-compact VeRPO.

This module deliberately lives outside the optimizer loop.  Its four stages
are immutable and independently runnable:

``plan``
    Score base student draws on the visible feedback tests, retain only groups
    where every draw passes zero visible cases, and select ``K`` candidates by
    deterministic greedy max-min code diversity.

``diagnose``
    Build an exact F2 grounding catalogue and request one independently
    validated diagnosis-plus-steps artifact per group. The diagnosis-only arm
    reuses that exact diagnosis with only the prescriptive steps removed.

``materialize``
    Produce candidate-rank-specific conditioning plans for
    ``plain_resample``, ``compiler_only``, ``diagnosis_only``, and
    ``diagnosis_and_steps``.  Every task reserves ``R`` repair slots. Rejected
    diagnoses stay present as ``generate=false`` rows, so the scorer cannot
    shrink the denominator.

``score``
    Score every generated repair on visible tests.  Select exactly one repair
    per task/arm using a predeclared visible-only rule, then evaluate that same
    repair and its originating base candidate on the sealed development
    reward-holdback. Missing/rejected slots count as failures. Only student
    repairs that fully pass both suites are exported as RS-SFT targets;
    partial execution preferences are exported separately and explicitly
    marked off-policy.

The diagnosis stage has no holdback argument and constructs its API payload
from a strict visible-only whitelist. The scoring stage has no judge object.
Consequently private holdback/reference Dart cannot cross the API boundary by
accidental object reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence


PATCH_ROOT = Path(__file__).resolve().parents[2]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation.direct_compact_qwen_inference import (
    RESCUE_ARMS,
    RESCUE_CONDITIONING_SCHEMA,
)
from scripts.training.direct_compact_verpo import score_dart_candidate
from scripts.training.verpo_judge_antigravity import VerpoJudge
from scripts.training.verpo_rescue_grounding import (
    GROUNDING_SCHEMA,
    GroundingCatalog,
    build_grounding_catalog,
    validate_diagnosis_item,
)


PILOT_PLAN_SCHEMA = "direct-compact-verpo-rescue-pilot-plan-v1"
DIAGNOSIS_ARTIFACT_SCHEMA = "direct-compact-verpo-rescue-diagnoses-v1"
DIAGNOSIS_JOURNAL_SCHEMA = "direct-compact-verpo-rescue-diagnosis-journal-v1"
MATERIALIZATION_SCHEMA = "direct-compact-verpo-rescue-materialization-v1"
INFERENCE_BUNDLE_SCHEMA = "direct-compact-verpo-rescue-inference-bundle-v1"
SCORE_ARTIFACT_SCHEMA = "direct-compact-verpo-rescue-score-v1"
RS_SFT_TARGET_SCHEMA = "direct-compact-verpo-rescue-rs-sft-target-v1"
PREFERENCE_PAIR_SCHEMA = "direct-compact-verpo-rescue-preference-pair-v1"
CONDITIONING_TEXT_SCHEMA = "direct-compact-verpo-rescue-context-v1"
FEEDBACK_VIEW_REPORT_SCHEMA = "verpo-train-feedback-view-v1"

PLAN_HASH_FIELD = "plan_sha256"
DIAGNOSIS_HASH_FIELD = "diagnosis_artifact_sha256"
MATERIALIZATION_HASH_FIELD = "materialization_sha256"
BUNDLE_HASH_FIELD = "bundle_sha256"
SCORE_HASH_FIELD = "score_artifact_sha256"
DIAGNOSIS_JOURNAL_GENESIS_SHA256 = "0" * 64

ARM_ORDER = (
    "plain_resample",
    "compiler_only",
    "diagnosis_only",
    "diagnosis_and_steps",
)
GUIDANCE_MODES = ("diagnosis_only", "diagnosis_and_steps")
JUDGE_CALL_MODE = "diagnosis_and_steps"
FORBIDDEN_API_KEYS = frozenset(
    {
        "acceptance_tests",
        "hidden_tests",
        "reward_holdback_tests",
        "reference",
        "reference_code",
        "reference_dart",
        "gold",
        "gold_code",
        "solution",
        "target",
        "target_code",
    }
)
CHECKPOINT_BINDING_FIELDS = (
    "decoder_model",
    "decoder_revision",
    "model_config_sha256",
    "decoder_adapter",
    "decoder_adapter_sha256",
    "source_overlay_sha256",
    "contract_sha256",
    "codebook_sha256",
    "codec_sha256",
    "tokenizer_json_sha256",
    "direct_prompt_mode",
)
REPAIR_GENERATION_FIELDS = (
    "dataset_sha256",
    "alignment_sha256",
    "selected_role",
    "decoder_model",
    "decoder_revision",
    "decoder_adapter_sha256",
    "source_overlay_sha256",
    "contract_sha256",
    "tokenizer_json_sha256",
    "direct_prompt_mode",
    "attn_implementation",
    "max_new_tokens",
    "num_samples",
    "temperature",
    "top_p",
    "top_k",
    "batch_size",
    "limit",
    "seed",
    "bf16",
    "fp16",
    "precision",
    "sampling_seed_policy",
    "started_without_terminal_policy",
    "resampled_slots",
)

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?|"
    r"===|!==|==|!=|<=|>=|=>|&&|\|\||\+\+|--|<<|>>|"
    r"[^\s]",
    re.UNICODE,
)


class RescueError(ValueError):
    """An artifact or experimental invariant failed closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, *, include_path: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RescueError(f"artifact does not exist: {resolved}")
    value = {
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if include_path:
        value["path"] = str(resolved)
    return value


def _require_exact_file_record(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Require path, byte count, and SHA-256 to match an upstream seal."""

    if not isinstance(expected, Mapping):
        raise RescueError(f"{label} has no sealed file record")
    observed = file_record(path)
    expected_path = Path(str(expected.get("path") or "")).expanduser().resolve()
    if (
        Path(str(observed["path"])).resolve() != expected_path
        or observed["sha256"] != expected.get("sha256")
        or observed["size_bytes"] != expected.get("size_bytes")
    ):
        raise RescueError(f"{label} differs from feedback-view report")
    return observed


def validate_feedback_view_report(
    report_path: str | Path,
    expected_report_sha256: str,
    *,
    expected_outputs: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Validate an exact feedback-view report and selected output records."""

    expected_sha = str(expected_report_sha256).strip().lower()
    if not _SHA_RE.fullmatch(expected_sha):
        raise RescueError("expected feedback-view report SHA-256 is invalid")
    resolved = Path(report_path).expanduser().resolve()
    report_record = file_record(resolved)
    if report_record["sha256"] != expected_sha:
        raise RescueError("feedback-view report hash mismatch")
    report = read_json(resolved)
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != FEEDBACK_VIEW_REPORT_SCHEMA
        or report.get("status") != "complete"
        or not isinstance(report.get("outputs"), Mapping)
    ):
        raise RescueError("feedback-view report schema/status is invalid")
    output_records: dict[str, dict[str, Any]] = {}
    for name, output_path in expected_outputs.items():
        output_records[name] = _require_exact_file_record(
            output_path,
            report["outputs"].get(name),
            label=f"feedback-view {name}",
        )
    return {
        "report": dict(report),
        "report_record": report_record,
        "output_records": output_records,
    }


def artifact_digest(value: Mapping[str, Any], hash_field: str) -> str:
    if not isinstance(value, Mapping):
        raise RescueError("artifact must be an object")
    return canonical_sha256(
        {key: child for key, child in value.items() if key != hash_field}
    )


def seal_artifact(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    body = {key: child for key, child in value.items() if key != hash_field}
    return {**body, hash_field: canonical_sha256(body)}


def require_artifact_digest(
    value: Mapping[str, Any],
    *,
    schema: str,
    hash_field: str,
) -> str:
    if value.get("schema") != schema:
        raise RescueError(f"unexpected artifact schema; expected {schema}")
    expected = str(value.get(hash_field) or "")
    observed = artifact_digest(value, hash_field)
    if not _SHA_RE.fullmatch(expected) or expected != observed:
        raise RescueError(f"{schema} self-digest mismatch")
    return expected


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueError(f"cannot read JSON artifact {path}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RescueError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RescueError(
                        f"{path}:{line_number}: JSONL row is not an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueError(f"cannot read JSONL artifact {path}: {exc}") from exc
    return rows


def write_json_new(path: str | Path, value: Any, *, canonical: bool = False) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = (
        canonical_json(value)
        if canonical
        else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    )
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not canonical:
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_new(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RescueError(f"{label} must be an integer >= {minimum}")
    return value


def _task_id(row: Mapping[str, Any], label: str) -> str:
    value = row.get("task_id") or row.get("id")
    if not isinstance(value, str) or not value:
        raise RescueError(f"{label} has no task identity")
    return value


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RescueError(f"{label} row {position} is not an object")
        identity = _task_id(row, f"{label} row {position}")
        if identity in result:
            raise RescueError(f"{label} has duplicate task {identity!r}")
        result[identity] = row
    return result


def _has_forbidden_api_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_API_KEYS or "holdback" in normalized:
                return normalized
            nested = _has_forbidden_api_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _has_forbidden_api_key(child)
            if nested is not None:
                return nested
    return None


def assert_api_safe_payload(value: Mapping[str, Any]) -> None:
    forbidden = _has_forbidden_api_key(value)
    if forbidden is not None:
        raise RescueError(
            f"provider payload contains forbidden private/reference key {forbidden!r}"
        )
    if set(value) != {
        "source",
        "source_sha256",
        "source_format_guide",
        "tests",
        "candidates",
        "reference_catalog",
        "reference_catalog_sha256",
        "guidance_mode",
    }:
        raise RescueError("provider payload differs from visible-only whitelist")


def checkpoint_binding(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise RescueError("inference provenance must be an object")
    binding = {
        field: provenance.get(field) for field in CHECKPOINT_BINDING_FIELDS
    }
    if not isinstance(binding["decoder_model"], str) or not binding[
        "decoder_model"
    ]:
        raise RescueError("inference provenance has no decoder model")
    for field in (
        "source_overlay_sha256",
        "contract_sha256",
        "codebook_sha256",
        "codec_sha256",
        "tokenizer_json_sha256",
    ):
        if not isinstance(binding[field], str) or not _SHA_RE.fullmatch(
            binding[field]
        ):
            raise RescueError(f"inference provenance has invalid {field}")
    adapter_sha = binding.get("decoder_adapter_sha256")
    model_config_sha = binding.get("model_config_sha256")
    if (
        adapter_sha is not None
        and (
            not isinstance(adapter_sha, str)
            or not _SHA_RE.fullmatch(adapter_sha)
        )
    ):
        raise RescueError("inference provenance adapter digest is invalid")
    if (
        model_config_sha is not None
        and (
            not isinstance(model_config_sha, str)
            or not _SHA_RE.fullmatch(model_config_sha)
        )
    ):
        raise RescueError("inference provenance model-config digest is invalid")
    return binding


def checkpoint_fingerprint(provenance: Mapping[str, Any]) -> dict[str, Any]:
    binding = checkpoint_binding(provenance)
    return {
        "binding": binding,
        "sha256": canonical_sha256(binding),
    }


def repair_generation_binding(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: provenance.get(field) for field in REPAIR_GENERATION_FIELDS
    }


def _normalized_code_tokens(code: str) -> tuple[str, ...]:
    # Comments and whitespace should not dominate structural diversity.
    without_block = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    without_line = re.sub(r"//[^\r\n]*", " ", without_block)
    return tuple(token.lower() for token in _TOKEN_RE.findall(without_line))


def _token_ngrams(code: str, width: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = _normalized_code_tokens(code)
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset((token,) for token in tokens)
    return frozenset(
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def code_distance(left: str, right: str) -> float:
    """Normalized token-trigram Jaccard distance in ``[0, 1]``."""

    first = _token_ngrams(left)
    second = _token_ngrams(right)
    if not first and not second:
        return 0.0
    union = first | second
    return 1.0 - (len(first & second) / len(union))


def realized_code_diversity(candidates: Sequence[str]) -> dict[str, Any]:
    """Report realized code diversity without treating it as a covariate."""

    values = list(candidates)
    if any(not isinstance(value, str) for value in values):
        raise RescueError("diversity candidates must be text")
    pairwise = [
        code_distance(values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return {
        "programs": len(values),
        "unique_normalized_programs": len(
            {_normalized_code_tokens(value) for value in values}
        ),
        "minimum_pairwise_distance": min(pairwise) if pairwise else None,
        "mean_pairwise_distance": (
            sum(pairwise) / len(pairwise) if pairwise else None
        ),
        "distance": "normalized_token_trigram_jaccard",
        "used_to_adjust_rescue_outcome": False,
    }


def mcnemar_power_plan(
    sample_size: int,
    *,
    minimum_detectable_difference: float = 0.05,
    assumed_discordant_fraction: float = 0.10,
    alpha: float = 0.05,
    desired_power: float = 0.80,
) -> dict[str, Any]:
    """Conservative continuity-corrected normal approximation for McNemar.

    Power depends on the *paired discordance*, which is unknown before the
    pilot. The assumption is therefore explicit and must be replaced by the
    observed paired table when sizing the confirmatory run.
    """

    _plain_int(sample_size, "McNemar sample_size")
    values = (
        minimum_detectable_difference,
        assumed_discordant_fraction,
        alpha,
        desired_power,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise RescueError("McNemar power parameters must be numeric")
    difference = float(minimum_detectable_difference)
    discordance = float(assumed_discordant_fraction)
    alpha_value = float(alpha)
    target = float(desired_power)
    if (
        not 0.0 < difference < discordance <= 1.0
        or not 0.0 < alpha_value < 1.0
        or not 0.0 < target < 1.0
    ):
        raise RescueError(
            "McNemar power requires 0<difference<discordance<=1 and "
            "0<alpha,power<1"
        )
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha_value / 2.0)

    def approximate_power(count: int) -> float:
        if count <= 0:
            return 0.0
        # A one-pair continuity correction is material in the sparse,
        # low-rescue regime and avoids the optimistic uncorrected estimate.
        noncentrality = max(
            0.0,
            (count * difference - 1.0)
            / ((count * discordance) ** 0.5),
        )
        return normal.cdf(-critical - noncentrality) + (
            1.0 - normal.cdf(critical - noncentrality)
        )

    required = 1
    while required < 1_000_000 and approximate_power(required) < target:
        required += 1
    if required >= 1_000_000:
        raise RescueError("McNemar power target exceeds safety search limit")
    achieved = approximate_power(sample_size)
    return {
        "test": "paired_two_sided_mcnemar",
        "approximation": (
            "continuity-corrected normal; replace assumed discordance with "
            "pilot paired table before confirmatory run"
        ),
        "primary_contrast": (
            "plain_resample_vs_diagnosis_and_steps_visible_group_rescue"
        ),
        "minimum_detectable_absolute_difference": difference,
        "assumed_discordant_fraction": discordance,
        "alpha_two_sided": alpha_value,
        "desired_power": target,
        "sample_size": sample_size,
        "approximate_achieved_power": achieved,
        "approximate_required_groups": required,
        "adequately_powered_under_assumption": sample_size >= required,
        "warning": (
            None
            if sample_size >= required
            else (
                "Underpowered for the predeclared primary contrast under the "
                "stated discordance assumption; intermediate arms are "
                "descriptive and a confirmatory sample must be resized from "
                "the observed paired table."
            )
        ),
        "experimental_unit": "unique_dead_task_group",
        "repair_candidates_are_not_independent_units": True,
    }


def paired_mcnemar_contrast(
    task_arm_results: Sequence[Mapping[str, Any]],
    *,
    control_arm: str = "plain_resample",
    treatment_arm: str = "diagnosis_and_steps",
) -> dict[str, Any]:
    """Compute the sealed task-paired visible-rescue McNemar contrast."""

    by_task: dict[str, dict[str, bool]] = {}
    for row in task_arm_results:
        if not isinstance(row, Mapping):
            raise RescueError("task-arm result is not an object")
        arm = str(row.get("arm") or "")
        if arm not in {control_arm, treatment_arm}:
            continue
        task_id = _task_id(row, "task-arm result")
        if arm in by_task.setdefault(task_id, {}):
            raise RescueError(
                f"paired McNemar input duplicates {task_id!r}/{arm}"
            )
        rescued = row.get("visible_rescued")
        if not isinstance(rescued, bool):
            raise RescueError("paired McNemar outcome is not boolean")
        by_task[task_id][arm] = rescued
    incomplete = sorted(
        task_id
        for task_id, arms in by_task.items()
        if set(arms) != {control_arm, treatment_arm}
    )
    if incomplete:
        raise RescueError(
            "paired McNemar contrast lacks both arms for tasks: "
            + ", ".join(incomplete[:8])
        )
    if not by_task:
        raise RescueError("paired McNemar contrast has zero task pairs")
    table = Counter()
    for arms in by_task.values():
        control = bool(arms[control_arm])
        treatment = bool(arms[treatment_arm])
        table[
            (
                "control_rescued" if control else "control_failed",
                "treatment_rescued" if treatment else "treatment_failed",
            )
        ] += 1
    control_only = int(
        table[("control_rescued", "treatment_failed")]
    )
    treatment_only = int(
        table[("control_failed", "treatment_rescued")]
    )
    discordant = control_only + treatment_only
    if discordant:
        smaller = min(control_only, treatment_only)
        exact_p = min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, value)
                for value in range(smaller + 1)
            )
            / (2**discordant),
        )
        continuity_statistic = (
            max(0, abs(treatment_only - control_only) - 1) ** 2
            / discordant
        )
    else:
        exact_p = 1.0
        continuity_statistic = 0.0
    pairs = len(by_task)
    treatment_rate = sum(
        int(arms[treatment_arm]) for arms in by_task.values()
    ) / pairs
    control_rate = sum(
        int(arms[control_arm]) for arms in by_task.values()
    ) / pairs
    return {
        "schema": "paired-mcnemar-visible-rescue-v1",
        "predeclared": True,
        "experimental_unit": "unique_dead_task_group",
        "control_arm": control_arm,
        "treatment_arm": treatment_arm,
        "outcome": "group_visible_rescued_itt",
        "pairs": pairs,
        "table": {
            "neither_rescued": int(
                table[("control_failed", "treatment_failed")]
            ),
            "control_only_rescued": control_only,
            "treatment_only_rescued": treatment_only,
            "both_rescued": int(
                table[("control_rescued", "treatment_rescued")]
            ),
        },
        "discordant_pairs": discordant,
        "control_rescue_rate": control_rate,
        "treatment_rescue_rate": treatment_rate,
        "absolute_rate_difference_treatment_minus_control": (
            treatment_rate - control_rate
        ),
        "continuity_corrected_chi_square_statistic": (
            continuity_statistic
        ),
        "exact_two_sided_binomial_p_value": exact_p,
        "zero_discordance_p_value_policy": "p=1",
        "repair_candidates_are_not_independent_units": True,
    }


def max_min_diverse_indices(
    candidates: Sequence[str],
    k: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select ``K`` slots by deterministic greedy max-min dissimilarity.

    The first two slots are the globally most distant pair. Ties use content
    hashes; rollout position is only a final tie-break for byte-identical
    draws, for which every diversity statistic is necessarily identical.
    """

    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise RescueError("candidates must be a sequence")
    _plain_int(k, "diversity k", minimum=1)
    values = list(candidates)
    if len(values) < k:
        raise RescueError(
            f"cannot select {k} candidates from only {len(values)} draws"
        )
    if any(not isinstance(value, str) for value in values):
        raise RescueError("candidate code must be text")

    hashes = [sha256_text(value) for value in values]
    distances: dict[tuple[int, int], float] = {}

    def distance(left: int, right: int) -> float:
        key = (min(left, right), max(left, right))
        if key not in distances:
            distances[key] = code_distance(values[left], values[right])
        return distances[key]

    means = [
        (
            sum(
                distance(index, other)
                for other in range(len(values))
                if other != index
            )
            / max(1, len(values) - 1)
        )
        for index in range(len(values))
    ]
    if k == 1:
        selected = [
            min(
                range(len(values)),
                key=lambda index: (-means[index], hashes[index], index),
            )
        ]
    else:
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
            remaining = [
                index for index in range(len(values)) if index not in selected
            ]
            chosen = min(
                remaining,
                key=lambda index: (
                    -min(distance(index, member) for member in selected),
                    -means[index],
                    hashes[index],
                    index,
                ),
            )
            selected.append(chosen)

    pairwise = [
        distance(selected[left], selected[right])
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]
    return selected, {
        "schema": "normalized-token-trigram-jaccard-max-min-v1",
        "selected_count": len(selected),
        "selected_unique_code_sha256": len(
            {hashes[index] for index in selected}
        ),
        "minimum_pairwise_distance": min(pairwise) if pairwise else None,
        "mean_pairwise_distance": (
            sum(pairwise) / len(pairwise) if pairwise else None
        ),
        "all_pairwise_distances": pairwise,
        "rollout_index_used_as_diversity_measure": False,
    }


def _sanitize_score_detail(
    value: Mapping[str, Any],
    *,
    include_diagnostic: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RescueError("candidate scorer returned a non-object")
    passes = value.get("test_passes")
    if (
        not isinstance(passes, list)
        or not passes
        or any(not isinstance(item, bool) for item in passes)
    ):
        raise RescueError("candidate scorer returned invalid per-test outcomes")
    compiled = value.get("compiled")
    full_pass = value.get("full_pass")
    if not isinstance(compiled, bool) or not isinstance(full_pass, bool):
        raise RescueError("candidate scorer returned invalid compile/full flags")
    if full_pass != bool(compiled and all(passes)):
        raise RescueError("candidate scorer full-pass flag disagrees with tests")
    result = {
        "compiled": compiled,
        "full_pass": full_pass,
        "test_passes": list(passes),
        "passed_tests": sum(passes),
        "total_tests": len(passes),
    }
    if include_diagnostic:
        result["diagnostic"] = str(value.get("diagnostic") or "")[:4000]
    return result


def _score_many(
    jobs: Sequence[tuple[str, str, str]],
    *,
    timeout: int,
    stability_runs: int,
    workers: int,
    score_fn: Callable[..., Mapping[str, Any]],
    include_diagnostic: bool,
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    def run(job: tuple[str, str, str]) -> dict[str, Any]:
        candidate, tests, identity = job
        return _sanitize_score_detail(
            score_fn(
                candidate,
                tests,
                identity,
                timeout=timeout,
                stability_runs=stability_runs,
            ),
            include_diagnostic=include_diagnostic,
        )

    if workers <= 1:
        return [run(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(run, jobs))


def _f2_source(
    row: Mapping[str, Any],
    *,
    expected_system_prompt_sha256: str,
) -> tuple[str, str]:
    text = row.get("text")
    if (
        row.get("representation_schema") != "lossless-semantic-f2"
        or not isinstance(text, str)
        or not text.startswith("F2\n")
        or row.get("text_sha256") != sha256_text(text)
        or row.get("system_prompt_sha256")
        != expected_system_prompt_sha256
    ):
        raise RescueError(f"{_task_id(row, 'F2 row')}: invalid exact F2 row")
    return text, str(row["text_sha256"])


def _stable_group_rank(task_id: str, seed: int) -> str:
    return canonical_sha256(
        {
            "schema": "direct-compact-verpo-rescue-pilot-order-v1",
            "seed": seed,
            "task_id": task_id,
        }
    )


def build_pilot_plan(
    inference_rows: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    f2_rows: Sequence[Mapping[str, Any]],
    *,
    inference_provenance: Mapping[str, Any],
    source_format_guide: str,
    source_format_guide_sha256: str,
    input_records: Mapping[str, Mapping[str, Any]],
    select_k: int,
    repairs_per_candidate: int,
    max_groups: int = 0,
    seed: int = 260103525,
    timeout: int = 20,
    stability_runs: int = 1,
    workers: int = 1,
    mcnemar_minimum_difference: float = 0.05,
    mcnemar_assumed_discordance: float = 0.10,
    mcnemar_alpha: float = 0.05,
    mcnemar_power: float = 0.80,
    score_fn: Callable[..., Mapping[str, Any]] = score_dart_candidate,
) -> dict[str, Any]:
    """Build a hash-bound visible-only pilot plan from base inference."""

    _plain_int(select_k, "select_k", minimum=1)
    _plain_int(repairs_per_candidate, "repairs_per_candidate", minimum=1)
    _plain_int(max_groups, "max_groups")
    _plain_int(seed, "seed")
    _plain_int(timeout, "timeout", minimum=1)
    _plain_int(stability_runs, "stability_runs", minimum=1)
    _plain_int(workers, "workers", minimum=1)
    if (
        not isinstance(source_format_guide, str)
        or not source_format_guide.strip()
        or sha256_text(source_format_guide) != source_format_guide_sha256
    ):
        raise RescueError("F2 source-format guide/hash binding is invalid")
    if not isinstance(input_records, Mapping):
        raise RescueError("pilot input records are missing")

    fingerprint = checkpoint_fingerprint(inference_provenance)
    rollout_by_id = _index_unique(rollout_rows, "visible rollout")
    f2_by_id = _index_unique(f2_rows, "F2")
    inference_by_id = _index_unique(inference_rows, "base inference")
    ordered_ids = [_task_id(row, "base inference row") for row in inference_rows]

    jobs: list[tuple[str, str, str]] = []
    coordinates: list[tuple[str, int]] = []
    predictions_by_id: dict[str, list[str]] = {}
    for task_id in ordered_ids:
        rollout = rollout_by_id.get(task_id)
        f2 = f2_by_id.get(task_id)
        inference = inference_by_id[task_id]
        if rollout is None or f2 is None:
            raise RescueError(f"{task_id}: missing visible rollout or exact F2")
        if any(
            key in rollout
            for key in (
                "acceptance_tests",
                "hidden_tests",
                "reward_holdback_tests",
            )
        ):
            raise RescueError(f"{task_id}: private tests leaked into rollout")
        visible_tests = rollout.get("feedback_tests")
        predictions = inference.get("predictions")
        if (
            not isinstance(visible_tests, str)
            or not visible_tests.strip()
            or not isinstance(predictions, list)
            or len(predictions) < select_k
            or any(not isinstance(value, str) for value in predictions)
        ):
            raise RescueError(f"{task_id}: invalid visible tests/predictions")
        _f2_source(
            f2,
            expected_system_prompt_sha256=source_format_guide_sha256,
        )
        predictions_by_id[task_id] = list(predictions)
        for candidate_index, candidate in enumerate(predictions):
            jobs.append(
                (
                    candidate,
                    visible_tests,
                    f"rescue-base-{task_id}-{candidate_index}",
                )
            )
            coordinates.append((task_id, candidate_index))

    scored = _score_many(
        jobs,
        timeout=timeout,
        stability_runs=stability_runs,
        workers=workers,
        score_fn=score_fn,
        include_diagnostic=True,
    )
    details_by_id: dict[str, list[dict[str, Any]]] = {
        task_id: [] for task_id in ordered_ids
    }
    for (task_id, candidate_index), detail in zip(
        coordinates, scored, strict=True
    ):
        if candidate_index != len(details_by_id[task_id]):
            raise RescueError("base scoring order changed")
        details_by_id[task_id].append(detail)

    eligible_ids = [
        task_id
        for task_id in ordered_ids
        if all(
            int(detail["passed_tests"]) == 0
            for detail in details_by_id[task_id]
        )
    ]
    ranked_eligible = sorted(
        eligible_ids,
        key=lambda task_id: (_stable_group_rank(task_id, seed), task_id),
    )
    selected_task_ids = (
        ranked_eligible[:max_groups] if max_groups else ranked_eligible
    )
    if not selected_task_ids:
        raise RescueError("base inference contains no visible all-zero group")
    groups: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        predictions = predictions_by_id[task_id]
        details = details_by_id[task_id]
        selected, diversity = max_min_diverse_indices(predictions, select_k)
        rollout = rollout_by_id[task_id]
        f2 = f2_by_id[task_id]
        source, source_sha = _f2_source(
            f2,
            expected_system_prompt_sha256=source_format_guide_sha256,
        )
        candidate_rows = []
        for candidate_rank, base_index in enumerate(selected):
            candidate = predictions[base_index]
            detail = details[base_index]
            candidate_rows.append(
                {
                    "candidate_rank": candidate_rank,
                    "base_candidate_index": base_index,
                    "candidate": candidate,
                    "candidate_sha256": sha256_text(candidate),
                    "visible_detail": detail,
                }
            )
        catalog = build_grounding_catalog(
            source,
            [row["candidate"] for row in candidate_rows],
            diagnostics=[
                row["visible_detail"]["diagnostic"] for row in candidate_rows
            ],
        )
        groups.append(
            {
                "task_id": task_id,
                "source": source,
                "source_sha256": source_sha,
                "source_record_sha256": canonical_sha256(dict(f2)),
                "source_format_guide": source_format_guide,
                "source_format_guide_sha256": source_format_guide_sha256,
                "visible_tests": str(rollout["feedback_tests"]),
                "visible_tests_sha256": sha256_text(
                    str(rollout["feedback_tests"])
                ),
                "candidates": candidate_rows,
                "diversity": diversity,
                "grounding_catalog_sha256": catalog.catalog_sha256,
            }
        )

    total_draws = sum(len(value) for value in predictions_by_id.values())
    body = {
        "schema": PILOT_PLAN_SCHEMA,
        "status": "complete",
        "policy": {
            "eligibility": (
                "every base draw passes zero visible feedback-test cases"
            ),
            "candidate_selection": (
                "normalized token-trigram Jaccard greedy max-min; content-hash "
                "ties; rollout position only for byte-identical draws"
            ),
            "task_selection": (
                "task-bound SHA-256 rank after eligibility; zero max_groups "
                "means all eligible groups"
            ),
            "seed": seed,
            "select_k": select_k,
            "repairs_per_candidate": repairs_per_candidate,
            "max_groups": max_groups,
            "visible_only": True,
        },
        "checkpoint": fingerprint,
        "inputs": {key: dict(value) for key, value in input_records.items()},
        "groups": groups,
        "funnel": {
            "inference_groups_seen": len(ordered_ids),
            "base_draws_scored": total_draws,
            "visible_all_zero_groups": len(eligible_ids),
            "pilot_groups_selected": len(groups),
            "base_candidates_selected": len(groups) * select_k,
            "groups_excluded_nonzero_visible": len(ordered_ids) - len(eligible_ids),
        },
        "budget": {
            "judge_calls_planned": len(groups),
            "arms": len(ARM_ORDER),
            "candidate_ranks_per_group": select_k,
            "repairs_per_candidate": repairs_per_candidate,
            "repair_slots_planned_total": (
                len(groups)
                * len(ARM_ORDER)
                * select_k
                * repairs_per_candidate
            ),
            "primary_itt_unit": "task_arm",
            "slot_itt_denominator_fixed_before_diagnosis": True,
        },
        "power": mcnemar_power_plan(
            len(groups),
            minimum_detectable_difference=mcnemar_minimum_difference,
            assumed_discordant_fraction=mcnemar_assumed_discordance,
            alpha=mcnemar_alpha,
            desired_power=mcnemar_power,
        ),
        "privacy": {
            "provider_visible_tests_only": True,
            "reward_holdback_opened": False,
            "reference_dart_copied_to_plan": False,
            "acceptance_tests_copied_to_plan": False,
            "final_175_holdout_touched": False,
        },
    }
    return seal_artifact(body, PLAN_HASH_FIELD)


def _validate_plan(plan: Mapping[str, Any]) -> str:
    digest = require_artifact_digest(
        plan,
        schema=PILOT_PLAN_SCHEMA,
        hash_field=PLAN_HASH_FIELD,
    )
    groups = plan.get("groups")
    policy = plan.get("policy")
    checkpoint = plan.get("checkpoint")
    if (
        not isinstance(groups, list)
        or not groups
        or not isinstance(policy, Mapping)
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("sha256")
        != canonical_sha256(checkpoint.get("binding"))
    ):
        raise RescueError("pilot plan structure/checkpoint binding is invalid")
    select_k = _plain_int(policy.get("select_k"), "plan select_k", minimum=1)
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise RescueError("pilot plan group is not an object")
        task_id = _task_id(group, "pilot plan group")
        if task_id in seen:
            raise RescueError(f"pilot plan duplicates task {task_id!r}")
        seen.add(task_id)
        candidates = group.get("candidates")
        source = group.get("source")
        visible_tests = group.get("visible_tests")
        if (
            not isinstance(source, str)
            or group.get("source_sha256") != sha256_text(source)
            or not isinstance(visible_tests, str)
            or group.get("visible_tests_sha256")
            != sha256_text(visible_tests)
            or not isinstance(candidates, list)
            or len(candidates) != select_k
        ):
            raise RescueError(f"{task_id}: pilot plan source/candidates changed")
        for rank, candidate in enumerate(candidates):
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("candidate_rank") != rank
                or not isinstance(candidate.get("candidate"), str)
                or candidate.get("candidate_sha256")
                != sha256_text(candidate["candidate"])
                or int(
                    (candidate.get("visible_detail") or {}).get(
                        "passed_tests", -1
                    )
                )
                != 0
            ):
                raise RescueError(f"{task_id}: candidate rank/binding changed")
    return digest


def _judge_payload(
    group: Mapping[str, Any],
    *,
    catalog: GroundingCatalog,
    guidance_mode: str,
) -> dict[str, Any]:
    payload = {
        "source": str(group["source"]),
        "source_sha256": str(group["source_sha256"]),
        "source_format_guide": str(group["source_format_guide"]),
        "tests": str(group["visible_tests"]),
        "candidates": [
            {
                "group_index": int(candidate["base_candidate_index"]),
                "candidate": str(candidate["candidate"]),
                "diagnostic": str(
                    candidate["visible_detail"].get("diagnostic") or ""
                ),
                "compiled": bool(
                    candidate["visible_detail"].get("compiled")
                ),
                "full_pass": False,
            }
            for candidate in group["candidates"]
        ],
        "reference_catalog": catalog,
        "reference_catalog_sha256": catalog.catalog_sha256,
        "guidance_mode": guidance_mode,
    }
    assert_api_safe_payload(
        {
            **payload,
            "reference_catalog": catalog.to_prompt_dict(),
        }
    )
    return payload


def _derive_diagnosis_only(
    diagnosis_and_steps: Mapping[str, Any],
) -> dict[str, Any]:
    """Hold diagnosis content fixed and remove only prescriptive steps."""

    diagnoses = diagnosis_and_steps.get("diagnoses")
    if not isinstance(diagnoses, list):
        raise RescueError("diagnosis-and-steps result has no diagnosis list")
    derived_rows: list[dict[str, Any]] = []
    for raw in diagnoses:
        if not isinstance(raw, Mapping):
            raise RescueError("diagnosis-and-steps row is not an object")
        row = dict(raw)
        row["repair_steps"] = []
        derived_rows.append(row)
    result = dict(diagnosis_and_steps)
    result["guidance_mode"] = "diagnosis_only"
    result["diagnoses"] = derived_rows
    result["derived_from_guidance_mode"] = JUDGE_CALL_MODE
    result["derivation"] = "exact_same_diagnosis_with_repair_steps_omitted"
    return result


def _diagnosis_call_key(
    *,
    plan_sha256: str,
    group: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "schema": "direct-compact-verpo-rescue-diagnosis-call-v1",
            "source_plan_sha256": plan_sha256,
            "task_id": str(group["task_id"]),
            "guidance_mode": JUDGE_CALL_MODE,
            "grounding_catalog_sha256": str(
                group["grounding_catalog_sha256"]
            ),
        }
    )


def _journal_event_body(
    *,
    event: str,
    event_index: int,
    previous_event_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DIAGNOSIS_JOURNAL_SCHEMA,
        "event": event,
        "event_index": event_index,
        "previous_event_sha256": previous_event_sha256,
        **dict(payload),
    }


def _append_diagnosis_journal_event(
    path: str | Path,
    *,
    event: str,
    payload: Mapping[str, Any],
    cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one fsync'd hash-chained event without rewriting prior calls."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cursor is None:
        existing = _load_diagnosis_journal_events(destination)
        event_index = len(existing)
        previous = (
            str(existing[-1]["journal_event_sha256"])
            if existing
            else DIAGNOSIS_JOURNAL_GENESIS_SHA256
        )
    else:
        event_index = _plain_int(
            cursor.get("event_index"),
            "diagnosis journal cursor event_index",
        )
        previous = str(cursor.get("head_sha256") or "")
        expected_size = _plain_int(
            cursor.get("size_bytes"),
            "diagnosis journal cursor size",
        )
        observed_size = destination.stat().st_size if destination.exists() else 0
        if (
            not _SHA_RE.fullmatch(previous)
            or observed_size != expected_size
        ):
            raise RescueError(
                "diagnosis journal changed outside this append process"
            )
    body = _journal_event_body(
        event=event,
        event_index=event_index,
        previous_event_sha256=previous,
        payload=payload,
    )
    sealed = {
        **body,
        "journal_event_sha256": canonical_sha256(body),
    }
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(sealed) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if cursor is not None:
        cursor["event_index"] = event_index + 1
        cursor["head_sha256"] = sealed["journal_event_sha256"]
        cursor["size_bytes"] = destination.stat().st_size
    return sealed


def _load_diagnosis_journal_events(
    path: str | Path,
) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return []
    rows = read_jsonl(source)
    previous = DIAGNOSIS_JOURNAL_GENESIS_SHA256
    for index, row in enumerate(rows):
        expected = str(row.get("journal_event_sha256") or "")
        body = {
            key: value
            for key, value in row.items()
            if key != "journal_event_sha256"
        }
        if (
            row.get("schema") != DIAGNOSIS_JOURNAL_SCHEMA
            or row.get("event_index") != index
            or row.get("previous_event_sha256") != previous
            or not _SHA_RE.fullmatch(expected)
            or canonical_sha256(body) != expected
        ):
            raise RescueError(
                f"diagnosis journal event {index} failed its hash chain"
            )
        previous = expected
    return rows


def _judge_contract(judge: Any, *, max_total_calls: int) -> dict[str, Any]:
    telemetry = (
        judge.telemetry()
        if callable(getattr(judge, "telemetry", None))
        else {}
    )
    contract = {
        "model": str(telemetry.get("model") or judge.model),
        "base_url": str(telemetry.get("base_url") or judge.base_url),
        "api_style": str(telemetry.get("api_style") or judge.api_style),
        "concurrency": int(getattr(judge, "concurrency", 1)),
        "max_tokens": int(telemetry.get("max_tokens")),
        "timeout_seconds": float(telemetry.get("timeout_seconds")),
        "max_retries": int(telemetry.get("max_retries")),
        "completion_retries": int(
            telemetry.get("completion_retries_allowed")
        ),
        "retry_max_tokens": int(getattr(judge, "retry_max_tokens")),
        "thinking_mode": str(telemetry.get("thinking_mode")),
        "reasoning_effort": str(telemetry.get("reasoning_effort")),
        "reasoning_mode": str(telemetry.get("reasoning_mode")),
        "chat_json_schema": bool(telemetry.get("chat_json_schema")),
        "fail_closed": bool(telemetry.get("fail_closed")),
        "max_total_calls": max_total_calls,
        "one_provider_call_per_dead_group": True,
        "one_billed_attempt_per_dead_group": True,
        "sdk_transport_retries": 0,
        "guidance_mode_sent": JUDGE_CALL_MODE,
        "diagnosis_only_is_derived": True,
    }
    if (
        contract["max_retries"] != 0
        or contract["completion_retries"] != 0
        or contract["retry_max_tokens"] != contract["max_tokens"]
    ):
        raise RescueError(
            "judge resolved retry settings violate one-billed-attempt contract"
        )
    return contract


def _diagnosis_journal_contract(
    plan: Mapping[str, Any],
    *,
    plan_sha256: str,
    judge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_plan_sha256": plan_sha256,
        "ordered_call_keys": [
            _diagnosis_call_key(plan_sha256=plan_sha256, group=group)
            for group in plan["groups"]
        ],
        "judge": dict(judge_contract),
        "grounding_validator_schema": GROUNDING_SCHEMA,
        "result_schema": DIAGNOSIS_ARTIFACT_SCHEMA,
    }


def _validate_receipt_attestation(
    attestation: Any,
    *,
    prior_count: int,
    prior_head: str,
    prior_response_ids: Sequence[str],
) -> tuple[int, str, list[str]]:
    """Validate embedded receipts and recover exact resume constructor state."""

    if attestation is None:
        return prior_count, prior_head, list(prior_response_ids)
    if not isinstance(attestation, Mapping):
        raise RescueError("diagnosis terminal has malformed receipt attestation")
    receipts = attestation.get("receipts")
    if (
        not isinstance(receipts, list)
        or attestation.get("receipt_count_before_step") != prior_count
        or attestation.get("previous_receipt_chain_sha256") != prior_head
        or attestation.get("receipt_count_this_step") != len(receipts)
    ):
        raise RescueError("diagnosis receipt-attestation cursor changed")
    count = prior_count
    head = prior_head
    response_ids = list(prior_response_ids)
    response_id_set = set(response_ids)
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise RescueError("diagnosis receipt is not an object")
        receipt = dict(raw)
        receipt_sha = str(receipt.pop("receipt_sha256", ""))
        if (
            receipt.get("receipt_index") != count + 1
            or receipt.get("previous_receipt_sha256") != head
            or not _SHA_RE.fullmatch(receipt_sha)
            or canonical_sha256(receipt) != receipt_sha
        ):
            raise RescueError("diagnosis provider receipt chain changed")
        count += 1
        head = receipt_sha
        response = receipt.get("response")
        validation = receipt.get("validation")
        response_id = (
            response.get("id") if isinstance(response, Mapping) else None
        )
        accepted = (
            validation.get("accepted")
            if isinstance(validation, Mapping)
            else False
        )
        if isinstance(response_id, str) and response_id and accepted is True:
            response_id_sha = sha256_text(response_id)
            if response_id_sha in response_id_set:
                raise RescueError(
                    "accepted provider response ID repeats across resume"
                )
            response_id_set.add(response_id_sha)
            response_ids.append(response_id_sha)
    if (
        attestation.get("cumulative_receipt_count") != count
        or attestation.get("cumulative_receipt_chain_sha256") != head
    ):
        raise RescueError("diagnosis receipt-attestation tail changed")
    return count, head, response_ids


def inspect_diagnosis_journal(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover completed calls and refuse ambiguous started calls."""

    events = _load_diagnosis_journal_events(path)
    if not events:
        return {
            "exists": False,
            "events": [],
            "completed_results": {},
            "receipt_count": 0,
            "receipt_chain_sha256": DIAGNOSIS_JOURNAL_GENESIS_SHA256,
            "response_id_sha256s": [],
            "complete": None,
        }
    header = events[0]
    if (
        header.get("event") != "journal_header"
        or header.get("contract") != dict(expected_contract)
        or header.get("contract_sha256")
        != canonical_sha256(expected_contract)
    ):
        raise RescueError("diagnosis journal contract differs from this run")
    ordered = list(expected_contract["ordered_call_keys"])
    completed: dict[str, Mapping[str, Any]] = {}
    receipt_count = 0
    receipt_head = DIAGNOSIS_JOURNAL_GENESIS_SHA256
    response_ids: list[str] = []
    active: str | None = None
    complete_event: Mapping[str, Any] | None = None
    next_call = 0
    for event in events[1:]:
        kind = event.get("event")
        if kind == "diagnosis_started":
            if active is not None or next_call >= len(ordered):
                raise RescueError("diagnosis journal start ordering is invalid")
            call_key = str(event.get("call_key") or "")
            if call_key != ordered[next_call]:
                raise RescueError("diagnosis journal call order changed")
            active = call_key
        elif kind == "diagnosis_terminal":
            call_key = str(event.get("call_key") or "")
            result = event.get("result")
            if (
                active != call_key
                or not isinstance(result, Mapping)
                or event.get("result_sha256") != canonical_sha256(result)
            ):
                raise RescueError("diagnosis journal terminal is invalid")
            (
                receipt_count,
                receipt_head,
                response_ids,
            ) = _validate_receipt_attestation(
                event.get("receipt_attestation"),
                prior_count=receipt_count,
                prior_head=receipt_head,
                prior_response_ids=response_ids,
            )
            completed[call_key] = dict(result)
            active = None
            next_call += 1
        elif kind == "diagnosis_complete":
            if (
                active is not None
                or next_call != len(ordered)
                or complete_event is not None
                or event.get("completed_call_keys") != ordered
            ):
                raise RescueError("diagnosis completion event is invalid")
            complete_event = event
        else:
            raise RescueError(f"unknown diagnosis journal event {kind!r}")
        if complete_event is not None and event is not events[-1]:
            raise RescueError("diagnosis journal has events after completion")
    if active is not None:
        raise RescueError(
            "diagnosis journal ends with a started but non-terminal paid call; "
            "provider idempotency is unavailable, so automatic retry is forbidden"
        )
    return {
        "exists": True,
        "events": events,
        "completed_results": completed,
        "receipt_count": receipt_count,
        "receipt_chain_sha256": receipt_head,
        "response_id_sha256s": response_ids,
        "complete": complete_event,
    }


class _JournalReplayJudge:
    """No-network judge facade used when every call is journaled."""

    def __init__(self, telemetry: Mapping[str, Any]):
        self._snapshot = dict(telemetry)

    def diagnose_group(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RescueError("journal replay attempted an uncompleted provider call")

    def telemetry(self) -> dict[str, Any]:
        return dict(self._snapshot)


def diagnose_pilot_plan(
    plan: Mapping[str, Any],
    *,
    judge: Any,
    receipt_journal_record: Mapping[str, Any] | None = None,
    completed_results: Mapping[str, Mapping[str, Any]] | None = None,
    before_call: Callable[[Mapping[str, Any]], None] | None = None,
    after_call: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute only the visible/API-safe diagnosis stage."""

    plan_sha = _validate_plan(plan)
    if not hasattr(judge, "diagnose_group"):
        raise RescueError("judge object has no diagnose_group method")
    rows: list[dict[str, Any]] = []
    accepted = Counter()
    rejected = Counter()
    rejection_causes: Counter[str] = Counter()
    for group in plan["groups"]:
        candidates = group["candidates"]
        catalog = build_grounding_catalog(
            str(group["source"]),
            [str(candidate["candidate"]) for candidate in candidates],
            diagnostics=[
                str(candidate["visible_detail"].get("diagnostic") or "")
                for candidate in candidates
            ],
        )
        if catalog.catalog_sha256 != group["grounding_catalog_sha256"]:
            raise RescueError(
                f"{group['task_id']}: grounding catalogue changed after planning"
            )
        call_key = _diagnosis_call_key(
            plan_sha256=plan_sha,
            group=group,
        )
        existing = (
            completed_results.get(call_key)
            if isinstance(completed_results, Mapping)
            else None
        )
        if existing is not None:
            result = dict(existing)
        else:
            payload = _judge_payload(
                group,
                catalog=catalog,
                guidance_mode=JUDGE_CALL_MODE,
            )
            receipt_cursor = int(
                (judge.telemetry() or {}).get("receipt_count", 0)
            )
            call_context = {
                "call_key": call_key,
                "task_id": group["task_id"],
                "guidance_mode": JUDGE_CALL_MODE,
                "grounding_catalog_sha256": catalog.catalog_sha256,
                "provider_payload_sha256": canonical_sha256(
                    {
                        **payload,
                        "reference_catalog": catalog.to_prompt_dict(),
                    }
                ),
                "receipt_cursor": receipt_cursor,
            }
            if before_call is not None:
                before_call(call_context)
            try:
                result = judge.diagnose_group(
                    payload,
                    guidance_mode=JUDGE_CALL_MODE,
                    item_validator=validate_diagnosis_item,
                    validator_schema_version=GROUNDING_SCHEMA,
                )
            except Exception as exc:
                result = {
                    "schema": "verpo-judge-diagnose-result-v1",
                    "guidance_mode": JUDGE_CALL_MODE,
                    "diagnoses": [
                        {
                            "group_index": int(
                                candidate["base_candidate_index"]
                            ),
                            "accepted": False,
                            "rejection_reasons": ["judge_call_failure"],
                            "fault_class": None,
                            "edit_location": None,
                            "evidence": [],
                            "explanation": "",
                            "repair_steps": [],
                        }
                        for candidate in candidates
                    ],
                    "failure_class": type(exc).__name__,
                    "failure_message_sha256": sha256_text(str(exc)),
                }
            attestation = (
                judge.receipt_attestation_since(receipt_cursor)
                if callable(
                    getattr(judge, "receipt_attestation_since", None)
                )
                else None
            )
            if after_call is not None:
                after_call(
                    {
                        **call_context,
                        "result": dict(result),
                        "result_sha256": canonical_sha256(result),
                        "receipt_attestation": attestation,
                    }
                )
        if not isinstance(result, Mapping):
            raise RescueError(
                f"{group['task_id']} {JUDGE_CALL_MODE}: judge result is not an object"
            )
        result = dict(result)
        diagnoses = result.get("diagnoses")
        if not isinstance(diagnoses, list) or len(diagnoses) != len(
            candidates
        ):
            raise RescueError(
                f"{group['task_id']} {JUDGE_CALL_MODE}: diagnosis coverage differs"
            )
        for expected, diagnosis in zip(candidates, diagnoses, strict=True):
            if (
                not isinstance(diagnosis, Mapping)
                or diagnosis.get("group_index")
                != expected["base_candidate_index"]
                or not isinstance(diagnosis.get("accepted"), bool)
            ):
                raise RescueError(
                    f"{group['task_id']} {JUDGE_CALL_MODE}: diagnosis identity changed"
                )
            for mode in GUIDANCE_MODES:
                if diagnosis["accepted"]:
                    accepted[mode] += 1
                else:
                    rejected[mode] += 1
            if not diagnosis["accepted"]:
                reasons = diagnosis.get("rejection_reasons")
                if not isinstance(reasons, list) or not reasons:
                    raise RescueError("rejected diagnosis has no reason")
                rejection_causes.update(str(reason) for reason in reasons)
        mode_results = {
            "diagnosis_and_steps": result,
            "diagnosis_only": _derive_diagnosis_only(result),
        }
        rows.append(
            {
                "task_id": group["task_id"],
                "grounding_catalog_sha256": catalog.catalog_sha256,
                "modes": mode_results,
            }
        )

    telemetry = (
        judge.telemetry() if callable(getattr(judge, "telemetry", None)) else {}
    )
    body = {
        "schema": DIAGNOSIS_ARTIFACT_SCHEMA,
        "status": "complete",
        "source_plan_sha256": plan_sha,
        "grounding_validator_schema": GROUNDING_SCHEMA,
        "rows": rows,
        "funnel": {
            "groups_attempted": len(plan["groups"]),
            "calls_planned": len(plan["groups"]),
            "calls_reused_from_journal": (
                len(completed_results or {})
            ),
            "diagnoses_requested": (
                len(plan["groups"])
                * int(plan["policy"]["select_k"])
            ),
            "accepted_by_mode": dict(accepted),
            "rejected_by_mode": dict(rejected),
            "rejection_causes": dict(sorted(rejection_causes.items())),
        },
        "judge_telemetry": telemetry,
        "receipt_journal": (
            dict(receipt_journal_record)
            if receipt_journal_record is not None
            else None
        ),
        "privacy": {
            "visible_tests_sent": True,
            "reward_holdback_argument_exists": False,
            "reward_holdback_sent": False,
            "reference_dart_sent": False,
            "plaintext_reasoning_required_or_persisted": False,
            "diagnosis_only_derived_without_second_provider_call": True,
        },
    }
    return seal_artifact(body, DIAGNOSIS_HASH_FIELD)


def _validate_diagnoses(
    diagnoses: Mapping[str, Any],
    *,
    plan_sha: str,
) -> str:
    digest = require_artifact_digest(
        diagnoses,
        schema=DIAGNOSIS_ARTIFACT_SCHEMA,
        hash_field=DIAGNOSIS_HASH_FIELD,
    )
    if diagnoses.get("source_plan_sha256") != plan_sha:
        raise RescueError("diagnosis artifact belongs to another pilot plan")
    return digest


def _diagnosis_lookup(
    diagnosis_row: Mapping[str, Any],
    *,
    mode: str,
) -> dict[int, Mapping[str, Any]]:
    modes = diagnosis_row.get("modes")
    result = modes.get(mode) if isinstance(modes, Mapping) else None
    diagnoses = result.get("diagnoses") if isinstance(result, Mapping) else None
    if not isinstance(diagnoses, list):
        return {}
    lookup: dict[int, Mapping[str, Any]] = {}
    for value in diagnoses:
        if not isinstance(value, Mapping):
            continue
        index = value.get("group_index")
        if type(index) is int and index >= 0 and index not in lookup:
            lookup[index] = value
    return lookup


def _conditioning_text(
    *,
    arm: str,
    candidate: Mapping[str, Any],
    diagnosis: Mapping[str, Any] | None,
) -> str:
    if arm == "plain_resample":
        return ""
    context: dict[str, Any] = {
        "schema": CONDITIONING_TEXT_SCHEMA,
        "arm": arm,
        "base_candidate": str(candidate["candidate"]),
        "visible_execution_diagnostic": str(
            candidate["visible_detail"].get("diagnostic") or ""
        ),
        "instruction": (
            "Return only a complete corrected Dart compilation unit. "
            "Do not return prose or fences."
        ),
    }
    if arm in {"diagnosis_only", "diagnosis_and_steps"}:
        if not isinstance(diagnosis, Mapping) or diagnosis.get("accepted") is not True:
            raise RescueError("cannot condition on a rejected diagnosis")
        context["judge_feedback"] = {
            "fault_class": diagnosis.get("fault_class"),
            "edit_location": diagnosis.get("edit_location"),
            "evidence": diagnosis.get("evidence"),
            "explanation": diagnosis.get("explanation"),
        }
        if arm == "diagnosis_and_steps":
            context["judge_feedback"]["repair_steps"] = diagnosis.get(
                "repair_steps"
            )
    return "\n\nRESCUE_CONTEXT_JSON\n" + canonical_json(context) + "\n"


def plan_key(arm: str, candidate_rank: int) -> str:
    return f"{arm}:rank{candidate_rank:03d}"


def materialize_conditioning_plans(
    plan: Mapping[str, Any],
    diagnoses: Mapping[str, Any],
) -> dict[str, Any]:
    """Build all ``4 x K`` fixed-slot inference plans in memory."""

    plan_sha = _validate_plan(plan)
    diagnosis_sha = _validate_diagnoses(diagnoses, plan_sha=plan_sha)
    select_k = int(plan["policy"]["select_k"])
    repairs = int(plan["policy"]["repairs_per_candidate"])
    diagnosis_by_id = _index_unique(diagnoses["rows"], "diagnoses")
    plans: dict[str, dict[str, Any]] = {}
    plan_records: list[dict[str, Any]] = []
    arm_funnel: dict[str, dict[str, int]] = {}

    for arm in ARM_ORDER:
        arm_generate = 0
        arm_rejected = 0
        for candidate_rank in range(select_k):
            rows: list[dict[str, Any]] = []
            for group in plan["groups"]:
                task_id = str(group["task_id"])
                candidate = group["candidates"][candidate_rank]
                diagnosis: Mapping[str, Any] | None = None
                rejection_reasons: list[str] = []
                generate = True
                if arm in {"diagnosis_only", "diagnosis_and_steps"}:
                    diagnosis_row = diagnosis_by_id.get(task_id)
                    lookup = (
                        _diagnosis_lookup(diagnosis_row, mode=arm)
                        if diagnosis_row is not None
                        else {}
                    )
                    diagnosis = lookup.get(
                        int(candidate["base_candidate_index"])
                    )
                    if (
                        diagnosis is None
                        or diagnosis.get("accepted") is not True
                    ):
                        generate = False
                        raw_reasons = (
                            diagnosis.get("rejection_reasons")
                            if isinstance(diagnosis, Mapping)
                            else ["diagnosis_missing"]
                        )
                        rejection_reasons = sorted(
                            {
                                str(reason)
                                for reason in (
                                    raw_reasons
                                    if isinstance(raw_reasons, list)
                                    else ["diagnosis_rejected"]
                                )
                                if str(reason)
                            }
                        ) or ["diagnosis_rejected"]
                conditioning = (
                    _conditioning_text(
                        arm=arm,
                        candidate=candidate,
                        diagnosis=diagnosis,
                    )
                    if generate
                    else ""
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "generate": generate,
                        "conditioning": conditioning,
                        "conditioning_sha256": sha256_text(conditioning),
                        "rejection_reasons": rejection_reasons,
                    }
                )
                arm_generate += int(generate)
                arm_rejected += int(not generate)
            inference_plan = {
                "schema": RESCUE_CONDITIONING_SCHEMA,
                "arm": arm,
                "base_candidate_rank": candidate_rank,
                "repairs_per_candidate": repairs,
                "source_plan_sha256": plan_sha,
                "rows": rows,
            }
            key = plan_key(arm, candidate_rank)
            plans[key] = inference_plan
            plan_records.append(
                {
                    "key": key,
                    "arm": arm,
                    "base_candidate_rank": candidate_rank,
                    "conditioning_plan_sha256": canonical_sha256(
                        inference_plan
                    ),
                    "rows": len(rows),
                    "generatable_rows": sum(
                        int(row["generate"]) for row in rows
                    ),
                    "rejected_rows": sum(
                        int(not row["generate"]) for row in rows
                    ),
                    "planned_repair_slots": len(rows) * repairs,
                    "generated_repair_slots_requested": sum(
                        int(row["generate"]) for row in rows
                    )
                    * repairs,
                }
            )
        arm_funnel[arm] = {
            "task_candidate_rows": len(plan["groups"]) * select_k,
            "generatable_rows": arm_generate,
            "rejected_rows": arm_rejected,
            "planned_repair_slots": (
                len(plan["groups"]) * select_k * repairs
            ),
            "generated_repair_slots_requested": arm_generate * repairs,
            "rejected_repair_slots": arm_rejected * repairs,
        }

    manifest_body = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "complete",
        "source_plan_sha256": plan_sha,
        "diagnosis_artifact_sha256": diagnosis_sha,
        "conditioning_schema": RESCUE_CONDITIONING_SCHEMA,
        "arms": list(ARM_ORDER),
        "candidate_ranks": select_k,
        "repairs_per_candidate": repairs,
        "plans": plan_records,
        "funnel_by_arm": arm_funnel,
        "invariants": {
            "all_four_arms_materialized": True,
            "every_task_candidate_reserves_fixed_R_slots": True,
            "rejected_diagnoses_preserved_generate_false": True,
            "rejected_slots_remain_in_itt_denominator": True,
            "conditioning_contains_no_holdback": True,
            "conditioning_contains_no_reference_dart": True,
        },
    }
    return {
        "manifest": seal_artifact(
            manifest_body, MATERIALIZATION_HASH_FIELD
        ),
        "plans": plans,
    }


def write_materialized_plans(
    materialized: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist canonical plan bytes and a path-bound manifest."""

    directory = Path(output_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    manifest = dict(materialized["manifest"])
    plans = materialized["plans"]
    records = []
    for record in manifest["plans"]:
        key = str(record["key"])
        path = directory / (key.replace(":", "__") + ".json")
        write_json_new(path, plans[key], canonical=True)
        observed = file_record(path)
        if observed["sha256"] != record["conditioning_plan_sha256"]:
            raise RescueError(f"{key}: canonical plan/file digest differs")
        records.append({**record, "artifact": observed})
    manifest["plans"] = records
    manifest = seal_artifact(manifest, MATERIALIZATION_HASH_FIELD)
    write_json_new(directory / "materialization.json", manifest)
    return manifest


def _validate_materialized(
    materialized: Mapping[str, Any],
    *,
    plan_sha: str,
) -> Mapping[str, Any]:
    manifest = materialized.get("manifest")
    plans = materialized.get("plans")
    if not isinstance(manifest, Mapping) or not isinstance(plans, Mapping):
        raise RescueError("materialized plans are malformed")
    require_artifact_digest(
        manifest,
        schema=MATERIALIZATION_SCHEMA,
        hash_field=MATERIALIZATION_HASH_FIELD,
    )
    if manifest.get("source_plan_sha256") != plan_sha:
        raise RescueError("materialization belongs to another pilot plan")
    for record in manifest.get("plans") or []:
        key = str(record.get("key") or "")
        value = plans.get(key)
        if (
            not isinstance(value, Mapping)
            or canonical_sha256(value)
            != record.get("conditioning_plan_sha256")
            or value.get("schema") != RESCUE_CONDITIONING_SCHEMA
        ):
            raise RescueError(f"materialized conditioning plan {key!r} changed")
    return manifest


def _validate_repair_run(
    run: Mapping[str, Any],
    *,
    conditioning_plan: Mapping[str, Any],
    conditioning_plan_sha256: str,
    checkpoint_sha256: str,
    expected_generation_binding: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    outputs = run.get("outputs")
    provenance = run.get("provenance")
    if not isinstance(outputs, list) or not isinstance(provenance, Mapping):
        raise RescueError("repair run must contain outputs and provenance")
    output_file_sha = str(run.get("output_file_sha256") or "")
    if (
        not _SHA_RE.fullmatch(output_file_sha)
        or provenance.get("output_sha256") != output_file_sha
    ):
        raise RescueError("repair output/provenance digest binding failed")
    if checkpoint_fingerprint(provenance)["sha256"] != checkpoint_sha256:
        raise RescueError("repair output used a different student checkpoint")
    rescue = provenance.get("rescue_conditioning")
    if (
        not isinstance(rescue, Mapping)
        or rescue.get("schema") != RESCUE_CONDITIONING_SCHEMA
        or rescue.get("sha256") != conditioning_plan_sha256
        or rescue.get("arm") != conditioning_plan["arm"]
        or rescue.get("base_candidate_rank")
        != conditioning_plan["base_candidate_rank"]
        or rescue.get("repairs_per_candidate")
        != conditioning_plan["repairs_per_candidate"]
        or rescue.get("source_plan_sha256")
        != conditioning_plan["source_plan_sha256"]
        or provenance.get("prediction_token_ids_persisted") is not True
    ):
        raise RescueError("repair provenance differs from conditioning plan")
    generation = repair_generation_binding(provenance)
    if (
        generation.get("selected_role") != "fit"
        or generation.get("batch_size") != 1
        or generation.get("limit") != 0
        or generation.get("num_samples")
        != conditioning_plan["repairs_per_candidate"]
        or generation.get("resampled_slots") != 0
        or generation.get("sampling_seed_policy")
        != (
            "sha256(base_seed,source_plan_sha256,"
            "base_candidate_rank,task_id)"
        )
        or generation.get("started_without_terminal_policy")
        != "retry_identical_seeded_batch_with_hash_chained_receipt"
        or not isinstance(generation.get("attn_implementation"), str)
        or not generation["attn_implementation"]
        or type(generation.get("bf16")) is not bool
        or type(generation.get("fp16")) is not bool
        or generation.get("precision") not in {"bf16", "fp16", "fp32"}
        or generation.get("precision")
        != (
            "bf16"
            if generation.get("bf16") is True
            else (
                "fp16"
                if generation.get("fp16") is True
                else "fp32"
            )
        )
        or (
            generation.get("bf16") is True
            and generation.get("fp16") is True
        )
        or any(
            not _SHA_RE.fullmatch(str(generation.get(field) or ""))
            for field in (
                "dataset_sha256",
                "alignment_sha256",
            )
        )
    ):
        raise RescueError(
            "repair inference lacks paired precision/RNG/resample provenance"
        )
    if (
        expected_generation_binding is not None
        and generation != expected_generation_binding
    ):
        raise RescueError("repair arms used different generation settings")

    by_id = _index_unique(outputs, "repair inference")
    expected_ids = {
        str(row["task_id"])
        for row in conditioning_plan["rows"]
        if row["generate"]
    }
    if set(by_id) != expected_ids:
        raise RescueError("repair inference task coverage differs from plan")
    repairs = int(conditioning_plan["repairs_per_candidate"])
    conditioning_by_id = {
        str(row["task_id"]): row for row in conditioning_plan["rows"]
    }
    for task_id, row in by_id.items():
        predictions = row.get("predictions")
        token_ids = row.get("prediction_token_ids")
        if (
            not isinstance(predictions, list)
            or len(predictions) != repairs
            or any(not isinstance(value, str) for value in predictions)
            or not isinstance(token_ids, list)
            or len(token_ids) != repairs
            or any(
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                for tokens in token_ids
            )
            or row.get("conditioning_sha256")
            != conditioning_by_id[task_id]["conditioning_sha256"]
        ):
            raise RescueError(f"{task_id}: repair output slots are invalid")
    return by_id, generation


def _visible_selection_key(slot: Mapping[str, Any]) -> tuple[Any, ...]:
    detail = slot["visible_detail"]
    return (
        -int(detail["passed_tests"]),
        -int(bool(detail["full_pass"])),
        -int(bool(detail["compiled"])),
        str(slot["repair_sha256"]),
        int(slot["base_candidate_rank"]),
        int(slot["repair_rank"]),
    )


def _private_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    # Never persist private diagnostics or source tests.
    return {
        "compiled": bool(detail["compiled"]),
        "full_pass": bool(detail["full_pass"]),
        "test_passes": list(detail["test_passes"]),
        "passed_tests": int(detail["passed_tests"]),
        "total_tests": int(detail["total_tests"]),
        "diagnostic_persisted": False,
    }


def score_rescue_outputs(
    plan: Mapping[str, Any],
    materialized: Mapping[str, Any],
    repair_runs: Mapping[str, Mapping[str, Any]],
    holdback_rows: Sequence[Mapping[str, Any]],
    *,
    private_holdback_commitment: Mapping[str, Any],
    timeout: int = 20,
    stability_runs: int = 1,
    workers: int = 1,
    score_fn: Callable[..., Mapping[str, Any]] = score_dart_candidate,
) -> dict[str, Any]:
    """Score fixed slots and return report plus separate training exports."""

    _plain_int(workers, "workers", minimum=1)
    plan_sha = _validate_plan(plan)
    manifest = _validate_materialized(materialized, plan_sha=plan_sha)
    if not isinstance(repair_runs, Mapping):
        raise RescueError("repair_runs must be keyed by arm/rank")
    if (
        not isinstance(private_holdback_commitment, Mapping)
        or not _SHA_RE.fullmatch(
            str(private_holdback_commitment.get("sha256") or "")
        )
        or type(private_holdback_commitment.get("size_bytes")) is not int
        or private_holdback_commitment["size_bytes"] < 0
        or "path" in private_holdback_commitment
    ):
        raise RescueError(
            "private holdback commitment must contain hash/size but no path"
        )
    holdback_by_id = _index_unique(holdback_rows, "private reward holdback")
    for group in plan["groups"]:
        private = holdback_by_id.get(str(group["task_id"]))
        tests = (
            private.get("reward_holdback_tests")
            if isinstance(private, Mapping)
            else None
        )
        if not isinstance(tests, str) or not tests.strip():
            raise RescueError(
                f"{group['task_id']}: no private reward-holdback tests"
            )

    plan_records = {
        str(record["key"]): record for record in manifest["plans"]
    }
    checkpoint_sha = str(plan["checkpoint"]["sha256"])
    generation_binding: dict[str, Any] | None = None
    validated_runs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for key, run in repair_runs.items():
        if key not in plan_records or key not in materialized["plans"]:
            raise RescueError(f"repair run has unknown plan key {key!r}")
        by_id, observed_generation = _validate_repair_run(
            run,
            conditioning_plan=materialized["plans"][key],
            conditioning_plan_sha256=plan_records[key][
                "conditioning_plan_sha256"
            ],
            checkpoint_sha256=checkpoint_sha,
            expected_generation_binding=generation_binding,
        )
        if generation_binding is None:
            generation_binding = observed_generation
        validated_runs[key] = by_id

    select_k = int(plan["policy"]["select_k"])
    repairs = int(plan["policy"]["repairs_per_candidate"])
    group_by_id = {str(group["task_id"]): group for group in plan["groups"]}
    visible_cache: dict[tuple[str, str], dict[str, Any]] = {}
    private_cache: dict[tuple[str, str], dict[str, Any]] = {}
    slot_results: list[dict[str, Any]] = []
    task_arm_results: list[dict[str, Any]] = []
    rs_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    preferences: dict[tuple[str, str, str], dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}

    # Precompute each unique generated visible slot in stable order. This is
    # the large scoring surface and is safe to parallelize because it opens no
    # private bytes and selection still happens only after every score exists.
    visible_job_map: dict[
        tuple[str, str], tuple[str, str, str]
    ] = {}
    for run_key in sorted(validated_runs):
        conditioning_plan = materialized["plans"][run_key]
        arm_name = str(conditioning_plan["arm"])
        candidate_rank = int(conditioning_plan["base_candidate_rank"])
        for task_id in sorted(validated_runs[run_key]):
            output = validated_runs[run_key][task_id]
            tests = str(group_by_id[task_id]["visible_tests"])
            for repair_rank, code in enumerate(output["predictions"]):
                cache_key = (sha256_text(str(code)), sha256_text(tests))
                visible_job_map.setdefault(
                    cache_key,
                    (
                        str(code),
                        tests,
                        (
                            f"rescue-visible-{arm_name}-{task_id}-"
                            f"{candidate_rank}-{repair_rank}"
                        ),
                    ),
                )
    visible_job_keys = list(visible_job_map)
    visible_details = _score_many(
        [visible_job_map[key] for key in visible_job_keys],
        timeout=timeout,
        stability_runs=stability_runs,
        workers=workers,
        score_fn=score_fn,
        include_diagnostic=True,
    )
    visible_cache.update(
        zip(visible_job_keys, visible_details, strict=True)
    )

    def visible_score(code: str, tests: str, identity: str) -> dict[str, Any]:
        del identity
        key = (sha256_text(code), sha256_text(tests))
        if key not in visible_cache:
            raise RescueError(
                "visible repair was not present in the sealed inference run"
            )
        return dict(visible_cache[key])

    def private_score_pair(
        repair_code: str,
        base_code: str,
        tests: str,
        *,
        identity_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        values = (repair_code, base_code)
        keys = [
            (sha256_text(code), sha256_text(tests)) for code in values
        ]
        missing_keys: list[tuple[str, str]] = []
        missing_jobs: list[tuple[str, str, str]] = []
        for role, code, key in zip(
            ("repair", "base"), values, keys, strict=True
        ):
            if key not in private_cache and key not in missing_keys:
                missing_keys.append(key)
                missing_jobs.append(
                    (code, tests, f"{identity_prefix}-{role}")
                )
        if missing_jobs:
            details = _score_many(
                missing_jobs,
                timeout=timeout,
                stability_runs=stability_runs,
                workers=workers,
                score_fn=score_fn,
                include_diagnostic=False,
            )
            private_cache.update(
                zip(missing_keys, details, strict=True)
            )
        return dict(private_cache[keys[0]]), dict(private_cache[keys[1]])

    for arm in ARM_ORDER:
        funnel = Counter()
        cell_counts = Counter()
        for group in plan["groups"]:
            task_id = str(group["task_id"])
            visible_tests = str(group["visible_tests"])
            private_tests = str(
                holdback_by_id[task_id]["reward_holdback_tests"]
            )
            task_slots: list[dict[str, Any]] = []
            for candidate_rank in range(select_k):
                key = plan_key(arm, candidate_rank)
                conditioning_plan = materialized["plans"][key]
                row_by_id = {
                    str(row["task_id"]): row
                    for row in conditioning_plan["rows"]
                }
                conditioning_row = row_by_id[task_id]
                base_candidate = group["candidates"][candidate_rank]
                output = (
                    validated_runs.get(key, {}).get(task_id)
                    if conditioning_row["generate"]
                    else None
                )
                predictions = (
                    list(output["predictions"])
                    if output is not None
                    else []
                )
                token_ids = (
                    list(output["prediction_token_ids"])
                    if output is not None
                    else []
                )
                for repair_rank in range(repairs):
                    funnel["planned_slots"] += 1
                    if not conditioning_row["generate"]:
                        status = "diagnosis_rejected"
                        funnel["diagnosis_rejected_slots"] += 1
                        repair = None
                    elif repair_rank >= len(predictions):
                        status = "missing_output"
                        funnel["missing_output_slots"] += 1
                        repair = None
                    else:
                        status = "generated"
                        repair = predictions[repair_rank]
                        funnel["generated_slots"] += 1
                    slot: dict[str, Any] = {
                        "task_id": task_id,
                        "arm": arm,
                        "base_candidate_rank": candidate_rank,
                        "base_candidate_sha256": base_candidate[
                            "candidate_sha256"
                        ],
                        "repair_rank": repair_rank,
                        "status": status,
                        "planned": True,
                        "conditioning_sha256": conditioning_row[
                            "conditioning_sha256"
                        ],
                        "rejection_reasons": list(
                            conditioning_row["rejection_reasons"]
                        ),
                        "repair": repair,
                        "repair_sha256": (
                            sha256_text(repair)
                            if isinstance(repair, str)
                            else None
                        ),
                        "prediction_token_ids_sha256": (
                            canonical_sha256(token_ids[repair_rank])
                            if repair_rank < len(token_ids)
                            else None
                        ),
                        "visible_detail": None,
                    }
                    if repair is not None:
                        detail = visible_score(
                            repair,
                            visible_tests,
                            (
                                f"rescue-visible-{arm}-{task_id}-"
                                f"{candidate_rank}-{repair_rank}"
                            ),
                        )
                        slot["visible_detail"] = detail
                        funnel["visible_compiled_slots"] += int(
                            detail["compiled"]
                        )
                        funnel["visible_any_pass_slots"] += int(
                            detail["passed_tests"] > 0
                        )
                        funnel["visible_full_pass_slots"] += int(
                            detail["full_pass"]
                        )
                        task_slots.append(slot)
                    slot_results.append(slot)

            selected = (
                min(task_slots, key=_visible_selection_key)
                if task_slots
                else None
            )
            funnel["attempted_groups"] += 1
            if selected is None:
                funnel["groups_without_generated_repair"] += 1
                task_arm_results.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "selected_slot": None,
                        "visible_rescued": False,
                        "holdback_evaluated": False,
                        "visible_holdback_cell": "no_effect",
                        "realized_repair_diversity": realized_code_diversity(
                            []
                        ),
                    }
                )
                cell_counts["no_effect"] += 1
                continue

            funnel["groups_with_generated_repair"] += 1
            selected_detail = selected["visible_detail"]
            visible_up = int(selected_detail["passed_tests"]) > 0
            funnel["groups_visible_rescued"] += int(visible_up)
            base_rank = int(selected["base_candidate_rank"])
            base = group["candidates"][base_rank]
            repair_private, base_private = private_score_pair(
                str(selected["repair"]),
                str(base["candidate"]),
                private_tests,
                identity_prefix=(
                    f"rescue-private-{arm}-{task_id}-{base_rank}"
                ),
            )
            holdback_delta = (
                int(repair_private["passed_tests"])
                - int(base_private["passed_tests"])
            )
            holdback_up = holdback_delta > 0
            task_diversity = realized_code_diversity(
                [str(slot["repair"]) for slot in task_slots]
            )
            if task_diversity["mean_pairwise_distance"] is not None:
                funnel["diversity_tasks_with_pairs"] += 1
                funnel["sum_task_mean_pairwise_distance"] += float(
                    task_diversity["mean_pairwise_distance"]
                )
            funnel["sum_task_unique_normalized_programs"] += int(
                task_diversity["unique_normalized_programs"]
            )
            if visible_up and holdback_up:
                cell = "genuine_rescue"
            elif visible_up:
                cell = "visible_only_overfit"
            elif holdback_up:
                cell = "private_only_check"
            else:
                cell = "no_effect"
            cell_counts[cell] += 1
            funnel["selected_holdback_improved"] += int(holdback_up)
            funnel["selected_visible_and_holdback_full"] += int(
                selected_detail["full_pass"] and repair_private["full_pass"]
            )
            selected_summary = {
                "base_candidate_rank": base_rank,
                "repair_rank": int(selected["repair_rank"]),
                "repair_sha256": selected["repair_sha256"],
                "selection_rule": (
                    "max visible passed tests, then visible full/compile, "
                    "then repair SHA-256; no holdback input"
                ),
                "visible_detail": selected_detail,
                "holdback_detail": _private_summary(repair_private),
                "base_holdback_detail": _private_summary(base_private),
                "visible_delta_passed_tests": int(
                    selected_detail["passed_tests"]
                ),
                "holdback_delta_passed_tests": holdback_delta,
            }
            task_arm_results.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "selected_slot": selected_summary,
                    "visible_rescued": visible_up,
                    "holdback_evaluated": True,
                    "visible_holdback_cell": cell,
                    "realized_repair_diversity": task_diversity,
                }
            )

            code = str(selected["repair"])
            code_sha = str(selected["repair_sha256"])
            if selected_detail["full_pass"] and repair_private["full_pass"]:
                export_key = (task_id, code_sha)
                contributor = {
                    "arm": arm,
                    "base_candidate_rank": base_rank,
                    "repair_rank": int(selected["repair_rank"]),
                }
                if export_key not in rs_candidates:
                    rs_candidates[export_key] = {
                        "schema": RS_SFT_TARGET_SCHEMA,
                        "task_id": task_id,
                        "code": code,
                        "code_sha256": code_sha,
                        "target_mode": "final_dart_code_only",
                        "reasoning_in_target": False,
                        "student_checkpoint_sha256": checkpoint_sha,
                        "source_plan_sha256": plan_sha,
                        "visible_full_pass": True,
                        "development_reward_holdback_full_pass": True,
                        "development_reward_holdback_tests_sha256": sha256_text(
                            private_tests
                        ),
                        "development_holdback_consumed_for_transfer_selection": True,
                        "final_175_holdout_touched": False,
                        "provider_saw_development_holdback": False,
                        "contributors": [contributor],
                    }
                else:
                    rs_candidates[export_key]["contributors"].append(
                        contributor
                    )
            elif visible_up and holdback_up:
                rejected_code = str(base["candidate"])
                rejected_sha = str(base["candidate_sha256"])
                preference_key = (task_id, code_sha, rejected_sha)
                if preference_key not in preferences:
                    preferences[preference_key] = {
                        "schema": PREFERENCE_PAIR_SCHEMA,
                        "task_id": task_id,
                        "chosen": code,
                        "chosen_sha256": code_sha,
                        "rejected": rejected_code,
                        "rejected_sha256": rejected_sha,
                        "chosen_visible_passed_tests": int(
                            selected_detail["passed_tests"]
                        ),
                        "rejected_visible_passed_tests": 0,
                        "chosen_holdback_delta_passed_tests": holdback_delta,
                        "off_policy": True,
                        "different_conditioning_prefixes": True,
                        "eligible_for_on_policy_verpo_update": False,
                        "kept_separate_from_rs_sft_targets": True,
                        "source_plan_sha256": plan_sha,
                    }

        attempted = int(funnel["attempted_groups"])
        planned = int(funnel["planned_slots"])
        metrics[arm] = {
            **dict(funnel),
            "visible_change_x_private_change_2x2": {
                key: int(cell_counts[key])
                for key in (
                    "genuine_rescue",
                    "visible_only_overfit",
                    "private_only_check",
                    "no_effect",
                )
            },
            "visible_only_overfit_is_observational_label": True,
            "realized_code_diversity": {
                "tasks_with_generated_programs": int(
                    funnel["groups_with_generated_repair"]
                ),
                "tasks_with_pairwise_distance": int(
                    funnel["diversity_tasks_with_pairs"]
                ),
                "mean_of_task_mean_pairwise_distance": (
                    funnel["sum_task_mean_pairwise_distance"]
                    / funnel["diversity_tasks_with_pairs"]
                    if funnel["diversity_tasks_with_pairs"]
                    else None
                ),
                "mean_unique_normalized_programs_per_attempted_group": (
                    funnel["sum_task_unique_normalized_programs"] / attempted
                    if attempted
                    else 0.0
                ),
                "used_to_adjust_rescue_outcome": False,
            },
            "group_visible_rescue_rate_itt": (
                funnel["groups_visible_rescued"] / attempted
                if attempted
                else 0.0
            ),
            "slot_visible_any_pass_rate_itt": (
                funnel["visible_any_pass_slots"] / planned
                if planned
                else 0.0
            ),
            "slot_visible_full_pass_rate_itt": (
                funnel["visible_full_pass_slots"] / planned
                if planned
                else 0.0
            ),
            "itt_denominator_includes_rejected_and_missing": True,
        }

    rs_rows = sorted(
        rs_candidates.values(),
        key=lambda row: (row["task_id"], row["code_sha256"]),
    )
    preference_rows = sorted(
        preferences.values(),
        key=lambda row: (
            row["task_id"],
            row["chosen_sha256"],
            row["rejected_sha256"],
        ),
    )
    primary_contrast = paired_mcnemar_contrast(task_arm_results)
    report_body = {
        "schema": SCORE_ARTIFACT_SCHEMA,
        "status": "complete",
        "source_plan_sha256": plan_sha,
        "materialization_sha256": manifest[MATERIALIZATION_HASH_FIELD],
        "student_checkpoint_sha256": checkpoint_sha,
        "repair_generation_binding": generation_binding,
        "predeclared_power": dict(plan.get("power") or {}),
        "primary_paired_contrast": primary_contrast,
        "private_reward_holdback_commitment": dict(
            private_holdback_commitment
        ),
        "selection_policy": {
            "same_repair_visible_and_holdback": True,
            "holdback_used_in_selection": False,
            "rule": (
                "max visible passed tests; then visible full-pass; then "
                "visible compile; then repair SHA-256; then sealed slot rank"
            ),
            "base_comparator": (
                "the selected repair's originating base candidate"
            ),
        },
        "metrics_by_arm": metrics,
        "slot_results": slot_results,
        "task_arm_results": task_arm_results,
        "exports": {
            "rs_sft_rows": len(rs_rows),
            "partial_preference_rows": len(preference_rows),
            "rs_sft_requires_full_visible_and_holdback": True,
            "preference_pairs_are_separate_off_policy": True,
        },
        "privacy": {
            "holdback_test_source_persisted": False,
            "holdback_diagnostic_persisted": False,
            "holdback_exposed_to_provider": False,
            "reference_dart_exposed_to_provider": False,
            "final_175_holdout_touched": False,
            "development_holdback_is_now_consumed_for_transfer_selection": True,
        },
    }
    return {
        "report": seal_artifact(report_body, SCORE_HASH_FIELD),
        "rs_sft_targets": rs_rows,
        "preference_pairs": preference_rows,
    }


def _f2_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    f2_record: Mapping[str, Any],
) -> tuple[str, str]:
    contract = manifest.get("f2_prompt_contract")
    output = manifest.get("output")
    if (
        not isinstance(contract, Mapping)
        or contract.get("representation_schema") != "lossless-semantic-f2"
        or contract.get("all_rows_within_limit") is not True
        or not isinstance(output, Mapping)
        or output.get("sha256") != f2_record.get("sha256")
    ):
        raise RescueError("F2 manifest/output contract is invalid")
    guide = contract.get("system_prompt")
    guide_sha = str(contract.get("system_prompt_sha256") or "")
    if (
        not isinstance(guide, str)
        or not guide.strip()
        or sha256_text(guide) != guide_sha
    ):
        raise RescueError("F2 manifest system prompt/hash is invalid")
    return guide, guide_sha


def _plan_command(args: argparse.Namespace) -> None:
    inference_path = Path(args.base_inference).expanduser().resolve()
    provenance_path = Path(args.base_provenance).expanduser().resolve()
    rollout_path = Path(args.rollout).expanduser().resolve()
    f2_path = Path(args.f2).expanduser().resolve()
    f2_manifest_path = Path(args.f2_manifest).expanduser().resolve()
    feedback = validate_feedback_view_report(
        args.feedback_view_report,
        args.expected_feedback_view_report_sha256,
        expected_outputs={
            "rollout": rollout_path,
            "f2": f2_path,
            "f2_manifest": f2_manifest_path,
        },
    )
    inference = read_json(inference_path)
    provenance = read_json(provenance_path)
    if not isinstance(inference, list) or not isinstance(provenance, Mapping):
        raise RescueError("base inference/provenance shapes are invalid")
    inference_record = file_record(inference_path)
    if provenance.get("output_sha256") != inference_record["sha256"]:
        raise RescueError("base inference output/provenance hash mismatch")
    f2_record = file_record(f2_path)
    f2_manifest = read_json(f2_manifest_path)
    if not isinstance(f2_manifest, Mapping):
        raise RescueError("F2 manifest is not an object")
    guide, guide_sha = _f2_manifest_contract(
        f2_manifest, f2_record=f2_record
    )
    plan = build_pilot_plan(
        inference,
        read_jsonl(rollout_path),
        read_jsonl(f2_path),
        inference_provenance=provenance,
        source_format_guide=guide,
        source_format_guide_sha256=guide_sha,
        input_records={
            "base_inference": inference_record,
            "base_provenance": file_record(provenance_path),
            "visible_rollout": file_record(rollout_path),
            "exact_f2": f2_record,
            "exact_f2_manifest": file_record(f2_manifest_path),
            "feedback_view_report": feedback["report_record"],
        },
        select_k=args.select_k,
        repairs_per_candidate=args.repairs_per_candidate,
        max_groups=args.max_groups,
        seed=args.seed,
        timeout=args.reward_timeout,
        stability_runs=args.stability_runs,
        workers=args.workers,
        mcnemar_minimum_difference=args.mcnemar_minimum_difference,
        mcnemar_assumed_discordance=args.mcnemar_assumed_discordance,
        mcnemar_alpha=args.mcnemar_alpha,
        mcnemar_power=args.mcnemar_power,
    )
    write_json_new(args.output, plan)


def _diagnose_command(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    if not isinstance(plan, Mapping):
        raise RescueError("pilot plan is not an object")
    plan_sha = _validate_plan(plan)
    journal_path = Path(args.receipt_journal).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    planned_calls = len(plan["groups"])
    max_total_calls = args.max_calls if args.max_calls > 0 else planned_calls
    if max_total_calls != planned_calls:
        raise RescueError(
            "diagnosis call budget must equal one call per selected dead group"
        )
    if args.max_retries != 0:
        raise RescueError(
            "paid rescue diagnosis requires max_retries=0: without provider "
            "idempotency, neither SDK transport retries nor completion "
            "retries can prove exactly one billed attempt per group"
        )

    # Resolve environment-backed defaults before sealing the journal contract.
    probe = VerpoJudge(
        model=args.model or None,
        base_url=args.base_url or None,
        api_style=args.api_style or None,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        completion_retries=0,
        retry_max_tokens=args.max_tokens,
        thinking_mode=args.thinking_mode or None,
        reasoning_effort=args.reasoning_effort or None,
        max_calls=max_total_calls,
        fail_closed=True,
    )
    judge_contract = _judge_contract(
        probe, max_total_calls=max_total_calls
    )
    contract = _diagnosis_journal_contract(
        plan,
        plan_sha256=plan_sha,
        judge_contract=judge_contract,
    )
    state = inspect_diagnosis_journal(
        journal_path,
        expected_contract=contract,
    )
    if not state["exists"]:
        _append_diagnosis_journal_event(
            journal_path,
            event="journal_header",
            payload={
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        state = inspect_diagnosis_journal(
            journal_path,
            expected_contract=contract,
        )

    complete = state["complete"]
    if complete is not None:
        artifact = complete.get("diagnosis_artifact")
        if (
            not isinstance(artifact, Mapping)
            or complete.get("diagnosis_artifact_sha256")
            != canonical_sha256(artifact)
        ):
            raise RescueError(
                "completed diagnosis journal has no valid sealed artifact"
            )
        require_artifact_digest(
            artifact,
            schema=DIAGNOSIS_ARTIFACT_SCHEMA,
            hash_field=DIAGNOSIS_HASH_FIELD,
        )
        if output_path.exists():
            observed = read_json(output_path)
            if observed != artifact:
                raise RescueError(
                    "diagnosis output differs from completed journal"
                )
        else:
            write_json_new(output_path, artifact)
        return
    if output_path.exists():
        raise RescueError(
            "diagnosis output exists before a journal completion event"
        )
    journal_cursor = {
        "event_index": len(state["events"]),
        "head_sha256": str(
            state["events"][-1]["journal_event_sha256"]
        ),
        "size_bytes": journal_path.stat().st_size,
    }

    completed_results = dict(state["completed_results"])
    remaining = planned_calls - len(completed_results)
    if remaining <= 0:
        judge: Any = _JournalReplayJudge(judge_contract)
    else:
        # VerpoJudge creates receipt files exclusively. Each process therefore
        # gets a fresh segment while the result journal remains append-only.
        segment_index = 0
        while True:
            receipt_segment = journal_path.with_name(
                f"{journal_path.name}.receipts-segment-"
                f"{segment_index:06d}.jsonl"
            )
            if not receipt_segment.exists():
                break
            segment_index += 1
        judge = VerpoJudge(
            model=probe.model,
            base_url=probe.base_url,
            api_style=probe.api_style,
            concurrency=probe.concurrency,
            max_tokens=probe.max_tokens,
            timeout_seconds=probe.timeout_seconds,
            max_retries=probe.max_retries,
            completion_retries=probe.completion_retries,
            retry_max_tokens=probe.retry_max_tokens,
            thinking_mode=probe.thinking_mode,
            reasoning_effort=probe.reasoning_effort,
            reasoning_mode=probe.reasoning_mode,
            chat_json_schema=probe.chat_json_schema,
            max_calls=remaining,
            fail_closed=True,
            receipt_chain_seed=state["receipt_chain_sha256"],
            receipt_index_offset=state["receipt_count"],
            prior_response_id_sha256s=state["response_id_sha256s"],
            receipt_journal_path=receipt_segment,
        )

    def before_call(context: Mapping[str, Any]) -> None:
        _append_diagnosis_journal_event(
            journal_path,
            event="diagnosis_started",
            payload=dict(context),
            cursor=journal_cursor,
        )

    def after_call(context: Mapping[str, Any]) -> None:
        _append_diagnosis_journal_event(
            journal_path,
            event="diagnosis_terminal",
            payload=dict(context),
            cursor=journal_cursor,
        )

    result = diagnose_pilot_plan(
        plan,
        judge=judge,
        completed_results=completed_results,
        before_call=before_call,
        after_call=after_call,
    )
    final_state = inspect_diagnosis_journal(
        journal_path,
        expected_contract=contract,
    )
    if len(final_state["completed_results"]) != planned_calls:
        raise RescueError("diagnosis journal is incomplete after execution")
    terminal_head = str(
        final_state["events"][-1]["journal_event_sha256"]
    )
    # Runtime segmentation and crash boundaries must not change the sealed
    # diagnosis artifact. All final telemetry is reconstructed from the
    # append-only terminal events.
    funnel = dict(result["funnel"])
    funnel["calls_reused_from_journal"] = planned_calls
    funnel["calls_reused_from_journal_semantics"] = (
        "all final results are loaded from hash-chained terminal events"
    )
    result["funnel"] = funnel
    result["judge_telemetry"] = {
        "contract": judge_contract,
        "terminal_calls": planned_calls,
        "receipt_count": final_state["receipt_count"],
        "receipt_chain_sha256": final_state["receipt_chain_sha256"],
        "unique_accepted_response_ids": len(
            final_state["response_id_sha256s"]
        ),
        "resume_segments_not_an_analysis_covariate": True,
    }
    result["receipt_journal"] = {
        "schema": DIAGNOSIS_JOURNAL_SCHEMA,
        "contract_sha256": canonical_sha256(contract),
        "terminal_journal_head_sha256": terminal_head,
        "terminal_event_count": len(final_state["events"]),
        "receipt_count": final_state["receipt_count"],
        "receipt_chain_sha256": final_state["receipt_chain_sha256"],
        "plaintext_reasoning_persisted": False,
    }
    result = seal_artifact(result, DIAGNOSIS_HASH_FIELD)
    _append_diagnosis_journal_event(
        journal_path,
        event="diagnosis_complete",
        payload={
            "completed_call_keys": list(contract["ordered_call_keys"]),
            "diagnosis_artifact": result,
            "diagnosis_artifact_sha256": canonical_sha256(result),
        },
        cursor=journal_cursor,
    )
    write_json_new(output_path, result)


def _materialize_command(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    diagnoses = read_json(args.diagnoses)
    if not isinstance(plan, Mapping) or not isinstance(diagnoses, Mapping):
        raise RescueError("plan/diagnoses must be objects")
    materialized = materialize_conditioning_plans(plan, diagnoses)
    write_materialized_plans(materialized, args.output_dir)


def _load_materialized_dir(path: str | Path) -> dict[str, Any]:
    directory = Path(path).expanduser().resolve()
    manifest = read_json(directory / "materialization.json")
    if not isinstance(manifest, Mapping):
        raise RescueError("materialization manifest is not an object")
    require_artifact_digest(
        manifest,
        schema=MATERIALIZATION_SCHEMA,
        hash_field=MATERIALIZATION_HASH_FIELD,
    )
    plans: dict[str, Any] = {}
    for record in manifest.get("plans") or []:
        artifact = record.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RescueError("materialization plan record has no artifact")
        plan_path = Path(str(artifact.get("path") or "")).resolve()
        if (
            not plan_path.is_file()
            or sha256_file(plan_path) != artifact.get("sha256")
            or artifact.get("sha256")
            != record.get("conditioning_plan_sha256")
        ):
            raise RescueError("materialized plan artifact changed")
        plans[str(record["key"])] = read_json(plan_path)
    return {"manifest": manifest, "plans": plans}


def build_inference_bundle(
    plan: Mapping[str, Any],
    materialized: Mapping[str, Any],
    repair_output_dir: str | Path,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Validate local repair outputs and seal their exact artifact set."""

    plan_sha = _validate_plan(plan)
    manifest = _validate_materialized(materialized, plan_sha=plan_sha)
    directory = Path(repair_output_dir).expanduser().resolve()
    if not directory.is_dir():
        raise RescueError(f"repair output directory does not exist: {directory}")
    records = {
        str(record["key"]): record for record in manifest["plans"]
    }
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    generation_binding: dict[str, Any] | None = None
    for key in sorted(records):
        record = records[key]
        stem = key.replace(":", "__")
        output_path = directory / f"{stem}.json"
        provenance_path = directory / f"{stem}.json.provenance.json"
        output_exists = output_path.is_file()
        provenance_exists = provenance_path.is_file()
        if output_exists != provenance_exists:
            raise RescueError(
                f"{key}: repair output/provenance are only partially present"
            )
        if not output_exists:
            if int(record.get("generatable_rows", 0)) > 0:
                missing.append(key)
            continue
        output_record = file_record(output_path)
        provenance_record = file_record(provenance_path)
        outputs = read_json(output_path)
        provenance = read_json(provenance_path)
        run = {
            "outputs": outputs,
            "provenance": provenance,
            "output_file_sha256": output_record["sha256"],
        }
        _by_id, observed_generation = _validate_repair_run(
            run,
            conditioning_plan=materialized["plans"][key],
            conditioning_plan_sha256=record["conditioning_plan_sha256"],
            checkpoint_sha256=str(plan["checkpoint"]["sha256"]),
            expected_generation_binding=generation_binding,
        )
        if generation_binding is None:
            generation_binding = observed_generation
        runs.append(
            {
                "key": key,
                "output": str(output_path),
                "provenance": str(provenance_path),
                "output_record": output_record,
                "provenance_record": provenance_record,
            }
        )
    if missing and not allow_missing:
        raise RescueError(
            "generatable conditioning plans lack inference outputs: "
            + ", ".join(missing)
        )
    body = {
        "schema": INFERENCE_BUNDLE_SCHEMA,
        "status": "complete" if not missing else "complete_with_missing_runs",
        "source_plan_sha256": plan_sha,
        "materialization_sha256": manifest[MATERIALIZATION_HASH_FIELD],
        "repair_output_directory": str(directory),
        "allow_missing": bool(allow_missing),
        "planned_runs": len(records),
        "generatable_runs": sum(
            int(record.get("generatable_rows", 0)) > 0
            for record in records.values()
        ),
        "present_runs": len(runs),
        "missing_generatable_runs": missing,
        "missing_runs_count_as_itt_failures": True,
        "repair_generation_binding": generation_binding,
        "runs": runs,
    }
    return seal_artifact(body, BUNDLE_HASH_FIELD)


def _bundle_command(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    if not isinstance(plan, Mapping):
        raise RescueError("pilot plan is not an object")
    materialized = _load_materialized_dir(args.materialized_dir)
    bundle = build_inference_bundle(
        plan,
        materialized,
        args.repair_output_dir,
        allow_missing=args.allow_missing,
    )
    write_json_new(args.output, bundle)


def _load_repair_bundle(
    path: str | Path,
    *,
    expected_plan_sha256: str,
    expected_materialization_sha256: str,
    expected_plan_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    bundle_path = Path(path).expanduser().resolve()
    value = read_json(bundle_path)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != INFERENCE_BUNDLE_SCHEMA
        or not isinstance(value.get("runs"), list)
    ):
        raise RescueError("repair inference bundle schema is invalid")
    require_artifact_digest(
        value,
        schema=INFERENCE_BUNDLE_SCHEMA,
        hash_field=BUNDLE_HASH_FIELD,
    )
    if (
        value.get("source_plan_sha256") != expected_plan_sha256
        or value.get("materialization_sha256")
        != expected_materialization_sha256
    ):
        raise RescueError("repair inference bundle belongs to another plan")
    known_keys = {
        str(record.get("key") or "") for record in expected_plan_records
    }
    generatable_keys = {
        str(record.get("key") or "")
        for record in expected_plan_records
        if int(record.get("generatable_rows", 0)) > 0
    }
    result: dict[str, dict[str, Any]] = {}
    for raw in value["runs"]:
        if not isinstance(raw, Mapping):
            raise RescueError("repair inference run entry is not an object")
        key = str(raw.get("key") or "")
        if not key or key in result or key not in known_keys:
            raise RescueError("repair inference bundle key is invalid/duplicate")
        output_path = Path(str(raw.get("output") or "")).expanduser().resolve()
        provenance_path = Path(
            str(raw.get("provenance") or "")
        ).expanduser().resolve()
        if not output_path.is_file() or not provenance_path.is_file():
            raise RescueError(f"{key}: repair output/provenance is missing")
        if (
            raw.get("output_record") != file_record(output_path)
            or raw.get("provenance_record") != file_record(provenance_path)
        ):
            raise RescueError(f"{key}: bundled repair artifact changed")
        outputs = read_json(output_path)
        provenance = read_json(provenance_path)
        if not isinstance(outputs, list) or not isinstance(provenance, Mapping):
            raise RescueError(f"{key}: repair artifacts have invalid shapes")
        result[key] = {
            "outputs": outputs,
            "provenance": provenance,
            "output_file_sha256": sha256_file(output_path),
        }
    observed_missing = sorted(generatable_keys - set(result))
    if (
        value.get("missing_generatable_runs") != observed_missing
        or (observed_missing and value.get("allow_missing") is not True)
    ):
        raise RescueError(
            "repair inference bundle missing-run accounting is invalid"
        )
    return result


def _score_command(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    if not isinstance(plan, Mapping):
        raise RescueError("pilot plan is not an object")
    materialized = _load_materialized_dir(args.materialized_dir)
    holdback_path = Path(args.private_holdback).expanduser().resolve()
    feedback = validate_feedback_view_report(
        args.feedback_view_report,
        args.expected_feedback_view_report_sha256,
        expected_outputs={"reward_holdback_private": holdback_path},
    )
    plan_feedback_record = (plan.get("inputs") or {}).get(
        "feedback_view_report"
    )
    if plan_feedback_record != feedback["report_record"]:
        raise RescueError(
            "score feedback-view report differs from the one sealed by plan"
        )
    holdback_record = dict(
        feedback["output_records"]["reward_holdback_private"]
    )
    holdback_record.pop("path", None)
    result = score_rescue_outputs(
        plan,
        materialized,
        _load_repair_bundle(
            args.repair_bundle,
            expected_plan_sha256=str(plan[PLAN_HASH_FIELD]),
            expected_materialization_sha256=str(
                materialized["manifest"][MATERIALIZATION_HASH_FIELD]
            ),
            expected_plan_records=materialized["manifest"]["plans"],
        ),
        read_jsonl(holdback_path),
        private_holdback_commitment=holdback_record,
        timeout=args.reward_timeout,
        stability_runs=args.stability_runs,
        workers=args.workers,
    )
    write_jsonl_new(args.rs_sft_output, result["rs_sft_targets"])
    write_jsonl_new(args.preferences_output, result["preference_pairs"])
    report = dict(result["report"])
    report["export_artifacts"] = {
        "rs_sft_targets": file_record(args.rs_sft_output),
        "preference_pairs": file_record(args.preferences_output),
    }
    report = seal_artifact(report, SCORE_HASH_FIELD)
    write_json_new(args.output, report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asynchronous fixed-budget direct-compact VeRPO rescue"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--base-inference", required=True)
    plan.add_argument("--base-provenance", required=True)
    plan.add_argument("--rollout", required=True)
    plan.add_argument("--f2", required=True)
    plan.add_argument("--f2-manifest", required=True)
    plan.add_argument("--feedback-view-report", required=True)
    plan.add_argument(
        "--expected-feedback-view-report-sha256",
        required=True,
    )
    plan.add_argument("--output", required=True)
    plan.add_argument("--select-k", type=int, default=4)
    plan.add_argument("--repairs-per-candidate", type=int, default=4)
    plan.add_argument("--max-groups", type=int, default=0)
    plan.add_argument("--seed", type=int, default=260103525)
    plan.add_argument("--reward-timeout", type=int, default=20)
    plan.add_argument("--stability-runs", type=int, default=1)
    plan.add_argument("--workers", type=int, default=8)
    plan.add_argument("--mcnemar-minimum-difference", type=float, default=0.05)
    plan.add_argument("--mcnemar-assumed-discordance", type=float, default=0.10)
    plan.add_argument("--mcnemar-alpha", type=float, default=0.05)
    plan.add_argument("--mcnemar-power", type=float, default=0.80)
    plan.set_defaults(func=_plan_command)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--plan", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--receipt-journal", required=True)
    diagnose.add_argument("--model", default="")
    diagnose.add_argument("--base-url", default="")
    diagnose.add_argument("--api-style", default="")
    diagnose.add_argument("--max-tokens", type=int, default=4096)
    diagnose.add_argument("--timeout-seconds", type=float, default=180.0)
    diagnose.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help=(
            "Must remain zero: provider idempotency is unavailable and each "
            "sealed dead group permits exactly one billed attempt."
        ),
    )
    diagnose.add_argument("--thinking-mode", default="provider_default")
    diagnose.add_argument("--reasoning-effort", default="high")
    diagnose.add_argument("--max-calls", type=int, default=0)
    diagnose.set_defaults(func=_diagnose_command)

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--plan", required=True)
    materialize.add_argument("--diagnoses", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.set_defaults(func=_materialize_command)

    bundle = sub.add_parser("bundle")
    bundle.add_argument("--plan", required=True)
    bundle.add_argument("--materialized-dir", required=True)
    bundle.add_argument("--repair-output-dir", required=True)
    bundle.add_argument("--output", required=True)
    bundle.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Seal absent generatable runs as ITT failures instead of failing "
            "bundle creation"
        ),
    )
    bundle.set_defaults(func=_bundle_command)

    score = sub.add_parser("score")
    score.add_argument("--plan", required=True)
    score.add_argument("--materialized-dir", required=True)
    score.add_argument("--repair-bundle", required=True)
    score.add_argument("--private-holdback", required=True)
    score.add_argument("--feedback-view-report", required=True)
    score.add_argument(
        "--expected-feedback-view-report-sha256",
        required=True,
    )
    score.add_argument("--output", required=True)
    score.add_argument("--rs-sft-output", required=True)
    score.add_argument("--preferences-output", required=True)
    score.add_argument("--reward-timeout", type=int, default=20)
    score.add_argument("--stability-runs", type=int, default=1)
    score.add_argument("--workers", type=int, default=8)
    score.set_defaults(func=_score_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


__all__ = [
    "ARM_ORDER",
    "DIAGNOSIS_ARTIFACT_SCHEMA",
    "DIAGNOSIS_JOURNAL_SCHEMA",
    "INFERENCE_BUNDLE_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "PILOT_PLAN_SCHEMA",
    "PREFERENCE_PAIR_SCHEMA",
    "RS_SFT_TARGET_SCHEMA",
    "SCORE_ARTIFACT_SCHEMA",
    "RescueError",
    "assert_api_safe_payload",
    "build_inference_bundle",
    "build_pilot_plan",
    "canonical_sha256",
    "checkpoint_fingerprint",
    "code_distance",
    "diagnose_pilot_plan",
    "inspect_diagnosis_journal",
    "materialize_conditioning_plans",
    "max_min_diverse_indices",
    "paired_mcnemar_contrast",
    "parse_args",
    "plan_key",
    "score_rescue_outputs",
    "validate_feedback_view_report",
    "write_materialized_plans",
]


if __name__ == "__main__":
    main()
