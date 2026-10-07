#!/usr/bin/env python3
"""Offline Qwen teacher collection and artifact materialization.

This module is deliberately independent of every optimizer/trainer.  It accepts
API-readable compact prompts which were serialized and round-trip verified
before collection, journals one API response per sample, and only then creates
immutable training-data views.

The teacher samples are Monte-Carlo sequence samples.  Top-5 log probabilities
are retained for audit and possible explicitly sparse objectives; they are
never represented as dense/full-vocabulary KL targets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


DEFAULT_MODEL = "qwen3.8-max-preview"
SAMPLES_PER_TASK = 8
TOP_LOGPROBS = 5
OBJECTIVE_MODE_REQUIRE_TOP5 = "require_top5"
OBJECTIVE_MODE_SEQUENCE_ONLY = "sequence_only"
OBJECTIVE_MODES = frozenset(
    {OBJECTIVE_MODE_REQUIRE_TOP5, OBJECTIVE_MODE_SEQUENCE_ONLY}
)
SAMPLE_SEED_ALGORITHM = (
    "sha256(seed_base NUL task_id NUL sample_index)[:8] mod 2^31"
)
NEGATIVE_TAIL_TOLERANCE = 1e-8
JOURNAL_SCHEMA = "qwen-direct-compact-teacher-journal-v1"
CANDIDATE_SCHEMA = "qwen-direct-compact-teacher-candidate-v1"
REJECTED_DRAW_SCHEMA = "qwen-direct-compact-teacher-rejected-draw-v1"
SLOT_STARTED_SCHEMA = "qwen-direct-compact-teacher-slot-started-v1"
ORPHAN_REISSUE_AUTHORIZATION_SCHEMA = (
    "qwen-direct-compact-teacher-orphan-reissue-authorization-v1"
)
ORPHAN_REISSUE_REAUTHORIZATION_SCHEMA = (
    "qwen-direct-compact-teacher-orphan-reissue-implementation-"
    "reauthorization-v1"
)
ORPHAN_REISSUE_ATTEMPT_SCHEMA = (
    "qwen-direct-compact-teacher-orphan-reissue-attempt-started-v1"
)
SLOT_TERMINAL_SCHEMA = "qwen-direct-compact-teacher-slot-terminal-v1"
VERIFICATION_SCHEMA = "qwen-direct-compact-teacher-verification-v1"
AUDIT_SCHEMA = "qwen-direct-compact-teacher-audit-v1"
PARSEABLE_SCHEMA = "qwen-direct-compact-mc-sequence-v1"
RS_SFT_SCHEMA = "qwen-direct-compact-rs-sft-v1"
TARGET_LENGTH_CONTRACT_SCHEMA = "qwen-sequence-target-length-contract-v1"
TARGET_LENGTH_EVIDENCE_SCHEMA = "qwen-sequence-target-length-evidence-v1"
JOURNAL_CHAIN_HEAD_SCHEMA = "qwen-teacher-journal-chain-head-v1"
JOURNAL_CHAIN_GENESIS = "0" * 64
JOURNAL_CHAIN_FIELDS = {
    "journal_event_index",
    "journal_previous_event_sha256",
    "journal_event_sha256",
}

REQUIRED_SERIALIZER_ATTESTATIONS = {
    "artifact_hashes",
    "row_contract_hashes",
    "codec_text_roundtrip",
    "codec_token_id_roundtrip",
    "student_constant_prefix",
    "per_task_instruction_dictionary_roundtrip",
    "compact_semantic_f2_roundtrip",
    "branch_targets_reconstructed_from_cfg",
    "visible_task_symbols_one_token",
}
F2_PROMPT_MANIFEST_SCHEMA = "verified-api-readable-compact-v2"
F2_REPRESENTATION_SCHEMA = "lossless-semantic-f2"
MODERATION_TRANSPORT_SCHEMA = "qwen-f2-unicode-escape-transport-v1"
MODERATION_TRANSPORT_SYSTEM_SUFFIX = (
    "Transport layer: in the user message, every ASCII escape ~UHHHH; "
    "denotes exactly the single Unicode code point U+HHHH. Expand every "
    "escape exactly before decoding F2."
)
FORBIDDEN_PROMPT_FIELDS = {
    "tests",
    "acceptance_tests",
    "feedback_tests",
    "hidden_tests",
    "test_code",
}
ALIBABA_MODEL_STUDIO_HOSTS = frozenset(
    {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
    }
)


class ArtifactError(RuntimeError):
    """Fail-closed artifact or provenance violation."""


def validate_alibaba_model_studio_base_url(
    value: str, *, token_plan_automation_authorized: bool = False
) -> str:
    """Allow only HTTPS Alibaba endpoints explicitly authorized for this run."""

    try:
        parsed = urlsplit(str(value or "").strip())
        port = parsed.port
    except ValueError as exc:
        raise ArtifactError("Qwen base URL is invalid") from exc
    hostname = (parsed.hostname or "").lower()
    normalized_path = parsed.path.rstrip("/")
    is_token_plan = (
        hostname.startswith("token-plan.")
        and hostname.endswith(".maas.aliyuncs.com")
    )
    approved_host = (
        hostname in ALIBABA_MODEL_STUDIO_HOSTS
        or (
            hostname.endswith(".maas.aliyuncs.com")
            and (not is_token_plan or token_plan_automation_authorized)
        )
    )
    if (
        parsed.scheme != "https"
        or not approved_host
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or normalized_path != "/compatible-mode/v1"
    ):
        raise ArtifactError(
            "Qwen collection requires an approved Alibaba Model Studio "
            "automation-capable HTTPS endpoint ending /compatible-mode/v1; "
            "Token Plan requires an explicit provider-authorization attestation "
            "and arbitrary endpoints are forbidden"
        )
    return f"https://{hostname}/compatible-mode/v1"


def objective_mode_from_header(payload: Mapping[str, Any]) -> str:
    """Return the sealed teacher objective mode.

    Journals written before the explicit mode field existed requested top-5
    logprobs unconditionally, so they are safely interpreted as
    ``require_top5``.  ``sequence_only`` is never inferred automatically.
    """

    raw = payload.get("objective_mode", OBJECTIVE_MODE_REQUIRE_TOP5)
    mode = str(raw or "").strip()
    if mode not in OBJECTIVE_MODES:
        raise ArtifactError(
            "teacher run header has unsupported objective_mode "
            f"{mode!r}; expected one of {sorted(OBJECTIVE_MODES)}"
        )
    return mode


def derived_sample_seed(seed_base: int, task_id: str, sample_index: int) -> int:
    if not 0 <= int(seed_base) < 2**31:
        raise ArtifactError("seed_base must be in [0, 2^31)")
    if not task_id or not 0 <= int(sample_index) < SAMPLES_PER_TASK:
        raise ArtifactError("cannot derive a seed for an invalid teacher slot")
    material = f"{int(seed_base)}\0{task_id}\0{int(sample_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (
        2**31
    )


def validate_mc_teacher_sampling(
    generation_parameters: Mapping[str, Any],
) -> None:
    """Require untempered, untruncated teacher-distribution sampling."""

    try:
        temperature = float(generation_parameters.get("temperature"))
        top_p = float(generation_parameters.get("top_p"))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "MC sequence KL requires explicit temperature=1 and top_p=1"
        ) from exc
    extra_body = generation_parameters.get("extra_body")
    if (
        temperature != 1.0
        or top_p != 1.0
        or not isinstance(extra_body, Mapping)
        or extra_body.get("top_k") != 101
    ):
        raise ArtifactError(
            "MC sequence KL requires temperature=1.0, top_p=1.0, and "
            "extra_body.top_k=101 (Alibaba's documented disabled setting); "
            "tempered or truncated sampling is a "
            "different objective"
        )


def validate_qwen38_sequence_sampling(
    requested_model: str,
    generation_parameters: Mapping[str, Any],
) -> None:
    extra_body = generation_parameters.get("extra_body")
    if (
        requested_model != DEFAULT_MODEL
        or generation_parameters.get("n") != 1
        or generation_parameters.get("max_tokens") != 12288
        or not isinstance(extra_body, Mapping)
        or extra_body.get("enable_thinking") is not True
        or extra_body.get("thinking_budget") != 8192
    ):
        raise ArtifactError(
            "sequence_only requires exact qwen3.8-max-preview K=8 independent "
            "n=1 draws with thinking_budget=8192 and max_tokens=12288"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactError(f"required file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_target_length_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    binding: "StudentTokenizerBinding",
) -> dict[str, Any]:
    """Bind collection to the exact trainer contract and tokenizer."""

    record = file_record(path)
    if record["sha256"] != expected_sha256.strip().lower():
        raise ArtifactError(
            "target contract hash mismatch: "
            f"expected {expected_sha256}, got {record['sha256']}"
        )
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read target contract: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ArtifactError("target contract is not an object")
    try:
        max_target_tokens = int(raw.get("max_target_tokens"))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "target contract lacks an integer max_target_tokens"
        ) from exc
    if max_target_tokens <= 0:
        raise ArtifactError("target contract max_target_tokens must be positive")
    if (
        str(raw.get("tokenizer_json_sha256") or "")
        != binding.tokenizer_record["sha256"]
    ):
        raise ArtifactError(
            "target contract tokenizer differs from the pinned student tokenizer"
        )
    return {
        "schema": TARGET_LENGTH_CONTRACT_SCHEMA,
        "trainer_contract": record,
        "trainer_contract_schema": str(raw.get("schema") or ""),
        "max_target_tokens": max_target_tokens,
        "student_tokenizer": dict(binding.tokenizer_record),
        "student_eos_token_id": binding.eos_token_id,
        "tokenization": {
            "add_special_tokens": False,
            "eos_policy": "append_exactly_once_if_final_token_is_not_eos",
            "matches_trainer_dataset_loader": True,
            "truncation_permitted": False,
            "overflow_filtering_permitted": False,
            "overflow_resampling_permitted": False,
        },
        "target_source": {
            "field": "choice.message.content",
            "reasoning_field": "choice.message.reasoning_content",
            "reasoning_excluded": True,
            "final_dart_code_only_required": True,
        },
    }


def validate_target_length_contract(
    value: Any,
    *,
    binding: "StudentTokenizerBinding",
    objective_mode: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError("run header lacks the target-length contract")
    trainer_contract = value.get("trainer_contract")
    student_tokenizer = value.get("student_tokenizer")
    tokenization = value.get("tokenization")
    target_source = value.get("target_source")
    try:
        max_target_tokens = int(value.get("max_target_tokens"))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "run header target-length limit is invalid"
        ) from exc
    if (
        value.get("schema") != TARGET_LENGTH_CONTRACT_SCHEMA
        or not isinstance(trainer_contract, Mapping)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(trainer_contract.get("sha256") or "")
        )
        or not isinstance(student_tokenizer, Mapping)
        or student_tokenizer.get("sha256")
        != binding.tokenizer_record.get("sha256")
        or int(value.get("student_eos_token_id", -1)) != binding.eos_token_id
        or max_target_tokens <= 0
        or not isinstance(tokenization, Mapping)
        or tokenization.get("add_special_tokens") is not False
        or tokenization.get("eos_policy")
        != "append_exactly_once_if_final_token_is_not_eos"
        or tokenization.get("matches_trainer_dataset_loader") is not True
        or tokenization.get("truncation_permitted") is not False
        or tokenization.get("overflow_filtering_permitted") is not False
        or tokenization.get("overflow_resampling_permitted") is not False
        or not isinstance(target_source, Mapping)
        or target_source.get("field") != "choice.message.content"
        or target_source.get("reasoning_field")
        != "choice.message.reasoning_content"
        or target_source.get("reasoning_excluded") is not True
        or target_source.get("final_dart_code_only_required")
        != (objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5)
    ):
        raise ArtifactError("run header target-length contract failed")
    return dict(value)


def target_length_evidence(
    target: str,
    *,
    binding: "StudentTokenizerBinding",
    max_target_tokens: int,
) -> dict[str, Any]:
    """Tokenize exactly as DirectCompactJsonlDataset does, without truncation."""

    if not isinstance(target, str) or not target:
        raise ArtifactError("cannot audit an empty sequence target")
    try:
        encoded = binding.tokenizer.encode(target, add_special_tokens=False)
        content_ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    except Exception as exc:
        raise ArtifactError(
            "student tokenizer failed on a teacher sequence target"
        ) from exc
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in content_ids
    ):
        raise ArtifactError("student tokenizer returned invalid target token IDs")
    target_ids = [int(token_id) for token_id in content_ids]
    eos_appended = not target_ids or target_ids[-1] != binding.eos_token_id
    if eos_appended:
        target_ids.append(binding.eos_token_id)
    token_count = len(target_ids)
    return {
        "schema": TARGET_LENGTH_EVIDENCE_SCHEMA,
        "sequence_target_sha256": sha256_text(target),
        "content_token_count": len(content_ids),
        "eos_inclusive_target_token_count": token_count,
        "max_target_tokens": int(max_target_tokens),
        "within_contract": token_count <= int(max_target_tokens),
        "overflow_by_tokens": max(0, token_count - int(max_target_tokens)),
        "eos_token_id": binding.eos_token_id,
        "eos_appended": eos_appended,
        "final_token_is_eos": bool(
            target_ids and target_ids[-1] == binding.eos_token_id
        ),
        "add_special_tokens": False,
        "truncated": False,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ArtifactError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_event(path: str | Path, event: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    head_path = Path(str(destination) + ".chain-head.json")
    if any(field in event for field in JOURNAL_CHAIN_FIELDS):
        raise ArtifactError("caller may not supply journal-chain metadata")
    if destination.exists():
        if not head_path.is_file():
            raise ArtifactError(
                "existing journal has no durable chain head; resume is unsafe"
            )
        try:
            head = json.loads(head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("journal chain head is unreadable") from exc
        if (
            not isinstance(head, Mapping)
            or head.get("schema") != JOURNAL_CHAIN_HEAD_SCHEMA
            or int(head.get("journal_size_bytes", -1))
            != destination.stat().st_size
        ):
            raise ArtifactError(
                "journal bytes differ from the durable chain head"
            )
        event_index = int(head.get("event_count", -1))
        previous_sha256 = str(head.get("head_event_sha256") or "")
        if event_index < 0 or not re.fullmatch(
            r"[0-9a-f]{64}", previous_sha256
        ):
            raise ArtifactError("journal chain head is malformed")
    else:
        if head_path.exists():
            raise ArtifactError("orphan journal chain head exists")
        event_index = 0
        previous_sha256 = JOURNAL_CHAIN_GENESIS
    chained = {
        **dict(event),
        "journal_event_index": event_index,
        "journal_previous_event_sha256": previous_sha256,
    }
    event_sha256 = stable_sha256(chained)
    chained["journal_event_sha256"] = event_sha256
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(chained) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_write_json(
        head_path,
        {
            "schema": JOURNAL_CHAIN_HEAD_SCHEMA,
            "event_count": event_index + 1,
            "head_event_sha256": event_sha256,
            "journal_size_bytes": destination.stat().st_size,
        },
    )


def load_hash_chained_journal(path: str | Path) -> list[dict[str, Any]]:
    """Validate every event plus the durable external chain head."""

    source = Path(path)
    head_path = Path(str(source) + ".chain-head.json")
    if not source.exists():
        if head_path.exists():
            raise ArtifactError("orphan journal chain head exists")
        return []
    if not head_path.is_file():
        raise ArtifactError("journal has no durable chain head")
    rows = read_jsonl(source)
    previous = JOURNAL_CHAIN_GENESIS
    for index, row in enumerate(rows):
        observed_hash = str(row.get("journal_event_sha256") or "")
        if (
            row.get("journal_event_index") != index
            or row.get("journal_previous_event_sha256") != previous
            or not re.fullmatch(r"[0-9a-f]{64}", observed_hash)
        ):
            raise ArtifactError(
                f"journal hash chain breaks at event {index}"
            )
        unhashed = {
            key: value
            for key, value in row.items()
            if key != "journal_event_sha256"
        }
        if stable_sha256(unhashed) != observed_hash:
            raise ArtifactError(
                f"journal event hash mismatch at event {index}"
            )
        previous = observed_hash
    try:
        head = json.loads(head_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("journal chain head is unreadable") from exc
    if (
        not isinstance(head, Mapping)
        or head.get("schema") != JOURNAL_CHAIN_HEAD_SCHEMA
        or head.get("event_count") != len(rows)
        or head.get("head_event_sha256") != previous
        or head.get("journal_size_bytes") != source.stat().st_size
    ):
        raise ArtifactError(
            "journal chain head does not attest the complete journal"
        )
    return rows


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class PromptRow:
    task_id: str
    text: str
    text_sha256: str
    source_record_sha256: str
    source_schema: str
    representation_schema: str = ""
    system_prompt_sha256: str = ""


def load_verified_prompt_rows(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int | None = None,
) -> tuple[list[PromptRow], dict[str, Any]]:
    record = file_record(path)
    expected = expected_sha256.strip().lower()
    if record["sha256"] != expected:
        raise ArtifactError(
            f"prompt artifact hash mismatch: expected {expected}, "
            f"got {record['sha256']}"
        )
    raw_rows = read_jsonl(path)
    if expected_rows is not None and len(raw_rows) != int(expected_rows):
        raise ArtifactError(
            f"prompt artifact has {len(raw_rows)} rows, expected {expected_rows}"
        )
    prompts: list[PromptRow] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows):
        leaked = sorted(FORBIDDEN_PROMPT_FIELDS.intersection(row))
        if leaked:
            raise ArtifactError(
                f"prompt row {index} exposes verifier fields: {leaked}"
            )
        task_id = str(row.get("task_id") or "")
        text = row.get("text")
        if not task_id or not isinstance(text, str) or not text.strip():
            raise ArtifactError(f"prompt row {index} lacks task_id/text")
        if task_id in seen:
            raise ArtifactError(f"duplicate prompt task_id: {task_id}")
        seen.add(task_id)
        actual_text_hash = sha256_text(text)
        if str(row.get("text_sha256") or "") != actual_text_hash:
            raise ArtifactError(f"prompt row {task_id} text hash mismatch")
        verified = row.get("verified")
        if not isinstance(verified, Mapping):
            raise ArtifactError(
                f"prompt row {task_id} lacks serializer verification attestations"
            )
        missing = sorted(
            name
            for name in REQUIRED_SERIALIZER_ATTESTATIONS
            if verified.get(name) is not True
        )
        if missing:
            raise ArtifactError(
                f"prompt row {task_id} failed serializer attestations: {missing}"
            )
        if verified.get("opaque_custom_ids_in_text") is not False:
            raise ArtifactError(
                f"prompt row {task_id} does not attest opaque IDs were removed"
            )
        representation_schema = str(row.get("representation_schema") or "")
        system_prompt_sha256 = str(row.get("system_prompt_sha256") or "")
        if representation_schema != F2_REPRESENTATION_SCHEMA:
            raise ArtifactError(
                f"prompt row {task_id} is not the lossless F2 representation"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", system_prompt_sha256):
            raise ArtifactError(
                f"prompt row {task_id} lacks its F2 system-prompt binding"
            )
        prompts.append(
            PromptRow(
                task_id=task_id,
                text=text,
                text_sha256=actual_text_hash,
                source_record_sha256=stable_sha256(row),
                source_schema=str(row.get("schema") or ""),
                representation_schema=representation_schema,
                system_prompt_sha256=system_prompt_sha256,
            )
        )
    return prompts, record


def load_f2_prompt_contract(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    prompt_record: Mapping[str, Any],
    expected_rows: int,
    student_tokenizer_sha256: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load the exact manifest-bound F2 grammar prompt used by every API arm."""

    record = file_record(manifest_path)
    if record["sha256"] != expected_sha256.strip().lower():
        raise ArtifactError("F2 prompt-manifest hash mismatch")
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read F2 prompt manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ArtifactError("F2 prompt manifest is not an object")
    output = manifest.get("output")
    contract = manifest.get("f2_prompt_contract")
    invariants = manifest.get("invariants")
    required_invariants = {
        "all_artifact_hashes_verified",
        "all_row_contract_hashes_verified",
        "all_codec_roundtrips_verified",
        "all_student_constant_prefixes_verified",
        "all_f2_semantic_roundtrips_verified",
        "f2_system_prompt_self_contained_and_hashed",
        "all_complete_prompts_within_limit",
        "opaque_source_ids_expanded",
        "cfg_explicit",
        # The API teacher, RS-SFT teacher, and VeRPO judge must all see the
        # same complete user-function bundle as the student.  The old
        # single-function F2 artifacts also used ``lossless-semantic-f2``;
        # requiring these attestations prevents them from being accidentally
        # accepted after the multi-function representation migration.
        "all_user_functions_retained",
        "all_external_symbols_retained",
        "transfer_table_redundancy_proven",
        "train_dev_representation_contract_identical",
    }
    if (
        manifest.get("schema") != F2_PROMPT_MANIFEST_SCHEMA
        or int(manifest.get("rows", -1)) != int(expected_rows)
        or not isinstance(output, Mapping)
        or output.get("sha256") != prompt_record.get("sha256")
        or int(output.get("size_bytes", output.get("bytes", -1)))
        != int(prompt_record.get("size_bytes", -2))
        or not isinstance(contract, Mapping)
        or contract.get("representation_schema") != F2_REPRESENTATION_SCHEMA
        or contract.get("tokenizer_sha256") != student_tokenizer_sha256
        or contract.get("all_rows_within_limit") is not True
        or not isinstance(invariants, Mapping)
        or any(invariants.get(name) is not True for name in required_invariants)
    ):
        raise ArtifactError("F2 prompt manifest contract failed")
    system_prompt = contract.get("system_prompt")
    system_prompt_sha256 = str(contract.get("system_prompt_sha256") or "")
    if (
        not isinstance(system_prompt, str)
        or not system_prompt.strip()
        or sha256_text(system_prompt) != system_prompt_sha256
    ):
        raise ArtifactError("F2 system prompt/hash binding failed")
    return system_prompt, record, dict(manifest)


def build_messages(system_prompt: str, prompt: PromptRow) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.text},
    ]


def build_lossless_moderation_transport(
    messages: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Escape arbitrary F2 codebook symbols after a provider false positive.

    The transformation is deliberately activated only after Alibaba returns
    ``data_inspection_failed`` for the canonical request.  Tilde is escaped as
    well, making a single-pass decoder an exact inverse even if canonical F2
    text happens to contain text resembling the transport marker.
    """

    if (
        len(messages) != 2
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
    ):
        raise ArtifactError("moderation transport requires system+user messages")
    system = str(messages[0].get("content") or "")
    canonical_text = str(messages[1].get("content") or "")
    transported_text = "".join(
        character
        if ord(character) < 128 and character != "~"
        else f"~U{ord(character):04X};"
        for character in canonical_text
    )
    decoded = re.sub(
        r"~U([0-9A-F]{4,6});",
        lambda match: chr(int(match.group(1), 16)),
        transported_text,
    )
    if decoded != canonical_text:
        raise ArtifactError("moderation transport is not exactly reversible")
    transported_messages = [
        {
            "role": "system",
            "content": system + "\n" + MODERATION_TRANSPORT_SYSTEM_SUFFIX,
        },
        {"role": "user", "content": transported_text},
    ]
    transport = {
        "schema": MODERATION_TRANSPORT_SCHEMA,
        "reason": "provider_input_moderation_false_positive",
        "canonical_messages_sha256": stable_sha256(list(messages)),
        "transport_messages_sha256": stable_sha256(transported_messages),
        "transport_system_suffix_sha256": sha256_text(
            MODERATION_TRANSPORT_SYSTEM_SUFFIX
        ),
        "canonical_text_sha256": sha256_text(canonical_text),
        "transported_text_sha256": sha256_text(transported_text),
        "roundtrip_sha256": sha256_text(decoded),
        "non_ascii_codepoints_escaped": sum(
            ord(character) >= 128 for character in canonical_text
        ),
        "literal_tildes_escaped": canonical_text.count("~"),
        "reversible_non_ascii_escape": "~UHHHH;",
        "roundtrip_proven": True,
        "canonical_raw_byte_request_executed": False,
        "losslessly_equivalent_transport_request_executed": True,
        "moderation_error_code": "data_inspection_failed",
    }
    return transported_messages, transport


def attach_candidate_request_transport(
    candidate: Mapping[str, Any],
    request_transport: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash-bind an alternate, lossless request transport to a candidate."""

    metadata = {
        "schema",
        "event",
        "created_at",
        "candidate_id",
        "candidate_payload_sha256",
        *JOURNAL_CHAIN_FIELDS,
    }
    payload = {
        key: value for key, value in candidate.items() if key not in metadata
    }
    payload["request_transport"] = dict(request_transport)
    payload_sha256 = stable_sha256(payload)
    basis = {
        "task_id": payload.get("task_id"),
        "sample_index": payload.get("sample_index"),
        "prompt_sha256": payload.get("prompt_sha256"),
        "candidate_payload_sha256": payload_sha256,
    }
    return {
        "schema": candidate.get("schema"),
        "event": candidate.get("event"),
        "created_at": candidate.get("created_at"),
        "candidate_id": stable_sha256(basis),
        "candidate_payload_sha256": payload_sha256,
        **payload,
    }


def provider_error_code(error: Exception) -> str:
    """Extract a stable provider error code without retaining sensitive text."""

    direct = str(getattr(error, "code", "") or "")
    if direct:
        return direct
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        code = str(body.get("code") or "")
        if code:
            return code
        nested = body.get("error")
        if isinstance(nested, Mapping):
            return str(nested.get("code") or "")
    return ""


def count_prompt_tokens(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    chat_overhead_reserve: int,
) -> dict[str, int]:
    """Conservatively count readable prompt content with sealed chat reserve."""
    if chat_overhead_reserve < 0:
        raise ArtifactError("chat_overhead_reserve cannot be negative")
    counts = {"system_tokens": 0, "user_tokens": 0}
    for message in messages:
        role = str(message.get("role") or "")
        if role not in ("system", "user"):
            raise ArtifactError(f"unsupported prompt role {role!r}")
        encoded = tokenizer.encode(
            str(message.get("content") or ""), add_special_tokens=False
        )
        token_ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
        counts[f"{role}_tokens"] += len(token_ids)
    return {
        **counts,
        "chat_overhead_reserve": int(chat_overhead_reserve),
        "estimated_prompt_tokens": (
            counts["system_tokens"]
            + counts["user_tokens"]
            + int(chat_overhead_reserve)
        ),
    }


def extract_code(text: str, required_function: str) -> str:
    value = (text or "").strip()
    match = re.search(r"```(?:dart)?\s*(.*?)```", value, re.I | re.S)
    code = (match.group(1) if match else value).strip()
    if not code:
        return ""
    if required_function and not re.search(
        rf"\b{re.escape(required_function)}\s*\(", code
    ):
        return ""
    return code


def is_final_dart_code_only(text: str, required_function: str) -> bool:
    """Reject common response wrappers while allowing imperfect Dart draws."""

    value = (text or "").strip()
    if (
        not value
        or "```" in value
        or re.search(r"</?(?:think|analysis|reasoning)>", value, re.I)
    ):
        return False
    if required_function:
        # Require an actual top-level-looking function declaration line, not
        # merely prose that happens to mention ``fn0(``.
        declaration = re.compile(
            rf"(?m)^[ \t]*(?:"
            rf"[A-Za-z_$][\w$]*(?:[.?]<[^>\r\n]+>)?"
            rf"(?:[?<>\[\],.$\w]*)[ \t]+"
            rf")?{re.escape(required_function)}[ \t]*\([^)\r\n]*\)"
            rf"[ \t]*(?:async\*?|sync\*)?[ \t]*(?:=>|\{{)"
        )
        if declaration.search(value) is None:
            return False
    return extract_code(value, required_function) == value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} is not a float") from exc
    if not math.isfinite(result):
        raise ArtifactError(f"{label} is not finite")
    return result


def _raw_bytes(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ArtifactError(f"{label}.bytes is not an array")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255:
            raise ArtifactError(f"{label}.bytes has an invalid octet")
        result.append(int(item))
    return result


def _token_text(value: Any, label: str) -> str:
    result = _field(value, "token")
    if not isinstance(result, str):
        raise ArtifactError(f"{label}.token is not a string")
    return result


def _normalize_logprob_token(value: Any, label: str) -> dict[str, Any]:
    alternatives = _field(value, "top_logprobs", []) or []
    if not isinstance(alternatives, (list, tuple)):
        raise ArtifactError(f"{label}.top_logprobs is not an array")
    return {
        "token": _token_text(value, label),
        "bytes": _raw_bytes(_field(value, "bytes"), label),
        # Do not round: JSON emits the full Python float representation supplied
        # by the SDK.
        "logprob": _finite_float(_field(value, "logprob"), f"{label}.logprob"),
        "top_logprobs": [
            {
                "token": _token_text(alt, f"{label}.top[{index}]"),
                "bytes": _raw_bytes(_field(alt, "bytes"), f"{label}.top[{index}]"),
                "logprob": _finite_float(
                    _field(alt, "logprob"), f"{label}.top[{index}].logprob"
                ),
            }
            for index, alt in enumerate(alternatives)
        ],
    }


def backend_identity(candidate: Mapping[str, Any]) -> dict[str, str]:
    response = candidate.get("response") or {}
    return {
        "requested_model": str(candidate.get("requested_model") or ""),
        "returned_model": str(response.get("returned_model") or ""),
        "system_fingerprint": str(response.get("system_fingerprint") or ""),
    }


def backend_identity_sha256(candidate: Mapping[str, Any]) -> str:
    return stable_sha256(backend_identity(candidate))


def _candidate_payload_from_event(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "schema",
        "event",
        "created_at",
        "candidate_id",
        "candidate_payload_sha256",
        *JOURNAL_CHAIN_FIELDS,
    }
    return {key: value for key, value in row.items() if key not in metadata}


def _validate_candidate_event(row: Mapping[str, Any]) -> None:
    if row.get("schema") != CANDIDATE_SCHEMA:
        raise ArtifactError("journal teacher-candidate schema mismatch")
    payload = _candidate_payload_from_event(row)
    payload_sha = stable_sha256(payload)
    if row.get("candidate_payload_sha256") != payload_sha:
        raise ArtifactError("journal teacher-candidate payload hash mismatch")
    basis = {
        "task_id": payload.get("task_id"),
        "sample_index": payload.get("sample_index"),
        "prompt_sha256": payload.get("prompt_sha256"),
        "candidate_payload_sha256": payload_sha,
    }
    if row.get("candidate_id") != stable_sha256(basis):
        raise ArtifactError("journal teacher-candidate ID hash mismatch")


def _verification_payload_from_event(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "schema",
        "event",
        "created_at",
        "verification_payload_sha256",
        *JOURNAL_CHAIN_FIELDS,
    }
    return {key: value for key, value in row.items() if key not in metadata}


def _validate_verification_event(row: Mapping[str, Any]) -> None:
    if row.get("schema") != VERIFICATION_SCHEMA:
        raise ArtifactError("journal verification schema mismatch")
    if row.get("verification_payload_sha256") != stable_sha256(
        _verification_payload_from_event(row)
    ):
        raise ArtifactError("journal verification payload hash mismatch")


def normalize_response(
    response: Any,
    *,
    task_id: str,
    sample_index: int,
    prompt_sha256: str,
    requested_model: str,
    request_parameters: Mapping[str, Any],
    required_function: str,
) -> dict[str, Any]:
    choices = _field(response, "choices", []) or []
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise ArtifactError("teacher response must contain exactly one choice")
    choice = choices[0]
    message = _field(choice, "message", {}) or {}
    raw_content_value = _field(message, "content")
    if raw_content_value is None:
        raw_content = ""
    elif isinstance(raw_content_value, str):
        raw_content = raw_content_value
    else:
        raise ArtifactError("choice.message.content is not a string")
    reasoning_content_value = _field(message, "reasoning_content")
    if reasoning_content_value is None:
        reasoning_content = ""
    elif isinstance(reasoning_content_value, str):
        reasoning_content = reasoning_content_value
    else:
        raise ArtifactError("choice.message.reasoning_content is not a string")
    finish_reason = str(_field(choice, "finish_reason", "") or "")
    logprobs = _field(choice, "logprobs", {}) or {}
    token_values = _field(logprobs, "content", []) or []
    if not isinstance(token_values, (list, tuple)):
        raise ArtifactError("choice.logprobs.content is not an array")
    tokens = [
        _normalize_logprob_token(value, f"token[{index}]")
        for index, value in enumerate(token_values)
    ]
    if request_parameters.get("logprobs") is True:
        if int(request_parameters.get("top_logprobs", -1)) != TOP_LOGPROBS:
            raise ArtifactError("logprob response was requested with a non-top5 contract")
        if not tokens:
            raise ArtifactError("teacher response omitted requested content logprobs")
        reconstructed = bytearray()
        for index, token in enumerate(tokens):
            raw = token.get("bytes")
            top = token.get("top_logprobs")
            if raw is None:
                raise ArtifactError(f"token[{index}] omitted chosen raw bytes")
            if not isinstance(top, list) or len(top) != TOP_LOGPROBS:
                raise ArtifactError(f"token[{index}] omitted exact top-5 alternatives")
            reconstructed.extend(int(value) for value in raw)
            if any(alternative.get("bytes") is None for alternative in top):
                raise ArtifactError(
                    f"token[{index}] top-5 alternative omitted raw bytes"
                )
            top_mass = math.fsum(
                math.exp(float(alternative["logprob"])) for alternative in top
            )
            if 1.0 - top_mass < -NEGATIVE_TAIL_TOLERANCE:
                raise ArtifactError(
                    f"token[{index}] has materially negative inferred tail mass"
                )
    response_id = str(_field(response, "id", "") or "")
    returned_model = str(_field(response, "model", "") or "")
    if not response_id:
        raise ArtifactError("teacher response has no request ID")
    if not returned_model:
        raise ArtifactError("teacher response has no returned model ID")
    if not finish_reason:
        raise ArtifactError("teacher response has no finish_reason")
    created = _field(response, "created")
    system_fingerprint = str(_field(response, "system_fingerprint", "") or "")
    provider_reported_seed_value = _field(response, "seed")
    if provider_reported_seed_value is None:
        provider_reported_seed = None
    elif (
        isinstance(provider_reported_seed_value, int)
        and not isinstance(provider_reported_seed_value, bool)
        and 0 <= provider_reported_seed_value < 2**31
    ):
        provider_reported_seed = int(provider_reported_seed_value)
    else:
        raise ArtifactError("teacher response has an invalid reported seed")
    requested_seed = request_parameters.get("seed")
    if (
        provider_reported_seed is not None
        and requested_seed is not None
        and provider_reported_seed != int(requested_seed)
    ):
        raise ArtifactError(
            "teacher response explicitly reports a seed different from the request"
        )
    usage = _plain(_field(response, "usage"))
    if not isinstance(usage, Mapping) or int(usage.get("total_tokens") or 0) <= 0:
        raise ArtifactError("teacher response has zero or missing token usage")
    if request_parameters.get("logprobs") is True and bytes(
        reconstructed
    ) != raw_content.encode("utf-8"):
        raise ArtifactError(
            "chosen provider-token bytes do not reconstruct final content"
        )
    code = extract_code(raw_content, required_function)
    completion_attested = bool(
        finish_reason == "stop"
        and raw_content.strip()
        and response_id
        and returned_model
    )
    candidate_payload = {
        "task_id": task_id,
        "sample_index": int(sample_index),
        "prompt_sha256": prompt_sha256,
        "requested_model": requested_model,
        "request_parameters": dict(request_parameters),
        "response": {
            "request_id": response_id,
            "returned_model": returned_model,
            "system_fingerprint": system_fingerprint,
            "created": created,
            "service_tier": _field(response, "service_tier"),
            "provider_reported_seed": provider_reported_seed,
            "usage": usage,
            "finish_reason": finish_reason,
            "raw_content": raw_content,
            "raw_content_sha256": sha256_text(raw_content),
            "raw_reasoning_content": reasoning_content,
            "raw_reasoning_content_sha256": sha256_text(reasoning_content),
            "reasoning_content_present": bool(reasoning_content),
            "reasoning_logprobs_available": False,
        },
        "chosen_tokens_with_top_logprobs": tokens,
        "parse": {
            "parseable": bool(code),
            "required_function": required_function,
            "code": code,
            "code_sha256": sha256_text(code) if code else None,
            "normalization": (
                "trim_outer_whitespace"
                if code and code == raw_content.strip()
                else "extract_first_dart_fence_and_trim"
                if code
                else "not_extractable"
            ),
            "code_equals_trimmed_raw_content": bool(
                code and code == raw_content.strip()
            ),
        },
        "completion_attested": completion_attested,
        "backend_identity": {
            "requested_model": requested_model,
            "returned_model": returned_model,
            "system_fingerprint": system_fingerprint,
        },
    }
    candidate_payload_sha256 = stable_sha256(candidate_payload)
    candidate_basis = {
        "task_id": task_id,
        "sample_index": sample_index,
        "prompt_sha256": prompt_sha256,
        "candidate_payload_sha256": candidate_payload_sha256,
    }
    return {
        "schema": CANDIDATE_SCHEMA,
        "event": "teacher_candidate",
        "created_at": utc_now(),
        "candidate_id": stable_sha256(candidate_basis),
        "candidate_payload_sha256": candidate_payload_sha256,
        **candidate_payload,
    }


def make_rejected_draw_event(
    response: Any,
    *,
    task_id: str,
    sample_index: int,
    prompt_sha256: str,
    requested_model: str,
    request_parameters: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Seal a provider-produced draw that cannot enter the objective.

    The slot is consumed permanently. A resumed run must never call the API
    again for the same task/sample/seed, because doing so would condition the
    Monte Carlo target on passing post-response validation.
    """

    raw_response = _plain(response)
    payload = {
        "task_id": str(task_id),
        "sample_index": int(sample_index),
        "prompt_sha256": str(prompt_sha256),
        "requested_model": str(requested_model),
        "request_parameters": dict(request_parameters),
        "failure_kind": "invalid_teacher_response",
        "error_type": type(error).__name__,
        "error": _safe_error(error),
        "provider_response": raw_response,
        "provider_response_sha256": stable_sha256(raw_response),
        "provider_request_id": str(_field(response, "id", "") or ""),
        "returned_model": str(_field(response, "model", "") or ""),
        "terminal": True,
        "slot_consumed": True,
        "resample_permitted": False,
    }
    payload_sha = stable_sha256(payload)
    basis = {
        "task_id": payload["task_id"],
        "sample_index": payload["sample_index"],
        "prompt_sha256": payload["prompt_sha256"],
        "rejected_payload_sha256": payload_sha,
    }
    return {
        "schema": REJECTED_DRAW_SCHEMA,
        "event": "teacher_rejected_draw",
        "created_at": utc_now(),
        "rejected_draw_id": stable_sha256(basis),
        "rejected_payload_sha256": payload_sha,
        **payload,
    }


def _validate_rejected_draw_event(row: Mapping[str, Any]) -> None:
    if row.get("schema") != REJECTED_DRAW_SCHEMA:
        raise ArtifactError("journal rejected-draw schema mismatch")
    metadata = {
        "schema",
        "event",
        "created_at",
        "rejected_draw_id",
        "rejected_payload_sha256",
        *JOURNAL_CHAIN_FIELDS,
    }
    payload = {
        key: value for key, value in row.items() if key not in metadata
    }
    payload_sha = stable_sha256(payload)
    if row.get("rejected_payload_sha256") != payload_sha:
        raise ArtifactError("journal rejected-draw payload hash mismatch")
    basis = {
        "task_id": payload.get("task_id"),
        "sample_index": payload.get("sample_index"),
        "prompt_sha256": payload.get("prompt_sha256"),
        "rejected_payload_sha256": payload_sha,
    }
    if row.get("rejected_draw_id") != stable_sha256(basis):
        raise ArtifactError("journal rejected-draw ID hash mismatch")
    if (
        payload.get("terminal") is not True
        or payload.get("slot_consumed") is not True
        or payload.get("resample_permitted") is not False
    ):
        raise ArtifactError("journal rejected draw is not terminal/fail-closed")


def make_slot_started_event(
    *,
    task_id: str,
    sample_index: int,
    prompt_sha256: str,
    request_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "task_id": str(task_id),
        "sample_index": int(sample_index),
        "prompt_sha256": str(prompt_sha256),
        "request_parameters": dict(request_parameters),
        "provider_idempotency_key_available": False,
        "automatic_reissue_if_indeterminate": False,
    }
    payload_sha = stable_sha256(payload)
    return {
        "schema": SLOT_STARTED_SCHEMA,
        "event": "teacher_slot_started",
        "created_at": utc_now(),
        "slot_started_id": stable_sha256(
            {
                "task_id": payload["task_id"],
                "sample_index": payload["sample_index"],
                "prompt_sha256": payload["prompt_sha256"],
                "slot_started_payload_sha256": payload_sha,
            }
        ),
        "slot_started_payload_sha256": payload_sha,
        **payload,
    }


def make_orphan_reissue_authorization_event(
    started: Mapping[str, Any],
    *,
    original_run_header_sha256: str = "",
    original_collector_implementation: Mapping[str, Any] | None = None,
    recovery_collector_implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize one indeterminate logical slot to resume despite billing risk.

    The original ``teacher_slot_started`` row remains the request receipt.  This
    event does not replace it or create a second logical sample; it explicitly
    binds any recovery call to the original parameters and seed.
    """
    request_parameters = started.get("request_parameters")
    if not isinstance(request_parameters, Mapping):
        raise ArtifactError("orphan slot-started receipt has no request parameters")
    request_parameters = dict(request_parameters)
    payload = {
        "task_id": str(started.get("task_id") or ""),
        "sample_index": int(started.get("sample_index", -1)),
        "prompt_sha256": str(started.get("prompt_sha256") or ""),
        "slot_started_id": str(started.get("slot_started_id") or ""),
        "slot_started_payload_sha256": str(
            started.get("slot_started_payload_sha256") or ""
        ),
        "original_run_header_sha256": str(original_run_header_sha256),
        "original_collector_implementation": (
            dict(original_collector_implementation)
            if isinstance(original_collector_implementation, Mapping)
            else None
        ),
        "recovery_collector_implementation": (
            dict(recovery_collector_implementation)
            if isinstance(recovery_collector_implementation, Mapping)
            else None
        ),
        "collector_implementation_change_only_resume_permitted": True,
        "original_request_receipt": {
            "slot_started_id": str(started.get("slot_started_id") or ""),
            "slot_started_payload_sha256": str(
                started.get("slot_started_payload_sha256") or ""
            ),
            "request_parameters_sha256": stable_sha256(request_parameters),
            "requested_seed": request_parameters.get("seed"),
        },
        "reissue_request_parameters": request_parameters,
        "reissue_request_parameters_sha256": stable_sha256(request_parameters),
        "same_logical_task_and_sample_only": True,
        "same_request_parameters_and_seed_required": True,
        "provider_idempotency_key_available": False,
        "original_provider_request_may_have_billed_or_completed": True,
        "duplicate_provider_billing_risk_acknowledged": True,
        "single_terminal_logical_outcome_required": True,
        "completed_slots_must_not_be_reissued": True,
        "token_plan_automation_authorized": True,
        "authorization_scope": (
            "resume_this_orphan_until_one_terminal_logical_outcome"
        ),
    }
    payload_sha = stable_sha256(payload)
    return {
        "schema": ORPHAN_REISSUE_AUTHORIZATION_SCHEMA,
        "event": "teacher_slot_orphan_reissue_authorized",
        "created_at": utc_now(),
        "orphan_reissue_authorization_id": stable_sha256(
            {
                "slot_started_id": payload["slot_started_id"],
                "orphan_reissue_authorization_payload_sha256": payload_sha,
            }
        ),
        "orphan_reissue_authorization_payload_sha256": payload_sha,
        **payload,
    }


def make_orphan_reissue_reauthorization_event(
    started: Mapping[str, Any],
    prior_authorization: Mapping[str, Any],
    *,
    recovery_collector_implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicitly supersede an orphan authorization after code-only recovery.

    The logical slot, original header, request parameters, seed, and billing
    risk remain unchanged.  The append-only event binds the next attempt to
    both the prior authorization and the exact new collector implementation.
    """
    request_parameters = started.get("request_parameters")
    if not isinstance(request_parameters, Mapping):
        raise ArtifactError("orphan reauthorization lacks request parameters")
    previous_implementation = prior_authorization.get(
        "recovery_collector_implementation"
    )
    if not isinstance(previous_implementation, Mapping):
        raise ArtifactError(
            "orphan reauthorization lacks its prior recovery implementation"
        )
    request_parameters = dict(request_parameters)
    payload = {
        "task_id": str(started.get("task_id") or ""),
        "sample_index": int(started.get("sample_index", -1)),
        "prompt_sha256": str(started.get("prompt_sha256") or ""),
        "slot_started_id": str(started.get("slot_started_id") or ""),
        "slot_started_payload_sha256": str(
            started.get("slot_started_payload_sha256") or ""
        ),
        "prior_orphan_reissue_authorization_id": str(
            prior_authorization.get("orphan_reissue_authorization_id") or ""
        ),
        "original_run_header_sha256": str(
            prior_authorization.get("original_run_header_sha256") or ""
        ),
        "original_collector_implementation": prior_authorization.get(
            "original_collector_implementation"
        ),
        "previous_recovery_collector_implementation": dict(
            previous_implementation
        ),
        "recovery_collector_implementation": dict(
            recovery_collector_implementation
        ),
        "collector_implementation_change_only_resume_permitted": True,
        "reissue_request_parameters": request_parameters,
        "reissue_request_parameters_sha256": stable_sha256(
            request_parameters
        ),
        "requested_seed": request_parameters.get("seed"),
        "same_logical_task_and_sample_only": True,
        "same_request_parameters_and_seed_required": True,
        "original_or_prior_provider_request_may_have_billed_or_completed": True,
        "duplicate_provider_billing_risk_acknowledged": True,
        "single_terminal_logical_outcome_required": True,
        "completed_slots_must_not_be_reissued": True,
        "token_plan_automation_authorized": True,
        "authorization_scope": (
            "supersede_prior_orphan_authorization_for_code_only_recovery"
        ),
    }
    payload_sha = stable_sha256(payload)
    return {
        "schema": ORPHAN_REISSUE_REAUTHORIZATION_SCHEMA,
        "event": "teacher_slot_orphan_reissue_implementation_reauthorized",
        "created_at": utc_now(),
        "orphan_reissue_authorization_id": stable_sha256(
            {
                "slot_started_id": payload["slot_started_id"],
                "prior_orphan_reissue_authorization_id": payload[
                    "prior_orphan_reissue_authorization_id"
                ],
                "orphan_reissue_authorization_payload_sha256": payload_sha,
            }
        ),
        "orphan_reissue_authorization_payload_sha256": payload_sha,
        **payload,
    }


def make_orphan_reissue_attempt_event(
    started: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    attempt_index: int,
) -> dict[str, Any]:
    """Create one durable receipt immediately before a recovery API call."""
    if attempt_index < 1:
        raise ArtifactError("orphan reissue attempt index must be positive")
    request_parameters = started.get("request_parameters")
    recovery_implementation = authorization.get(
        "recovery_collector_implementation"
    )
    if not isinstance(request_parameters, Mapping):
        raise ArtifactError("orphan reissue attempt has no request parameters")
    if recovery_implementation is not None and not isinstance(
        recovery_implementation, Mapping
    ):
        raise ArtifactError(
            "orphan reissue attempt has an invalid recovery implementation"
        )
    request_parameters = dict(request_parameters)
    payload = {
        "task_id": str(started.get("task_id") or ""),
        "sample_index": int(started.get("sample_index", -1)),
        "prompt_sha256": str(started.get("prompt_sha256") or ""),
        "slot_started_id": str(started.get("slot_started_id") or ""),
        "orphan_reissue_authorization_id": str(
            authorization.get("orphan_reissue_authorization_id") or ""
        ),
        "attempt_index": int(attempt_index),
        "request_parameters": request_parameters,
        "request_parameters_sha256": stable_sha256(request_parameters),
        "requested_seed": request_parameters.get("seed"),
        "recovery_collector_implementation": (
            dict(recovery_implementation)
            if isinstance(recovery_implementation, Mapping)
            else None
        ),
        "same_logical_task_and_sample_only": True,
        "same_request_parameters_and_seed_required": True,
        "provider_idempotency_key_available": False,
        "original_or_prior_provider_request_may_have_billed_or_completed": True,
        "duplicate_provider_billing_risk_acknowledged": True,
        "attempt_outcome_indeterminate_until_single_terminal_event": True,
    }
    payload_sha = stable_sha256(payload)
    return {
        "schema": ORPHAN_REISSUE_ATTEMPT_SCHEMA,
        "event": "teacher_slot_orphan_reissue_attempt_started",
        "created_at": utc_now(),
        "orphan_reissue_attempt_id": stable_sha256(
            {
                "orphan_reissue_authorization_id": payload[
                    "orphan_reissue_authorization_id"
                ],
                "attempt_index": payload["attempt_index"],
                "orphan_reissue_attempt_payload_sha256": payload_sha,
            }
        ),
        "orphan_reissue_attempt_payload_sha256": payload_sha,
        **payload,
    }


def make_slot_terminal_event(
    started: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    outcome_type: str,
    orphan_reissue_attempt_id: str = "",
) -> dict[str, Any]:
    if outcome_type == "candidate":
        outcome_id = str(outcome.get("candidate_id") or "")
    elif outcome_type == "rejected_draw":
        outcome_id = str(outcome.get("rejected_draw_id") or "")
    else:
        raise ArtifactError("unsupported terminal slot outcome type")
    if not outcome_id:
        raise ArtifactError("terminal slot outcome has no durable ID")
    payload = {
        "task_id": str(started.get("task_id") or ""),
        "sample_index": int(started.get("sample_index", -1)),
        "prompt_sha256": str(started.get("prompt_sha256") or ""),
        "slot_started_id": str(started.get("slot_started_id") or ""),
        "outcome_type": outcome_type,
        "outcome_id": outcome_id,
        "automatic_reissue_permitted": False,
    }
    if orphan_reissue_attempt_id:
        payload["orphan_reissue_attempt_id"] = str(
            orphan_reissue_attempt_id
        )
    payload_sha = stable_sha256(payload)
    return {
        "schema": SLOT_TERMINAL_SCHEMA,
        "event": "teacher_slot_terminal",
        "created_at": utc_now(),
        "slot_terminal_id": stable_sha256(
            {
                "slot_started_id": payload["slot_started_id"],
                "outcome_type": outcome_type,
                "outcome_id": outcome_id,
                "slot_terminal_payload_sha256": payload_sha,
            }
        ),
        "slot_terminal_payload_sha256": payload_sha,
        **payload,
    }


def _lifecycle_payload(
    row: Mapping[str, Any],
    *,
    id_field: str,
    hash_field: str,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "schema",
            "event",
            "created_at",
            id_field,
            hash_field,
            *JOURNAL_CHAIN_FIELDS,
        }
    }


def _validate_slot_started_event(row: Mapping[str, Any]) -> None:
    if row.get("schema") != SLOT_STARTED_SCHEMA:
        raise ArtifactError("journal slot-started schema mismatch")
    payload = _lifecycle_payload(
        row,
        id_field="slot_started_id",
        hash_field="slot_started_payload_sha256",
    )
    payload_sha = stable_sha256(payload)
    if row.get("slot_started_payload_sha256") != payload_sha:
        raise ArtifactError("journal slot-started payload hash mismatch")
    expected_id = stable_sha256(
        {
            "task_id": payload.get("task_id"),
            "sample_index": payload.get("sample_index"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "slot_started_payload_sha256": payload_sha,
        }
    )
    if row.get("slot_started_id") != expected_id:
        raise ArtifactError("journal slot-started ID hash mismatch")
    if (
        payload.get("provider_idempotency_key_available") is not False
        or payload.get("automatic_reissue_if_indeterminate") is not False
    ):
        raise ArtifactError("slot-started event permits unsafe reissue")


def _validate_orphan_reissue_authorization_event(
    row: Mapping[str, Any],
) -> None:
    if row.get("schema") != ORPHAN_REISSUE_AUTHORIZATION_SCHEMA:
        raise ArtifactError("journal orphan-reissue authorization schema mismatch")
    payload = _lifecycle_payload(
        row,
        id_field="orphan_reissue_authorization_id",
        hash_field="orphan_reissue_authorization_payload_sha256",
    )
    payload_sha = stable_sha256(payload)
    if row.get("orphan_reissue_authorization_payload_sha256") != payload_sha:
        raise ArtifactError(
            "journal orphan-reissue authorization payload hash mismatch"
        )
    expected_id = stable_sha256(
        {
            "slot_started_id": payload.get("slot_started_id"),
            "orphan_reissue_authorization_payload_sha256": payload_sha,
        }
    )
    if row.get("orphan_reissue_authorization_id") != expected_id:
        raise ArtifactError(
            "journal orphan-reissue authorization ID hash mismatch"
        )
    request_parameters = payload.get("reissue_request_parameters")
    receipt = payload.get("original_request_receipt")
    if not isinstance(request_parameters, Mapping) or not isinstance(
        receipt, Mapping
    ):
        raise ArtifactError(
            "journal orphan-reissue authorization lacks the original request receipt"
        )
    parameters_sha256 = stable_sha256(dict(request_parameters))
    if (
        payload.get("reissue_request_parameters_sha256") != parameters_sha256
        or receipt.get("request_parameters_sha256") != parameters_sha256
        or receipt.get("requested_seed") != request_parameters.get("seed")
        or receipt.get("slot_started_id") != payload.get("slot_started_id")
        or receipt.get("slot_started_payload_sha256")
        != payload.get("slot_started_payload_sha256")
    ):
        raise ArtifactError(
            "journal orphan-reissue authorization changed the request receipt"
        )
    required_true = (
        "same_logical_task_and_sample_only",
        "same_request_parameters_and_seed_required",
        "original_provider_request_may_have_billed_or_completed",
        "duplicate_provider_billing_risk_acknowledged",
        "single_terminal_logical_outcome_required",
        "completed_slots_must_not_be_reissued",
        "token_plan_automation_authorized",
        "collector_implementation_change_only_resume_permitted",
    )
    if (
        any(payload.get(key) is not True for key in required_true)
        or payload.get("provider_idempotency_key_available") is not False
        or payload.get("authorization_scope")
        != "resume_this_orphan_until_one_terminal_logical_outcome"
    ):
        raise ArtifactError(
            "journal orphan-reissue authorization lacks the fail-closed risk contract"
        )


def _validate_orphan_reissue_reauthorization_event(
    row: Mapping[str, Any],
) -> None:
    if row.get("schema") != ORPHAN_REISSUE_REAUTHORIZATION_SCHEMA:
        raise ArtifactError("journal orphan reauthorization schema mismatch")
    payload = _lifecycle_payload(
        row,
        id_field="orphan_reissue_authorization_id",
        hash_field="orphan_reissue_authorization_payload_sha256",
    )
    payload_sha = stable_sha256(payload)
    if row.get("orphan_reissue_authorization_payload_sha256") != payload_sha:
        raise ArtifactError(
            "journal orphan reauthorization payload hash mismatch"
        )
    expected_id = stable_sha256(
        {
            "slot_started_id": payload.get("slot_started_id"),
            "prior_orphan_reissue_authorization_id": payload.get(
                "prior_orphan_reissue_authorization_id"
            ),
            "orphan_reissue_authorization_payload_sha256": payload_sha,
        }
    )
    if row.get("orphan_reissue_authorization_id") != expected_id:
        raise ArtifactError("journal orphan reauthorization ID hash mismatch")
    parameters = payload.get("reissue_request_parameters")
    old_implementation = payload.get(
        "previous_recovery_collector_implementation"
    )
    new_implementation = payload.get("recovery_collector_implementation")
    if (
        not isinstance(parameters, Mapping)
        or payload.get("reissue_request_parameters_sha256")
        != stable_sha256(dict(parameters))
        or payload.get("requested_seed") != parameters.get("seed")
        or not isinstance(old_implementation, Mapping)
        or not isinstance(new_implementation, Mapping)
        or old_implementation == new_implementation
        or not str(payload.get("prior_orphan_reissue_authorization_id") or "")
    ):
        raise ArtifactError(
            "journal orphan reauthorization changed its request or lacks "
            "an exact implementation transition"
        )
    required_true = (
        "collector_implementation_change_only_resume_permitted",
        "same_logical_task_and_sample_only",
        "same_request_parameters_and_seed_required",
        "original_or_prior_provider_request_may_have_billed_or_completed",
        "duplicate_provider_billing_risk_acknowledged",
        "single_terminal_logical_outcome_required",
        "completed_slots_must_not_be_reissued",
        "token_plan_automation_authorized",
    )
    if (
        any(payload.get(key) is not True for key in required_true)
        or payload.get("authorization_scope")
        != "supersede_prior_orphan_authorization_for_code_only_recovery"
    ):
        raise ArtifactError(
            "journal orphan reauthorization lacks the fail-closed risk contract"
        )


def _validate_orphan_reissue_attempt_event(row: Mapping[str, Any]) -> None:
    if row.get("schema") != ORPHAN_REISSUE_ATTEMPT_SCHEMA:
        raise ArtifactError("journal orphan-reissue attempt schema mismatch")
    payload = _lifecycle_payload(
        row,
        id_field="orphan_reissue_attempt_id",
        hash_field="orphan_reissue_attempt_payload_sha256",
    )
    payload_sha = stable_sha256(payload)
    if row.get("orphan_reissue_attempt_payload_sha256") != payload_sha:
        raise ArtifactError("journal orphan-reissue attempt payload hash mismatch")
    expected_id = stable_sha256(
        {
            "orphan_reissue_authorization_id": payload.get(
                "orphan_reissue_authorization_id"
            ),
            "attempt_index": payload.get("attempt_index"),
            "orphan_reissue_attempt_payload_sha256": payload_sha,
        }
    )
    if row.get("orphan_reissue_attempt_id") != expected_id:
        raise ArtifactError("journal orphan-reissue attempt ID hash mismatch")
    request_parameters = payload.get("request_parameters")
    if (
        not isinstance(request_parameters, Mapping)
        or payload.get("request_parameters_sha256")
        != stable_sha256(dict(request_parameters))
        or payload.get("requested_seed") != request_parameters.get("seed")
        or not isinstance(payload.get("attempt_index"), int)
        or payload.get("attempt_index") < 1
        or payload.get("same_logical_task_and_sample_only") is not True
        or payload.get("same_request_parameters_and_seed_required") is not True
        or payload.get("provider_idempotency_key_available") is not False
        or payload.get(
            "original_or_prior_provider_request_may_have_billed_or_completed"
        )
        is not True
        or payload.get("duplicate_provider_billing_risk_acknowledged")
        is not True
        or payload.get(
            "attempt_outcome_indeterminate_until_single_terminal_event"
        )
        is not True
    ):
        raise ArtifactError(
            "journal orphan-reissue attempt lacks the exact-request risk contract"
        )


def _validate_slot_terminal_event(row: Mapping[str, Any]) -> None:
    if row.get("schema") != SLOT_TERMINAL_SCHEMA:
        raise ArtifactError("journal slot-terminal schema mismatch")
    payload = _lifecycle_payload(
        row,
        id_field="slot_terminal_id",
        hash_field="slot_terminal_payload_sha256",
    )
    payload_sha = stable_sha256(payload)
    if row.get("slot_terminal_payload_sha256") != payload_sha:
        raise ArtifactError("journal slot-terminal payload hash mismatch")
    expected_id = stable_sha256(
        {
            "slot_started_id": payload.get("slot_started_id"),
            "outcome_type": payload.get("outcome_type"),
            "outcome_id": payload.get("outcome_id"),
            "slot_terminal_payload_sha256": payload_sha,
        }
    )
    if row.get("slot_terminal_id") != expected_id:
        raise ArtifactError("journal slot-terminal ID hash mismatch")
    if (
        payload.get("outcome_type") not in {"candidate", "rejected_draw"}
        or payload.get("automatic_reissue_permitted") is not False
    ):
        raise ArtifactError("slot-terminal event has an unsafe outcome")


@dataclass
class JournalState:
    header: dict[str, Any] | None
    candidates: dict[str, dict[str, Any]]
    rejections: dict[str, dict[str, Any]]
    starts: dict[tuple[str, str, int], dict[str, Any]]
    terminals: dict[tuple[str, str, int], dict[str, Any]]
    slots: dict[tuple[str, str, int], str]
    verifications: dict[str, dict[str, Any]]
    error_counts: dict[tuple[str, str, int], int]
    reissue_authorizations: dict[
        tuple[str, str, int], dict[str, Any]
    ] = field(default_factory=dict)
    reissue_attempts: dict[
        tuple[str, str, int], list[dict[str, Any]]
    ] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_indeterminate_slots: bool = False,
    ) -> "JournalState":
        header: dict[str, Any] | None = None
        candidates: dict[str, dict[str, Any]] = {}
        rejections: dict[str, dict[str, Any]] = {}
        starts: dict[tuple[str, str, int], dict[str, Any]] = {}
        terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
        slots: dict[tuple[str, str, int], str] = {}
        verifications: dict[str, dict[str, Any]] = {}
        error_counts: dict[tuple[str, str, int], int] = {}
        reissue_authorizations: dict[
            tuple[str, str, int], dict[str, Any]
        ] = {}
        reissue_attempts: dict[
            tuple[str, str, int], list[dict[str, Any]]
        ] = {}
        source = Path(path)
        if not source.exists():
            return cls(
                header,
                candidates,
                rejections,
                starts,
                terminals,
                slots,
                verifications,
                error_counts,
                reissue_authorizations,
                reissue_attempts,
            )
        for row in load_hash_chained_journal(source):
            event = row.get("event")
            if event == "run_header":
                if header is not None:
                    raise ArtifactError("journal contains multiple run headers")
                if row.get("schema") != JOURNAL_SCHEMA:
                    raise ArtifactError("journal run-header schema mismatch")
                if row.get("header_sha256") != stable_sha256(
                    row.get("payload") or {}
                ):
                    raise ArtifactError("journal run-header hash mismatch")
                header = row
            elif event == "teacher_candidate":
                _validate_candidate_event(row)
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id or candidate_id in candidates:
                    raise ArtifactError(
                        "journal has a missing/duplicate candidate_id"
                    )
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                if slot in slots:
                    raise ArtifactError(f"journal has duplicate successful slot: {slot}")
                candidates[candidate_id] = row
                slots[slot] = candidate_id
            elif event == "teacher_rejected_draw":
                _validate_rejected_draw_event(row)
                rejected_id = str(row.get("rejected_draw_id") or "")
                if not rejected_id or rejected_id in rejections:
                    raise ArtifactError(
                        "journal has a missing/duplicate rejected_draw_id"
                    )
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                if slot in slots:
                    raise ArtifactError(
                        f"journal has duplicate consumed slot: {slot}"
                    )
                rejections[rejected_id] = row
                slots[slot] = rejected_id
            elif event == "teacher_slot_started":
                _validate_slot_started_event(row)
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                if slot in starts:
                    raise ArtifactError(
                        f"journal has duplicate slot-started event: {slot}"
                    )
                starts[slot] = row
            elif event == "teacher_slot_orphan_reissue_authorized":
                _validate_orphan_reissue_authorization_event(row)
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                started = starts.get(slot)
                if started is None:
                    raise ArtifactError(
                        "journal orphan-reissue authorization has no prior "
                        f"slot-started receipt: {slot}"
                    )
                if (
                    slot in reissue_authorizations
                    or slot in terminals
                    or slot in slots
                ):
                    raise ArtifactError(
                        "journal orphan-reissue authorization is duplicate or "
                        f"targets a completed slot: {slot}"
                    )
                request_parameters = dict(
                    started.get("request_parameters") or {}
                )
                if (
                    row.get("slot_started_id")
                    != started.get("slot_started_id")
                    or row.get("slot_started_payload_sha256")
                    != started.get("slot_started_payload_sha256")
                    or row.get("reissue_request_parameters")
                    != request_parameters
                    or row.get("reissue_request_parameters_sha256")
                    != stable_sha256(request_parameters)
                ):
                    raise ArtifactError(
                        "journal orphan-reissue authorization is not bound to "
                        f"the original request receipt: {slot}"
                    )
                header_payload = (
                    header.get("payload")
                    if isinstance(header, Mapping)
                    else None
                )
                if (
                    not isinstance(header_payload, Mapping)
                    or row.get("original_run_header_sha256")
                    != header.get("header_sha256")
                    or row.get("original_collector_implementation")
                    != header_payload.get("implementation")
                    or not isinstance(
                        header_payload.get("provider_authorization"),
                        Mapping,
                    )
                    or header_payload["provider_authorization"].get(
                        "token_plan_automation_authorized"
                    )
                    is not True
                ):
                    raise ArtifactError(
                        "journal orphan-reissue authorization is not bound to "
                        "the original run header/collector implementation"
                    )
                reissue_authorizations[slot] = row
            elif (
                event
                == "teacher_slot_orphan_reissue_implementation_reauthorized"
            ):
                _validate_orphan_reissue_reauthorization_event(row)
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                started = starts.get(slot)
                prior = reissue_authorizations.get(slot)
                if (
                    started is None
                    or prior is None
                    or slot in terminals
                    or slot in slots
                ):
                    raise ArtifactError(
                        "journal orphan reauthorization has no live prior "
                        f"authorization: {slot}"
                    )
                request_parameters = dict(
                    started.get("request_parameters") or {}
                )
                header_payload = (
                    header.get("payload")
                    if isinstance(header, Mapping)
                    else None
                )
                if (
                    row.get("slot_started_id")
                    != started.get("slot_started_id")
                    or row.get("slot_started_payload_sha256")
                    != started.get("slot_started_payload_sha256")
                    or row.get("prior_orphan_reissue_authorization_id")
                    != prior.get("orphan_reissue_authorization_id")
                    or row.get(
                        "previous_recovery_collector_implementation"
                    )
                    != prior.get("recovery_collector_implementation")
                    or row.get("reissue_request_parameters")
                    != request_parameters
                    or row.get("reissue_request_parameters_sha256")
                    != stable_sha256(request_parameters)
                    or not isinstance(header_payload, Mapping)
                    or row.get("original_run_header_sha256")
                    != header.get("header_sha256")
                    or row.get("original_collector_implementation")
                    != header_payload.get("implementation")
                    or not isinstance(
                        header_payload.get("provider_authorization"),
                        Mapping,
                    )
                    or header_payload["provider_authorization"].get(
                        "token_plan_automation_authorized"
                    )
                    is not True
                ):
                    raise ArtifactError(
                        "journal orphan reauthorization is not bound to its "
                        f"prior authorization and original request: {slot}"
                    )
                reissue_authorizations[slot] = row
            elif event == "teacher_slot_orphan_reissue_attempt_started":
                _validate_orphan_reissue_attempt_event(row)
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                started = starts.get(slot)
                authorization = reissue_authorizations.get(slot)
                prior_attempts = reissue_attempts.setdefault(slot, [])
                if started is None or authorization is None:
                    raise ArtifactError(
                        "journal orphan-reissue attempt has no prior start/"
                        f"authorization receipt: {slot}"
                    )
                if slot in terminals or slot in slots:
                    raise ArtifactError(
                        "journal orphan-reissue attempt targets a completed "
                        f"logical slot: {slot}"
                    )
                if (
                    row.get("slot_started_id")
                    != started.get("slot_started_id")
                    or row.get("orphan_reissue_authorization_id")
                    != authorization.get(
                        "orphan_reissue_authorization_id"
                    )
                    or row.get("attempt_index") != len(prior_attempts) + 1
                    or row.get("request_parameters")
                    != started.get("request_parameters")
                    or row.get("request_parameters_sha256")
                    != stable_sha256(
                        dict(started.get("request_parameters") or {})
                    )
                    or row.get("recovery_collector_implementation")
                    != authorization.get(
                        "recovery_collector_implementation"
                    )
                ):
                    raise ArtifactError(
                        "journal orphan-reissue attempt is not the next exact "
                        f"attempt for the authorized request receipt: {slot}"
                    )
                prior_attempts.append(row)
            elif event == "teacher_slot_terminal":
                _validate_slot_terminal_event(row)
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                if slot in terminals:
                    raise ArtifactError(
                        f"journal has duplicate slot-terminal event: {slot}"
                    )
                terminals[slot] = row
            elif event == "verification":
                _validate_verification_event(row)
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id or candidate_id in verifications:
                    raise ArtifactError(
                        "journal has a missing/duplicate verification"
                    )
                verifications[candidate_id] = row
            elif event == "teacher_error":
                slot = (
                    str(row.get("task_id") or ""),
                    str(row.get("prompt_sha256") or ""),
                    int(row.get("sample_index", -1)),
                )
                error_counts[slot] = error_counts.get(slot, 0) + 1
            else:
                raise ArtifactError(f"journal has unknown event type: {event!r}")
        dangling = sorted(set(verifications).difference(candidates))
        if dangling:
            raise ArtifactError(
                f"journal verifies unknown candidates: {dangling[:3]}"
            )
        lifecycle_slots = (
            set(starts)
            | set(terminals)
            | set(slots)
            | set(reissue_authorizations)
            | set(reissue_attempts)
        )
        for slot in sorted(lifecycle_slots):
            started = starts.get(slot)
            terminal = terminals.get(slot)
            outcome_id = slots.get(slot)
            if started is None:
                raise ArtifactError(
                    f"journal outcome has no durable slot-started event: {slot}"
                )
            if terminal is None:
                if allow_indeterminate_slots:
                    continue
                if slot in reissue_authorizations:
                    raise ArtifactError(
                        "journal contains an authorized but unfinished orphan "
                        "slot; explicit duplicate-billing-risk recovery opt-in "
                        f"is required: {slot}"
                    )
                raise ArtifactError(
                    "journal contains an indeterminate paid slot with no "
                    f"terminal event; automatic reissue is forbidden: {slot}"
                )
            if (
                terminal.get("slot_started_id")
                != started.get("slot_started_id")
                or terminal.get("outcome_id") != outcome_id
            ):
                raise ArtifactError(
                    f"journal slot lifecycle/outcome binding failed: {slot}"
                )
            outcome_type = terminal.get("outcome_type")
            if (
                outcome_type == "candidate"
                and outcome_id not in candidates
            ) or (
                outcome_type == "rejected_draw"
                and outcome_id not in rejections
            ):
                raise ArtifactError(
                    f"journal terminal slot points to the wrong outcome: {slot}"
                )
            attempts = reissue_attempts.get(slot, [])
            terminal_attempt_id = str(
                terminal.get("orphan_reissue_attempt_id") or ""
            )
            if attempts:
                if terminal_attempt_id != str(
                    attempts[-1].get("orphan_reissue_attempt_id") or ""
                ):
                    raise ArtifactError(
                        "journal terminal slot is not bound to the latest "
                        f"orphan-reissue attempt: {slot}"
                    )
            elif terminal_attempt_id:
                raise ArtifactError(
                    "journal terminal slot names an unknown orphan-reissue "
                    f"attempt: {slot}"
                )
        return cls(
            header,
            candidates,
            rejections,
            starts,
            terminals,
            slots,
            verifications,
            error_counts,
            reissue_authorizations,
            reissue_attempts,
        )


def ensure_run_header(
    journal_path: str | Path,
    header_payload: Mapping[str, Any],
    *,
    allow_indeterminate_slots: bool = False,
    allow_recovery_implementation_change: bool = False,
) -> JournalState:
    expected = {
        "schema": JOURNAL_SCHEMA,
        "event": "run_header",
        "payload": dict(header_payload),
    }
    expected["header_sha256"] = stable_sha256(expected["payload"])
    state = JournalState.load(
        journal_path,
        allow_indeterminate_slots=allow_indeterminate_slots,
    )
    if state.header is None:
        row = dict(expected)
        row["created_at"] = utc_now()
        append_event(journal_path, row)
        return JournalState.load(
            journal_path,
            allow_indeterminate_slots=allow_indeterminate_slots,
        )
    observed = {
        "schema": state.header.get("schema"),
        "event": state.header.get("event"),
        "payload": state.header.get("payload"),
        "header_sha256": state.header.get("header_sha256"),
    }
    if observed != expected:
        observed_payload = observed.get("payload")
        expected_payload = expected.get("payload")
        same_except_implementation = False
        differing_payload_keys: list[str] = []
        if isinstance(observed_payload, Mapping) and isinstance(
            expected_payload, Mapping
        ):
            observed_without_implementation = dict(observed_payload)
            expected_without_implementation = dict(expected_payload)
            observed_without_implementation.pop("implementation", None)
            expected_without_implementation.pop("implementation", None)
            # Parallelism affects only how many independent, already-sealed
            # n=1 requests are in flight.  Permit a recovery process to raise
            # these two operational limits without weakening any request,
            # sampling, endpoint, retry, timeout, prompt, or objective binding.
            # Decreases and missing/type-changed controls remain fail-closed.
            parallelism_increase_only = True
            observed_transport = observed_without_implementation.get(
                "transport"
            )
            expected_transport = expected_without_implementation.get(
                "transport"
            )
            if isinstance(observed_transport, Mapping) and isinstance(
                expected_transport, Mapping
            ):
                observed_transport = dict(observed_transport)
                expected_transport = dict(expected_transport)
                for field in ("api_workers", "local_verifier_workers"):
                    old = observed_transport.pop(field, None)
                    new = expected_transport.pop(field, None)
                    if old == new:
                        continue
                    if (
                        isinstance(old, bool)
                        or isinstance(new, bool)
                        or not isinstance(old, int)
                        or not isinstance(new, int)
                        or old < 1
                        or new < old
                    ):
                        parallelism_increase_only = False
                observed_without_implementation["transport"] = (
                    observed_transport
                )
                expected_without_implementation["transport"] = (
                    expected_transport
                )
            elif observed_transport != expected_transport:
                parallelism_increase_only = False
            same_except_implementation = (
                parallelism_increase_only
                and
                observed_without_implementation
                == expected_without_implementation
            )
            differing_payload_keys = sorted(
                key
                for key in (
                    set(observed_without_implementation)
                    | set(expected_without_implementation)
                )
                if observed_without_implementation.get(key)
                != expected_without_implementation.get(key)
            )
        if not (
            allow_recovery_implementation_change
            and same_except_implementation
        ):
            raise ArtifactError(
                "resume configuration/header differs from the append-only "
                "journal; non-implementation payload keys="
                f"{differing_payload_keys}"
            )
    return state


def make_verification_event(
    candidate: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    compiled = result.get("compiled")
    passed = result.get("passed")
    completion = result.get("harness_completion_attested")
    if not isinstance(compiled, bool) or not isinstance(passed, bool):
        raise ArtifactError("verifier must return boolean compiled/passed")
    if passed and not compiled:
        raise ArtifactError("verifier cannot pass an uncompiled candidate")
    if passed and completion is not True:
        raise ArtifactError(
            "verifier pass lacks independent harness completion attestation"
        )
    for key in ("verifier_id", "verifier_sha256", "tests_sha256"):
        if not str(result.get(key) or ""):
            raise ArtifactError(f"verifier result lacks {key}")
    diagnostic = str(result.get("diagnostic") or "")
    payload = {
        "candidate_id": str(candidate["candidate_id"]),
        "task_id": str(candidate["task_id"]),
        "compiled": compiled,
        "passed": passed,
        "harness_completion_attested": completion is True,
        # Hidden-harness output is audit-only and can contain oracle values.
        # Seal only its digest/presence; never copy it into training artifacts.
        "diagnostic_present": bool(diagnostic),
        "diagnostic_sha256": sha256_text(diagnostic),
        "verifier_id": str(result["verifier_id"]),
        "verifier_sha256": str(result["verifier_sha256"]),
        "tests_sha256": str(result["tests_sha256"]),
    }
    return {
        "schema": VERIFICATION_SCHEMA,
        "event": "verification",
        "created_at": utc_now(),
        "verification_payload_sha256": stable_sha256(payload),
        **payload,
    }


def _safe_error(exc: BaseException) -> dict[str, str]:
    # Do not serialize repr(exc), provider bodies, request objects, headers, or
    # credentials. A digest still lets operators correlate repeated failures.
    message = str(exc)
    return {
        "type": type(exc).__name__,
        "message": "provider_call_failed; inspect provider-side request logs",
        "message_sha256": sha256_text(message),
    }


def collect_candidates(
    *,
    prompts: Sequence[PromptRow],
    client: Any,
    journal_path: str | Path,
    header_payload: Mapping[str, Any],
    system_prompt: str,
    requested_model: str,
    generation_parameters: Mapping[str, Any],
    required_function: str,
    verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    max_retries: int = 2,
    allow_homogeneous_shards: bool = False,
    workers: int = 1,
    verifier_workers: int | None = None,
    progress_every: int = 0,
    seed_base: int | None = None,
    require_returned_model_exact: bool = False,
    authorize_orphan_reissue_with_duplicate_billing_risk: bool = False,
) -> JournalState:
    """Collect exactly eight independent API calls per task.

    ``client`` is injected to make this function testable without a network.
    It must expose ``client.chat.completions.create``.
    """
    if int(header_payload.get("samples_per_task", -1)) != SAMPLES_PER_TASK:
        raise ArtifactError("production collection requires K=8")
    if authorize_orphan_reissue_with_duplicate_billing_risk:
        provider_authorization = header_payload.get("provider_authorization")
        if (
            not isinstance(provider_authorization, Mapping)
            or provider_authorization.get(
                "token_plan_automation_authorized"
            )
            is not True
        ):
            raise ArtifactError(
                "orphan reissue requires the sealed Token Plan automation "
                "authorization attestation"
            )
    if (
        header_payload.get("returned_model_must_equal_requested", False)
        is not bool(require_returned_model_exact)
    ):
        raise ArtifactError(
            "returned-model equality policy differs from the sealed run header"
        )
    validate_mc_teacher_sampling(generation_parameters)
    objective_mode = objective_mode_from_header(header_payload)
    transport_contract = header_payload.get("transport")
    length_policy = (
        transport_contract.get("length_capped_response_policy")
        if isinstance(transport_contract, Mapping)
        else None
    )
    length_capacities = (
        list(length_policy.get("max_token_capacities") or [])
        if isinstance(length_policy, Mapping)
        else []
    )
    if (
        not length_capacities
        or length_capacities[0] != generation_parameters.get("max_tokens")
        or length_capacities != sorted(set(length_capacities))
        or any(not isinstance(value, int) or value <= 0 for value in length_capacities)
        or length_policy.get("same_task_draw_only") is not True
        or length_policy.get("completed_draws_reissued") is not False
        or length_policy.get("capped_responses_retained_by_hash") is not True
    ):
        raise ArtifactError("run header has an invalid length-escalation policy")
    if objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5:
        if int(generation_parameters.get("top_logprobs", -1)) != TOP_LOGPROBS:
            raise ArtifactError("require_top5 collection needs top_logprobs=5")
        if generation_parameters.get("logprobs") is not True:
            raise ArtifactError("require_top5 collection needs logprobs=true")
        extra_body = generation_parameters.get("extra_body")
        if (
            not isinstance(extra_body, Mapping)
            or extra_body.get("enable_thinking") is not False
        ):
            raise ArtifactError(
                "require_top5 needs enable_thinking=false so teacher content "
                "logprobs and student logits use the same visible prefix"
            )
    elif (
        "top_logprobs" in generation_parameters
        or generation_parameters.get("logprobs") not in {None, False}
    ):
        raise ArtifactError(
            "sequence_only collection must omit provider logprob request fields"
        )
    if workers < 1:
        raise ArtifactError("workers must be positive")
    if verifier_workers is None:
        verifier_workers = workers
    if verifier_workers < 1:
        raise ArtifactError("verifier_workers must be positive")
    if progress_every < 0:
        raise ArtifactError("progress_every cannot be negative")
    if seed_base is not None and not 0 <= seed_base < 2**31:
        raise ArtifactError("seed_base must be in [0, 2^31)")
    if seed_base is not None:
        seed_contract = header_payload.get("sampling_seed_contract")
        if (
            not isinstance(seed_contract, Mapping)
            or seed_contract.get("algorithm") != SAMPLE_SEED_ALGORITHM
            or seed_contract.get("seed_base") != int(seed_base)
            or seed_contract.get("unique_seed_required_per_task_draw") is not True
        ):
            raise ArtifactError(
                "seeded collection requires the exact sampling_seed_contract "
                "in the sealed run header"
            )
        for prompt in prompts:
            task_seeds = {
                derived_sample_seed(seed_base, prompt.task_id, sample_index)
                for sample_index in range(SAMPLES_PER_TASK)
            }
            if len(task_seeds) != SAMPLES_PER_TASK:
                raise ArtifactError(
                    f"sealed seed derivation collided within task {prompt.task_id}"
                )
    state = ensure_run_header(
        journal_path,
        header_payload,
        allow_indeterminate_slots=(
            authorize_orphan_reissue_with_duplicate_billing_risk
        ),
        allow_recovery_implementation_change=(
            authorize_orphan_reissue_with_duplicate_billing_risk
        ),
    )
    identities = {
        canonical_json(backend_identity(candidate))
        for candidate in state.candidates.values()
    }
    if len(identities) > 1 and not allow_homogeneous_shards:
        raise ArtifactError(
            "journal already contains multiple returned model/backend identities"
        )
    journal_write_lock = RLock()

    def append_journal_event(event: Mapping[str, Any]) -> None:
        # Recovery workers write a receipt immediately before each provider
        # call. Serialize those writes with the main thread's candidate,
        # terminal, error, and verification events so the durable hash chain
        # always has exactly one writer at a time.
        with journal_write_lock:
            append_event(journal_path, event)

    def record_error(event: Mapping[str, Any]) -> None:
        append_journal_event(event)
        slot = (
            str(event.get("task_id") or ""),
            str(event.get("prompt_sha256") or ""),
            int(event.get("sample_index", -1)),
        )
        state.error_counts[slot] = state.error_counts.get(slot, 0) + 1

    def record_candidate(candidate: Mapping[str, Any]) -> None:
        candidate_row = dict(candidate)
        candidate_id = str(candidate_row["candidate_id"])
        request_id = str(
            (candidate_row.get("response") or {}).get("request_id") or ""
        )
        existing_request_ids = {
            str((row.get("response") or {}).get("request_id") or "")
            for row in state.candidates.values()
        }
        if not request_id or request_id in existing_request_ids:
            raise ArtifactError(
                "collector observed a missing/duplicate provider request ID"
            )
        slot = (
            str(candidate_row["task_id"]),
            str(candidate_row["prompt_sha256"]),
            int(candidate_row["sample_index"]),
        )
        candidate_slots = {
            (
                str(row.get("task_id") or ""),
                str(row.get("prompt_sha256") or ""),
                int(row.get("sample_index", -1)),
            )
            for row in state.candidates.values()
        }
        if (
            candidate_id in state.candidates
            or slot in state.slots
            or slot in candidate_slots
        ):
            raise ArtifactError(f"collector attempted to duplicate slot {slot}")
        append_journal_event(candidate_row)
        state.candidates[candidate_id] = candidate_row

    def record_rejection(rejection: Mapping[str, Any]) -> None:
        row = dict(rejection)
        rejected_id = str(row.get("rejected_draw_id") or "")
        slot = (
            str(row.get("task_id") or ""),
            str(row.get("prompt_sha256") or ""),
            int(row.get("sample_index", -1)),
        )
        if (
            not rejected_id
            or rejected_id in state.rejections
            or slot in state.slots
        ):
            raise ArtifactError(
                f"collector attempted to duplicate rejected slot {slot}"
            )
        append_journal_event(row)
        state.rejections[rejected_id] = row

    def record_started(started: Mapping[str, Any]) -> None:
        row = dict(started)
        slot = (
            str(row.get("task_id") or ""),
            str(row.get("prompt_sha256") or ""),
            int(row.get("sample_index", -1)),
        )
        if slot in state.starts or slot in state.slots:
            raise ArtifactError(
                f"collector attempted to restart paid slot {slot}"
            )
        append_journal_event(row)
        state.starts[slot] = row

    def record_reissue_authorization(
        authorization: Mapping[str, Any],
    ) -> None:
        row = dict(authorization)
        slot = (
            str(row.get("task_id") or ""),
            str(row.get("prompt_sha256") or ""),
            int(row.get("sample_index", -1)),
        )
        if (
            slot in state.reissue_authorizations
            or slot in state.terminals
            or slot in state.slots
            or slot not in state.starts
        ):
            raise ArtifactError(
                f"collector attempted an invalid orphan reissue authorization {slot}"
            )
        append_journal_event(row)
        state.reissue_authorizations[slot] = row

    def record_reissue_reauthorization(
        authorization: Mapping[str, Any],
    ) -> None:
        row = dict(authorization)
        slot = (
            str(row.get("task_id") or ""),
            str(row.get("prompt_sha256") or ""),
            int(row.get("sample_index", -1)),
        )
        prior = state.reissue_authorizations.get(slot)
        if (
            prior is None
            or slot in state.terminals
            or slot in state.slots
            or slot not in state.starts
            or row.get("prior_orphan_reissue_authorization_id")
            != prior.get("orphan_reissue_authorization_id")
        ):
            raise ArtifactError(
                "collector attempted an invalid orphan reauthorization "
                f"{slot}"
            )
        append_journal_event(row)
        state.reissue_authorizations[slot] = row

    def record_next_reissue_attempt(
        started: Mapping[str, Any],
        authorization: Mapping[str, Any],
    ) -> dict[str, Any]:
        slot = (
            str(started.get("task_id") or ""),
            str(started.get("prompt_sha256") or ""),
            int(started.get("sample_index", -1)),
        )
        with journal_write_lock:
            attempts = state.reissue_attempts.setdefault(slot, [])
            if (
                slot not in state.reissue_authorizations
                or slot in state.terminals
                or slot in state.slots
                or state.reissue_authorizations[slot].get(
                    "orphan_reissue_authorization_id"
                )
                != authorization.get("orphan_reissue_authorization_id")
            ):
                raise ArtifactError(
                    "collector attempted an invalid orphan reissue attempt "
                    f"{slot}"
                )
            row = make_orphan_reissue_attempt_event(
                started,
                authorization,
                attempt_index=len(attempts) + 1,
            )
            append_event(journal_path, row)
            attempts.append(row)
        return row

    def record_terminal(
        started: Mapping[str, Any],
        outcome: Mapping[str, Any],
        *,
        outcome_type: str,
    ) -> None:
        slot = (
            str(started.get("task_id") or ""),
            str(started.get("prompt_sha256") or ""),
            int(started.get("sample_index", -1)),
        )
        attempts = state.reissue_attempts.get(slot, [])
        row = make_slot_terminal_event(
            started,
            outcome,
            outcome_type=outcome_type,
            orphan_reissue_attempt_id=(
                str(
                    attempts[-1].get("orphan_reissue_attempt_id") or ""
                )
                if attempts
                else ""
            ),
        )
        if slot in state.terminals:
            raise ArtifactError(
                f"collector attempted to duplicate terminal slot {slot}"
            )
        existing_outcome_id = state.slots.get(slot)
        if (
            existing_outcome_id is not None
            and existing_outcome_id != str(row["outcome_id"])
        ):
            raise ArtifactError(
                f"collector attempted to replace terminal slot outcome {slot}"
            )
        append_journal_event(row)
        state.terminals[slot] = row
        state.slots[slot] = str(row["outcome_id"])

    def record_verification(verification: Mapping[str, Any]) -> None:
        verification_row = dict(verification)
        candidate_id = str(verification_row["candidate_id"])
        if candidate_id in state.verifications:
            raise ArtifactError(
                f"collector attempted to duplicate verification {candidate_id}"
            )
        append_journal_event(verification_row)
        state.verifications[candidate_id] = verification_row

    def verify_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        return make_verification_event(candidate, verifier(candidate))

    # A crash after the durable candidate/rejection row but before its terminal
    # lifecycle row is safe to finish locally. Never redraw such a slot.
    for slot in sorted(set(state.slots).difference(state.terminals)):
        started = state.starts[slot]
        outcome_id = state.slots[slot]
        if outcome_id in state.candidates:
            record_terminal(
                started,
                state.candidates[outcome_id],
                outcome_type="candidate",
            )
        elif outcome_id in state.rejections:
            record_terminal(
                started,
                state.rejections[outcome_id],
                outcome_type="rejected_draw",
            )
        else:  # defensive: JournalState.load already validates this relation
            raise ArtifactError(f"orphan slot has an unknown outcome: {slot}")

    if state.rejections:
        first = next(iter(state.rejections.values()))
        raise ArtifactError(
            "journal contains a consumed rejected teacher draw; this sealed "
            "Monte Carlo run is permanently failed and cannot resample "
            f"task={first.get('task_id')} sample={first.get('sample_index')}"
        )

    # Finish all already-durable offline verification before making another
    # paid request. Verification workers never write the journal; the calling
    # thread remains the only writer.
    existing_unverified = [
        candidate
        for candidate_id, candidate in state.candidates.items()
        if candidate_id not in state.verifications
    ]
    if existing_unverified:
        with ThreadPoolExecutor(max_workers=verifier_workers) as executor:
            futures = [
                executor.submit(verify_candidate, candidate)
                for candidate in existing_unverified
            ]
            for future in futures:
                record_verification(future.result())

    # Breadth-first order gives every task its first independent draw before
    # spending quota on the second draw. This is also the useful order for
    # monitoring an interrupted harvest.
    pending: list[tuple[PromptRow, list[dict[str, str]], str, int]] = []
    for sample_index in range(SAMPLES_PER_TASK):
        for prompt in prompts:
            messages = build_messages(system_prompt, prompt)
            prompt_sha256 = stable_sha256(messages)
            slot = (prompt.task_id, prompt_sha256, sample_index)
            if slot not in state.slots:
                pending.append((prompt, messages, prompt_sha256, sample_index))

    def request_parameters_for(
        prompt: PromptRow,
        sample_index: int,
    ) -> dict[str, Any]:
        result = dict(generation_parameters)
        if seed_base is not None:
            result["seed"] = derived_sample_seed(
                seed_base, prompt.task_id, sample_index
            )
        return result

    def call_slot(
        work: tuple[PromptRow, list[dict[str, str]], str, int],
        started: Mapping[str, Any],
        reissue_authorization: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        prompt, messages, prompt_sha256, sample_index = work
        errors: list[dict[str, Any]] = []
        base_request_parameters = dict(started["request_parameters"])
        active_messages: Sequence[Mapping[str, str]] = messages
        request_transport: dict[str, Any] | None = None
        for capacity_index, capacity in enumerate(length_capacities):
            request_parameters = {
                **base_request_parameters,
                "max_tokens": int(capacity),
            }
            payload = {
                "model": requested_model,
                "messages": active_messages,
                **request_parameters,
            }
            advance_capacity = False
            for retry_index in range(max_retries + 1):
                response: Any = None
                try:
                    if reissue_authorization is not None:
                        # One receipt per actual provider-call attempt, including
                        # transport retries and length-capacity escalation.
                        record_next_reissue_attempt(
                            started,
                            reissue_authorization,
                        )
                    response = client.chat.completions.create(**payload)
                    candidate = normalize_response(
                        response,
                        task_id=prompt.task_id,
                        sample_index=sample_index,
                        prompt_sha256=prompt_sha256,
                        requested_model=requested_model,
                        request_parameters=request_parameters,
                        required_function=required_function,
                    )
                    returned_model = str(
                        (candidate.get("response") or {}).get("returned_model") or ""
                    )
                    if (
                        require_returned_model_exact
                        and returned_model != requested_model
                    ):
                        raise ArtifactError(
                            "provider returned a model ID different from the exact "
                            "requested model"
                        )
                    if candidate.get("completion_attested") is not True:
                        raise ArtifactError(
                            "teacher response must finish with stop and contain "
                            "nonempty final content"
                        )
                    if request_transport is not None:
                        candidate = attach_candidate_request_transport(
                            candidate,
                            request_transport,
                        )
                    return {
                        "candidate": candidate,
                        "errors": errors,
                        "terminal_failure": False,
                    }
                except ArtifactError as exc:
                    choices = _field(response, "choices", []) if response is not None else []
                    choice = choices[0] if isinstance(choices, (list, tuple)) and len(choices) == 1 else {}
                    finish_reason = str(_field(choice, "finish_reason", "") or "")
                    if finish_reason == "length" and capacity_index + 1 < len(
                        length_capacities
                    ):
                        plain_response = _plain(response)
                        errors.append(
                            {
                                "schema": JOURNAL_SCHEMA,
                                "event": "teacher_error",
                                "created_at": utc_now(),
                                "task_id": prompt.task_id,
                                "sample_index": sample_index,
                                "prompt_sha256": prompt_sha256,
                                "retry_index": retry_index,
                                "failure_kind": "length_capped_response",
                                "provider_request_id": str(
                                    _field(response, "id", "") or ""
                                ),
                                "provider_response_sha256": stable_sha256(
                                    plain_response
                                ),
                                "attempted_max_tokens": int(capacity),
                                "next_max_tokens": int(
                                    length_capacities[capacity_index + 1]
                                ),
                                "error": _safe_error(exc),
                            }
                        )
                        advance_capacity = True
                        break
                    # A malformed/unsupported non-length response is a consumed
                    # draw. A final-capacity length response is retained as a
                    # terminal rejection rather than silently truncated.
                    if response is None:
                        errors.append(
                            {
                                "schema": JOURNAL_SCHEMA,
                                "event": "teacher_error",
                                "created_at": utc_now(),
                                "task_id": prompt.task_id,
                                "sample_index": sample_index,
                                "prompt_sha256": prompt_sha256,
                                "retry_index": retry_index,
                                "error": _safe_error(exc),
                            }
                        )
                        continue
                    return {
                        "candidate": None,
                        "rejection": make_rejected_draw_event(
                            response,
                            task_id=prompt.task_id,
                            sample_index=sample_index,
                            prompt_sha256=prompt_sha256,
                            requested_model=requested_model,
                            request_parameters=request_parameters,
                            error=exc,
                        ),
                        "errors": errors,
                        "terminal_failure": True,
                        "failure_kind": "invalid_teacher_response",
                        "task_id": prompt.task_id,
                        "sample_index": sample_index,
                    }
                except Exception as exc:
                    error_code = provider_error_code(exc)
                    if (
                        error_code == "data_inspection_failed"
                        and request_transport is None
                    ):
                        active_messages, request_transport = (
                            build_lossless_moderation_transport(messages)
                        )
                        payload["messages"] = active_messages
                        errors.append(
                            {
                                "schema": JOURNAL_SCHEMA,
                                "event": "teacher_error",
                                "created_at": utc_now(),
                                "task_id": prompt.task_id,
                                "sample_index": sample_index,
                                "prompt_sha256": prompt_sha256,
                                "retry_index": retry_index,
                                "failure_kind": (
                                    "provider_input_moderation_false_positive"
                                ),
                                "provider_error_code": error_code,
                                "canonical_messages_sha256": (
                                    request_transport[
                                        "canonical_messages_sha256"
                                    ]
                                ),
                                "transport_messages_sha256": (
                                    request_transport[
                                        "transport_messages_sha256"
                                    ]
                                ),
                                "error": _safe_error(exc),
                            }
                        )
                        # This is the same semantic F2 prompt and exact sampled
                        # slot/seed, encoded through the reversible ASCII
                        # transport. It is never used unless the canonical
                        # provider request was explicitly blocked.
                        continue
                    errors.append(
                        {
                            "schema": JOURNAL_SCHEMA,
                            "event": "teacher_error",
                            "created_at": utc_now(),
                            "task_id": prompt.task_id,
                            "sample_index": sample_index,
                            "prompt_sha256": prompt_sha256,
                            "retry_index": retry_index,
                            "provider_error_code": error_code or None,
                            "error": _safe_error(exc),
                        }
                    )
            if advance_capacity:
                continue
            return {
                "candidate": None,
                "rejection": None,
                "errors": errors,
                "terminal_failure": True,
                "failure_kind": "provider_call_failed",
                "task_id": prompt.task_id,
                "sample_index": sample_index,
            }
        return {
            "candidate": None,
            "rejection": None,
            "errors": errors,
            "terminal_failure": True,
            "failure_kind": "provider_call_failed",
            "task_id": prompt.task_id,
            "sample_index": sample_index,
        }

    fatal_messages: list[str] = []
    api_futures: dict[
        Future[dict[str, Any]],
        tuple[
            tuple[PromptRow, list[dict[str, str]], str, int],
            dict[str, Any],
        ],
    ] = {}
    verification_futures: dict[Future[dict[str, Any]], str] = {}
    next_pending = 0
    completed_at_start = len(state.candidates)
    next_progress = (
        ((completed_at_start // progress_every) + 1) * progress_every
        if progress_every
        else 0
    )
    verification_backlog_limit = max(verifier_workers * 4, verifier_workers)

    def drain_verifications(*, block: bool) -> None:
        nonlocal verification_futures
        if not verification_futures:
            return
        done, _ = wait(
            set(verification_futures),
            return_when=FIRST_COMPLETED,
            timeout=None if block else 0,
        )
        for future in done:
            candidate_id = verification_futures.pop(future)
            try:
                record_verification(future.result())
            except Exception as exc:
                fatal_messages.append(
                    "local verifier failed for durable candidate "
                    f"{candidate_id} ({type(exc).__name__}); journal is resumable"
                )

    def submit_until_full(executor: ThreadPoolExecutor) -> None:
        nonlocal next_pending
        while (
            not fatal_messages
            and next_pending < len(pending)
            and len(api_futures) < workers
            and len(verification_futures) < verification_backlog_limit
        ):
            work = pending[next_pending]
            next_pending += 1
            prompt, _messages, prompt_sha256, sample_index = work
            slot = (prompt.task_id, prompt_sha256, sample_index)
            expected_request_parameters = request_parameters_for(
                prompt, sample_index
            )
            started = state.starts.get(slot)
            reissue_authorization: Mapping[str, Any] | None = None
            if started is None:
                started = make_slot_started_event(
                    task_id=prompt.task_id,
                    sample_index=sample_index,
                    prompt_sha256=prompt_sha256,
                    request_parameters=expected_request_parameters,
                )
                # fsync intent before issuing the paid request.
                record_started(started)
            else:
                if not authorize_orphan_reissue_with_duplicate_billing_risk:
                    raise ArtifactError(
                        "journal contains an orphan paid slot; explicit "
                        "duplicate-billing-risk recovery opt-in is required"
                    )
                if started.get("request_parameters") != expected_request_parameters:
                    raise ArtifactError(
                        "orphan reissue request parameters/seed differ from "
                        f"the original request receipt: {slot}"
                    )
                authorization = state.reissue_authorizations.get(slot)
                if authorization is None:
                    authorization = make_orphan_reissue_authorization_event(
                        started,
                        original_run_header_sha256=str(
                            (state.header or {}).get("header_sha256") or ""
                        ),
                        original_collector_implementation=(
                            ((state.header or {}).get("payload") or {}).get(
                                "implementation"
                            )
                        ),
                        recovery_collector_implementation=header_payload.get(
                            "implementation"
                        ),
                    )
                    # This fsynced event records that the original provider
                    # request may already have billed or completed. A crash
                    # after it remains the same authorized logical reissue.
                    record_reissue_authorization(authorization)
                elif authorization.get(
                    "recovery_collector_implementation"
                ) != header_payload.get("implementation"):
                    authorization = (
                        make_orphan_reissue_reauthorization_event(
                            started,
                            authorization,
                            recovery_collector_implementation=(
                                header_payload["implementation"]
                            ),
                        )
                    )
                    record_reissue_reauthorization(authorization)
                reissue_authorization = authorization
            api_futures[
                executor.submit(
                    call_slot,
                    work,
                    started,
                    reissue_authorization,
                )
            ] = (
                work,
                started,
            )

    with (
        ThreadPoolExecutor(max_workers=workers) as api_executor,
        ThreadPoolExecutor(max_workers=verifier_workers) as verifier_executor,
    ):
        submit_until_full(api_executor)
        while api_futures or (
            not fatal_messages and next_pending < len(pending)
        ):
            if not api_futures:
                # API scheduling can be throttled by a full local-verification
                # backlog. Make room, then resume without dropping pending
                # paid-request slots.
                drain_verifications(block=True)
                submit_until_full(api_executor)
                continue
            done, _ = wait(set(api_futures), return_when=FIRST_COMPLETED)
            for future in done:
                work, started = api_futures.pop(future)
                prompt, _, _, sample_index = work
                try:
                    result = future.result()
                except Exception as exc:  # defensive: call_slot is self-contained
                    fatal_messages.append(
                        "teacher worker crashed for "
                        f"{prompt.task_id} sample {sample_index} "
                        f"({type(exc).__name__}); journal is resumable"
                    )
                    continue
                for event in result["errors"]:
                    record_error(event)
                candidate = result.get("candidate")
                if candidate is None:
                    rejection = result.get("rejection")
                    if rejection is not None:
                        record_rejection(rejection)
                        record_terminal(
                            started,
                            rejection,
                            outcome_type="rejected_draw",
                        )
                    fatal_messages.append(
                        "teacher call failed for "
                        f"{result['task_id']} sample {result['sample_index']} "
                        f"({result['failure_kind']}); the paid slot is sealed "
                        "and automatic resampling is forbidden"
                    )
                    continue
                record_candidate(candidate)
                record_terminal(
                    started,
                    candidate,
                    outcome_type="candidate",
                )
                observed_identity = canonical_json(backend_identity(candidate))
                identities.add(observed_identity)
                if len(identities) > 1 and not allow_homogeneous_shards:
                    fatal_messages.append(
                        "requested alias returned a different model/backend; "
                        "candidate was preserved, collection stopped"
                    )
                candidate_id = str(candidate["candidate_id"])
                verification_futures[
                    verifier_executor.submit(verify_candidate, candidate)
                ] = candidate_id
                if progress_every and len(state.candidates) >= next_progress:
                    print(
                        "QWEN_TEACHER_PROGRESS "
                        f"candidates={len(state.candidates)} "
                        f"verified={len(state.verifications)} "
                        f"total={len(prompts) * SAMPLES_PER_TASK}",
                        flush=True,
                    )
                    next_progress += progress_every
            drain_verifications(block=False)
            while (
                not fatal_messages
                and next_pending < len(pending)
                and len(verification_futures) >= verification_backlog_limit
            ):
                drain_verifications(block=True)
            submit_until_full(api_executor)

        while verification_futures:
            drain_verifications(block=True)

    if fatal_messages:
        raise ArtifactError(fatal_messages[0])
    if next_pending != len(pending):
        raise ArtifactError(
            "collection stopped before all slots were scheduled; journal is resumable"
        )
    return state


class StudentTokenizerBinding:
    """Strict raw-byte to exactly-one-student-token binding."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        eos_token_id: int,
        tokenizer_record: Mapping[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.tokenizer_record = dict(tokenizer_record)
        vocab_size = int(tokenizer.get_vocab_size())
        if not 0 <= self.eos_token_id < vocab_size:
            raise ArtifactError(
                f"EOS token id {self.eos_token_id} is outside vocabulary {vocab_size}"
            )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        eos_token_id: int,
    ) -> "StudentTokenizerBinding":
        record = file_record(path)
        if record["sha256"] != expected_sha256.strip().lower():
            raise ArtifactError(
                "student tokenizer hash mismatch: "
                f"expected {expected_sha256}, got {record['sha256']}"
            )
        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover - dependency failure
            raise ArtifactError("the tokenizers package is required") from exc
        tokenizer = Tokenizer.from_file(record["path"])
        return cls(
            tokenizer,
            eos_token_id=eos_token_id,
            tokenizer_record=record,
        )

    def map_bytes(self, raw_octets: Sequence[int] | None) -> tuple[int | None, str | None]:
        if raw_octets is None:
            return None, "provider_bytes_missing"
        raw = bytes(int(value) for value in raw_octets)
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None, "provider_bytes_not_standalone_utf8"
        try:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
        except Exception as exc:
            return None, f"student_tokenizer_encode_error:{type(exc).__name__}"
        if len(ids) != 1:
            return None, f"student_mapping_is_{len(ids)}_tokens"
        token_id = int(ids[0])
        try:
            decoded = str(
                self.tokenizer.decode([token_id], skip_special_tokens=False)
            ).encode("utf-8")
        except Exception as exc:
            return None, f"student_tokenizer_decode_error:{type(exc).__name__}"
        if decoded != raw:
            return None, "student_roundtrip_bytes_mismatch"
        return token_id, None


def audit_candidate_tokens(
    candidate: Mapping[str, Any],
    binding: StudentTokenizerBinding,
    *,
    negative_tail_tolerance: float = NEGATIVE_TAIL_TOLERANCE,
) -> dict[str, Any]:
    token_rows: list[dict[str, Any]] = []
    materially_negative = 0
    tiny_negative = 0
    chosen_mapped = 0
    alternatives_mapped = 0
    alternatives_total = 0
    exact_top5 = 0
    chosen_in_top = 0
    eos_positions: list[int] = []
    tokens = candidate.get("chosen_tokens_with_top_logprobs") or []
    chosen_byte_fragments: list[bytes] = []
    chosen_bytes_complete = True
    for position, token in enumerate(tokens):
        raw_chosen = token.get("bytes")
        if raw_chosen is None:
            chosen_bytes_complete = False
        else:
            chosen_byte_fragments.append(bytes(int(value) for value in raw_chosen))
        chosen_id, chosen_error = binding.map_bytes(token.get("bytes"))
        if chosen_id is not None:
            chosen_mapped += 1
            if chosen_id == binding.eos_token_id:
                eos_positions.append(position)
        top = token.get("top_logprobs") or []
        if len(top) == TOP_LOGPROBS:
            exact_top5 += 1
        top_mass = math.fsum(math.exp(float(item["logprob"])) for item in top)
        raw_tail = 1.0 - top_mass  # retain raw value; never max(0, tail)
        is_material = raw_tail < -abs(float(negative_tail_tolerance))
        is_tiny = raw_tail < 0.0 and not is_material
        materially_negative += int(is_material)
        tiny_negative += int(is_tiny)
        alternative_rows: list[dict[str, Any]] = []
        selected_in_top = False
        for alternative in top:
            token_id, mapping_error = binding.map_bytes(alternative.get("bytes"))
            alternatives_total += 1
            alternatives_mapped += int(token_id is not None)
            if (
                alternative.get("bytes") == token.get("bytes")
                and float(alternative["logprob"]) == float(token["logprob"])
            ):
                selected_in_top = True
            alternative_rows.append(
                {
                    "student_token_id": token_id,
                    "mapping_error": mapping_error,
                }
            )
        chosen_in_top += int(selected_in_top)
        token_rows.append(
            {
                "position": position,
                "chosen_student_token_id": chosen_id,
                "chosen_mapping_error": chosen_error,
                "top_alternative_mappings": alternative_rows,
                "top_logprob_count": len(top),
                "top_probability_mass": top_mass,
                "tail_probability_mass_raw": raw_tail,
                "materially_negative_tail": is_material,
                "tiny_negative_tail_within_tolerance": is_tiny,
                "chosen_present_in_top": selected_in_top,
            }
        )
    token_count = len(tokens)
    mapping_complete = token_count > 0 and chosen_mapped == token_count
    top_mapping_complete = (
        alternatives_total > 0 and alternatives_mapped == alternatives_total
    )
    reconstructed_bytes = (
        b"".join(chosen_byte_fragments) if chosen_bytes_complete else None
    )
    expected_content_bytes = str(
        (candidate.get("response") or {}).get("raw_content") or ""
    ).encode("utf-8")
    content_bytes_match = bool(
        reconstructed_bytes is not None
        and reconstructed_bytes == expected_content_bytes
    )
    return {
        "tokens": token_rows,
        "summary": {
            "chosen_token_count": token_count,
            "chosen_tokens_mapped_one_to_one": chosen_mapped,
            "chosen_mapping_coverage": (
                chosen_mapped / token_count if token_count else 0.0
            ),
            "top_alternatives": alternatives_total,
            "top_alternatives_mapped_one_to_one": alternatives_mapped,
            "top_mapping_coverage": (
                alternatives_mapped / alternatives_total
                if alternatives_total
                else 0.0
            ),
            "positions_with_exactly_top5": exact_top5,
            "positions_with_chosen_in_top": chosen_in_top,
            "materially_negative_tail_positions": materially_negative,
            "tiny_negative_tail_positions": tiny_negative,
            "logged_eos_positions": eos_positions,
            "logged_eos_covered": bool(eos_positions),
            "chosen_bytes_complete": chosen_bytes_complete,
            "chosen_bytes_reconstruct_raw_content": content_bytes_match,
            "reconstructed_content_bytes_sha256": (
                hashlib.sha256(reconstructed_bytes).hexdigest()
                if reconstructed_bytes is not None
                else None
            ),
            "raw_content_bytes_sha256": hashlib.sha256(
                expected_content_bytes
            ).hexdigest(),
            "chosen_mapping_complete": mapping_complete,
            "top_mapping_complete": top_mapping_complete,
            "top5_count_complete": token_count > 0 and exact_top5 == token_count,
            "tail_valid": materially_negative == 0,
            "negative_tail_tolerance": float(negative_tail_tolerance),
        },
    }


def _output_for_shard(path: Path, identity_sha: str, split: bool) -> Path:
    if not split:
        return path
    return path.with_name(path.stem + f".shard-{identity_sha[:12]}" + path.suffix)


def materialize_artifacts(
    *,
    journal_path: str | Path,
    binding: StudentTokenizerBinding,
    parseable_output: str | Path,
    rs_sft_output: str | Path,
    audit_output: str | Path,
    allow_homogeneous_shards: bool = False,
    negative_tail_tolerance: float = NEGATIVE_TAIL_TOLERANCE,
) -> dict[str, Any]:
    state = JournalState.load(journal_path)
    if state.rejections:
        first = next(iter(state.rejections.values()))
        raise ArtifactError(
            "teacher journal contains a consumed rejected draw; exact-K "
            "sequence distillation is permanently failed without resampling: "
            f"task={first.get('task_id')} sample={first.get('sample_index')}"
        )
    if state.header is None:
        raise ArtifactError("journal has no run header")
    payload = state.header.get("payload") or {}
    expected_tasks = [str(value) for value in payload.get("task_ids") or []]
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise ArtifactError("run header has an invalid task set")
    if int(payload.get("samples_per_task", -1)) != SAMPLES_PER_TASK:
        raise ArtifactError("run header is not a K=8 collection")
    prompt_bindings = payload.get("prompt_bindings")
    if not isinstance(prompt_bindings, list) or len(prompt_bindings) != len(
        expected_tasks
    ):
        raise ArtifactError("run header has no exact per-task prompt bindings")
    expected_prompt_hashes: dict[str, str] = {}
    for item in prompt_bindings:
        if not isinstance(item, Mapping):
            raise ArtifactError("run header prompt binding is not an object")
        task_id = str(item.get("task_id") or "")
        prompt_hash = str(item.get("request_messages_sha256") or "")
        if (
            task_id not in expected_tasks
            or task_id in expected_prompt_hashes
            or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash)
        ):
            raise ArtifactError("run header has an invalid prompt binding")
        expected_prompt_hashes[task_id] = prompt_hash
    if set(expected_prompt_hashes) != set(expected_tasks):
        raise ArtifactError("run header prompt bindings do not cover every task")
    expected_parameters = payload.get("generation_parameters")
    if not isinstance(expected_parameters, Mapping):
        raise ArtifactError("run header has no generation-parameter seal")
    validate_mc_teacher_sampling(expected_parameters)
    objective_mode = objective_mode_from_header(payload)
    target_contract = validate_target_length_contract(
        payload.get("target_length_contract"),
        binding=binding,
        objective_mode=objective_mode,
    )
    max_target_tokens = int(target_contract["max_target_tokens"])
    if objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5:
        if (
            expected_parameters.get("logprobs") is not True
            or int(expected_parameters.get("top_logprobs", -1))
            != TOP_LOGPROBS
        ):
            raise ArtifactError(
                "require_top5 journal does not seal logprobs=true/top_logprobs=5"
            )
        extra_body = expected_parameters.get("extra_body")
        if (
            not isinstance(extra_body, Mapping)
            or extra_body.get("enable_thinking") is not False
        ):
            raise ArtifactError(
                "require_top5 journal enables hidden reasoning, so teacher and "
                "student token prefixes are not aligned"
            )
    elif (
        "top_logprobs" in expected_parameters
        or expected_parameters.get("logprobs") not in {None, False}
    ):
        raise ArtifactError(
            "sequence_only journal contains provider logprob request fields"
        )
    expected_requested_model = str(payload.get("requested_model") or "")
    if not expected_requested_model:
        raise ArtifactError("run header has no requested model")
    if objective_mode == OBJECTIVE_MODE_SEQUENCE_ONLY:
        validate_qwen38_sequence_sampling(
            expected_requested_model, expected_parameters
        )
    seed_contract = payload.get("sampling_seed_contract")
    if seed_contract is None:
        seed_base: int | None = None
    elif (
        isinstance(seed_contract, Mapping)
        and seed_contract.get("algorithm") == SAMPLE_SEED_ALGORITHM
        and seed_contract.get("unique_seed_required_per_task_draw") is True
        and isinstance(seed_contract.get("seed_base"), int)
        and not isinstance(seed_contract.get("seed_base"), bool)
        and 0 <= int(seed_contract["seed_base"]) < 2**31
    ):
        seed_base = int(seed_contract["seed_base"])
    else:
        raise ArtifactError("run header has an invalid sampling_seed_contract")

    groups: dict[str, list[dict[str, Any]]] = {}
    provider_request_ids: set[str] = set()
    for candidate in state.candidates.values():
        task_id = str(candidate.get("task_id") or "")
        if candidate.get("prompt_sha256") != expected_prompt_hashes.get(task_id):
            raise ArtifactError(
                f"journal candidate {candidate.get('candidate_id')} has a "
                "prompt hash outside the sealed task binding"
            )
        candidate_parameters = dict(expected_parameters)
        if seed_base is not None:
            candidate_parameters["seed"] = derived_sample_seed(
                seed_base,
                task_id,
                int(candidate.get("sample_index", -1)),
            )
        observed_parameters = dict(candidate.get("request_parameters") or {})
        observed_capacity = observed_parameters.get("max_tokens")
        length_policy = (payload.get("transport") or {}).get(
            "length_capped_response_policy"
        ) or {}
        allowed_capacities = list(length_policy.get("max_token_capacities") or [])
        if observed_capacity not in allowed_capacities:
            raise ArtifactError("journal candidate used an unsealed output capacity")
        candidate_parameters["max_tokens"] = observed_capacity
        if canonical_json(observed_parameters) != canonical_json(candidate_parameters):
            raise ArtifactError("journal candidate request parameters changed")
        if str(candidate.get("requested_model") or "") != expected_requested_model:
            raise ArtifactError("journal candidate requested-model seal changed")
        request_id = str((candidate.get("response") or {}).get("request_id") or "")
        if not request_id or request_id in provider_request_ids:
            raise ArtifactError(
                "journal contains a missing/duplicate provider request ID"
            )
        provider_request_ids.add(request_id)
        identity_sha = backend_identity_sha256(candidate)
        groups.setdefault(identity_sha, []).append(candidate)
    if len(groups) > 1 and not allow_homogeneous_shards:
        raise ArtifactError(
            "alias/backend changed within the artifact; explicitly split "
            "homogeneous shards before production use"
        )

    parseable_base = Path(parseable_output)
    rs_base = Path(rs_sft_output)
    shard_reports: list[dict[str, Any]] = []
    aggregate = {
        "candidates": 0,
        "sequence": 0,
        "parseable": 0,
        "rs_sft": 0,
        "completion_attested": 0,
        "chosen_tokens": 0,
        "chosen_mapped": 0,
        "top_alternatives": 0,
        "top_mapped": 0,
        "material_negative": 0,
        "eos_sequences": 0,
        "sequences_with_chosen_mapping": 0,
        "sequences_with_top_mapping": 0,
        "sequences_with_exact_top5": 0,
        "sequences_with_content_byte_reconstruction": 0,
        "positions_with_exact_top5": 0,
        "positions_with_chosen_in_top": 0,
        "targets_within_contract": 0,
        "targets_final_dart_code_only": 0,
        "targets_reasoning_excluded": 0,
    }
    target_length_checks: list[dict[str, Any]] = []
    overflow_diagnostics: list[dict[str, Any]] = []
    non_code_diagnostics: list[dict[str, Any]] = []
    split = len(groups) > 1
    for identity_sha, candidates in sorted(groups.items()):
        parseable_rows: list[dict[str, Any]] = []
        rs_rows: list[dict[str, Any]] = []
        candidates.sort(key=lambda row: (str(row["task_id"]), int(row["sample_index"])))
        for candidate in candidates:
            token_audit = audit_candidate_tokens(
                candidate,
                binding,
                negative_tail_tolerance=negative_tail_tolerance,
            )
            summary = token_audit["summary"]
            parse = candidate.get("parse") or {}
            verification = state.verifications.get(str(candidate["candidate_id"]))
            parseable = parse.get("parseable") is True
            attested = candidate.get("completion_attested") is True
            raw_content = str(candidate["response"].get("raw_content") or "")
            sequence_target = raw_content.strip() if attested else ""
            sequence_eligible = bool(attested and sequence_target)
            length_evidence = (
                target_length_evidence(
                    sequence_target,
                    binding=binding,
                    max_target_tokens=max_target_tokens,
                )
                if sequence_eligible
                else None
            )
            final_dart_code_only = bool(
                sequence_eligible
                and parseable
                and parse.get("code_equals_trimmed_raw_content") is True
                and str(parse.get("code") or "") == sequence_target
                and is_final_dart_code_only(
                    sequence_target,
                    str(parse.get("required_function") or ""),
                )
            )
            reasoning_excluded = bool(
                sequence_eligible
                and sha256_text(sequence_target)
                == sha256_text(raw_content.strip())
            )
            if length_evidence is not None:
                diagnostic_basis = {
                    "task_id": str(candidate["task_id"]),
                    "sample_index": int(candidate["sample_index"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    **length_evidence,
                    "final_dart_code_only": final_dart_code_only,
                    "reasoning_excluded": reasoning_excluded,
                }
                target_length_checks.append(diagnostic_basis)
                if length_evidence["within_contract"] is not True:
                    overflow_diagnostics.append(diagnostic_basis)
                if not final_dart_code_only:
                    non_code_diagnostics.append(
                        {
                            "task_id": str(candidate["task_id"]),
                            "sample_index": int(candidate["sample_index"]),
                            "candidate_id": str(candidate["candidate_id"]),
                            "sequence_target_sha256": sha256_text(
                                sequence_target
                            ),
                            "parseable": parseable,
                            "target_normalization": parse.get(
                                "normalization"
                            ),
                            "code_equals_trimmed_raw_content": bool(
                                parse.get("code_equals_trimmed_raw_content")
                            ),
                        }
                    )
            verifier_pass = bool(
                verification
                and verification.get("passed") is True
                and verification.get("harness_completion_attested") is True
            )
            row = {
                "schema": PARSEABLE_SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "task_id": candidate["task_id"],
                "sample_index": candidate["sample_index"],
                "prompt_sha256": candidate["prompt_sha256"],
                "backend_identity": candidate["backend_identity"],
                "request_parameters": candidate["request_parameters"],
                "response": candidate["response"],
                "raw_content": raw_content,
                "sequence_target": sequence_target,
                "sequence_target_sha256": (
                    sha256_text(sequence_target) if sequence_eligible else None
                ),
                "target_length_evidence": length_evidence,
                "target_content_contract": {
                    "source_field": "choice.message.content",
                    "reasoning_field": "choice.message.reasoning_content",
                    "reasoning_excluded": reasoning_excluded,
                    "final_dart_code_only": final_dart_code_only,
                },
                "code": parse.get("code") or "",
                "code_sha256": parse.get("code_sha256"),
                "target_normalization": parse.get("normalization"),
                "code_equals_trimmed_raw_content": bool(
                    parse.get("code_equals_trimmed_raw_content")
                ),
                "chosen_tokens_with_top_logprobs": candidate[
                    "chosen_tokens_with_top_logprobs"
                ],
                "student_token_mapping_audit": token_audit,
                "eligibility": {
                    "sft": sequence_eligible,
                    # Sequence Monte Carlo forward-KL/NLL uses the independently
                    # sampled text as the target. It does not pretend the sparse
                    # provider logprobs are a dense teacher distribution.
                    "mc_sequence_forward_kl": sequence_eligible,
                    "sparse_top5_tail_objective": bool(
                        objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
                        and
                        parseable
                        and summary["chosen_mapping_complete"]
                        and summary["top_mapping_complete"]
                        and summary["top5_count_complete"]
                        and summary["tail_valid"]
                    ),
                    "dense_full_vocabulary_kl": False,
                    "objective_mode": objective_mode,
                    "note": (
                        "Every completion-attested teacher final-content draw "
                        "is retained after trim_outer_whitespace only. Top-5 "
                        "plus audited tail is not a dense teacher distribution."
                    ),
                },
            }
            if sequence_eligible:
                parseable_rows.append(row)
            if parseable and attested and verifier_pass:
                rs_rows.append(
                    {
                        "schema": RS_SFT_SCHEMA,
                        "candidate_id": candidate["candidate_id"],
                        "task_id": candidate["task_id"],
                        "sample_index": candidate["sample_index"],
                        "prompt_sha256": candidate["prompt_sha256"],
                        "code": parse["code"],
                        "code_sha256": parse["code_sha256"],
                        "completion_attested": True,
                        "verification": {
                            key: verification[key]
                            for key in (
                                "compiled",
                                "passed",
                                "harness_completion_attested",
                                "verifier_id",
                                "verifier_sha256",
                                "tests_sha256",
                                "diagnostic_present",
                                "diagnostic_sha256",
                                "verification_payload_sha256",
                            )
                        },
                        "backend_identity": candidate["backend_identity"],
                    }
                )
            aggregate["candidates"] += 1
            aggregate["sequence"] += int(sequence_eligible)
            aggregate["parseable"] += int(parseable)
            aggregate["rs_sft"] += int(parseable and attested and verifier_pass)
            aggregate["completion_attested"] += int(attested)
            aggregate["chosen_tokens"] += summary["chosen_token_count"]
            aggregate["chosen_mapped"] += summary[
                "chosen_tokens_mapped_one_to_one"
            ]
            aggregate["top_alternatives"] += summary["top_alternatives"]
            aggregate["top_mapped"] += summary[
                "top_alternatives_mapped_one_to_one"
            ]
            aggregate["material_negative"] += summary[
                "materially_negative_tail_positions"
            ]
            aggregate["eos_sequences"] += int(summary["logged_eos_covered"])
            aggregate["sequences_with_chosen_mapping"] += int(
                summary["chosen_mapping_complete"]
            )
            aggregate["sequences_with_top_mapping"] += int(
                summary["top_mapping_complete"]
            )
            aggregate["sequences_with_exact_top5"] += int(
                summary["top5_count_complete"]
            )
            aggregate["sequences_with_content_byte_reconstruction"] += int(
                summary["chosen_bytes_reconstruct_raw_content"]
            )
            aggregate["positions_with_exact_top5"] += summary[
                "positions_with_exactly_top5"
            ]
            aggregate["positions_with_chosen_in_top"] += summary[
                "positions_with_chosen_in_top"
            ]
            aggregate["targets_within_contract"] += int(
                bool(
                    length_evidence
                    and length_evidence["within_contract"] is True
                )
            )
            aggregate["targets_final_dart_code_only"] += int(
                final_dart_code_only
            )
            aggregate["targets_reasoning_excluded"] += int(
                reasoning_excluded
            )

        parseable_path = _output_for_shard(parseable_base, identity_sha, split)
        rs_path = _output_for_shard(rs_base, identity_sha, split)
        atomic_write_jsonl(parseable_path, parseable_rows)
        atomic_write_jsonl(rs_path, rs_rows)
        exemplar = candidates[0] if candidates else {}
        shard_reports.append(
            {
                "backend_identity_sha256": identity_sha,
                "backend_identity": backend_identity(exemplar) if exemplar else {},
                "candidates": len(candidates),
                "sequence_rows": len(parseable_rows),
                "parseable_rows": len(parseable_rows),
                "rs_sft_rows": len(rs_rows),
                "parseable_output": file_record(parseable_path),
                "rs_sft_output": file_record(rs_path),
            }
        )

    counts_by_task: dict[str, int] = {task_id: 0 for task_id in expected_tasks}
    indices_by_task: dict[str, set[int]] = {
        task_id: set() for task_id in expected_tasks
    }
    requested_seeds_by_task: dict[str, set[int]] = {
        task_id: set() for task_id in expected_tasks
    }
    sequence_hashes_by_task: dict[str, set[str]] = {
        task_id: set() for task_id in expected_tasks
    }
    provider_reported_seed_candidates = 0
    for candidate in state.candidates.values():
        task_id = str(candidate["task_id"])
        if task_id not in counts_by_task:
            raise ArtifactError(f"journal candidate has unknown task_id {task_id}")
        counts_by_task[task_id] += 1
        indices_by_task[task_id].add(int(candidate["sample_index"]))
        request_seed = (candidate.get("request_parameters") or {}).get("seed")
        if request_seed is not None:
            requested_seeds_by_task[task_id].add(int(request_seed))
        reported_seed = (candidate.get("response") or {}).get(
            "provider_reported_seed"
        )
        if reported_seed is not None:
            provider_reported_seed_candidates += 1
            if request_seed is None or int(reported_seed) != int(request_seed):
                raise ArtifactError(
                    "provider-reported seed differs from the sealed request"
                )
        if candidate.get("completion_attested") is True:
            final_sequence = str(
                (candidate.get("response") or {}).get("raw_content") or ""
            ).strip()
            if final_sequence:
                sequence_hashes_by_task[task_id].add(
                    sha256_text(final_sequence)
                )
    incomplete = {
        task_id: count
        for task_id, count in counts_by_task.items()
        if count != SAMPLES_PER_TASK
    }
    invalid_sample_indices = {
        task_id: sorted(indices)
        for task_id, indices in indices_by_task.items()
        if indices != set(range(SAMPLES_PER_TASK))
    }
    invalid_requested_seed_counts = {
        task_id: len(values)
        for task_id, values in requested_seeds_by_task.items()
        if seed_base is not None and len(values) != SAMPLES_PER_TASK
    }
    unique_sequence_counts_by_task = {
        task_id: len(sequence_hashes_by_task[task_id])
        for task_id in expected_tasks
    }
    tasks_with_duplicate_draws = sorted(
        task_id
        for task_id, unique_count in unique_sequence_counts_by_task.items()
        if 0 < unique_count < SAMPLES_PER_TASK
    )
    tasks_with_all_draws_identical = sorted(
        task_id
        for task_id, unique_count in unique_sequence_counts_by_task.items()
        if unique_count == 1
    )
    # A single deterministic-looking task is legitimate.  The production
    # pilot has exactly 16 tasks; only the corpus-wide pattern in which every
    # one of those tasks returns one repeated final sequence is pathological.
    pathological_all_tasks_identical = bool(
        len(expected_tasks) >= 16
        and not incomplete
        and len(tasks_with_all_draws_identical) == len(expected_tasks)
    )
    provider_seed_honor_attested = bool(
        seed_base is not None
        and state.candidates
        and provider_reported_seed_candidates == len(state.candidates)
    )
    all_verified = len(state.verifications) == len(state.candidates)
    chosen_coverage = (
        aggregate["chosen_mapped"] / aggregate["chosen_tokens"]
        if aggregate["chosen_tokens"]
        else 0.0
    )
    top_coverage = (
        aggregate["top_mapped"] / aggregate["top_alternatives"]
        if aggregate["top_alternatives"]
        else 0.0
    )
    sequence_common_ready = bool(
        not incomplete
        and not invalid_sample_indices
        and not invalid_requested_seed_counts
        and all_verified
        and (len(groups) <= 1 or allow_homogeneous_shards)
        and aggregate["candidates"] > 0
        and aggregate["sequence"] == aggregate["candidates"]
        and aggregate["targets_within_contract"] == aggregate["candidates"]
        and aggregate["targets_reasoning_excluded"] == aggregate["candidates"]
    )
    final_code_ready = bool(
        sequence_common_ready
        and aggregate["targets_final_dart_code_only"]
        == aggregate["candidates"]
    )
    top5_contract_ready = bool(
        objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
        and final_code_ready
        and (expected_parameters.get("extra_body") or {}).get(
            "enable_thinking"
        )
        is False
        and aggregate["material_negative"] == 0
        and aggregate["sequences_with_content_byte_reconstruction"]
        == aggregate["candidates"]
    )
    sequence_contract_ready = bool(
        sequence_common_ready
        and not pathological_all_tasks_identical
        and (
            objective_mode == OBJECTIVE_MODE_SEQUENCE_ONLY
            or top5_contract_ready
        )
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "created_at": utc_now(),
        "objective_mode": objective_mode,
        "provider_logprobs_required": (
            objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
        ),
        "journal": file_record(journal_path),
        "journal_chain_head": file_record(
            Path(str(journal_path) + ".chain-head.json")
        ),
        "run_header_sha256": state.header.get("header_sha256"),
        "prompt_artifact": payload.get("prompt_artifact"),
        "prompt_manifest": payload.get("prompt_manifest"),
        "f2_prompt_contract": payload.get("f2_prompt_contract"),
        "pilot_quality_gate": payload.get("pilot_quality_gate"),
        "pilot_quality_gate_contract": payload.get(
            "pilot_quality_gate_contract"
        ),
        "expected_tasks": len(expected_tasks),
        "samples_per_task": SAMPLES_PER_TASK,
        "incomplete_task_sample_counts": incomplete,
        "invalid_task_sample_indices": invalid_sample_indices,
        "teacher_error_events": sum(state.error_counts.values()),
        "all_candidates_independently_verified": all_verified,
        "unique_provider_request_ids": len(provider_request_ids),
        "sampling": {
            "separate_n1_requests": True,
            "requested_seed_algorithm": (
                SAMPLE_SEED_ALGORITHM if seed_base is not None else None
            ),
            "requested_seed_base": seed_base,
            "distinct_requested_seeds_per_task": {
                task_id: len(requested_seeds_by_task[task_id])
                for task_id in expected_tasks
            },
            "invalid_requested_seed_counts": invalid_requested_seed_counts,
            "provider_reported_seed_candidates": (
                provider_reported_seed_candidates
            ),
            "provider_seed_honor_attested": provider_seed_honor_attested,
            "provider_seed_honor_assumed": False,
            "provider_seed_honor_note": (
                "Attested only when every response explicitly echoes the exact "
                "requested seed. Distinct request IDs plus temperature=1 "
                "sampling remain valid when the provider does not echo a seed."
            ),
            "unique_final_sequences_per_task": (
                unique_sequence_counts_by_task
            ),
            "minimum_unique_final_sequences_per_task": min(
                unique_sequence_counts_by_task.values(), default=0
            ),
            "maximum_unique_final_sequences_per_task": max(
                unique_sequence_counts_by_task.values(), default=0
            ),
            "tasks_with_any_duplicate_draws": tasks_with_duplicate_draws,
            "tasks_with_all_k8_draws_identical": (
                tasks_with_all_draws_identical
            ),
            "pathological_all_tasks_have_identical_k8_draws": (
                pathological_all_tasks_identical
            ),
            "duplicates_filtered_from_sequence_training": False,
        },
        "homogeneous_backend_shards": shard_reports,
        "student_tokenizer": binding.tokenizer_record,
        "student_eos_token_id": binding.eos_token_id,
        "target_length_gate": {
            "schema": TARGET_LENGTH_EVIDENCE_SCHEMA,
            "passed": bool(
                aggregate["candidates"] > 0
                and aggregate["sequence"] == aggregate["candidates"]
                and not overflow_diagnostics
                and (
                    objective_mode == OBJECTIVE_MODE_SEQUENCE_ONLY
                    or not non_code_diagnostics
                )
                and aggregate["targets_reasoning_excluded"]
                == aggregate["candidates"]
            ),
            "target_contract": target_contract,
            "targets_checked": len(target_length_checks),
            "targets_within_contract": aggregate[
                "targets_within_contract"
            ],
            "targets_final_dart_code_only": aggregate[
                "targets_final_dart_code_only"
            ],
            "targets_reasoning_excluded": aggregate[
                "targets_reasoning_excluded"
            ],
            "overflow_count": len(overflow_diagnostics),
            "overflow_diagnostics": overflow_diagnostics,
            "non_code_target_count": len(non_code_diagnostics),
            "final_dart_code_only_required": (
                objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
            ),
            "non_code_target_diagnostics": non_code_diagnostics,
            "evidence_sha256": stable_sha256(target_length_checks),
            "failure_policy": {
                "truncate": False,
                "filter_draw": False,
                "resample_draw": False,
                "continue_to_build": False,
                "continue_to_gpu": False,
            },
        },
        "coverage": {
            "candidates": aggregate["candidates"],
            "sequence_candidates": aggregate["sequence"],
            "parseable_candidates": aggregate["parseable"],
            "completion_attested_candidates": aggregate["completion_attested"],
            "rs_sft_candidates": aggregate["rs_sft"],
            "chosen_tokens": aggregate["chosen_tokens"],
            "chosen_tokens_one_to_one_mapped": aggregate["chosen_mapped"],
            "chosen_mapping_coverage": chosen_coverage,
            "top_alternatives": aggregate["top_alternatives"],
            "top_alternatives_one_to_one_mapped": aggregate["top_mapped"],
            "top_mapping_coverage": top_coverage,
            "sequences_with_complete_chosen_mapping": aggregate[
                "sequences_with_chosen_mapping"
            ],
            "sequences_with_complete_top_mapping": aggregate[
                "sequences_with_top_mapping"
            ],
            "sequences_with_exact_top5_at_every_position": aggregate[
                "sequences_with_exact_top5"
            ],
            "sequences_whose_chosen_bytes_reconstruct_raw_content": aggregate[
                "sequences_with_content_byte_reconstruction"
            ],
            "positions_with_exactly_top5": aggregate[
                "positions_with_exact_top5"
            ],
            "positions_with_chosen_token_in_top5": aggregate[
                "positions_with_chosen_in_top"
            ],
            "sequences_with_logged_eos": aggregate["eos_sequences"],
            "logged_eos_sequence_coverage": (
                aggregate["eos_sequences"] / aggregate["candidates"]
                if aggregate["candidates"]
                else 0.0
            ),
            "materially_negative_tail_positions": aggregate["material_negative"],
            "negative_tail_tolerance": float(negative_tail_tolerance),
            "targets_within_trainer_contract": aggregate[
                "targets_within_contract"
            ],
            "targets_final_dart_code_only": aggregate[
                "targets_final_dart_code_only"
            ],
            "targets_reasoning_excluded": aggregate[
                "targets_reasoning_excluded"
            ],
        },
        "capabilities": {
            "monte_carlo_sequence_forward_kl": True,
            "correctness_conditioned_rs_sft": True,
            "sparse_top5_plus_tail": (
                objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
                and
                (expected_parameters.get("extra_body") or {}).get(
                    "enable_thinking"
                )
                is False
                and
                aggregate["material_negative"] == 0
                and aggregate["sequences_with_content_byte_reconstruction"]
                == aggregate["candidates"]
                and aggregate["candidates"] > 0
                and aggregate["sequences_with_chosen_mapping"]
                == aggregate["candidates"]
                and aggregate["sequences_with_top_mapping"]
                == aggregate["candidates"]
                and aggregate["sequences_with_exact_top5"]
                == aggregate["candidates"]
                and chosen_coverage == 1.0
                and top_coverage == 1.0
            ),
            "dense_full_vocabulary_kl": False,
            "dense_kl_claim_forbidden": True,
            "content_logprob_prefix_fully_visible_to_student": bool(
                objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
                and (expected_parameters.get("extra_body") or {}).get(
                    "enable_thinking"
                )
                is False
            ),
        },
        "production_readiness": {
            "mc_sequence_forward_kl_nll": sequence_contract_ready,
            "sparse_top5_plus_tail": bool(
                final_code_ready
                and top5_contract_ready
                and aggregate["sequences_with_content_byte_reconstruction"]
                == aggregate["candidates"]
                and aggregate["candidates"] > 0
                and aggregate["sequences_with_chosen_mapping"]
                == aggregate["candidates"]
                and aggregate["sequences_with_top_mapping"]
                == aggregate["candidates"]
                and aggregate["sequences_with_exact_top5"]
                == aggregate["candidates"]
            ),
            "correctness_conditioned_rs_sft": bool(
                final_code_ready
            ),
            "dense_full_vocabulary_kl": False,
        },
        "production_ready": sequence_contract_ready,
        "production_failures": [
            reason
            for condition, reason in (
                (bool(incomplete), "one or more tasks does not have exactly K=8"),
                (
                    bool(invalid_sample_indices),
                    "one or more tasks does not have exact sample indices 0..7",
                ),
                (
                    bool(invalid_requested_seed_counts),
                    "one or more tasks does not have eight distinct requested seeds",
                ),
                (not all_verified, "one or more candidates lacks verification"),
                (
                    len(groups) > 1 and not allow_homogeneous_shards,
                    "returned model/backend identity changed",
                ),
                (
                    objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
                    and aggregate["material_negative"] > 0,
                    "materially negative top-5 tail mass observed",
                ),
                (
                    objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
                    and
                    aggregate["sequences_with_content_byte_reconstruction"]
                    != aggregate["candidates"],
                    "chosen provider-token bytes do not reconstruct raw content",
                ),
                (
                    aggregate["candidates"] == 0,
                    "teacher journal contains no candidates",
                ),
                (
                    aggregate["sequence"] != aggregate["candidates"],
                    "one or more K=8 teacher draws lacks complete nonempty "
                    "final content",
                ),
                (
                    bool(overflow_diagnostics),
                    "one or more EOS-inclusive teacher targets exceeds the "
                    f"trainer max_target_tokens={max_target_tokens}; no "
                    "truncation, filtering, or resampling is permitted",
                ),
                (
                    objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
                    and bool(non_code_diagnostics),
                    "one or more teacher final-content draws is not exactly "
                    "final Dart code; reasoning/prose/fences cannot enter the "
                    "sequence target",
                ),
                (
                    aggregate["targets_reasoning_excluded"]
                    != aggregate["candidates"],
                    "one or more targets is not structurally bound only to "
                    "choice.message.content",
                ),
                (
                    pathological_all_tasks_identical,
                    "every task returned one identical final sequence across "
                    "all K=8 requests; stochastic sampling contract is suspect",
                ),
            )
            if condition
        ],
    }
    atomic_write_json(audit_output, audit)
    return audit
