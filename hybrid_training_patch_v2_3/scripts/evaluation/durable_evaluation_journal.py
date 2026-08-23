"""Append-only, hash-chained journals for exact-resume evaluation work."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

HEAD_SCHEMA = "durable-evaluation-journal-head-v1"
GENESIS_SHA256 = "0" * 64
CHAIN_FIELDS = frozenset(
    {
        "journal_event_index",
        "journal_previous_event_sha256",
        "journal_event_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write(Path(path), canonical_bytes(value) + b"\n")


def append_event(path: str | Path, event: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    head_path = Path(str(destination) + ".chain-head.json")
    if any(field in event for field in CHAIN_FIELDS):
        raise ValueError("caller may not supply journal-chain metadata")
    if destination.exists():
        rows = load_journal(destination)
        event_index = len(rows)
        previous = (
            rows[-1]["journal_event_sha256"] if rows else GENESIS_SHA256
        )
    else:
        if head_path.exists():
            raise ValueError("orphan evaluation journal chain head exists")
        event_index = 0
        previous = GENESIS_SHA256
    chained = {
        **dict(event),
        "journal_event_index": event_index,
        "journal_previous_event_sha256": previous,
    }
    chained["journal_event_sha256"] = canonical_sha256(chained)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_bytes(chained).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_write_json(
        head_path,
        {
            "schema": HEAD_SCHEMA,
            "event_count": event_index + 1,
            "head_event_sha256": chained["journal_event_sha256"],
            "journal_size_bytes": destination.stat().st_size,
        },
    )
    return chained


def load_journal(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    head_path = Path(str(source) + ".chain-head.json")
    if not source.exists():
        if head_path.exists():
            raise ValueError("orphan evaluation journal chain head exists")
        return []
    if not head_path.is_file():
        raise ValueError("evaluation journal has no durable chain head")
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"{source}:{line_number}: blank journal event"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source}:{line_number}: event is not an object"
                )
            rows.append(value)
    previous = GENESIS_SHA256
    for index, row in enumerate(rows):
        observed = str(row.get("journal_event_sha256") or "")
        unhashed = {
            key: value
            for key, value in row.items()
            if key != "journal_event_sha256"
        }
        if (
            row.get("journal_event_index") != index
            or row.get("journal_previous_event_sha256") != previous
            or not re.fullmatch(r"[0-9a-f]{64}", observed)
            or canonical_sha256(unhashed) != observed
        ):
            raise ValueError(
                f"evaluation journal hash chain breaks at event {index}"
            )
        previous = observed
    head = json.loads(head_path.read_text(encoding="utf-8"))
    if (
        not isinstance(head, dict)
        or head.get("schema") != HEAD_SCHEMA
        or head.get("event_count") != len(rows)
        or head.get("head_event_sha256") != previous
        or head.get("journal_size_bytes") != source.stat().st_size
    ):
        raise ValueError("evaluation journal chain head is inconsistent")
    return rows


def journal_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    rows = load_journal(source)
    head_path = Path(str(source) + ".chain-head.json")
    head = json.loads(head_path.read_text(encoding="utf-8"))
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "chain_head_path": str(head_path),
        "chain_head_sha256": sha256_file(head_path),
        "event_count": len(rows),
        "head_event_sha256": head["head_event_sha256"],
    }


def require_exact_or_write(path: str | Path, value: Any) -> None:
    """Publish a deterministic JSON value, never replacing different bytes."""
    destination = Path(path)
    payload = canonical_bytes(value) + b"\n"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"existing artifact differs: {destination}")
        return
    _atomic_write(destination, payload)


def require_unique_slots(slots: Sequence[str]) -> None:
    if not slots or len(slots) != len(set(slots)) or any(not slot for slot in slots):
        raise ValueError("evaluation slot identities must be unique and nonempty")
