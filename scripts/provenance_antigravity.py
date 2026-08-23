"""Run-level provenance helpers for Antigravity training and inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_commit(config: Any) -> str | None:
    value = getattr(config, "_commit_hash", None)
    return str(value) if value else None


def git_state(root: str | Path) -> dict[str, Any]:
    root = Path(root)

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain") if commit else None
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def graph_environment() -> dict[str, str]:
    # Do not redact ordinary configuration names such as
    # GRAPH_QWEN_PREFIX_TOKENS. Only credential-bearing variables are secret.
    sensitive = {
        "GRAPH_HF_TOKEN",
        "GRAPH_HUGGINGFACE_TOKEN",
        "GRAPH_API_KEY",
        "GRAPH_ACCESS_TOKEN",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if (key.startswith("GRAPH_") or key == "EVAL_PASS_STABILITY_RUNS")
        and key not in sensitive
    }


def file_record(path: str | Path, *, required: bool = True) -> dict[str, Any] | None:
    path = Path(path).resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def runtime_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        record.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - provenance must not hide the run
        record["torch_error"] = repr(exc)
    return record
