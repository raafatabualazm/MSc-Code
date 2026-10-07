#!/usr/bin/env python3
"""Resume legacy mixed RS-SFT checkpoints without changing their sealed trainer.

The original trainer placed ``WarmstartIdentity.exact_lora_targets`` into the
in-memory run contract as a tuple. JSON checkpoints necessarily store that
field as a list, and the legacy resume path compared the two Python objects
directly. This narrowly scoped launcher JSON-normalizes only that dataclass
before invoking the otherwise byte-identical, hash-bound trainer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.training import t5gemma2_mixed_rs_sft as trainer

_CHECKPOINT_RE = re.compile(r"checkpoint-optstep-([0-9]{6,})\Z")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flag_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"compatibility launcher requires exactly one {flag}")
    return argv[positions[0] + 1]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _assert_sealed_resume(argv: Sequence[str]) -> tuple[Path, str]:
    output_dir = Path(_flag_value(argv, "--output_dir")).expanduser().resolve()
    checkpoint = Path(
        _flag_value(argv, "--resume_checkpoint")
    ).expanduser().resolve()
    root_contract = _read_object(output_dir / "run_contract.json")
    checkpoint_contract = _read_object(checkpoint / "run_contract.json")
    root_sha = _canonical_sha256(root_contract)
    if root_sha != _canonical_sha256(checkpoint_contract):
        raise ValueError("compatibility resume contracts differ")

    expected_trainer_sha = str(
        root_contract.get("runtime", {}).get("trainer_sha256", "")
    )
    observed_trainer_sha = hashlib.sha256(
        Path(trainer.__file__).resolve().read_bytes()
    ).hexdigest()
    if not expected_trainer_sha or observed_trainer_sha != expected_trainer_sha:
        raise ValueError("compatibility resume trainer hash differs")

    latest = _read_object(output_dir / "latest_checkpoint.json")
    recorded = Path(str(latest.get("path", ""))).expanduser().resolve()
    if (
        recorded != checkpoint
        or str(latest.get("run_contract_sha256", "")) != root_sha
    ):
        raise ValueError("compatibility resume pointer binding differs")
    return checkpoint, root_sha


def _json_compatible_asdict(value: Any) -> dict[str, Any]:
    result = dataclasses.asdict(value)
    if isinstance(value, trainer.WarmstartIdentity):
        targets = result.get("exact_lora_targets")
        if not isinstance(targets, tuple):
            raise TypeError("legacy warm-start LoRA targets are not a tuple")
        result["exact_lora_targets"] = list(targets)
    return result


def _prune_superseded_checkpoints(
    output_dir: Path,
    latest_checkpoint: Path,
    *,
    retain: int = 2,
) -> list[str]:
    output_dir = output_dir.expanduser().resolve()
    latest_checkpoint = latest_checkpoint.expanduser().resolve()
    if retain < 2 or latest_checkpoint.parent != output_dir:
        raise ValueError("unsafe mixed-checkpoint retention request")
    pointer = _read_object(output_dir / "latest_checkpoint.json")
    recorded = Path(str(pointer.get("path", ""))).expanduser().resolve()
    if recorded != latest_checkpoint:
        raise ValueError("checkpoint retention pointer differs")

    checkpoints: list[tuple[int, Path]] = []
    for candidate in output_dir.iterdir():
        match = _CHECKPOINT_RE.fullmatch(candidate.name)
        if match is None:
            continue
        resolved = candidate.resolve()
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or resolved.parent != output_dir
            or resolved.name != candidate.name
        ):
            raise ValueError(f"unsafe checkpoint path: {candidate}")
        checkpoints.append((int(match.group(1)), resolved))
    checkpoints.sort()
    if not checkpoints or checkpoints[-1][1] != latest_checkpoint:
        raise ValueError("latest checkpoint is not the newest sealed checkpoint")

    removed: list[str] = []
    for _update, candidate in checkpoints[:-retain]:
        if candidate == latest_checkpoint or candidate.parent != output_dir:
            raise ValueError("refusing unsafe checkpoint deletion")
        shutil.rmtree(candidate)
        removed.append(candidate.name)
    return removed


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    checkpoint, contract_sha = _assert_sealed_resume(arguments)
    output_dir = Path(_flag_value(arguments, "--output_dir")).expanduser().resolve()
    trainer.asdict = _json_compatible_asdict
    original_save_checkpoint = trainer._save_checkpoint

    def save_checkpoint_with_retention(**kwargs: Any) -> Path:
        destination = original_save_checkpoint(**kwargs)
        removed_after_save = _prune_superseded_checkpoints(
            Path(kwargs["output_dir"]),
            destination,
        )
        if removed_after_save:
            print(
                json.dumps(
                    {
                        "event": "mixed_rs_sft_checkpoint_retention",
                        "latest_checkpoint": str(destination),
                        "removed": removed_after_save,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return destination

    trainer._save_checkpoint = save_checkpoint_with_retention
    removed_at_start = _prune_superseded_checkpoints(output_dir, checkpoint)
    wrapper_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "event": "mixed_rs_sft_resume_compat",
                "checkpoint": str(checkpoint),
                "removed_superseded_checkpoints": removed_at_start,
                "run_contract_sha256": contract_sha,
                "wrapper_sha256": wrapper_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return trainer.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
