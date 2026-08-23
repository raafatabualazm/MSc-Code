#!/usr/bin/env python3
"""Install the fail-closed hybrid-training v2.1 overlay into an Antigravity checkout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import time
from pathlib import Path

FILES = (
    "scripts/training/graph_positions.py",
    "scripts/training/checkpoint_contract.py",
    "scripts/training/hybrid_data_controls.py",
    "scripts/training/prepare_hybrid_training_data_antigravity.py",
    "scripts/training/build_balanced_sft_mix_antigravity.py",
    "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
    "scripts/training/graph_grpo_decompiler_antigravity.py",
    "scripts/training/teacher_repair_dataset_antigravity.py",
    "scripts/training/run_hybrid_curriculum_antigravity.py",
    "scripts/evaluation/audit_grpo_reward_antigravity.py",
    "scripts/evaluation/prepare_neutral_evaluation_antigravity.py",
    "scripts/evaluation/probe_graph_representations_antigravity.py",
    "scripts/evaluation/functional_graph_gate_antigravity.py",
    "scripts/evaluation/graph_inference_antigravity.py",
    "scripts/evaluation/run_sweeps_antigravity.py",
)

# These are project-owned dependencies, not replacements supplied by the patch.
PREREQUISITES = (
    "models/graphcodebert_tensor_builder.py",
    "models/graph_data_collator.py",
    "models/hierarchical_graph_encoder_antigravity.py",
    "models/pyg_cfg_dataset.py",
    "scripts/data/cfg_extractor.py",
    "scripts/data/dfg_extractor.py",
    "scripts/evaluation/graph_compile_at_k_antigravity.py",
    "scripts/provenance_antigravity.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_patch_source(source_root: Path) -> None:
    missing = [relative for relative in FILES if not (source_root / relative).is_file()]
    if missing:
        raise FileNotFoundError("patch archive is incomplete: " + ", ".join(missing))
    for relative in FILES:
        source = source_root / relative
        if source.suffix == ".py":
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-prerequisite-check",
        action="store_true",
        help="Allow installation into a partial checkout; runtime imports may still fail.",
    )
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parent
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.is_dir():
        parser.error(f"project root does not exist: {project_root}")
    validate_patch_source(source_root)

    if not args.no_prerequisite_check:
        missing = [relative for relative in PREREQUISITES if not (project_root / relative).is_file()]
        if missing:
            parser.error(
                "project checkout is missing required Antigravity files:\n  "
                + "\n  ".join(missing)
                + "\nUse --no-prerequisite-check only for a deliberate partial/staging install."
            )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_root = project_root / ".hybrid_patch_backups" / stamp
    changed = 0
    for relative in FILES:
        source = source_root / relative
        target = project_root / relative
        source_digest = digest(source)
        if target.is_file() and digest(target) == source_digest:
            print(f"unchanged {relative}")
            continue
        print(f"install   {relative}")
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.no_backup:
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        if digest(target) != source_digest:
            raise RuntimeError(f"post-copy digest mismatch for {relative}")
        changed += 1

    if args.dry_run:
        print(f"dry run only; {len(FILES)} patch files validated, no files changed")
    else:
        print(f"installed {changed} changed file(s); verified SHA-256 after every copy")
        if backup_root.exists():
            print(f"backups: {backup_root}")


if __name__ == "__main__":
    main()
