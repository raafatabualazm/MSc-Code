#!/usr/bin/env python3
"""Install the all-length hierarchical hybrid-training v2.3 overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import time
from pathlib import Path

FILES = (
    "configs/verpo_rescue_launch.example.json",
    "models/direct_compact_causal.py",
    "models/graph_data_collator.py",
    "scripts/training/direct_compact_qwen_decompiler.py",
    "scripts/training/direct_compact_verpo.py",
    "scripts/training/direct_compact_verpo_rescue.py",
    "scripts/training/verpo_rescue_grounding.py",
    "scripts/training/build_verpo_rescue_transfer.py",
    "scripts/training/launch_direct_compact_verpo_rescue.py",
    "scripts/training/graph_positions.py",
    "scripts/training/checkpoint_contract.py",
    "scripts/training/hybrid_data_controls.py",
    "scripts/training/prepare_hybrid_training_data_antigravity.py",
    "scripts/training/build_hierarchical_long_training_antigravity.py",
    "scripts/training/build_balanced_sft_mix_antigravity.py",
    "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
    "scripts/training/graph_grpo_decompiler_antigravity.py",
    "scripts/training/verpo_judge_antigravity.py",
    "scripts/training/build_verpo_repair_dataset_antigravity.py",
    "scripts/training/teacher_repair_dataset_antigravity.py",
    "scripts/training/run_hybrid_curriculum_antigravity.py",
    "scripts/evaluation/audit_grpo_reward_antigravity.py",
    "scripts/evaluation/durable_evaluation_journal.py",
    "scripts/evaluation/direct_compact_qwen_inference.py",
    "scripts/evaluation/graph_compile_at_k_antigravity.py",
    "scripts/evaluation/prepare_neutral_evaluation_antigravity.py",
    "scripts/evaluation/probe_graph_representations_antigravity.py",
    "scripts/evaluation/functional_graph_gate_antigravity.py",
    "scripts/evaluation/report_token_lengths_antigravity.py",
    "scripts/evaluation/graph_inference_antigravity.py",
    "scripts/evaluation/run_sweeps_antigravity.py",
)

# These are project-owned dependencies, not replacements supplied by the patch.
PREREQUISITES = (
    "../frontier_ceiling_patch_v1/frontier_f2.py",
    "models/graphcodebert_tensor_builder.py",
    "models/hierarchical_graph_encoder_antigravity.py",
    "models/pyg_cfg_dataset.py",
    "scripts/data/cfg_extractor.py",
    "scripts/data/dfg_extractor.py",
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
    manifest_path = source_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    for relative in FILES:
        source = source_root / relative
        record = records.get(relative)
        if record is None:
            raise ValueError(f"patch manifest does not cover installer file: {relative}")
        if int(record.get("size_bytes", -1)) != source.stat().st_size:
            raise ValueError(f"patch manifest size mismatch: {relative}")
        if str(record.get("sha256", "")).lower() != digest(source):
            raise ValueError(f"patch manifest SHA-256 mismatch: {relative}")
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
