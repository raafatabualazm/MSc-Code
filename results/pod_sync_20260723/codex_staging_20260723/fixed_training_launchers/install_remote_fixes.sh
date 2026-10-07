#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s STAGING_DIRECTORY\n' "$0" >&2
  exit 2
fi

STAGING_ROOT="$(realpath "$1")"
HYBRID_SOURCE="${STAGING_ROOT}/hybrid_training_patch_v2_3"
KD_SOURCE="${STAGING_ROOT}/true_kd_patch_v1"
LAUNCHER_SOURCE="${STAGING_ROOT}/fixed_training_launchers"

for required_path in \
  "${HYBRID_SOURCE}/MANIFEST.json" \
  "${KD_SOURCE}/scripts/training/true_distribution_kd_antigravity.py" \
  "${LAUNCHER_SOURCE}/run_finish_rs_sft.sh" \
  "${LAUNCHER_SOURCE}/run_verpo_v2.sh"; do
  if [[ ! -f "${required_path}" ]]; then
    printf 'Incomplete staging tree; missing %s\n' "${required_path}" >&2
    exit 2
  fi
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="/workspace/backups/codex_rs_verpo_kd_${STAMP}"
PATCH_ARCHIVE="/workspace/codex_patches/hybrid_training_patch_v2_3_fixed_20260723"

LIVE_FILES=(
  scripts/training/run_hybrid_curriculum_antigravity.py
  scripts/training/graph_grpo_decompiler_antigravity.py
  scripts/training/verpo_judge_antigravity.py
  scripts/training/build_verpo_repair_dataset_antigravity.py
  scripts/training/teacher_repair_dataset_antigravity.py
  scripts/training/hybrid_data_controls.py
  scripts/evaluation/functional_graph_gate_antigravity.py
)

mkdir -p "${BACKUP_ROOT}/live" /workspace/backups /workspace/codex_patches
for relative in "${LIVE_FILES[@]}"; do
  target="/workspace/${relative}"
  if [[ -f "${target}" ]]; then
    mkdir -p "${BACKUP_ROOT}/live/$(dirname "${relative}")"
    cp -a "${target}" "${BACKUP_ROOT}/live/${relative}"
  fi
done

for existing in \
  /workspace/run_finish_rs_sft.sh \
  /workspace/run_verpo_v1.sh \
  /workspace/run_verpo_v2.sh \
  /workspace/run_rs_sft_then_verpo.sh \
  /workspace/run_true_kd.sh \
  /workspace/run_dense_full_kd.sh \
  /workspace/soft_kd_trainer.py \
  /workspace/build_softkd_data.py; do
  if [[ -f "${existing}" ]]; then
    cp -a "${existing}" "${BACKUP_ROOT}/$(basename "${existing}")"
  fi
done
if [[ -d /workspace/true_kd_patch_v1 ]]; then
  cp -a /workspace/true_kd_patch_v1 "${BACKUP_ROOT}/true_kd_patch_v1"
fi
if [[ -d "${PATCH_ARCHIVE}" ]]; then
  cp -a "${PATCH_ARCHIVE}" "${BACKUP_ROOT}/hybrid_patch_archive"
fi

for relative in "${LIVE_FILES[@]}"; do
  source="${HYBRID_SOURCE}/${relative}"
  target="/workspace/${relative}"
  if [[ ! -f "${source}" ]]; then
    printf 'Hybrid patch is missing live file %s\n' "${relative}" >&2
    exit 2
  fi
  install -D -m 0644 "${source}" "${target}"
done

rm -rf "${PATCH_ARCHIVE}.incoming"
mkdir -p "${PATCH_ARCHIVE}.incoming"
cp -a "${HYBRID_SOURCE}/." "${PATCH_ARCHIVE}.incoming/"
if [[ -d "${PATCH_ARCHIVE}" ]]; then
  mv "${PATCH_ARCHIVE}" "${BACKUP_ROOT}/hybrid_patch_archive_replaced"
fi
mv "${PATCH_ARCHIVE}.incoming" "${PATCH_ARCHIVE}"

rm -rf /workspace/true_kd_patch_v1.incoming
mkdir -p /workspace/true_kd_patch_v1.incoming
cp -a "${KD_SOURCE}/." /workspace/true_kd_patch_v1.incoming/
if [[ -d /workspace/true_kd_patch_v1 ]]; then
  mv /workspace/true_kd_patch_v1 "${BACKUP_ROOT}/true_kd_patch_v1_replaced"
fi
mv /workspace/true_kd_patch_v1.incoming /workspace/true_kd_patch_v1

install -m 0755 \
  "${LAUNCHER_SOURCE}/run_finish_rs_sft.sh" \
  /workspace/run_finish_rs_sft.sh
install -m 0755 \
  "${LAUNCHER_SOURCE}/run_verpo_v2.sh" \
  /workspace/run_verpo_v2.sh
install -m 0755 \
  "${LAUNCHER_SOURCE}/run_verpo_v2.sh" \
  /workspace/run_verpo_v1.sh
install -m 0755 \
  "${LAUNCHER_SOURCE}/run_rs_sft_then_verpo.sh" \
  /workspace/run_rs_sft_then_verpo.sh
install -m 0755 \
  "${LAUNCHER_SOURCE}/run_true_kd.sh" \
  /workspace/run_true_kd.sh
install -m 0755 \
  "${LAUNCHER_SOURCE}/run_dense_full_kd.sh" \
  /workspace/run_dense_full_kd.sh
install -m 0755 \
  "${LAUNCHER_SOURCE}/soft_kd_trainer.py" \
  /workspace/soft_kd_trainer.py
install -m 0755 \
  "${LAUNCHER_SOURCE}/build_softkd_data.py" \
  /workspace/build_softkd_data.py
install -m 0644 \
  "${LAUNCHER_SOURCE}/README.md" \
  /workspace/FIXED_RS_SFT_VERPO_RUNBOOK.md

chmod 0755 \
  /workspace/true_kd_patch_v1/run_true_kd.sh \
  /workspace/true_kd_patch_v1/run_dense_full_kd.sh

{
  printf 'installed_at_utc=%s\n' "${STAMP}"
  printf 'backup_root=%s\n' "${BACKUP_ROOT}"
  printf 'hybrid_patch_archive=%s\n' "${PATCH_ARCHIVE}"
  sha256sum \
    /workspace/scripts/training/run_hybrid_curriculum_antigravity.py \
    /workspace/scripts/training/graph_grpo_decompiler_antigravity.py \
    /workspace/scripts/training/verpo_judge_antigravity.py \
    /workspace/scripts/training/build_verpo_repair_dataset_antigravity.py \
    /workspace/scripts/training/teacher_repair_dataset_antigravity.py \
    /workspace/scripts/training/hybrid_data_controls.py \
    /workspace/scripts/evaluation/functional_graph_gate_antigravity.py \
    /workspace/true_kd_patch_v1/scripts/training/true_distribution_kd_antigravity.py
} > /workspace/codex_rs_verpo_kd_install_receipt.txt

printf 'Installed fixes. Backup: %s\n' "${BACKUP_ROOT}"
