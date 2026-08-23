#!/usr/bin/env bash
# Atomically install the canonical Qwen -> OpenAI RS-SFT -> VeRPO runtime.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s STAGING_DIRECTORY\n' "$0" >&2
  exit 2
fi

STAGING_ROOT="$(realpath "$1")"
HYBRID_SOURCE="${STAGING_ROOT}/hybrid_training_patch_v2_3"
KD_SOURCE="${STAGING_ROOT}/true_kd_patch_v1"
LAUNCHER_SOURCE="${STAGING_ROOT}/fixed_training_launchers"
SERIALIZER_SOURCE="${STAGING_ROOT}/serialize_compact_inputs.py"
DATA_SOURCE="${STAGING_ROOT}/scripts/data"
FRONTIER_F2_SOURCE="${STAGING_ROOT}/frontier_ceiling_patch_v1/frontier_f2.py"
RECOVERED_WARMSTART_RECEIPT_SOURCE="${LAUNCHER_SOURCE}/direct_compact_fn0_self_seal_recovery.json"
PINNED_FRONTIER_F2_SHA256=097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce
RECOVERED_WARMSTART_RECEIPT_SHA256=41e0dd7ecf68ebb0b560c66266d686b49afe9e59c6390f1e4079854dca6a7c9b

REQUIRED=(
  "${HYBRID_SOURCE}/MANIFEST.json"
  "${HYBRID_SOURCE}/scripts/training/collect_qwen_direct_compact_teacher.py"
  "${HYBRID_SOURCE}/scripts/training/qwen_direct_compact_teacher_artifact.py"
  "${HYBRID_SOURCE}/scripts/training/probe_qwen_teacher_contract.py"
  "${HYBRID_SOURCE}/scripts/training/build_qwen_sequence_kd.py"
  "${HYBRID_SOURCE}/scripts/training/build_qwen_sparse_topk_tail_auxiliary.py"
  "${HYBRID_SOURCE}/scripts/training/direct_compact_sparse_topk_tail.py"
  "${HYBRID_SOURCE}/scripts/training/direct_compact_qwen_decompiler.py"
  "${HYBRID_SOURCE}/scripts/preprocessing/build_multifunction_executable_view.py"
  "${HYBRID_SOURCE}/scripts/preprocessing/build_multifunction_binary_compact.py"
  "${HYBRID_SOURCE}/scripts/preprocessing/build_verpo_feedback_view.py"
  "${HYBRID_SOURCE}/scripts/evaluation/prepare_direct_compact_eval.py"
  "${HYBRID_SOURCE}/scripts/evaluation/durable_evaluation_journal.py"
  "${HYBRID_SOURCE}/scripts/evaluation/direct_compact_qwen_inference.py"
  "${HYBRID_SOURCE}/scripts/evaluation/score_direct_compact_passk.py"
  "${HYBRID_SOURCE}/scripts/evaluation/graph_compile_at_k_antigravity.py"
  "${HYBRID_SOURCE}/scripts/evaluation/audit_qwen_direct_compact_teacher.py"
  "${HYBRID_SOURCE}/scripts/evaluation/validate_direct_compact_training_stage.py"
  "${HYBRID_SOURCE}/scripts/evaluation/seal_post_qwen_evaluation_suite.py"
  "${HYBRID_SOURCE}/scripts/run_qwen_sequence_kd_warmstart.sh"
  "${HYBRID_SOURCE}/scripts/run_qwen_sparse_topk_tail_warmstart.sh"
  "${HYBRID_SOURCE}/scripts/training/collect_chatgpt_compact_rs.py"
  "${HYBRID_SOURCE}/scripts/training/build_direct_compact_rs_sft.py"
  "${HYBRID_SOURCE}/scripts/training/direct_compact_verpo.py"
  "${HYBRID_SOURCE}/scripts/training/verpo_judge_antigravity.py"
  "${HYBRID_SOURCE}/scripts/training/seal_post_qwen_chain.py"
  "${KD_SOURCE}/scripts/training/true_distribution_kd_antigravity.py"
  "${LAUNCHER_SOURCE}/run_qwen38_sequence_kd.sh"
  "${LAUNCHER_SOURCE}/run_collect_chatgpt_compact_rs.sh"
  "${LAUNCHER_SOURCE}/run_finish_rs_sft.sh"
  "${LAUNCHER_SOURCE}/run_verpo_v2.sh"
  "${LAUNCHER_SOURCE}/run_rs_sft_then_verpo.sh"
  "${LAUNCHER_SOURCE}/qwen38_kd_supervisor.sh"
  "${LAUNCHER_SOURCE}/qwen38_kd.supervisor.conf"
  "${LAUNCHER_SOURCE}/qwen38_gold_only_supervisor.sh"
  "${LAUNCHER_SOURCE}/qwen38_gold_only.supervisor.conf"
  "${LAUNCHER_SOURCE}/post_qwen_pipeline_supervisor.sh"
  "${LAUNCHER_SOURCE}/post_qwen_pipeline.supervisor.conf"
  "${LAUNCHER_SOURCE}/multifunction_extract_v2_supervisor.sh"
  "${LAUNCHER_SOURCE}/multifunction_extract_v2.conf"
  "${LAUNCHER_SOURCE}/multifunction_constants_v2_supervisor.sh"
  "${LAUNCHER_SOURCE}/multifunction_constants_v2.conf"
  "${LAUNCHER_SOURCE}/multifunction_build_v2_supervisor.sh"
  "${LAUNCHER_SOURCE}/multifunction_build_v2.conf"
  "${LAUNCHER_SOURCE}/multifunction_executable_view_supervisor.sh"
  "${LAUNCHER_SOURCE}/multifunction_executable_view.supervisor.conf"
  "${DATA_SOURCE}/build_multifunction_compact_v2.py"
  "${DATA_SOURCE}/build_dart_user_symbol_attestation.py"
  "${DATA_SOURCE}/extract_dart_aot_user_function_bundle.py"
  "${DATA_SOURCE}/extract_attested_binary_pool_constants.py"
  "${DATA_SOURCE}/gdb_dump_attested_pool_offsets.py"
  "${FRONTIER_F2_SOURCE}"
  "${RECOVERED_WARMSTART_RECEIPT_SOURCE}"
  "${SERIALIZER_SOURCE}"
)
for required_path in "${REQUIRED[@]}"; do
  if [[ ! -f "${required_path}" ]]; then
    printf 'Incomplete staging tree; missing %s\n' "${required_path}" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "${FRONTIER_F2_SOURCE}" | awk '{print $1}')" \
  != "${PINNED_FRONTIER_F2_SHA256}" ]]; then
  printf 'Pinned frontier_f2.py hash mismatch in staging\n' >&2
  exit 2
fi
if [[ "$(sha256sum "${RECOVERED_WARMSTART_RECEIPT_SOURCE}" | awk '{print $1}')" \
  != "${RECOVERED_WARMSTART_RECEIPT_SHA256}" ]]; then
  printf 'Recovered warm-start receipt hash mismatch in staging\n' >&2
  exit 2
fi
PYTHONPATH="${HYBRID_SOURCE}" /venv/main/bin/python - \
  /workspace/artifacts/direct_compact_fn0_real_sft_v1_self_sealed_recovered \
  "${RECOVERED_WARMSTART_RECEIPT_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

from models.direct_compact_causal import sha256_artifact, sha256_file
from scripts.training.direct_compact_qwen_decompiler import (
    validate_self_sealed_checkpoint,
)

checkpoint = Path(sys.argv[1]).resolve()
receipt_path = Path(sys.argv[2]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("schema") != "direct-compact-self-seal-recovery-v1"
    or Path(str(receipt.get("recovered_checkpoint") or "")).resolve()
    != checkpoint
    or (receipt.get("invariants") or {}).get(
        "validate_self_sealed_checkpoint_passed"
    )
    is not True
):
    raise SystemExit("recovered warm-start receipt identity is invalid")
paths = validate_self_sealed_checkpoint(checkpoint)
artifacts = receipt.get("artifacts") or {}
observed = {
    "decoder_adapter_sha256": sha256_artifact(paths["adapter"]),
    "source_embedding_overlay_sha256": sha256_file(paths["overlay"]),
    "run_provenance_sha256": sha256_file(paths["provenance"]),
}
if any(artifacts.get(key) != value for key, value in observed.items()):
    raise SystemExit("recovered warm-start differs from its sealed receipt")
PY

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="/workspace/backups/codex_training_runtime_${STAMP}"
HYBRID_TARGET="/workspace/hybrid_training_patch_v2_3"
KD_TARGET="/workspace/true_kd_patch_v1"
HYBRID_INCOMING="/workspace/.hybrid_training_patch_v2_3.incoming.${STAMP}.$$"
KD_INCOMING="/workspace/.true_kd_patch_v1.incoming.${STAMP}.$$"

if [[ -e "${HYBRID_INCOMING}" || -e "${KD_INCOMING}" ]]; then
  printf 'Refusing to reuse an incoming installation path\n' >&2
  exit 2
fi

mkdir -p "${BACKUP_ROOT}" /workspace/backups
mkdir -p "${HYBRID_INCOMING}" "${KD_INCOMING}"
cp -a "${HYBRID_SOURCE}/." "${HYBRID_INCOMING}/"
cp -a "${KD_SOURCE}/." "${KD_INCOMING}/"

BACKUP_FILES=(
  /workspace/run_qwen38_sequence_kd.sh
  /workspace/run_collect_chatgpt_compact_rs.sh
  /workspace/run_finish_rs_sft.sh
  /workspace/run_verpo_v2.sh
  /workspace/run_rs_sft_then_verpo.sh
  /workspace/run_verpo_graph_legacy.sh
  /workspace/run_finish_graph_rs_sft_legacy.sh
  /workspace/run_true_kd.sh
  /workspace/run_dense_full_kd.sh
  /workspace/soft_kd_trainer.py
  /workspace/build_softkd_data.py
  /workspace/serialize_compact_inputs.py
  /workspace/FIXED_RS_SFT_VERPO_RUNBOOK.md
  /opt/supervisor-scripts/qwen38_kd.sh
  /etc/supervisor/conf.d/qwen38_kd.conf
  /opt/supervisor-scripts/qwen38_gold_only.sh
  /etc/supervisor/conf.d/qwen38_gold_only.conf
  /opt/supervisor-scripts/post_qwen_pipeline.sh
  /etc/supervisor/conf.d/post_qwen_pipeline.conf
  /opt/supervisor-scripts/multifunction_extract_v2.sh
  /etc/supervisor/conf.d/multifunction_extract_v2.conf
  /opt/supervisor-scripts/multifunction_constants_v2.sh
  /etc/supervisor/conf.d/multifunction_constants_v2.conf
  /opt/supervisor-scripts/multifunction_build_v2.sh
  /etc/supervisor/conf.d/multifunction_build_v2.conf
  /opt/supervisor-scripts/multifunction_executable_view.sh
  /etc/supervisor/conf.d/multifunction_executable_view.conf
  /workspace/scripts/data/build_multifunction_compact_v2.py
  /workspace/scripts/data/build_dart_user_symbol_attestation.py
  /workspace/scripts/data/extract_dart_aot_user_function_bundle.py
  /workspace/scripts/data/extract_attested_binary_pool_constants.py
  /workspace/scripts/data/gdb_dump_attested_pool_offsets.py
  /workspace/frontier_ceiling_patch_v1/frontier_f2.py
  /workspace/direct_compact_fn0_self_seal_recovery.json
)
for existing in "${BACKUP_FILES[@]}"; do
  if [[ -f "${existing}" ]]; then
    relative="${existing#/}"
    mkdir -p "${BACKUP_ROOT}/files/$(dirname "${relative}")"
    cp -a "${existing}" "${BACKUP_ROOT}/files/${relative}"
  fi
done
if [[ -L /workspace/frontier_ceiling_patch_v1 \
   || -L /workspace/frontier_ceiling_patch_v1/frontier_f2.py \
   || ( -e /workspace/frontier_ceiling_patch_v1 \
        && ! -d /workspace/frontier_ceiling_patch_v1 ) ]]; then
  printf 'Refusing unsafe frontier package/file target shape\n' >&2
  exit 2
fi
mkdir -p /workspace/frontier_ceiling_patch_v1
# Install only the hash-pinned F2 module. Never replace or prune the unrelated
# frontier package around it.
install -m 0644 \
  "${FRONTIER_F2_SOURCE}" \
  /workspace/frontier_ceiling_patch_v1/frontier_f2.py
install -m 0644 \
  "${RECOVERED_WARMSTART_RECEIPT_SOURCE}" \
  /workspace/direct_compact_fn0_self_seal_recovery.json

if [[ -e "${HYBRID_TARGET}" ]]; then
  mv "${HYBRID_TARGET}" "${BACKUP_ROOT}/hybrid_training_patch_v2_3"
fi
mv "${HYBRID_INCOMING}" "${HYBRID_TARGET}"

if [[ -e "${KD_TARGET}" ]]; then
  mv "${KD_TARGET}" "${BACKUP_ROOT}/true_kd_patch_v1"
fi
mv "${KD_INCOMING}" "${KD_TARGET}"

install -m 0755 "${SERIALIZER_SOURCE}" /workspace/serialize_compact_inputs.py
mkdir -p /workspace/scripts/data
for data_script in \
  build_multifunction_compact_v2.py \
  build_dart_user_symbol_attestation.py \
  extract_dart_aot_user_function_bundle.py \
  extract_attested_binary_pool_constants.py \
  gdb_dump_attested_pool_offsets.py
do
  install -m 0755 \
    "${DATA_SOURCE}/${data_script}" \
    "/workspace/scripts/data/${data_script}"
done

LAUNCHERS=(
  run_qwen38_sequence_kd.sh
  run_collect_chatgpt_compact_rs.sh
  run_finish_rs_sft.sh
  run_verpo_v2.sh
  run_rs_sft_then_verpo.sh
  run_verpo_graph_legacy.sh
  run_finish_graph_rs_sft_legacy.sh
  run_true_kd.sh
  run_dense_full_kd.sh
  soft_kd_trainer.py
  build_softkd_data.py
)
for launcher in "${LAUNCHERS[@]}"; do
  install -m 0755 "${LAUNCHER_SOURCE}/${launcher}" "/workspace/${launcher}"
done
install -m 0644 \
  "${LAUNCHER_SOURCE}/README.md" \
  /workspace/FIXED_RS_SFT_VERPO_RUNBOOK.md

mkdir -p /opt/supervisor-scripts /etc/supervisor/conf.d
install -m 0755 \
  "${LAUNCHER_SOURCE}/qwen38_kd_supervisor.sh" \
  /opt/supervisor-scripts/qwen38_kd.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/qwen38_kd.supervisor.conf" \
  /etc/supervisor/conf.d/qwen38_kd.conf
install -m 0755 \
  "${LAUNCHER_SOURCE}/qwen38_gold_only_supervisor.sh" \
  /opt/supervisor-scripts/qwen38_gold_only.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/qwen38_gold_only.supervisor.conf" \
  /etc/supervisor/conf.d/qwen38_gold_only.conf
install -m 0755 \
  "${LAUNCHER_SOURCE}/post_qwen_pipeline_supervisor.sh" \
  /opt/supervisor-scripts/post_qwen_pipeline.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/post_qwen_pipeline.supervisor.conf" \
  /etc/supervisor/conf.d/post_qwen_pipeline.conf
install -m 0755 \
  "${LAUNCHER_SOURCE}/multifunction_extract_v2_supervisor.sh" \
  /opt/supervisor-scripts/multifunction_extract_v2.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/multifunction_extract_v2.conf" \
  /etc/supervisor/conf.d/multifunction_extract_v2.conf
install -m 0755 \
  "${LAUNCHER_SOURCE}/multifunction_constants_v2_supervisor.sh" \
  /opt/supervisor-scripts/multifunction_constants_v2.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/multifunction_constants_v2.conf" \
  /etc/supervisor/conf.d/multifunction_constants_v2.conf
install -m 0755 \
  "${LAUNCHER_SOURCE}/multifunction_build_v2_supervisor.sh" \
  /opt/supervisor-scripts/multifunction_build_v2.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/multifunction_build_v2.conf" \
  /etc/supervisor/conf.d/multifunction_build_v2.conf
install -m 0755 \
  "${LAUNCHER_SOURCE}/multifunction_executable_view_supervisor.sh" \
  /opt/supervisor-scripts/multifunction_executable_view.sh
install -m 0644 \
  "${LAUNCHER_SOURCE}/multifunction_executable_view.supervisor.conf" \
  /etc/supervisor/conf.d/multifunction_executable_view.conf

chmod 0755 \
  "${KD_TARGET}/run_true_kd.sh" \
  "${KD_TARGET}/run_dense_full_kd.sh"

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl reread
  supervisorctl update
fi

RECEIPT="/workspace/codex_training_runtime_install_receipt.txt"
{
  printf 'installed_at_utc=%s\n' "${STAMP}"
  printf 'backup_root=%s\n' "${BACKUP_ROOT}"
  printf 'hybrid_target=%s\n' "${HYBRID_TARGET}"
  printf 'true_kd_target=%s\n' "${KD_TARGET}"
  printf 'recovered_warmstart=%s\n' \
    /workspace/artifacts/direct_compact_fn0_real_sft_v1_self_sealed_recovered
  printf 'recovered_warmstart_source_modified=false\n'
  printf 'frontier_package_replaced=false\n'
  sha256sum \
    "${HYBRID_TARGET}/scripts/training/collect_qwen_direct_compact_teacher.py" \
    "${HYBRID_TARGET}/scripts/training/probe_qwen_teacher_contract.py" \
    "${HYBRID_TARGET}/scripts/training/qwen_direct_compact_teacher_artifact.py" \
    "${HYBRID_TARGET}/scripts/training/build_qwen_sequence_kd.py" \
    "${HYBRID_TARGET}/scripts/training/build_qwen_sparse_topk_tail_auxiliary.py" \
    "${HYBRID_TARGET}/scripts/training/direct_compact_sparse_topk_tail.py" \
    "${HYBRID_TARGET}/scripts/training/direct_compact_qwen_decompiler.py" \
    "${HYBRID_TARGET}/scripts/preprocessing/build_multifunction_executable_view.py" \
    "${HYBRID_TARGET}/scripts/preprocessing/build_multifunction_binary_compact.py" \
    "${HYBRID_TARGET}/scripts/preprocessing/build_verpo_feedback_view.py" \
    "${HYBRID_TARGET}/scripts/evaluation/prepare_direct_compact_eval.py" \
    "${HYBRID_TARGET}/scripts/evaluation/durable_evaluation_journal.py" \
    "${HYBRID_TARGET}/scripts/evaluation/direct_compact_qwen_inference.py" \
    "${HYBRID_TARGET}/scripts/evaluation/score_direct_compact_passk.py" \
    "${HYBRID_TARGET}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
    "${HYBRID_TARGET}/scripts/evaluation/audit_qwen_direct_compact_teacher.py" \
    "${HYBRID_TARGET}/scripts/evaluation/validate_direct_compact_training_stage.py" \
    "${HYBRID_TARGET}/scripts/evaluation/seal_post_qwen_evaluation_suite.py" \
    "${HYBRID_TARGET}/scripts/run_qwen_sequence_kd_warmstart.sh" \
    "${HYBRID_TARGET}/scripts/run_qwen_sparse_topk_tail_warmstart.sh" \
    "${HYBRID_TARGET}/scripts/training/collect_chatgpt_compact_rs.py" \
    "${HYBRID_TARGET}/scripts/training/build_direct_compact_rs_sft.py" \
    "${HYBRID_TARGET}/scripts/training/direct_compact_verpo.py" \
    "${HYBRID_TARGET}/scripts/training/verpo_judge_antigravity.py" \
    "${HYBRID_TARGET}/scripts/training/seal_post_qwen_chain.py" \
    /workspace/serialize_compact_inputs.py \
    /workspace/run_qwen38_sequence_kd.sh \
    /opt/supervisor-scripts/qwen38_kd.sh \
    /etc/supervisor/conf.d/qwen38_kd.conf \
    /opt/supervisor-scripts/qwen38_gold_only.sh \
    /etc/supervisor/conf.d/qwen38_gold_only.conf \
    /opt/supervisor-scripts/post_qwen_pipeline.sh \
    /etc/supervisor/conf.d/post_qwen_pipeline.conf \
    /opt/supervisor-scripts/multifunction_extract_v2.sh \
    /etc/supervisor/conf.d/multifunction_extract_v2.conf \
    /opt/supervisor-scripts/multifunction_constants_v2.sh \
    /etc/supervisor/conf.d/multifunction_constants_v2.conf \
    /opt/supervisor-scripts/multifunction_build_v2.sh \
    /etc/supervisor/conf.d/multifunction_build_v2.conf \
    /opt/supervisor-scripts/multifunction_executable_view.sh \
    /etc/supervisor/conf.d/multifunction_executable_view.conf \
    /workspace/scripts/data/build_multifunction_compact_v2.py \
    /workspace/scripts/data/build_dart_user_symbol_attestation.py \
    /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
    /workspace/scripts/data/extract_attested_binary_pool_constants.py \
    /workspace/scripts/data/gdb_dump_attested_pool_offsets.py \
    /workspace/frontier_ceiling_patch_v1/frontier_f2.py \
    /workspace/direct_compact_fn0_self_seal_recovery.json \
    /workspace/run_collect_chatgpt_compact_rs.sh \
    /workspace/run_finish_rs_sft.sh \
    /workspace/run_verpo_v2.sh \
    /workspace/run_rs_sft_then_verpo.sh
} > "${RECEIPT}"

printf 'Installed canonical training runtime. Backup: %s\n' "${BACKUP_ROOT}"
printf 'Receipt: %s\n' "${RECEIPT}"
