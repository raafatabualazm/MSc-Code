#!/usr/bin/env bash
# Create a quiescent, checksum-manifested workspace snapshot for local archival.
#
# Credential .env files and model-checkpoint payloads are deliberately excluded
# from the ordinary archive. Their remote paths are recorded in the snapshot
# metadata without reading or copying their contents.
set -euo pipefail

snapshot_id="${1:-20260727-final}"
source_root="/workspace"
meta_root="/root/workspace_snapshot_${snapshot_id}_meta"
archive="/root/workspace_snapshot_${snapshot_id}.tar.zst"
ready="${archive}.ready"

if [[ ! "${snapshot_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'Invalid snapshot id: %s\n' "${snapshot_id}" >&2
  exit 2
fi
if [[ -e "${meta_root}" || -e "${archive}" || -e "${ready}" ]]; then
  printf 'Snapshot output already exists; refusing to overwrite: %s\n' \
    "${snapshot_id}" >&2
  exit 2
fi

writer_pattern='deepseek64_continuation_v1|qwen37_capacity_fallback_v6|qwen37_capacity_length_scheduler_v7|qwen37_capacity_adopted_outcome_monitor_v1|qwen37_capacity_length_repair_v7'
if pgrep -af "${writer_pattern}" >/tmp/workspace_snapshot_writers.txt; then
  printf 'Project writers are still active; refusing an inconsistent snapshot:\n' >&2
  cat /tmp/workspace_snapshot_writers.txt >&2
  exit 3
fi

secret_paths=(
  "workspace/Anthropic.env"
  "workspace/OpenAI.env"
  "workspace/Qwen.env"
  "workspace/Qwen_fallback.env"
  "workspace/Qwen_fallback_exhausted_20260726.env"
  "workspace/data.env"
  "workspace/experiment_workspace/OpenAI.env"
  "workspace/experiment_workspace/Qwen.env"
  "workspace/experiment_workspace/data.env"
)

checkpoint_name_patterns=(
  "*.safetensors"
  "*.ckpt"
  "pytorch_model*.bin"
  "adapter_model*.bin"
  "optimizer.pt"
  "scheduler.pt"
  "rng_state*.pth"
  "source_embedding_overlay.pt"
  "training_args.bin"
)

mkdir -p "${meta_root}"
printf '%s\n' "${secret_paths[@]}" >"${meta_root}/excluded_secret_paths.txt"
{
  printf 'snapshot_id=%s\n' "${snapshot_id}"
  printf 'hostname=%s\n' "$(hostname -f)"
  printf 'source_root=%s\n' "${source_root}"
  printf 'started_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'archive_compression=zstd-level-3-multithreaded\n'
  printf 'secret_policy=credential-env-files-excluded-names-recorded-only\n'
  printf 'checkpoint_policy=model-weight-and-optimizer-payloads-excluded-paths-recorded-only\n'
} >"${meta_root}/snapshot_metadata.txt"

cd /
find_args=(workspace -xdev -type f)
for secret in "${secret_paths[@]}"; do
  find_args+=(! -path "${secret}")
done
for pattern in "${checkpoint_name_patterns[@]}"; do
  find_args+=(! -name "${pattern}")
done

checkpoint_expr=('(')
first_pattern=1
for pattern in "${checkpoint_name_patterns[@]}"; do
  if (( first_pattern == 0 )); then
    checkpoint_expr+=(-o)
  fi
  checkpoint_expr+=(-name "${pattern}")
  first_pattern=0
done
checkpoint_expr+=(')')

find workspace -xdev -type f "${checkpoint_expr[@]}" -printf '%p\n' \
  | LC_ALL=C sort \
  >"${meta_root}/excluded_model_checkpoint_paths.txt"
find workspace -xdev -type f "${checkpoint_expr[@]}" -printf '%s\t%p\n' \
  | LC_ALL=C sort -t $'\t' -k2,2 \
  >"${meta_root}/excluded_model_checkpoint_files.tsv"

find "${find_args[@]}" -printf '%p\0' \
  | LC_ALL=C sort -z \
  | xargs -0 -r sha256sum \
  >"${meta_root}/workspace_files.sha256"
find workspace -xdev -type l -printf '%p -> %l\n' \
  | LC_ALL=C sort \
  >"${meta_root}/workspace_symlinks.txt"

file_count="$(wc -l <"${meta_root}/workspace_files.sha256")"
symlink_count="$(wc -l <"${meta_root}/workspace_symlinks.txt")"
checkpoint_file_count="$(
  wc -l <"${meta_root}/excluded_model_checkpoint_paths.txt"
)"
checkpoint_apparent_bytes="$(
  awk -F $'\t' '{total += $1} END {printf "%.0f", total}' \
    "${meta_root}/excluded_model_checkpoint_files.tsv"
)"
apparent_bytes="$(
  find "${find_args[@]}" -printf '%s\n' \
    | awk '{total += $1} END {printf "%.0f", total}'
)"
{
  printf 'regular_file_count=%s\n' "${file_count}"
  printf 'symlink_count=%s\n' "${symlink_count}"
  printf 'regular_file_apparent_bytes=%s\n' "${apparent_bytes}"
  printf 'excluded_model_checkpoint_file_count=%s\n' \
    "${checkpoint_file_count}"
  printf 'excluded_model_checkpoint_apparent_bytes=%s\n' \
    "${checkpoint_apparent_bytes}"
  printf 'manifest_completed_at=%s\n' "$(date -u +%FT%TZ)"
} >>"${meta_root}/snapshot_metadata.txt"

tar_args=(
  --numeric-owner
  --acls
  --xattrs
  --sparse
)
for secret in "${secret_paths[@]}"; do
  tar_args+=(--exclude="${secret}")
done
tar_args+=(
  "--exclude-from=${meta_root}/excluded_model_checkpoint_paths.txt"
)

tar "${tar_args[@]}" -I 'zstd -T0 -3' -cf "${archive}" \
  -C / workspace \
  -C /root "$(basename "${meta_root}")"

archive_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
archive_bytes="$(stat -c %s "${archive}")"
{
  printf 'snapshot_id=%s\n' "${snapshot_id}"
  printf 'archive=%s\n' "${archive}"
  printf 'archive_bytes=%s\n' "${archive_bytes}"
  printf 'archive_sha256=%s\n' "${archive_sha256}"
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
} >"${ready}"

printf 'WORKSPACE_SNAPSHOT_READY id=%s bytes=%s sha256=%s archive=%s\n' \
  "${snapshot_id}" "${archive_bytes}" "${archive_sha256}" "${archive}"
