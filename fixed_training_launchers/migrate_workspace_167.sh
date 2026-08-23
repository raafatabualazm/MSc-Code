#!/usr/bin/env bash
# Copy the exact resumable Qwen3.8/fit2776 state off the rented GPU host.
set -Eeuo pipefail

destination_host="root@167.172.150.125"
destination_root="/workspace/experiment_workspace"
transfer_key="/root/.ssh/codex_transfer_167_20260724"
ssh_transport=(
  ssh
  -i "${transfer_key}"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)

test -f "${transfer_key}"
"${ssh_transport[@]}" "${destination_host}" \
  "mkdir -p '${destination_root}' '${destination_root}/deployment_snapshot'"

sources=(
  /workspace/./multifunction_v1
  /workspace/./hybrid_training_patch_v2_3
  /workspace/./scripts
  /workspace/./artifacts/direct_compact_qwen38_inline_cfg_v2
  /workspace/./artifacts/direct_compact_qwen38_supplement1196
  /workspace/./artifacts/direct_compact_qwen38_union2776
  /workspace/./logs
  /workspace/./Qwen.env
  /workspace/./OpenAI.env
  /workspace/./data.env
  /workspace/./requirements.txt
  /workspace/./dart_3.12.2-1_amd64.deb
  /workspace/./FIXED_RS_SFT_VERPO_RUNBOOK.md
  /workspace/./FULL_HANDOFF.md
  /workspace/./RESULTS_SUMMARY_20260723.md
)

while IFS= read -r launcher; do
  sources+=("/workspace/./${launcher#/workspace/}")
done < <(find /workspace -maxdepth 1 -type f -name 'run_*.sh' -print | sort)

rsync_transport="$(
  printf '%q ' "${ssh_transport[@]}"
)"
rsync -aHz --compress-level=3 --relative --partial --append-verify \
  --info=progress2,stats2 \
  -e "${rsync_transport% }" \
  "${sources[@]}" \
  "${destination_host}:${destination_root}/"

rsync -aH --relative \
  -e "${rsync_transport% }" \
  /opt/supervisor-scripts/./qwen38_kd.sh \
  /opt/supervisor-scripts/./fit2776_supplement_pipeline.sh \
  /opt/supervisor-scripts/./fit2776_parent_harvest_gate.sh \
  /opt/supervisor-scripts/./fit2776_union_post_pipeline.sh \
  "${destination_host}:${destination_root}/deployment_snapshot/"

rsync -aH --relative \
  -e "${rsync_transport% }" \
  /etc/supervisor/conf.d/./qwen38_kd.conf \
  /etc/supervisor/conf.d/./fit2776_supplement_pipeline.conf \
  /etc/supervisor/conf.d/./fit2776_parent_harvest_gate.conf \
  /etc/supervisor/conf.d/./fit2776_union_post_pipeline.conf \
  "${destination_host}:${destination_root}/deployment_snapshot/"

# A checksum pass catches any source-side change or interrupted/corrupt copy.
rsync -aHcz --compress-level=3 --relative --partial --info=stats2 \
  -e "${rsync_transport% }" \
  "${sources[@]}" \
  "${destination_host}:${destination_root}/"

"${ssh_transport[@]}" "${destination_host}" \
  "chmod 600 '${destination_root}/Qwen.env' \
    '${destination_root}/OpenAI.env' '${destination_root}/data.env'"

printf 'MIGRATION_167_COMPLETE destination=%s:%s\n' \
  "${destination_host}" "${destination_root}"
