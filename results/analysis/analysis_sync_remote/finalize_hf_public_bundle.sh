#!/usr/bin/env bash
set -euo pipefail

python3 /tmp/prepare_hf_public_bundle.py

for destination in \
    /workspace/hf_public_t5gemma2_verpo_artifacts_v1 \
    /workspace/hf_public_t5gemma2_verpo_latest_v1
do
    cp /tmp/HF_MODEL_CARD.md "$destination/README.md"
    cp /tmp/HF_NOTICE.txt "$destination/NOTICE"
    curl -fsSL --retry 3 https://ai.google.dev/gemma/terms \
        -o "$destination/GEMMA_TERMS.html"
    curl -fsSL --retry 3 https://ai.google.dev/gemma/prohibited_use_policy \
        -o "$destination/GEMMA_PROHIBITED_USE_POLICY.html"
done

printf 'LATEST_FILES %s\n' "$(find /workspace/hf_public_t5gemma2_verpo_latest_v1 -type f | wc -l)"
printf 'ALL_FILES %s\n' "$(find /workspace/hf_public_t5gemma2_verpo_artifacts_v1 -type f | wc -l)"
df -h /workspace | tail -1
