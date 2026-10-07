#!/bin/bash
set -Eeuo pipefail

source /venv/main/bin/activate
cd /workspace/hybrid_training_patch_v2_3

OUTPUT=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2/direct_compact_multifunction_gold_sft_target24k
SOURCE=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2/direct_compact_multifunction_gold_sft
CONTRACT=/workspace/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json
CODEBOOK=/workspace/multifunction_v1/build/multifunction_inline_cfg_v2_codebook.json
CODEC=/workspace/scripts/data/build_multifunction_compact_v2.py
TOKENIZER=/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json

if [[ -e "${OUTPUT}" ]]; then
  python -m scripts.training.direct_compact_qwen_decompiler \
    --output_dir "${OUTPUT}" \
    --contract "${CONTRACT}" \
    --codebook "${CODEBOOK}" \
    --codec_artifact "${CODEC}" \
    --tokenizer_json "${TOKENIZER}" \
    --warmstart_checkpoint "${SOURCE}" \
    --validate_migrated_warmstart_only
else
  python -m scripts.training.direct_compact_qwen_decompiler \
    --output_dir "${OUTPUT}" \
    --contract "${CONTRACT}" \
    --codebook "${CODEBOOK}" \
    --codec_artifact "${CODEC}" \
    --tokenizer_json "${TOKENIZER}" \
    --warmstart_checkpoint "${SOURCE}" \
    --migrate_warmstart_only \
    --attn_implementation eager \
    --bf16
fi

printf 'CAPACITY_GOLD_MIGRATION_COMPLETE output=%s\n' "${OUTPUT}"
