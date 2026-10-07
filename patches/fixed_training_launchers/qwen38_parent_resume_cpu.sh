#!/usr/bin/env bash
# Resume the sealed 1,580-task Qwen3.8 teacher journal on an API/CPU host.
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PATCH_ROOT="${PATCH_ROOT:-${WORKSPACE}/hybrid_training_patch_v2_3}"
PYTHON="${PYTHON:-${WORKSPACE}/.venv/bin/python}"
QWEN_ENV_FILE="${QWEN_ENV_FILE:-${WORKSPACE}/Qwen.env}"
TOKENIZER_JSON="${WORKSPACE}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
OUTPUT_ROOT="${WORKSPACE}/artifacts/direct_compact_qwen38_inline_cfg_v2"

load_qwen_credentials() {
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ ! "${line}" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
      printf 'Qwen.env contains a non-KEY=VALUE line\n' >&2
      return 2
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "${key}" in
      API_KEY|QWEN_API_KEY|DASHSCOPE_ENDPOINT|QWEN_BASE_URL|QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED)
        printf -v "${key}" '%s' "${value}"
        export "${key}"
        ;;
      *)
        # Auxiliary Alibaba endpoint credentials are intentionally ignored by
        # this Qwen3.8 Token Plan resume.
        ;;
    esac
  done < "${QWEN_ENV_FILE}"
  export QWEN_API_KEY="${QWEN_API_KEY:-${API_KEY:-}}"
  [[ -n "${QWEN_API_KEY}" ]] || {
    printf 'QWEN_API_KEY/API_KEY is missing\n' >&2
    return 2
  }
}

load_qwen_credentials
export PATH="/usr/lib/dart/bin:${PATH}"
export PYTHONPATH="${PATCH_ROOT}:${WORKSPACE}"
export PYTHONUNBUFFERED=1

cd "${PATCH_ROOT}"
exec "${PYTHON}" -m scripts.training.collect_qwen_direct_compact_teacher \
  --token-plan-automation-authorized \
  --authorize-orphan-reissue-with-duplicate-billing-risk \
  --prompt-jsonl "${WORKSPACE}/multifunction_v1/build/train_multifunction_binary_f2.jsonl" \
  --expected-prompt-sha256 f86032533f7f3e1bade9280e5bf6e28858daf25e34167837fec160a34b2370ba \
  --expected-prompt-rows 1580 \
  --prompt-manifest "${WORKSPACE}/multifunction_v1/build/train_multifunction_binary_f2.jsonl.manifest.json" \
  --expected-prompt-manifest-sha256 231227e9aa6f31c793ff097e8167972975e9f94871c9ea682d6a0dc996ed4af8 \
  --verifier-jsonl "${WORKSPACE}/multifunction_v1/build/train_multifunction_binary.jsonl" \
  --expected-verifier-sha256 3cf3dd5b0fc950f7516434b2c90ce2bd6334178ace962e4442026bd33f476e18 \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
  --student-eos-token-id 151645 \
  --target-contract "${WORKSPACE}/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json" \
  --expected-target-contract-sha256 f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f \
  --model qwen3.8-max-preview \
  --objective-mode sequence_only \
  --enable-thinking \
  --required-function fn0 \
  --base-url https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1 \
  --api-key-env QWEN_API_KEY \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 101 \
  --max-tokens 12288 \
  --length-max-token-escalation 16384 24576 \
  --thinking-budget 8192 \
  --seed-base 44 \
  --max-prompt-tokens 12000 \
  --chat-overhead-reserve 256 \
  --timeout-seconds 600 \
  --verifier-timeout-seconds 45 \
  --max-retries 8 \
  --workers 32 \
  --verifier-workers 16 \
  --progress-every 50 \
  --journal "${OUTPUT_ROOT}/qwen_teacher.journal.jsonl" \
  --parseable-output "${OUTPUT_ROOT}/qwen_teacher.parseable.mc_sequence.jsonl" \
  --rs-sft-output "${OUTPUT_ROOT}/qwen_teacher.verified_only.rs_sft.jsonl" \
  --audit-output "${OUTPUT_ROOT}/qwen_teacher.audit.json" \
  --quality-gate-json "${OUTPUT_ROOT}/quality_pilot/qwen_teacher.pilot.quality_gate.json" \
  --expected-quality-gate-sha256 4a730f46c610f0b333cb334fdbd4ffe896c3ee4e1f14e1ffa54b42a5eadfc30c \
  "$@"
