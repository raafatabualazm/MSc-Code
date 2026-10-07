#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
WARMSTART="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_compiler_verpo_smoke_2epoch_v3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -d "${WARMSTART}" ]] || [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_VERPO_SMOKE_BLOCKED warm-start or Dart runtime is absent" >&2
  exit 78
fi
printf '%s  %s\n' \
  562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f \
  "${WARMSTART}/run_contract.json" \
  83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc \
  "${WARMSTART}/adapter/adapter_model.safetensors" \
  c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3 \
  "${WARMSTART}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d \
  "${WARMSTART}/tokenizer/tokenizer.json" \
  11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  "${FEEDBACK_DIR}/verpo_feedback_view.public.json" \
  | sha256sum -c -

/venv/main/bin/python - "${WARMSTART}/run_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
contract = json.loads(path.read_text(encoding="utf-8"))
canonical = hashlib.sha256(
    json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
if (
    canonical
    != "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3"
    or contract.get("schema") != "t5gemma2-enriched-sft-run-v1"
    or contract.get("architecture") != "native_encoder_decoder"
    or contract.get("dataset", {}).get("rows") != 2776
    or contract.get("lora", {}).get("encoder_and_decoder_trainable") is not True
    or len(contract.get("lora", {}).get("targets") or []) != 476
):
    raise SystemExit("T5GEMMA_VERPO_SMOKE_BLOCKED warm-start contract differs")
PY

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
if [[ -s "${OUTPUT_DIR}/result.json" ]] \
  && /usr/bin/jq -e \
    '.schema == "t5gemma2-compiler-feedback-verpo-run-v1"
     and .status == "complete"
     and .updates == 1
     and .latest_checkpoint == "checkpoint-optstep-000001"' \
    "${OUTPUT_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_VERPO_SMOKE_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

resume_args=()
if [[ -s "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  resume_checkpoint="$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")"
  if [[ -z "${resume_checkpoint}" || ! -d "${resume_checkpoint}" ]]; then
    echo "T5GEMMA_VERPO_SMOKE_BLOCKED invalid checkpoint pointer" >&2
    exit 78
  fi
  resume_args=(--resume_checkpoint "${resume_checkpoint}")
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_compiler_feedback_verpo.py \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --rollout_seal "${FEEDBACK_DIR}/verpo_rollout_feedback.seal.json" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --f2_manifest "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl.manifest.json" \
  --feedback_public_manifest "${FEEDBACK_DIR}/verpo_feedback_view.public.json" \
  --expected_feedback_public_manifest_sha256 11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  --compact_contract "${WORKSPACE}/multifunction_v1/expanded2776/executable_target24k/compact_contract.json" \
  --warmstart_checkpoint "${WARMSTART}" \
  --output_dir "${OUTPUT_DIR}" \
  --group_size 4 \
  --repair_group_size 4 \
  --max_repair_parents 2 \
  --tasks_per_update 1 \
  --max_updates 1 \
  --temperature 0.8 \
  --max_new_tokens 4096 \
  --max_source_tokens 32768 \
  --max_target_tokens 32768 \
  --verpo_alpha 2.0 \
  --local_weight 1.0 \
  --compile_weight 0.25 \
  --learning_rate 1e-6 \
  --ppo_clip 0.0 \
  --sft_replay_weight 0.02 \
  --on_policy_logprob_tolerance 2e-4 \
  --reward_workers 4 \
  --reward_timeout 30 \
  --reward_stability_runs 1 \
  --checkpoint_interval 1 \
  --keep_last_checkpoints 2 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16 \
  "${resume_args[@]}"
