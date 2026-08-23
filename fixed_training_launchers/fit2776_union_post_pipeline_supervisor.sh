#!/bin/bash
# Wait for both sealed Qwen parent harvests, train the expanded student, then
# execute the predeclared GPT RS-SFT -> DeepSeek VeRPO chain.
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

source /venv/main/bin/activate
mkdir -p /workspace/logs
cd /workspace
trap 'status=$?; if (( status != 0 )); then printf "FIT2776_UNION_POST_RETRYABLE_EXIT status=%s\n" "${status}" >&2; sleep 60; fi' EXIT

patch=/workspace/hybrid_training_patch_v2_3
export PYTHONPATH="${patch}:/workspace"
export PYTHONUNBUFFERED=1
expansion=/workspace/multifunction_v1/expanded2776
parent_root=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2
supplement_root=/workspace/artifacts/direct_compact_qwen38_supplement1196
union_root=/workspace/artifacts/direct_compact_qwen38_union2776

while [[ ! -f "${expansion}/build/expansion_build.seal.json" \
      || ! -f "${expansion}/qwen_2776_supplement.derivation.json" ]]; do
  printf 'FIT2776_UNION_WAIT representation_or_derivation=true\n'
  sleep 60
done

# Build the train eligibility view once, then re-seal it under the already
# approved target24k capacity-only contract. These paths never enter Qwen's
# teacher journals and can overlap the API harvest.
base_executable="${expansion}/executable"
target_executable="${expansion}/executable_target24k"
if [[ ! -f "${base_executable}/executable_view.build.json" ]]; then
  expansion_report="${expansion}/build/build_report.json"
  /venv/main/bin/python \
    "${patch}/scripts/preprocessing/build_multifunction_executable_view.py" \
    --parent-build-report "${expansion_report}" \
    --expected-parent-build-report-sha256 \
      "$(sha256sum "${expansion_report}" | awk '{print $1}')" \
    --output-dir "${base_executable}" 2>&1 \
    | tee -a /workspace/logs/fit2776_union_post_pipeline.log
fi
test -f "${base_executable}/executable_view.build.json" || {
  printf 'FIT2776_EXECUTABLE_BUILD_MISSING path=%s\n' \
    "${base_executable}/executable_view.build.json" >&2
  exit 2
}
rich_target_seal="${expansion}/fit2776_expansion.target24k.seal.json"
rich_target_receipt="${expansion}/fit2776_expansion.target24k.rebind.json"
/venv/main/bin/python \
  "${patch}/scripts/preprocessing/rebind_multifunction_parent_capacity.py" \
  --source-rich-seal \
    "${expansion}/build/train_multifunction_binary_expanded_2776.seal.json" \
  --source-dataset \
    "${expansion}/build/train_multifunction_binary_expanded_2776.jsonl" \
  --source-contract \
    /workspace/multifunction_v1/build/multifunction_inline_cfg_v2_contract.json \
  --target-dataset "${expansion}/fit2776_multifunction_binary.jsonl" \
  --target-contract \
    /workspace/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json \
  --generic-target-seal \
    "${expansion}/fit2776_multifunction_binary.target24k.seal.json" \
  --output-seal "${rich_target_seal}" \
  --output-receipt "${rich_target_receipt}" 2>&1 \
  | tee -a /workspace/logs/fit2776_union_post_pipeline.log
if [[ ! -f "${target_executable}/executable_view.build.json" ]]; then
  base_report="${base_executable}/executable_view.build.json"
  /venv/main/bin/python \
    "${patch}/scripts/preprocessing/migrate_multifunction_executable_capacity.py" \
    --source-dir "${base_executable}" \
    --expected-source-report-sha256 \
      "$(sha256sum "${base_report}" | awk '{print $1}')" \
    --target-contract \
      /workspace/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json \
    --target-parent-train-seal "${rich_target_seal}" \
    --output-dir "${target_executable}" 2>&1 \
    | tee -a /workspace/logs/fit2776_union_post_pipeline.log
fi
test -f "${target_executable}/executable_view.build.json" || {
  printf 'FIT2776_TARGET24K_EXECUTABLE_MISSING path=%s\n' \
    "${target_executable}/executable_view.build.json" >&2
  exit 2
}

while [[ ! -f "${parent_root}/qwen_mc_sequence_train.build.json" \
      || ! -f "${supplement_root}/qwen_mc_sequence_train.build.json" \
      || ! -f "${supplement_root}/qwen_cot_sft_train.build.json" ]]; do
  parent_status="$(supervisorctl status qwen38_kd 2>/dev/null || true)"
  supplement_status="$(
    supervisorctl status fit2776_supplement_pipeline 2>/dev/null || true
  )"
  if [[ "${supplement_status}" == *"EXITED"* \
     || "${supplement_status}" == *"STOPPED"* ]]; then
    printf 'FIT2776_UNION_RESTART supplement_pipeline_status=%s\n' \
      "${supplement_status}"
    supervisorctl start fit2776_supplement_pipeline
    supplement_status="$(
      supervisorctl status fit2776_supplement_pipeline 2>/dev/null || true
    )"
  elif [[ "${supplement_status}" == *"FATAL"* ]]; then
    printf 'FIT2776_UNION_FATAL supplement_pipeline_status=%s\n' \
      "${supplement_status}" >&2
    exit 2
  fi
  printf 'FIT2776_UNION_WAIT parent=%s supplement=%s\n' \
    "${parent_status:-unknown}" "${supplement_status:-unknown}"
  sleep 60
done

while supervisorctl status qwen38_kd 2>/dev/null \
  | grep -Eq ' (RUNNING|STARTING) '; do
  printf 'FIT2776_UNION_WAIT parent_gate_stopping_legacy_gpu_stage=true\n'
  sleep 10
done

bash /workspace/run_qwen38_union_2776.sh 2>&1 \
  | tee -a /workspace/logs/fit2776_union_post_pipeline.log
bash /workspace/run_qwen38_train_union_2776.sh 2>&1 \
  | tee -a /workspace/logs/fit2776_union_post_pipeline.log

executable_report="${target_executable}/executable_view.build.json"
export EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="$(
  sha256sum "${executable_report}" | awk '{print $1}'
)"
export EXPECTED_PARENT_FIT_ROWS=2776
export MULTIFUNCTION_EXECUTABLE_ROOT="${target_executable}"
export MULTIFUNCTION_BUILD="${expansion}"
export STUDENT_CHECKPOINT="${union_root}/direct_compact_qwen_cot_sft"
export QWEN_ROOT="${union_root}"
export QWEN_BUILD_MANIFEST="${union_root}/qwen_mc_sequence_train.build.json"

bash /workspace/run_collect_chatgpt_compact_rs.sh 2>&1 \
  | tee -a /workspace/logs/fit2776_union_post_pipeline.log
bash /workspace/run_rs_sft_then_verpo.sh 2>&1 \
  | tee -a /workspace/logs/fit2776_union_post_pipeline.log

printf 'FIT2776_FULL_CHAIN_COMPLETE fit=2776 heldout=175 evaluation=%s\n' \
  /workspace/artifacts/direct_compact_verpo_union2776_target24k/heldout175_evaluation/evaluation_suite.json
