#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="${T5GEMMA_HOLDBACK_AUDIT_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_HOLDBACK_AUDIT_PYTHON:-/venv/main/bin/python}"
FEEDBACK_ROOT="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
HARVEST_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1"
PROXY_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_proxy_reward_audit_v1"
OUTPUT_ROOT="${T5GEMMA_HOLDBACK_AUDIT_OUTPUT:-${WORKSPACE}/artifacts/t5gemma2_typed_holdback_alignment_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

blocked() { echo "T5GEMMA_TYPED_HOLDBACK_ALIGNMENT_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "CPU audit runtime is absent"

printf '%s  %s\n' \
  19a006e54117538f96750804d83f5fc4c74fab2dcec2137d1c770d311ead7f18 "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_holdback_alignment.py" \
  ecf779a68011910083af222c860cd43bc23bc675a623613511a07b2e7b823746 "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_proxy_reward_surface.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  232880791b108df96b4f01bc44a613c595cf4edaa738f6cb9a624412da5e50e4 "${PROJECT}/scripts/training/t5gemma2_compiler_feedback_verpo.py" \
  c4c72410333669f78d109d8848c70a79321ef42dba6e1a8344b138e8bfdbdb51 "${PROJECT}/scripts/training/seq2seq_verpo_core.py" \
  5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  6decd1ed1ecd3ce8e8a0bd6d861c30a26063c9d913957e361584413705f28a3b "${PROJECT}/scripts/preprocessing/build_verpo_feedback_view.py" \
  ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc "${HARVEST_ROOT}/harvest.journal.jsonl" \
  a8b41e0855cb73874c05ec0f57ca29c43449756b92a7cf7ccc034c4351d22a57 "${HARVEST_ROOT}/harvest.journal.jsonl.chain-head.json" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c "${FEEDBACK_ROOT}/verpo_rollout_feedback.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f "${FEEDBACK_ROOT}/reward_holdback.private.jsonl" \
  1a73ee3df03d1fda97d819e8536ab42435dfa1cbc802335987e21ed48cd196e2 "${FEEDBACK_ROOT}/verpo_feedback_view.build.json" \
  b63250b1db1ca53fdf033cd3935824b4a96a76c37ef4f1f390dabd72370be1f4 "${PROXY_ROOT}/reward_audit.journal.jsonl" \
  76b4bcc98ef7f16fd57a76d0501a7c91617c9ac80d1506b2beeb8763a2ab8172 "${PROXY_ROOT}/reward_audit.journal.jsonl.chain-head.json" \
  b0d73acb0391adea3844afa6f36589f4035e5fa4e73751f25836b318f43d9435 "${PROXY_ROOT}/reward_audit.summary.json" \
  | sha256sum -c - || blocked "sealed code or evidence differs"

mkdir -p "${OUTPUT_ROOT}"
for entry in "${OUTPUT_ROOT}"/* "${OUTPUT_ROOT}"/.[!.]* "${OUTPUT_ROOT}"/..?*; do
  [[ -e "${entry}" ]] || continue
  case "$(basename "${entry}")" in
    holdback_alignment.private.journal.jsonl|holdback_alignment.private.journal.jsonl.chain-head.json|holdback_alignment.summary.json) ;;
    *) blocked "foreign output exists: ${entry}" ;;
  esac
done

export PYTHONPATH="${PROJECT}"
export CUDA_VISIBLE_DEVICES=-1
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
exec nice -n 10 "${PYTHON_BIN}" \
  scripts/evaluation/audit_t5gemma2_typed_holdback_alignment.py \
  --harvest_journal "${HARVEST_ROOT}/harvest.journal.jsonl" \
  --feedback_jsonl "${FEEDBACK_ROOT}/verpo_rollout_feedback.jsonl" \
  --proxy_journal "${PROXY_ROOT}/reward_audit.journal.jsonl" \
  --proxy_summary "${PROXY_ROOT}/reward_audit.summary.json" \
  --holdback_jsonl "${FEEDBACK_ROOT}/reward_holdback.private.jsonl" \
  --feedback_build_report "${FEEDBACK_ROOT}/verpo_feedback_view.build.json" \
  --output_journal "${OUTPUT_ROOT}/holdback_alignment.private.journal.jsonl" \
  --output_summary "${OUTPUT_ROOT}/holdback_alignment.summary.json" \
  --dart_bin "${DART_BIN}" \
  --workers 8
