#!/usr/bin/env bash
set -euo pipefail

# Wait for the live residual harvest to seal, late-bind only that harvest's
# report/journal digests, and then replace this process with final mixed
# T5Gemma 2 RS-SFT.  All earlier production inputs remain immutable constants.
RESIDUAL_SERVICE=t5gemma-base-harvest-residual-chain-2epoch
RESIDUAL_DIR=/workspace/artifacts/t5gemma2_local_base_residual_unresolved4_v1
RESIDUAL_REPORT="${RESIDUAL_DIR}/harvest_report.json"
RESIDUAL_JOURNAL="${RESIDUAL_DIR}/harvest.journal.jsonl"
MIXED_LAUNCHER=/opt/supervisor-scripts/t5gemma2_mixed_rs_sft.sh

while true; do
  # supervisorctl returns nonzero for a clean EXITED program. Capture its
  # output without letting `set -e` turn normal completion into chain failure.
  status_line="$(supervisorctl status "${RESIDUAL_SERVICE}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING)
      sleep 30
      ;;
    EXITED|STOPPED)
      break
      ;;
    *)
      echo "T5GEMMA_MIXED_HANDOFF_BLOCKED unexpected ${RESIDUAL_SERVICE} state=${state:-missing}" >&2
      exit 78
      ;;
  esac
done

if [[ ! -s "${RESIDUAL_REPORT}" || ! -s "${RESIDUAL_JOURNAL}" ]]; then
  echo "T5GEMMA_MIXED_HANDOFF_BLOCKED residual report or journal is absent" >&2
  exit 78
fi
if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-local-rs-sft-pilot-report-v1"
   and .status == "complete"
   and .pilot.tasks == 1500
   and .pilot.accepted_unique_targets >= 0
   and .privacy_invariants.frontier_api_calls == false
   and .privacy_invariants.heldout_175_opened == false
   and .privacy_invariants.private_holdback_text_in_model_input == false
   and .journal.path != null
   and .journal.sha256 != null' \
  "${RESIDUAL_REPORT}" >/dev/null; then
  echo "T5GEMMA_MIXED_HANDOFF_BLOCKED residual report contract differs" >&2
  exit 78
fi
if ! /usr/bin/jq -e \
  'select(.event == "header"
          and .schema == "t5gemma2-local-rs-sft-pilot-journal-v1"
          and .contract.schema == "t5gemma2-local-rs-sft-pilot-v1"
          and .contract.schedule.pilot_tasks == 1500
          and .contract.sampling.base_samples == 4
          and .contract.sampling.repair_enabled == false
          and .contract.sampling.repair_samples == 0
          and .contract.no_frontier_api == true
          and .contract.heldout_175_opened == false)' \
  <(head -n 1 "${RESIDUAL_JOURNAL}") >/dev/null; then
  echo "T5GEMMA_MIXED_HANDOFF_BLOCKED residual journal header contract differs" >&2
  exit 78
fi

residual_report_sha="$(sha256sum "${RESIDUAL_REPORT}" | awk '{print $1}')"
residual_journal_sha="$(sha256sum "${RESIDUAL_JOURNAL}" | awk '{print $1}')"
recorded_journal_sha="$(/usr/bin/jq -r '.journal.sha256 // empty' "${RESIDUAL_REPORT}")"
if [[ ! "${residual_report_sha}" =~ ^[0-9a-f]{64}$ \
  || ! "${residual_journal_sha}" =~ ^[0-9a-f]{64}$ \
  || "${recorded_journal_sha}" != "${residual_journal_sha}" ]]; then
  echo "T5GEMMA_MIXED_HANDOFF_BLOCKED residual report/journal hash binding failed" >&2
  exit 78
fi
if ! /usr/bin/jq -e \
  'select(.event == "complete"
          and .schema == "t5gemma2-local-rs-sft-pilot-journal-v1")' \
  <(tail -n 1 "${RESIDUAL_JOURNAL}") >/dev/null; then
  echo "T5GEMMA_MIXED_HANDOFF_BLOCKED residual journal lacks terminal seal" >&2
  exit 78
fi

# Known production reports and their journals are checked independently here;
# the mixed trainer subsequently validates each report's complete provenance
# graph and every referenced target/F2/repair artifact before model loading.
known_bindings=(
  "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab /workspace/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json"
  "5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58 /workspace/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest.journal.jsonl"
  "8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50 /workspace/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json"
  "80a326b6b2b2c8bdb0cd745f9884ace91baf411971023b1fed2d98192a022024 /workspace/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest.journal.jsonl"
  "883aeeab6d6a62d4bab41c62f77c8578866cf1a3b9355c4fd74e1de06c048cae /workspace/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest_report.json"
  "680e9df0e05b39d1a7c41d9ebd50332d8ec59e87ce932d470853bc5c8eb6ace2 /workspace/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest.journal.jsonl"
  "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad /workspace/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue_report.json"
  "b2b6dfbb3d0a3efd5cbadee09e134c24fa7594f6df1238833d25a7b671c9af10 /workspace/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue.journal.jsonl"
  "99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4 /workspace/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1/api_rescue_report.json"
  "4bdeb9e6f5a0d3063b6d454d91bde65596ef788a7edd08d67045fa545b6481d6 /workspace/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1/api_rescue.journal.jsonl"
  "f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b /workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue_report.json"
  "49b97de386b759955497e3f9ab7b4358ca5e74ebf3a877fb6c7f3d98e39275b6 /workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue.journal.jsonl"
  "fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1 /workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue_report.json"
  "5c610a4073122e209e26af8e689a683258405c00e58a23c6e9a109c76f9c4c6c /workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue.journal.jsonl"
  "336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727 /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue_report.json"
  "33bf539f37beb285459511ee5349f8eec34b8335ff4c07339ce8a95467379cf0 /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue.journal.jsonl"
  "06af6f49ea45d485e6c61b0e4a8b783894ffb4a1491235c56fb2c0428cf0e683 /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue.journal.jsonl.chain-head.json"
  "aa22e905037222a34eb01964eb2f6b6a9826ffbb19376490ff1c130a2d8bf18b /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/direct_hard_targets.jsonl"
  "a8c9bc693a27d46c5d83d7b2beb4dddcdae6e1d46d64916d163688de3a3ba557 /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/direct_hard_targets_f2.jsonl"
  "77cae6c03ca0dd1e80e303afedf2fb551fd1e8ea7ceee0844ecf8448877b423e /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/repair_policy_sources.jsonl"
  "903fd33974f37fb6144267eac84e39f7d5d8ffcf437bf96db79920fd1f9b6924 /workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/repair_policy_targets.jsonl"
)
for binding in "${known_bindings[@]}"; do
  digest="${binding%% *}"
  path="${binding#* }"
  if [[ ! -s "${path}" ]]; then
    echo "T5GEMMA_MIXED_HANDOFF_BLOCKED missing pinned artifact ${path}" >&2
    exit 78
  fi
  printf '%s  %s\n' "${digest}" "${path}" | sha256sum -c -
done

export T5GEMMA_MIXED_LOCAL_REPORT_SPECS
T5GEMMA_MIXED_LOCAL_REPORT_SPECS="b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab=/workspace/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json;8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50=/workspace/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json;883aeeab6d6a62d4bab41c62f77c8578866cf1a3b9355c4fd74e1de06c048cae=/workspace/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest_report.json;${residual_report_sha}=${RESIDUAL_REPORT}"
export T5GEMMA_MIXED_API_REPORT_SPECS
T5GEMMA_MIXED_API_REPORT_SPECS="fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad=/workspace/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue_report.json;99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4=/workspace/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1/api_rescue_report.json;f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b=/workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue_report.json;fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1=/workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue_report.json;336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727=/workspace/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue_report.json"
export T5GEMMA_MIXED_ALLOW_EXPLORATORY_INPUTS=0
export T5GEMMA_MIXED_REQUIRE_LOCAL_PRODUCTION_FLOOR=0
export T5GEMMA_MIXED_MIN_DIRECT_TARGETS=200
export T5GEMMA_MIXED_MIN_REPAIR_TARGETS=71
export T5GEMMA_MIXED_OUTPUT_DIR=/workspace/artifacts/t5gemma2_4b4b_mixed_rs_sft_final_v1
export T5GEMMA_MIXED_GOLD_REPLAY_RATIO=3
export T5GEMMA_MIXED_EPOCHS=3
export T5GEMMA_MIXED_LEARNING_RATE=5e-5
export T5GEMMA_MIXED_RESUME_COMPAT=1

echo "T5GEMMA_MIXED_HANDOFF_SEALED residual_report_sha=${residual_report_sha} residual_journal_sha=${residual_journal_sha}"
exec "${MIXED_LAUNCHER}"
