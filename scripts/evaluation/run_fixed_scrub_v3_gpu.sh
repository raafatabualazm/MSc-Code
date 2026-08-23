#!/usr/bin/env bash
# Matched v3 opaque-signature evaluation on the frozen Regions16 checkpoint.
# The script waits for the unrelated ARM64 job, fails closed on every preflight,
# uses the pinned Dart 3.11.5 SDK, and never deletes or recycles the host.
set -Eeuo pipefail

ROOT="${1:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
RESULTS="$ROOT/results"
SWEEPS="$RESULTS/sweeps_antigravity"
LOGS="$ROOT/logs/fixed_scrub_v3"
DATA="$ROOT/data/testing"
BASE="qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_regions16"
CHECKPOINT="$ROOT/artifacts/$BASE/pytorch_model.bin"
CHECKPOINT_SHA256="e8e872608f22ae8e1c5607d6179feeb5f133401fb2a7d1fc40fe8894d8c347fc"
STAGING="${V3_STAGING:-$ROOT/fixed_scrub_v3_staging}"
STAGING_MANIFEST="$STAGING/fixed_scrub_v3_gpu_inputs.sha256"
INPUT_MANIFEST="$ROOT/fixed_scrub_v3_gpu_inputs.sha256"
QUEUE_STATUS="$RESULTS/fixed_scrub_v3_queue.status"
ARM64_STATUS="$RESULTS/arm64_regions16_s42.status"
CURRENT_ARM_STATUS=""
STAGE=bootstrap

mkdir -p "$RESULTS" "$SWEEPS" "$LOGS"

fail_status() {
  local rc=$?
  local now
  now="$(date -u +%FT%TZ)"
  printf 'FAILED stage=%s rc=%s time=%s\n' "$STAGE" "$rc" "$now" > "$QUEUE_STATUS"
  if [[ -n "$CURRENT_ARM_STATUS" ]]; then
    printf 'FAILED stage=%s rc=%s time=%s\n' "$STAGE" "$rc" "$now" > "$CURRENT_ARM_STATUS"
  fi
  exit "$rc"
}
trap fail_status ERR

STAGE=wait_arm64
printf 'QUEUED waiting_for=arm64_regions16_s42 time=%s\n' "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
while [[ -f "$ARM64_STATUS" ]] && grep -q '^RUNNING' "$ARM64_STATUS"; do
  sleep 60
done

STAGE=wait_vram
while true; do
  free_vram="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "$free_vram" =~ ^[0-9]+$ ]] && (( free_vram >= 60000 )); then
    break
  fi
  printf 'QUEUED waiting_for=vram free=%sMiB time=%s\n' "$free_vram" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
  sleep 60
done

STAGE=deploy_verified_inputs
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
test -s "$STAGING_MANIFEST"
(
  cd "$STAGING"
  sha256sum -c "$STAGING_MANIFEST"
) > "$LOGS/staging_manifest_check.log"
cp -a "$STAGING/scripts/." "$ROOT/scripts/"
cp -a "$STAGING/data/." "$ROOT/data/"
cp -a "$STAGING/results/." "$ROOT/results/"
cp -a "$STAGING/logs/." "$ROOT/logs/"
cp "$STAGING/generate_synthetic_tasks_parallel.py" "$ROOT/generate_synthetic_tasks_parallel.py"
cp "$STAGING_MANIFEST" "$INPUT_MANIFEST"

STAGE=preflight
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
cd "$ROOT"
test -s "$INPUT_MANIFEST"
sha256sum -c "$INPUT_MANIFEST" > "$LOGS/input_manifest_check.log"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$ROOT/toolchains/dartsdk-linux-x64-release.zip" | awk '{print $1}')" = "57f3ab5ac24883060b1ff12bcdac472ed76563ec7364e88f8a6d41e4f0db075f"
DART_VERSION="$(/home/zeus/dart-sdk/bin/dart --version 2>&1)"
[[ "$DART_VERSION" == "Dart SDK version: 3.11.5 (stable)"* ]]
printf '%s\n' "$DART_VERSION" > "$LOGS/dart_version.txt"
test -x "$PYTHON_BIN"

for arm in opaque_nameonly opaque_neutralexact; do
  test "$(wc -l < "$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_public.jsonl")" -eq 154
  test "$(wc -l < "$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_private.jsonl")" -eq 154
done

"$PYTHON_BIN" -m unittest \
  scripts.data.test_signature_scrubbed_eval \
  scripts.evaluation.test_rehydrate_signature_scrubbed_predictions \
  scripts.evaluation.test_protocol_integrity_antigravity \
  scripts.evaluation.test_fixed_scrub_v3_gates \
  scripts.evaluation.test_verify_fixed_scrub_v3_inference \
  scripts.evaluation.test_verify_fixed_scrub_v3_comparator \
  scripts.evaluation.test_write_fixed_scrub_v3_scoring_provenance \
  scripts.evaluation.test_project_fixed_scrub_v3_standalone_pool \
  scripts.evaluation.test_validate_fixed_scrub_v3_standalone_compile \
  scripts.evaluation.test_analyze_fixed_scrub_v3 \
  > "$LOGS/unit_tests.log" 2>&1

"$PYTHON_BIN" "$ROOT/scripts/evaluation/fixed_scrub_v3_gates.py" \
  --benchmark "$DATA/grpo_data_graphv2.jsonl" \
  --nameonly_private "$DATA/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_private.jsonl" \
  --nameonly_public "$DATA/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_public.jsonl" \
  --neutralexact_private "$DATA/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_private.jsonl" \
  --neutralexact_public "$DATA/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_public.jsonl" \
  --nameonly_summary "$DATA/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_private.jsonl.summary.json" \
  --neutralexact_summary "$DATA/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_private.jsonl.summary.json" \
  --target_name fn0 --expected_rows 154 --skip_dart \
  > "$LOGS/static_gates.log" 2>&1

"$PYTHON_BIN" "$ROOT/scripts/evaluation/verify_fixed_scrub_v3_comparator.py" \
  --predictions "$RESULTS/v3_comparator_rescore/comparator_predictions.json" \
  --provenance "$RESULTS/v3_comparator_rescore/comparator_predictions.json.provenance.json" \
  --dataset "$DATA/grpo_data_graphv2.jsonl" \
  --output "$RESULTS/v3_comparator_rescore/comparator_input_verification.json" \
  > "$LOGS/comparator_input_verification.log" 2>&1

export PATH="/home/zeus/dart-sdk/bin:/usr/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export GRAPH_ENCODER_MODEL=microsoft/graphcodebert-base GRAPH_DECODER_MODEL=Qwen/Qwen3-8B
export GRAPH_ENCODER_REVISION=2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d GRAPH_DECODER_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export GRAPH_ENCODER_PEFT=lora GRAPH_DECODER_PEFT=lora GRAPH_FREEZE_ENCODER=0 GRAPH_FREEZE_DECODER=0
export GRAPH_LORA_R=64 GRAPH_LORA_ALPHA=128 GRAPH_LOAD_4BIT=0 GRAPH_ATTN_IMPLEMENTATION=sdpa
export GRAPH_MAX_BLOCK_INSTRS=20 GRAPH_MAX_DATAFLOW_EDGES=0 GRAPH_POSITION_SCHEME=roberta
export GRAPH_BLOCK_POSITION_MODE=sinusoidal GRAPH_CAUSAL_POSITION_IDS=cumsum GRAPH_BLOCK_POOLING=cls
export GRAPH_BLOCK_VECTORS_PER_BLOCK=4 GRAPH_EDGE_ABLATION=full GRAPH_DFG_MODE=edges
export GRAPH_GLOBAL_ATTENTION_ABLATION=full GRAPH_GNN_ABLATION=identity GRAPH_GNN_LAYERS=4
export GRAPH_ADD_REVERSE_EDGES=1 GRAPH_AUTO_CFG=0 GRAPH_STRICT_GRAPH=1
export GRAPH_PROMPT_ASSEMBLY_MODE=none GRAPH_PROMPT_CLEAN_ASM=0 GRAPH_PROMPT_FIT_ASSEMBLY=0
export GRAPH_QWEN_PREFIX_TOKENS=64 GRAPH_QWEN_PREFIX_DYNAMIC=1 GRAPH_QWEN_PREFIX_MIN_TOKENS=4
export GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2=4 GRAPH_QWEN_PREFIX_GATE_MODE=token GRAPH_QWEN_PREFIX_GATE_INIT=0.2
export GRAPH_QWEN_PREFIX_RMS_MATCH=1 GRAPH_REGION_COMPRESSION=linear_residual GRAPH_REGION_MAX_BLOCKS=16
export GRAPH_USE_REASONING=0 GRAPH_SEED=42 GRAPH_QUIET=1 GRAPH_GRADIENT_CHECKPOINTING=1
export GRAPH_DECODER_PROMPT_MAX_LENGTH=2048 GRAPH_EVAL_GENERATION_BATCH_SIZE=10 GRAPH_EVAL_MAX_NEW_TOKENS=768
export EVAL_DART_WORKERS=64 EVAL_PASS_STABILITY_RUNS=1
unset GRAPH_QWEN_PREFIX_GATE_OVERRIDE

run_standalone_diagnostic() {
  local source_predictions="$1"
  local output_prefix="$2"
  local pool="${output_prefix}_standalone_candidate_pool.json"
  local projection_provenance="${pool}.provenance.json"
  local compile_stdout="${output_prefix}_legacy_standalone_aot_compile_at_k.txt"
  local result_provenance="${compile_stdout}.provenance.json"
  local pool_sha projection_sha scorer_sha

  "$PYTHON_BIN" "$ROOT/scripts/evaluation/project_fixed_scrub_v3_standalone_pool.py" \
    --predictions "$source_predictions" --output "$pool" \
    --provenance_output "$projection_provenance" > /dev/null
  pool_sha="$(sha256sum "$pool" | awk '{print $1}')"
  projection_sha="$(sha256sum "$projection_provenance" | awk '{print $1}')"
  scorer_sha="$(sha256sum "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" | awk '{print $1}')"

  "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
    --predictions "$pool" --k_values 1,5,10 --workers 64 --timeout 30 \
    --compile_mode legacy > "$compile_stdout"
  "$PYTHON_BIN" "$ROOT/scripts/evaluation/validate_fixed_scrub_v3_standalone_compile.py" \
    --source_predictions "$source_predictions" --candidate_pool "$pool" \
    --projection_provenance "$projection_provenance" --compile_stdout "$compile_stdout" \
    --scorer "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
    --projector "$ROOT/scripts/evaluation/project_fixed_scrub_v3_standalone_pool.py" \
    --dart_version_file "$LOGS/dart_version.txt" \
    --expected_pool_sha256 "$pool_sha" \
    --expected_projection_provenance_sha256 "$projection_sha" \
    --expected_scorer_sha256 "$scorer_sha" --output "$result_provenance" > /dev/null
}

run_arm() {
  local arm="$1"
  local signature_mode="$2"
  local analysis_arm="$3"
  local expected_prompt_digest="$4"
  local stem="${BASE}_sigscrub_v3_${arm}"
  local public="$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_public.jsonl"
  local private="$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_private.jsonl"
  local raw="$RESULTS/${stem}_raw_predictions.json"
  local scored="$RESULTS/${stem}_predictions.json"
  local arm_status="$RESULTS/${stem}.status"
  local log="$LOGS/${stem}.log"
  CURRENT_ARM_STATUS="$arm_status"

  {
    STAGE="${arm}_inference"
    printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
    printf 'RUNNING stage=inference time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_inference_antigravity.py" \
      --dataset "$public" --decoder_model Qwen/Qwen3-8B --output "$raw" \
      --checkpoint "$CHECKPOINT" \
      --decoder_revision b968826d9c46dd6066d109eabc6255188de91218 \
      --encoder_revision 2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d \
      --seed 42 --num_samples 10 --generation_batch_size 10 --max_new_tokens 768 \
      --decoder_prompt_max_length 2048 --graph_input_ablation none --graph_ablation_seed 42

    STAGE="${arm}_verify_inference"
    printf 'RUNNING stage=verify_inference time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/verify_fixed_scrub_v3_inference.py" \
      --public_dataset "$public" --raw_predictions "$raw" \
      --provenance "$raw.provenance.json" --checkpoint_sha256 "$CHECKPOINT_SHA256" \
      --expected_prompt_stream_sha256 "$expected_prompt_digest" \
      --output "$RESULTS/${stem}_inference_verification.json"

    STAGE="${arm}_rehydrate"
    printf 'RUNNING stage=rehydrate time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/rehydrate_signature_scrubbed_predictions.py" \
      --predictions "$raw" --public_dataset "$public" --private_dataset "$private" \
      --output "$scored" --expected_rows 154 --expected_samples 10 \
      --expected_signature_mode "$signature_mode" --expected_target_name fn0

    STAGE="${arm}_metrics"
    printf 'RUNNING stage=metrics time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_codebleu_antigravity.py" \
      --predictions "$scored" > "$RESULTS/${stem}_codebleu.txt"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_codebleu_antigravity.py" \
      --predictions "$scored" --compiled_only --workers 64 \
      > "$RESULTS/${stem}_legacy_standalone_aot_compiled_codebleu.txt"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
      --predictions "$scored" --k_values 1,5 --workers 64 --timeout 30 \
      --compile_mode jit_tests > "$RESULTS/${stem}_compile_at_k.txt"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_pass_at_k_antigravity.py" \
      --predictions "$scored" --k_values 1,5,10 --workers 64 --timeout 30 \
      > "$RESULTS/${stem}_pass_at_k.txt"
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/compile_statistical_results_antigravity.py" \
      --predictions "$scored" --output "$SWEEPS/${stem}_stats.csv" \
      --workers 64 --timeout 30 --compile_mode jit_tests
    "$PYTHON_BIN" "$ROOT/scripts/evaluation/write_fixed_scrub_v3_scoring_provenance.py" \
      --arm "$analysis_arm" --predictions "$scored" \
      --stats "$SWEEPS/${stem}_stats.csv" --checkpoint "$CHECKPOINT" \
      --inference_provenance "$raw.provenance.json" \
      --join_provenance "$scored.provenance.json" --public_dataset "$public" \
      --scorer "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
      --dart_version_file "$LOGS/dart_version.txt" \
      --output "$SWEEPS/${stem}_stats.csv.provenance.json"

    STAGE="${arm}_standalone_diagnostic"
    printf 'RUNNING stage=standalone_diagnostic time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
    run_standalone_diagnostic "$raw" "$RESULTS/${stem}"

    STAGE="${arm}_manifest"
    printf 'RUNNING stage=manifest time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
    sha256sum \
      "$public" "$private" "$CHECKPOINT" "$raw" "$raw.provenance.json" \
      "$RESULTS/${stem}_inference_verification.json" \
      "$scored" "$scored.provenance.json" \
      "$RESULTS/${stem}_codebleu.txt" \
      "$RESULTS/${stem}_legacy_standalone_aot_compiled_codebleu.txt" \
      "$RESULTS/${stem}_compile_at_k.txt" "$RESULTS/${stem}_pass_at_k.txt" \
      "$RESULTS/${stem}_standalone_candidate_pool.json" \
      "$RESULTS/${stem}_standalone_candidate_pool.json.provenance.json" \
      "$RESULTS/${stem}_legacy_standalone_aot_compile_at_k.txt" \
      "$RESULTS/${stem}_legacy_standalone_aot_compile_at_k.txt.provenance.json" \
      "$SWEEPS/${stem}_stats.csv" "$SWEEPS/${stem}_stats.csv.provenance.json" \
      > "$RESULTS/${stem}_sha256.txt"
    printf 'COMPLETE time=%s\n' "$(date -u +%FT%TZ)" > "$arm_status"
  } > "$log" 2>&1
  CURRENT_ARM_STATUS=""
}

run_arm opaque_nameonly name_only name_only e0c13c8169598851eb3363728bf792adb2389e420e3b9af6dd98834288ead622
run_arm opaque_neutralexact neutral_exact neutral_exact 5c4ae01d07e3ff962fc1c555ed1463365fca614bf4316cfb096fe96f99b64971

STAGE=comparator_rescore
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
PYTHON_BIN="$PYTHON_BIN" EVAL_DART_WORKERS=64 \
  bash "$ROOT/scripts/evaluation/run_fixed_scrub_v3_comparator_rescore.sh" \
  "$ROOT" "$RESULTS/v3_comparator_rescore/comparator_predictions.json" \
  "$RESULTS/v3_comparator_rescore"

"$PYTHON_BIN" "$ROOT/scripts/evaluation/write_fixed_scrub_v3_scoring_provenance.py" \
  --arm comparator \
  --predictions "$RESULTS/v3_comparator_rescore/comparator_predictions.json" \
  --stats "$RESULTS/v3_comparator_rescore/comparator_stats.csv" \
  --checkpoint "$CHECKPOINT" \
  --inference_provenance "$RESULTS/v3_comparator_rescore/comparator_predictions.json.provenance.json" \
  --scorer "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  --dart_version_file "$LOGS/dart_version.txt" \
  --output "$RESULTS/v3_comparator_rescore/comparator_stats.csv.provenance.json"

STAGE=comparator_standalone_diagnostic
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
run_standalone_diagnostic \
  "$RESULTS/v3_comparator_rescore/comparator_predictions.json" \
  "$RESULTS/v3_comparator_rescore/comparator"

STAGE=analysis
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
"$PYTHON_BIN" "$ROOT/scripts/evaluation/analyze_fixed_scrub_v3.py" \
  --comparator "$RESULTS/v3_comparator_rescore/comparator_predictions.json" \
  --comparator-stats "$RESULTS/v3_comparator_rescore/comparator_stats.csv" \
  --comparator-provenance "$RESULTS/v3_comparator_rescore/comparator_stats.csv.provenance.json" \
  --neutral-exact "$RESULTS/${BASE}_sigscrub_v3_opaque_neutralexact_predictions.json" \
  --neutral-exact-stats "$SWEEPS/${BASE}_sigscrub_v3_opaque_neutralexact_stats.csv" \
  --neutral-exact-provenance "$SWEEPS/${BASE}_sigscrub_v3_opaque_neutralexact_stats.csv.provenance.json" \
  --name-only "$RESULTS/${BASE}_sigscrub_v3_opaque_nameonly_predictions.json" \
  --name-only-stats "$SWEEPS/${BASE}_sigscrub_v3_opaque_nameonly_stats.csv" \
  --name-only-provenance "$SWEEPS/${BASE}_sigscrub_v3_opaque_nameonly_stats.csv.provenance.json" \
  --broken-tasks "$DATA/fixed_scrub_v3_known_broken_tasks.json" \
  --target-name fn0 --expected-tasks 154 --expected-candidates 10 \
  --output-json "$RESULTS/fixed_scrub_v3_analysis.json" \
  --output-markdown "$RESULTS/FIXED_SCRUB_V3_ANALYSIS.md"

STAGE=final_manifest
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
sha256sum \
  "$INPUT_MANIFEST" "$CHECKPOINT" \
  "$LOGS/staging_manifest_check.log" "$LOGS/input_manifest_check.log" "$LOGS/unit_tests.log" \
  "$LOGS/static_gates.log" "$LOGS/dart_version.txt" \
  "$RESULTS"/*sigscrub_v3* "$SWEEPS"/*sigscrub_v3* \
  "$RESULTS/v3_comparator_rescore"/* \
  "$RESULTS/fixed_scrub_v3_analysis.json" "$RESULTS/FIXED_SCRUB_V3_ANALYSIS.md" \
  "$LOGS"/* \
  > "$RESULTS/fixed_scrub_v3_final_sha256.txt"

printf 'COMPLETE time=%s\n' "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"
