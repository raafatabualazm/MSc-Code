#!/usr/bin/env bash
# Train only the exact 1,196-row gold continuation while API harvest runs.
#
# This is deliberately the first stage of run_qwen38_train_union_2776.sh and
# produces that launcher's exact EXPANDED_GOLD checkpoint.  It does not open a
# held-out dataset, wait for Qwen journals, or start sequence-KL/CoT training.
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
PATCH_ROOT="${PATCH_ROOT:-${WORKSPACE}/hybrid_training_patch_v2_3}"
LEGACY_QWEN_ROOT="${LEGACY_QWEN_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_inline_cfg_v2}"
LEGACY_BUILD_ROOT="${LEGACY_BUILD_ROOT:-${WORKSPACE}/multifunction_v1/build}"
EXPANDED_ROOT="${EXPANDED_ROOT:-${WORKSPACE}/multifunction_v1/expanded2776}"
UNION_ROOT="${UNION_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_union2776}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${WORKSPACE}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
CONTRACT="${CONTRACT:-${LEGACY_BUILD_ROOT}/multifunction_inline_cfg_v2_target24k_contract.json}"
CODEBOOK="${CODEBOOK:-${LEGACY_BUILD_ROOT}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-${WORKSPACE}/scripts/data/build_multifunction_compact_v2.py}"
DERIVATION="${DERIVATION:-${EXPANDED_ROOT}/qwen_2776_supplement.derivation.json}"

SUPPLEMENT_TRAIN="${SUPPLEMENT_TRAIN:-${EXPANDED_ROOT}/supplement1196_multifunction_binary.jsonl}"
SUPPLEMENT_SEAL="${SUPPLEMENT_SEAL:-${EXPANDED_ROOT}/supplement1196_multifunction_binary.target24k.seal.json}"
LEGACY_GOLD="${LEGACY_GOLD:-${LEGACY_QWEN_ROOT}/direct_compact_multifunction_gold_sft_target24k}"
LEGACY_SOURCE_GOLD="${LEGACY_SOURCE_GOLD:-${LEGACY_QWEN_ROOT}/direct_compact_multifunction_gold_sft}"
EXPANDED_GOLD="${EXPANDED_GOLD:-${UNION_ROOT}/direct_compact_multifunction_gold_sft_union2776}"

if (( $# != 0 )); then
  printf 'The sealed supplemental-gold stage accepts no positional arguments\n' >&2
  exit 2
fi
mkdir -p "${WORKSPACE}/locks" "${UNION_ROOT}"
# This is the same GPU-training lock used by the complete union launcher.  The
# API-only supplemental harvest has its own qwen38_supplement1196 lock, so the
# two may run concurrently without allowing two checkpoint writers.
exec 9>"${WORKSPACE}/locks/qwen38_union2776.lock"
if ! flock -n 9; then
  printf 'Another Qwen 2,776-task training chain holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[qwen38_gold_supplement1196] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

sha256_of() {
  local value
  value="$(sha256sum "$1")"
  printf '%s\n' "${value%% *}"
}

for required in \
  "${PATCH_ROOT}/scripts/training/direct_compact_qwen_decompiler.py" \
  "${PATCH_ROOT}/scripts/evaluation/validate_direct_compact_training_stage.py" \
  "${CONTRACT}" "${CODEBOOK}" "${CODEC}" "${TOKENIZER_JSON}" \
  "${DERIVATION}" "${SUPPLEMENT_TRAIN}" "${SUPPLEMENT_SEAL}" \
  "${LEGACY_GOLD}/decoder_adapter/adapter_config.json" \
  "${LEGACY_GOLD}/source_embedding_overlay.pt" \
  "${LEGACY_GOLD}/compact_contract.json" \
  "${LEGACY_GOLD}/run_provenance.json" \
  "${LEGACY_GOLD}/overlay_migration_receipt.json" \
  "${LEGACY_SOURCE_GOLD}/run_provenance.json"; do
  test -f "${required}" || {
    printf 'Required supplemental-gold input is missing: %s\n' \
      "${required}" >&2
    exit 2
  }
done

jq -e '
  .schema == "qwen-2776-supplement-derivation-v1"
  and .counts.fit_tasks == 2776
  and .counts.legacy_parent_tasks == 1580
  and .counts.supplement_tasks == 1196
  and .counts.heldout_tasks == 175
  and .heldout_intersection_count == 0
  and .invariants.live_journal_modified == false
' "${DERIVATION}" >/dev/null

EXPECTED_CONTRACT_SHA256="$(jq -er '.inputs.contract.sha256' "${DERIVATION}")"
if [[ "$(sha256_of "${CONTRACT}")" != "${EXPECTED_CONTRACT_SHA256}" ]]; then
  printf 'Gold-continuation contract differs from the 2,776-task derivation\n' >&2
  exit 2
fi

# Verify the exact supplemental data and legacy-1580 ancestry using the same
# records and invariants as run_qwen38_train_union_2776.sh.  This reads only
# fit artifacts and checkpoint provenance; the held-out dataset is never
# supplied to this process.
"${PYTHON}" - \
  "${DERIVATION}" \
  "${SUPPLEMENT_TRAIN}" "${SUPPLEMENT_SEAL}" \
  "${LEGACY_GOLD}" "${LEGACY_SOURCE_GOLD}" "${CONTRACT}" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def verify(record, path, label):
    path = pathlib.Path(path).resolve()
    if not isinstance(record, dict):
        raise SystemExit(f"{label} record is absent")
    if pathlib.Path(str(record.get("path") or "")).resolve() != path:
        raise SystemExit(f"{label} path differs from its sealed record")
    if record.get("sha256") != sha(path):
        raise SystemExit(f"{label} SHA-256 differs from its sealed record")
    size = record.get("size_bytes", record.get("bytes"))
    if int(size if size is not None else -1) != path.stat().st_size:
        raise SystemExit(f"{label} size differs from its sealed record")


(
    derivation_path,
    supplement_train,
    supplement_seal,
    legacy_gold,
    legacy_source_gold,
    contract,
) = sys.argv[1:]
derivation = load(derivation_path)
verify(
    derivation["outputs"]["supplement_compact"],
    supplement_train,
    "supplement compact",
)
verify(
    derivation["outputs"]["supplement_compact_seal"],
    supplement_seal,
    "supplement compact seal",
)
supplement_seal_value = load(supplement_seal)
if (
    supplement_seal_value.get("schema")
    != "compact-public-private-join-seal-v1"
    or supplement_seal_value.get("selected_role") != "fit"
    or int(supplement_seal_value.get("rows", -1)) != 1196
    or supplement_seal_value.get("output_sha256") != sha(supplement_train)
    or supplement_seal_value.get("contract_sha256") != sha(contract)
):
    raise SystemExit("supplemental target24k fit seal is not exact")

legacy_gold = pathlib.Path(legacy_gold).resolve()
legacy_source_gold = pathlib.Path(legacy_source_gold).resolve()
target_provenance_path = legacy_gold / "run_provenance.json"
receipt_path = legacy_gold / "overlay_migration_receipt.json"
source_provenance_path = legacy_source_gold / "run_provenance.json"
target = load(target_provenance_path)
receipt = load(receipt_path)
source = load(source_provenance_path)
source_record = target.get("warmstart_checkpoint") or {}
receipt_source = receipt.get("source_checkpoint") or {}
receipt_outputs = receipt.get("outputs") or {}
receipt_compatibility = receipt.get("contract_compatibility") or {}
legacy_inputs = derivation.get("inputs") or {}
expansion_seal_path = pathlib.Path(
    str(legacy_inputs["expansion_build_seal"]["path"])
).resolve()
verify(
    legacy_inputs["expansion_build_seal"],
    expansion_seal_path,
    "expansion build seal",
)
expansion_seal = load(expansion_seal_path)
base_parent_seal = (expansion_seal.get("artifacts") or {}).get(
    "parent_seal"
) or {}
if (
    target.get("schema") != "direct-compact-run-provenance-v1"
    or target.get("checkpoint_stage") != "contract-overlay-migration-only"
    or target.get("training_performed") is not False
    or target.get("heldout_loaded_during_migration") is not False
    or target.get("contract_sha256") != sha(contract)
    or target.get("overlay_migration_receipt_sha256") != sha(receipt_path)
    or pathlib.Path(str(source_record.get("path") or "")).resolve()
       != legacy_source_gold
    or source_record.get("provenance_sha256") != sha(source_provenance_path)
    or receipt.get("schema") != "direct-compact-overlay-migration-receipt-v1"
    or int(receipt.get("training_steps", -1)) != 0
    or receipt_source != source_record
    or (receipt.get("invariants") or {}) != {
        "changed_rows_use_new_codebook_mean_initialization": True,
        "decoder_adapter_tree_byte_identical": True,
        "heldout_data_opened": False,
        "new_contract_copied_byte_identically": True,
        "no_training_or_optimizer_step_performed": True,
        "old_overlay_row_reused_only_for_identical_expansion": True,
    }
    or source.get("schema") != "direct-compact-run-provenance-v1"
    or source.get("train_file_sha256")
       != legacy_inputs["legacy_compact"]["sha256"]
    or source.get("train_seal_sha256")
       != base_parent_seal.get("sha256")
    or int(source.get("train_sealed_rows", -1)) != 1580
    or source.get("heldout_loaded_during_training") is not False
    or source.get("contract_sha256")
       != legacy_inputs["candidate_contract"]["sha256"]
    or (source.get("loss_contract") or {}).get("primary_reduction")
       != "base_causal_lm_token_mean"
    or (source.get("loss_contract") or {}).get(
        "sequence_distribution_nll"
    ) is not False
    or source_record.get("contract_sha256") != source.get("contract_sha256")
    or source_record.get("decoder_adapter_sha256")
       != source.get("decoder_adapter_sha256")
    or source_record.get("source_overlay_sha256")
       != source.get("source_overlay_sha256")
    or target.get("decoder_adapter_sha256")
       != source_record.get("decoder_adapter_sha256")
    or target.get("source_overlay_sha256")
       != source_record.get("source_overlay_sha256")
    or receipt_outputs.get("decoder_adapter_sha256")
       != target.get("decoder_adapter_sha256")
    or receipt_outputs.get("source_overlay_sha256")
       != target.get("source_overlay_sha256")
    or receipt_outputs.get("compact_contract_sha256")
       != target.get("contract_sha256")
    or receipt_compatibility.get(
        "all_non_migratable_contract_fields_identical"
    ) is not True
    or int(receipt_compatibility.get("changed_expansion_rows", -1)) != 0
):
    raise SystemExit("legacy1580 gold checkpoint ancestry is not exact")
print(
    "QWEN_SUPPLEMENTAL_GOLD_INPUTS_AND_LEGACY_ANCESTRY_VERIFIED",
    flush=True,
)
PY

export PYTHONPATH="${PATCH_ROOT}:${WORKSPACE}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${PATCH_ROOT}"

# A self-sealed warm start includes the decoder LoRA, compact overlay, exact
# contract, and provenance.  Loading only the LoRA is rejected.
"${PYTHON}" -c \
  'import sys; from scripts.training.direct_compact_qwen_decompiler import validate_warmstart_checkpoint; validate_warmstart_checkpoint(sys.argv[1], contract_path=sys.argv[2]); print("LEGACY_GOLD_WARMSTART_VALID", flush=True)' \
  "${LEGACY_GOLD}" "${CONTRACT}"

validate_stage() {
  "${PYTHON}" -m scripts.evaluation.validate_direct_compact_training_stage \
    --checkpoint "${EXPANDED_GOLD}" \
    --contract "${CONTRACT}" \
    --train-file "${SUPPLEMENT_TRAIN}" \
    --train-seal "${SUPPLEMENT_SEAL}" \
    --expected-train-rows 1196 \
    --no-eval-during-training \
    --loss-mode token_mean \
    --base-warmstart "${LEGACY_GOLD}" \
    --stage-contract "${DERIVATION}" \
    --expected-stage-contract-sha256 "$(sha256_of "${DERIVATION}")"
}

if [[ -e "${EXPANDED_GOLD}" ]] && validate_stage; then
  printf 'QWEN_SUPPLEMENTAL_GOLD_STAGE_REUSE checkpoint=%s\n' \
    "${EXPANDED_GOLD}"
  exit 0
fi

args=(
  --train_file "${SUPPLEMENT_TRAIN}"
  --train_seal "${SUPPLEMENT_SEAL}"
  --no_eval_during_training
  --output_dir "${EXPANDED_GOLD}"
  --contract "${CONTRACT}"
  --codebook "${CODEBOOK}"
  --codec_artifact "${CODEC}"
  --tokenizer_json "${TOKENIZER_JSON}"
  --warmstart_checkpoint "${LEGACY_GOLD}"
  --stage_contract "${DERIVATION}"
  --expected_stage_contract_sha256 "$(sha256_of "${DERIVATION}")"
  --learning_rate 2e-5
  --epochs 1.0
  --batch_size 1
  --grad_accum 16
  --eval_strategy no
  --seed 44
  --logging_steps 1
  --save_steps 25
  --gradient_checkpointing
  --bf16
)
if [[ -d "${EXPANDED_GOLD}" ]]; then
  if compgen -G "${EXPANDED_GOLD}/checkpoint-*" >/dev/null; then
    # The trainer accepts only a checkpoint owned by this exact output,
    # warm-start, fit seal, contract, and stage contract.
    args+=(--resume_from_checkpoint auto)
  else
    printf 'Incomplete supplemental-gold output has no validated resume checkpoint: %s\n' \
      "${EXPANDED_GOLD}" >&2
    exit 2
  fi
fi

"${PYTHON}" -m scripts.training.direct_compact_qwen_decompiler \
  "${args[@]}"
validate_stage
printf 'QWEN_SUPPLEMENTAL_GOLD_STAGE_COMPLETE rows=1196 checkpoint=%s\n' \
  "${EXPANDED_GOLD}"
