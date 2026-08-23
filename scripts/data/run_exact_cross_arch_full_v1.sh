#!/usr/bin/env bash
set -euo pipefail

# Resumable orchestration for the frozen 1,975/219 neutral-source split.
# Each shard is independently transactional; this runner never publishes a
# merged corpus and caps Dart compilation at two concurrent shard workers.

ROOT="${ROOT:-/root/exact_cross_arch_pilot}"
OUT="${OUT:-/root/exact_cross_arch_full_v1}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
BUILDER="${BUILDER:-${ROOT}/scripts/data/build_exact_cross_arch_pairs.py}"
PRIVATE="${PRIVATE:-${ROOT}/input/master_dart_graphv2_signature_scrubbed_private.jsonl}"
SPLITS="${SPLITS:-${ROOT}/input/direct_compact_split_v1}"
DART="${DART:-/opt/dart-3.11.5/dart-sdk/bin/dart}"
DARTAOTRUNTIME="${DARTAOTRUNTIME:-/opt/dart-3.11.5/dart-sdk/bin/dartaotruntime}"
MAX_WORKERS="${MAX_WORKERS:-2}"

STATUS="${OUT}.status"
LOG="${OUT}.launcher.log"
mkdir -p "${OUT}/private" "${OUT}/logs"
exec 9>"${OUT}/private/launch.lock"
if ! flock -n 9; then
  printf '%s duplicate_launcher_refused\n' "$(date -u +%FT%TZ)" >"${STATUS}"
  exit 1
fi

status() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" | tee "${STATUS}"
}

on_exit() {
  rc=$?
  if (( rc != 0 )); then
    status "failed_exit_${rc}"
  fi
}
trap on_exit EXIT
rm -f "${OUT}/COMPLETE"

require_sha() {
  path="$1"
  expected="$2"
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    printf 'SHA mismatch for %s: %s != %s\n' "${path}" "${observed}" "${expected}" >&2
    exit 1
  fi
}

status "validating_frozen_inputs"
test "${MAX_WORKERS}" -ge 1
test "${MAX_WORKERS}" -le 2
test -x "${PYTHON}"
test -x "${DART}"
test -x "${DARTAOTRUNTIME}"
test "$("${PYTHON}" -c 'import networkx; print(networkx.__version__)')" = "3.6.1"
require_sha "${BUILDER}" "43563ccf1b28c790c7cd390ab0dcbcc239daea7b7f3cb77b09bd0449ff6796c1"
require_sha "${ROOT}/scripts/data/build_graph_v2_jsonl.py" "eefb68b6abdd7d5b3c4f5907ad03c000f62d0c5f0e11002a2fda72ca292bd673"
require_sha "${ROOT}/scripts/data/cfg_extractor.py" "daebbbfa7ac53fed9104e66396bc861bc837a8cea5a948548204d34439ee553c"
require_sha "${ROOT}/scripts/data/dfg_extractor.py" "603c052e8a79e7f6f689e97acdfc9c87245505b4fbf497bc2c49c2343fb0ed12"
require_sha "${PRIVATE}" "03acc6dd45c861fb6517048a4b8be66a428dcb5cde7cbf3ccdea302fdc08f52b"
require_sha "${SPLITS}/train_private_alignment.jsonl" "f6908c084d3fe3dce7fd7fcb0896cb900d46f5139942cbab96c7aef02af490ff"
require_sha "${SPLITS}/dev_private_alignment.jsonl" "c9c293317860c1b75c09c0ddf81ab3764a7daba9cf1ad133f04e98a8e701bef3"

SALT="${OUT}/private/shared_semantic_pair_salt"
if [[ ! -s "${SALT}" ]]; then
  umask 077
  head -c 32 /dev/urandom >"${SALT}"
fi
test "$(stat -c %s "${SALT}")" -eq 32

run_pair_shard() {
  split="$1"
  shard="$2"
  shard_name="$(printf '%03d' "${shard}")"
  shard_out="${OUT}/${split}/shard-${shard_name}"
  shard_log="${OUT}/logs/${split}-shard-${shard_name}.log"
  mkdir -p "$(dirname "${shard_out}")"
  "${PYTHON}" "${BUILDER}" \
    --private-input "${PRIVATE}" \
    --indices-file "${SPLITS}/${split}_private_alignment.jsonl" \
    --indices-file-format alignment-jsonl \
    --shard-size 64 \
    --shard-index "${shard}" \
    --output-dir "${shard_out}" \
    --pair-salt "${SALT}" \
    --dart "${DART}" \
    --dartaotruntime "${DARTAOTRUNTIME}" \
    --readelf /usr/bin/readelf \
    --objdump-x64 /usr/bin/x86_64-linux-gnu-objdump \
    --objdump-arm64 /usr/bin/aarch64-linux-gnu-objdump \
    --expected-dart-version 3.11.5 \
    --resume >>"${shard_log}" 2>&1
}
export -f run_pair_shard
export ROOT OUT PYTHON BUILDER PRIVATE SPLITS SALT DART DARTAOTRUNTIME

status "building_train_31_shards"
seq 0 30 | xargs -P"${MAX_WORKERS}" -I{} bash -c 'run_pair_shard train "$1"' _ {}

status "building_dev_4_shards"
seq 0 3 | xargs -P"${MAX_WORKERS}" -I{} bash -c 'run_pair_shard dev "$1"' _ {}

status "verifying_35_shards"
complete_count="$(find "${OUT}/train" "${OUT}/dev" -type f -name COMPLETE | wc -l)"
test "${complete_count}" -eq 35
if find "${OUT}/train" "${OUT}/dev" -type f -path '*/private/quarantine.jsonl' -size +0c | grep -q .; then
  echo "non-empty private quarantine found" >&2
  exit 1
fi
while IFS= read -r checksum_file; do
  shard_root="$(dirname "$(dirname "${checksum_file}")")"
  (cd "${shard_root}" && sha256sum -c private/SHA256SUMS.txt >/dev/null)
done < <(find "${OUT}/train" "${OUT}/dev" -type f -path '*/private/SHA256SUMS.txt' | sort)

tmp_complete="${OUT}/COMPLETE.tmp.$$"
printf '%s\n' "completed_utc=$(date -u +%FT%TZ)" "shards=35" "max_workers=${MAX_WORKERS}" >"${tmp_complete}"
mv -f "${tmp_complete}" "${OUT}/COMPLETE"
status "passed_35_shards"
