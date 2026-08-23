#!/usr/bin/env bash
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

PYTHON=/venv/main/bin/python
BUILDER=/workspace/hybrid_training_patch_v2_3/scripts/preprocessing/build_multifunction_executable_view.py
PARENT_REPORT=/workspace/multifunction_v1/build/build_report.json
OUTPUT_DIR=/workspace/multifunction_v1/executable

expected_builder_sha=395e894a2f3a433e2bd2f995c118ac7e332c1ce1e86053bb8d82a88ecd7ac5f8
expected_parent_sha=7a9fdd032fef34c43ac5e7b8217b6b0b4c986b7dfdf0f1b4b6897aec01df241f

observed_builder_sha="$(sha256sum "${BUILDER}" | awk '{print $1}')"
observed_parent_sha="$(sha256sum "${PARENT_REPORT}" | awk '{print $1}')"
if [[ "${observed_builder_sha}" != "${expected_builder_sha}" ]]; then
  printf 'Executable-view builder hash mismatch: expected %s, got %s\n' \
    "${expected_builder_sha}" "${observed_builder_sha}" >&2
  exit 2
fi
if [[ "${observed_parent_sha}" != "${expected_parent_sha}" ]]; then
  printf 'Parent build-report hash mismatch: expected %s, got %s\n' \
    "${expected_parent_sha}" "${observed_parent_sha}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite existing executable-view path: %s\n' \
    "${OUTPUT_DIR}" >&2
  exit 2
fi

pty "${PYTHON}" "${BUILDER}" \
  --parent-build-report "${PARENT_REPORT}" \
  --expected-parent-build-report-sha256 "${expected_parent_sha}" \
  --output-dir "${OUTPUT_DIR}"
