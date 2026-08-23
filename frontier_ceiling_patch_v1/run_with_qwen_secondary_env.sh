#!/usr/bin/env bash
# Map the secondary Alibaba credentials in Qwen.env to the conventional
# variables consumed by frontier_passk.py, without placing either secret in
# process arguments or logs.
set -euo pipefail

env_file="${QWEN_ENV_FILE:-/workspace/Qwen.env}"
if [[ ! -f "${env_file}" ]]; then
  printf 'missing Qwen environment file: %s\n' "${env_file}" >&2
  exit 2
fi

api_key2=""
endpoint2=""
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%$'\r'}"
  stripped="${line#"${line%%[![:space:]]*}"}"
  if [[ -z "${stripped}" || "${stripped}" == \#* || "${stripped}" != *=* ]]; then
    continue
  fi
  key="${stripped%%=*}"
  value="${stripped#*=}"
  key="${key%"${key##*[![:space:]]}"}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [[ ${#value} -ge 2 ]]; then
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi
  case "${key}" in
    API_KEY2)
      api_key2="${value}"
      ;;
    DASHSCOPE_ENDPOINT2)
      endpoint2="${value}"
      ;;
  esac
done < "${env_file}"

if [[ -z "${api_key2}" || -z "${endpoint2}" ]]; then
  printf 'Qwen.env must contain nonempty API_KEY2 and DASHSCOPE_ENDPOINT2\n' >&2
  exit 2
fi
if [[ "${endpoint2}" != https://* ]]; then
  printf 'DASHSCOPE_ENDPOINT2 must use https\n' >&2
  exit 2
fi

export QWEN_API_KEY="${api_key2}"
export QWEN_BASE_URL="${endpoint2%/}"
unset api_key2 endpoint2 value line stripped

exec "$@"
