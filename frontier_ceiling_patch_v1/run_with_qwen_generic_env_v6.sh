#!/usr/bin/env bash
# Load a generic Alibaba workspace credential without printing or persisting it.
set -euo pipefail

env_file="${QWEN_GENERIC_ENV_FILE:-/workspace/Qwen_fallback.env}"
if [[ ! -f "${env_file}" ]]; then
  printf 'missing generic Qwen environment file: %s\n' "${env_file}" >&2
  exit 2
fi

api_key=""
endpoint=""
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
    API_KEY)
      api_key="${value}"
      ;;
    DASHSCOPE_ENDPOINT)
      endpoint="${value}"
      ;;
  esac
done < "${env_file}"

if [[ -z "${api_key}" || -z "${endpoint}" ]]; then
  printf 'generic Qwen env must contain API_KEY and DASHSCOPE_ENDPOINT\n' >&2
  exit 2
fi
if [[ "${endpoint}" != https://* ]]; then
  printf 'DASHSCOPE_ENDPOINT must use https\n' >&2
  exit 2
fi

export QWEN_API_KEY="${api_key}"
export QWEN_BASE_URL="${endpoint%/}"
unset api_key endpoint value line stripped

exec "$@"
