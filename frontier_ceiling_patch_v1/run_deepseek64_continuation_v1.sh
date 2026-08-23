#!/usr/bin/env bash
set -euo pipefail

# Default is a no-provider-call preflight.  Paid execution requires both the
# explicit "run" mode and the runner's exact confirmation string.
mode="${1:-preflight}"
workspace="${WORKSPACE:-/workspace}"
python_bin="${PYTHON_BIN:-${workspace}/.venv/bin/python}"
entry="${workspace}/frontier_ceiling_patch_v1/deepseek64_continuation_v1.py"
contract="${workspace}/frontier_ceiling_patch_v1/deepseek64_continuation_contract_v1.json"
env_file="${DEEPSEEK_ENV_FILE:-${workspace}/data.env}"
run_root="${workspace}/artifacts/frontier_ceiling_two_enrichments/runs"
workers="${DEEPSEEK64_WORKERS:-4}"
address_space_limit="${DEEPSEEK64_ADDRESS_SPACE_LIMIT:-3G}"

if [[ ! "${workers}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSEEK64_WORKERS must be a positive integer" >&2
    exit 64
fi
if [[ -z "${address_space_limit}" ]]; then
    echo "DEEPSEEK64_ADDRESS_SPACE_LIMIT cannot be empty" >&2
    exit 64
fi

case "${mode}" in
    preflight|status)
        paid=0
        ;;
    adopt)
        # Local evaluator work only; this does not construct an API client.
        paid=0
        ;;
    run)
        paid=1
        if [[ "${DEEPSEEK64_PAID_CONFIRM:-}" != "YES_64K_CONTINUATION" ]]; then
            echo "Refusing paid launch: export DEEPSEEK64_PAID_CONFIRM=YES_64K_CONTINUATION" >&2
            exit 64
        fi
        ;;
    *)
        echo "usage: $0 {preflight|status|adopt|run}" >&2
        exit 64
        ;;
esac

[[ -x "${python_bin}" ]] || { echo "missing Python: ${python_bin}" >&2; exit 66; }
[[ -f "${entry}" ]] || { echo "missing runner: ${entry}" >&2; exit 66; }
[[ -f "${contract}" ]] || { echo "missing contract: ${contract}" >&2; exit 66; }

common=(
    "${python_bin}" "${entry}"
    --workspace "${workspace}"
    --contract "${contract}"
    --mode "${mode}"
    --workers "${workers}"
)
if [[ "${paid}" -eq 1 ]]; then
    common+=(--paid-confirmation YES_64K_CONTINUATION)
fi

launch_arm() {
    local arm="$1"
    local unit="frontier-deepseek64-${arm}-v1"
    local out
    case "${arm}" in
        opus)
            out="${run_root}/opus_real_deepseek_v4pro_k10_64k_continuation_v1"
            ;;
        codex)
            out="${run_root}/codex_multifunction_deepseek_v4pro_k10_64k_continuation_v1"
            ;;
        *)
            return 64
            ;;
    esac

    if [[ "${mode}" == "run" ]]; then
        systemd-run \
            --unit="${unit}" \
            --property=WorkingDirectory="${workspace}" \
            --property=Restart=no \
            --property=KillMode=mixed \
            --property=LimitAS="${address_space_limit}" \
            --collect \
            "${common[@]}" \
            --arm "${arm}" \
            --out "${out}" \
            --deepseek-env-file "${env_file}"
    else
        "${common[@]}" \
            --arm "${arm}" \
            --out "${out}" \
            --deepseek-env-file "${env_file}"
    fi
}

# The two arms have independent frozen sources and output locks.  Non-paid
# modes run serially for readable diagnostics; paid mode launches two systemd
# services.  Four workers per arm plus a 3 GiB per-process address-space
# ceiling are the safe defaults for the 8 GiB host; callers may override both
# explicitly.  The ceiling prevents pathological generated Dart programs from
# taking down the runner with a host-wide OOM.
launch_arm opus
launch_arm codex
