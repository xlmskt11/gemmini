#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
RUNTIME_CFG="${REPO_ROOT}/sims/firesim/deploy/config_runtime.yaml"
RESULTS_ROOT="${REPO_ROOT}/sims/firesim/deploy/results-workload"
SESSION_NAME="${FIRESIM_SCREEN_SESSION:-fsim0}"
SSH_AGENT_VARS="${HOME}/.ssh/AGENT_VARS"
N_PREDICT="${1:-1}"

load_ssh_agent() {
    if [[ -f "${SSH_AGENT_VARS}" ]]; then
        # shellcheck disable=SC1090
        source "${SSH_AGENT_VARS}"
    fi
}

read_host_spec() {
    python3 - "$1" <<'PY'
import re
import sys

cfg = open(sys.argv[1], "r", encoding="utf-8").read().splitlines()
for line in cfg:
    m = re.match(r"\s*-\s+([^:]+@[^:]+):(\d+):\s*(\S+)", line)
    if m:
        print(m.group(1))
        print(m.group(2))
        raise SystemExit(0)
raise SystemExit("Unable to find run_farm_hosts_to_use entry in config_runtime.yaml")
PY
}

screen_quote() {
    python3 - "$1" <<'PY'
import sys

payload = sys.argv[1] + "\n"
parts = ["$'"]
for ch in payload:
    o = ord(ch)
    if ch == "\\":
        parts.append("\\\\")
    elif ch == "'":
        parts.append("\\'")
    elif ch == "\n":
        parts.append("\\n")
    elif 32 <= o < 127:
        parts.append(ch)
    else:
        parts.append(f"\\x{o:02x}")
parts.append("'")
print("".join(parts))
PY
}

latest_uartlog() {
    find "${RESULTS_ROOT}" -path '*/llama-firesim0/uartlog' -print0 2>/dev/null | \
        xargs -0 ls -1t 2>/dev/null | head -n 1
}

send_line() {
    local line="$1"
    local quoted
    quoted=$(screen_quote "${line}")
    ssh -p "${PORT}" "${HOST}" "screen -S ${SESSION_NAME} -X stuff ${quoted}"
}

if ! [[ "${N_PREDICT}" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 [N_PREDICT]" >&2
    exit 1
fi

mapfile -t HOST_INFO < <(read_host_spec "${RUNTIME_CFG}")
HOST="${HOST_INFO[0]}"
PORT="${HOST_INFO[1]}"

load_ssh_agent
UARTLOG=$(latest_uartlog)
if [[ -z "${UARTLOG}" ]]; then
    echo "Unable to locate a running llama-firesim uartlog under ${RESULTS_ROOT}" >&2
    exit 1
fi

send_line ":arrival-benchmark ${N_PREDICT}"

until grep -q "ARRIVAL-BENCHMARK-DONE" "${UARTLOG}"; do
    sleep 5
done

send_line ":quit"

until grep -q "firemarshal workload run/command done" "${UARTLOG}"; do
    sleep 5
done

RESULT_DIR=$(dirname "${UARTLOG}")
echo "Arrival benchmark completed."
echo "Artifacts:"
echo "  ${RESULT_DIR}/arrival_request_summary.csv"
echo "  ${RESULT_DIR}/arrival_mode_summary.csv"
echo "  ${RESULT_DIR}/arrival_stage_summary.csv"
echo "  ${RESULT_DIR}/op_profile.csv"
echo "  ${RESULT_DIR}/stage_summary.csv"
echo "  ${RESULT_DIR}/backend_summary.csv"
echo "  ${RESULT_DIR}/summary.md"
