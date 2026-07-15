#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
RUNTIME_CFG="${1:-${REPO_ROOT}/sims/firesim/deploy/config_runtime.yaml}"
SESSION_NAME="${FIRESIM_SCREEN_SESSION:-fsim0}"
SSH_AGENT_VARS="${HOME}/.ssh/AGENT_VARS"

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

mapfile -t HOST_INFO < <(read_host_spec "${RUNTIME_CFG}")
HOST="${HOST_INFO[0]}"
PORT="${HOST_INFO[1]}"

load_ssh_agent
exec ssh -t -p "${PORT}" "${HOST}" "screen -r ${SESSION_NAME}"
