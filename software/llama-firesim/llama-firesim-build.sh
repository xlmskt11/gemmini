#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
MARSHAL_DIR="${REPO_ROOT}/software/firemarshal"
WORKLOAD_CFG="${SCRIPT_DIR}/llama-firesim.yaml"
CONDA_ENV_DIR="${REPO_ROOT}/.conda-env"
RISCV_TOOLS_DIR="${CONDA_ENV_DIR}/riscv-tools/bin"

setup_build_env() {
    if [[ -d "${CONDA_ENV_DIR}/bin" ]]; then
        export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
    fi

    if [[ -d "${RISCV_TOOLS_DIR}" ]]; then
        export PATH="${RISCV_TOOLS_DIR}:${PATH}"
    fi

    if ! command -v riscv64-unknown-linux-gnu-gcc >/dev/null 2>&1; then
        echo "Unable to locate riscv64-unknown-linux-gnu-gcc." >&2
        echo "Expected it in PATH or at ${RISCV_TOOLS_DIR}." >&2
        echo "Run from a Chipyard environment with '. ./env.sh' or install riscv-tools." >&2
        exit 1
    fi
}

setup_build_env
cd "${MARSHAL_DIR}"
./marshal build "${WORKLOAD_CFG}"
./marshal install "${WORKLOAD_CFG}"

echo "Installed FireMarshal workload and FireSim JSON for llama-firesim."
echo "Set sims/firesim/deploy/config_runtime.yaml workload_name to llama-firesim.json before runworkload."
