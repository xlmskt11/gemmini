#!/usr/bin/env bash

set -euo pipefail

DEFAULTS_FILE="/root/llama-firesim/llama-firesim-defaults.env"
if [[ -f "${DEFAULTS_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${DEFAULTS_FILE}"
fi

export GGML_GEMMINI_MAX_OFFLOADS="${GGML_GEMMINI_MAX_OFFLOADS:--1}"
export GGML_GEMMINI_PREPACK_WEIGHTS="${GGML_GEMMINI_PREPACK_WEIGHTS:-1}"
export GGML_GEMMINI_TRACE="${GGML_GEMMINI_TRACE:-0}"

CTX_SIZE="${LLAMA_FIRESIM_CTX_SIZE:-2048}"
N_PREDICT="${LLAMA_FIRESIM_N_PREDICT:-16}"
RESULTS_DIR="${LLAMA_FIRESIM_RESULTS_DIR:-/root/llama-results}"
SWEEP_PROMPT_MIN="${LLAMA_FIRESIM_SWEEP_PROMPT_MIN:-64}"
SWEEP_PROMPT_MAX="${LLAMA_FIRESIM_SWEEP_PROMPT_MAX:-256}"
SWEEP_PROMPT_STEP="${LLAMA_FIRESIM_SWEEP_PROMPT_STEP:-16}"
SWEEP_MASKS="${LLAMA_FIRESIM_SWEEP_MASKS:-}"

mkdir -p "${RESULTS_DIR}"

ARGS=(
    --backend gemmini \
    --ctx-size "${CTX_SIZE}" \
    --n-predict "${N_PREDICT}" \
    --results-dir "${RESULTS_DIR}" \
    --sweep \
    --sweep-prompt-min "${SWEEP_PROMPT_MIN}" \
    --sweep-prompt-max "${SWEEP_PROMPT_MAX}" \
    --sweep-prompt-step "${SWEEP_PROMPT_STEP}"
)
if [[ -n "${SWEEP_MASKS}" ]]; then
    ARGS+=(--sweep-masks "${SWEEP_MASKS}")
fi

exec /root/llama-firesim/llama-firesim-cli "${ARGS[@]}"
