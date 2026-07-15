#!/usr/bin/env bash

set -euo pipefail

export GGML_GEMMINI_ACTIVE_MASK="${GGML_GEMMINI_ACTIVE_MASK:-0xf}"
export GGML_GEMMINI_MAX_OFFLOADS="${GGML_GEMMINI_MAX_OFFLOADS:--1}"
export GGML_GEMMINI_PREPACK_WEIGHTS="${GGML_GEMMINI_PREPACK_WEIGHTS:-1}"
export GGML_GEMMINI_TRACE="${GGML_GEMMINI_TRACE:-0}"

CTX_SIZE="${LLAMA_FIRESIM_CTX_SIZE:-2048}"
N_PREDICT="${LLAMA_FIRESIM_N_PREDICT:-16}"
RESULTS_DIR="${LLAMA_FIRESIM_RESULTS_DIR:-/root/llama-results}"
SWEEP_PROMPT_MIN="${LLAMA_FIRESIM_SWEEP_PROMPT_MIN:-64}"
SWEEP_PROMPT_MAX="${LLAMA_FIRESIM_SWEEP_PROMPT_MAX:-256}"
SWEEP_PROMPT_STEP="${LLAMA_FIRESIM_SWEEP_PROMPT_STEP:-16}"
SWEEP_MASKS="${LLAMA_FIRESIM_SWEEP_MASKS:-1,3,7,15}"

mkdir -p "${RESULTS_DIR}"

exec /root/llama-firesim/llama-firesim-cli \
    --backend gemmini \
    --ctx-size "${CTX_SIZE}" \
    --n-predict "${N_PREDICT}" \
    --results-dir "${RESULTS_DIR}" \
    --sweep \
    --sweep-prompt-min "${SWEEP_PROMPT_MIN}" \
    --sweep-prompt-max "${SWEEP_PROMPT_MAX}" \
    --sweep-prompt-step "${SWEEP_PROMPT_STEP}" \
    --sweep-masks "${SWEEP_MASKS}"
