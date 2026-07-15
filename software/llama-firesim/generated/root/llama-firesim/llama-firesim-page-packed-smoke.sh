#!/usr/bin/env bash

set -euo pipefail

BIN="${LLAMA_FIRESIM_BIN:-/root/llama-firesim/llama-firesim-cli}"
ROWS="${LLAMA_FIRESIM_SMOKE_M:-35}"
COLS_OUT="${LLAMA_FIRESIM_SMOKE_N:-67}"
COLS_IN="${LLAMA_FIRESIM_SMOKE_K:-83}"

export GGML_GEMMINI_ACTIVE_MASK="${GGML_GEMMINI_ACTIVE_MASK:-0xf}"

run_case() {
    local mode="$1"
    local a="$2"
    local b="$3"
    local c="$4"
    local d="$5"

    echo "GEMMINI-PACKING-SMOKE-CASE,mode=${mode},a=${a},b=${b},c=${c},d=${d},shape=${ROWS}x${COLS_OUT}x${COLS_IN}"
    GGML_GEMMINI_PAGE_PACKED_A="${a}" \
    GGML_GEMMINI_PAGE_PACKED_B="${b}" \
    GGML_GEMMINI_PAGE_PACKED_C="${c}" \
    GGML_GEMMINI_PAGE_PACKED_D="${d}" \
        "${BIN}" --gemmini-mode "${mode}" --gemmini-smoke "${ROWS}" "${COLS_OUT}" "${COLS_IN}"
}

for mode in single multi; do
    for abc in 0 1 2 3 4 5 6 7; do
        run_case "${mode}" "$((abc & 1))" "$(((abc >> 1) & 1))" "$(((abc >> 2) & 1))" 0
    done

    run_case "${mode}" 1 1 1 1
done

echo "GEMMINI-PACKING-SMOKE-PASS"
