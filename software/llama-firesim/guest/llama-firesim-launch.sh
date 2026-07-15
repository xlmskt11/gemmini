#!/usr/bin/env bash

set -euo pipefail

DEFAULTS_FILE="/root/llama-firesim/llama-firesim-defaults.env"
if [[ -f "${DEFAULTS_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${DEFAULTS_FILE}"
fi

export GGML_GEMMINI_ACTIVE_MASK="${GGML_GEMMINI_ACTIVE_MASK:-0xf}"
export GGML_GEMMINI_MAX_OFFLOADS="${GGML_GEMMINI_MAX_OFFLOADS:--1}"
export GGML_GEMMINI_PREPACK_WEIGHTS="${GGML_GEMMINI_PREPACK_WEIGHTS:-1}"
export GGML_GEMMINI_TRACE="${GGML_GEMMINI_TRACE:-0}"
export GGML_GEMMINI_PREFAULT="${GGML_GEMMINI_PREFAULT:-0}"
export GGML_GEMMINI_WEIGHT_PACK="${GGML_GEMMINI_WEIGHT_PACK:-/root/models/model.gguf.gemmini-pack}"
export GGML_GEMMINI_PAGE_PACKED_A="${GGML_GEMMINI_PAGE_PACKED_A:-0}"
export GGML_GEMMINI_PAGE_PACKED_B="${GGML_GEMMINI_PAGE_PACKED_B:-0}"
export GGML_GEMMINI_PAGE_PACKED_C="${GGML_GEMMINI_PAGE_PACKED_C:-0}"
export GGML_GEMMINI_PAGE_PACKED_D="${GGML_GEMMINI_PAGE_PACKED_D:-0}"

BACKEND="${LLAMA_FIRESIM_BACKEND:-gemmini}"
CTX_SIZE="${LLAMA_FIRESIM_CTX_SIZE:-2048}"
N_PREDICT="${LLAMA_FIRESIM_N_PREDICT:-16}"
RESULTS_DIR="${LLAMA_FIRESIM_RESULTS_DIR:-/root/llama-results}"
MODE="${LLAMA_FIRESIM_MODE:-repl}"

mkdir -p "${RESULTS_DIR}"

case "${MODE}" in
    repl|interactive)
        exec /root/llama-firesim/llama-firesim-cli \
            --backend "${BACKEND}" \
            --ctx-size "${CTX_SIZE}" \
            --n-predict "${N_PREDICT}" \
            --results-dir "${RESULTS_DIR}" \
            --interactive
        ;;
    sweep)
        exec /root/llama-firesim/llama-firesim-sweep.sh
        ;;
    *)
        echo "unknown LLAMA_FIRESIM_MODE=${MODE}" >&2
        echo "valid values: repl, sweep" >&2
        exit 1
        ;;
esac
