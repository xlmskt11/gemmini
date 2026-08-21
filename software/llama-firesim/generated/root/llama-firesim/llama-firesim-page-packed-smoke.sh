#!/usr/bin/env bash

set -euo pipefail

DEFAULTS_FILE="/root/llama-firesim/llama-firesim-defaults.env"
if [[ -f "${DEFAULTS_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${DEFAULTS_FILE}"
fi

BIN="${LLAMA_FIRESIM_BIN:-/root/llama-firesim/llama-firesim-cli}"
ROWS="${LLAMA_FIRESIM_SMOKE_M:-35}"
COLS_OUT="${LLAMA_FIRESIM_SMOKE_N:-67}"
COLS_IN="${LLAMA_FIRESIM_SMOKE_K:-83}"
PROFILE="${LLAMA_FIRESIM_HW_PROFILE:-}"

case "${PROFILE}" in
    single|multi)
        ;;
    *)
        echo "LLAMA_FIRESIM_HW_PROFILE must be single or multi for page-packed smoke." >&2
        exit 1
        ;;
esac

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

run_boundary_case() {
    local rows="$1"
    local cols_in="$2"
    local expected_a="$3"
    local expected_b="$4"
    local expected_c="$5"
    local expected_d="$6"
    local cols_out=3
    local output

    echo "GEMMINI-PACKING-BOUNDARY,mode=${PROFILE},shape=${rows}x${cols_out}x${cols_in},expected=${expected_a}${expected_b}${expected_c}${expected_d}"
    output=$(
        GGML_GEMMINI_PAGE_PACKED_A=1 \
        GGML_GEMMINI_PAGE_PACKED_B=1 \
        GGML_GEMMINI_PAGE_PACKED_C=1 \
        GGML_GEMMINI_PAGE_PACKED_D=1 \
            "${BIN}" --gemmini-mode "${PROFILE}" --gemmini-smoke "${rows}" "${cols_out}" "${cols_in}"
    )
    printf '%s\n' "${output}"

    local expected="page_packed_a=${expected_a},page_packed_b=${expected_b},page_packed_c=${expected_c},page_packed_d=${expected_d}"
    if ! grep -Fq "${expected}" <<<"${output}"; then
        echo "page-packing boundary policy mismatch: expected ${expected}" >&2
        return 1
    fi
}

for abc in 0 1 2 3 4 5 6 7; do
    run_case "${PROFILE}" "$((abc & 1))" "$(((abc >> 1) & 1))" "$(((abc >> 2) & 1))" 0
done

run_case "${PROFILE}" 1 1 1 1

# Options remain enabled, but M <= 1 disables A/C/D and K <= 1 disables B.
run_boundary_case 1 1 0 0 0 0
run_boundary_case 1 2 0 1 0 0
run_boundary_case 2 1 1 0 1 1
run_boundary_case 2 2 1 1 1 1

echo "GEMMINI-PACKING-SMOKE-PASS"
