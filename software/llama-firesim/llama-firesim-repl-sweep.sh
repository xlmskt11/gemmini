#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
RUNTIME_CFG="${REPO_ROOT}/sims/firesim/deploy/config_runtime.yaml"
RESULTS_ROOT="${REPO_ROOT}/sims/firesim/deploy/results-workload"
IMAGE_PROFILE_STAMP="${REPO_ROOT}/software/firemarshal/images/firechip/llama-firesim/llama-firesim.img.hw-profile"
SESSION_NAME="${SCREEN_NAME:-${FIRESIM_SCREEN_SESSION:-fsim0}}"
SSH_AGENT_VARS="${HOME}/.ssh/AGENT_VARS"
UARTLOG_OVERRIDE="${LLAMA_FIRESIM_UARTLOG:-}"
REMOTE_UARTLOG="${LLAMA_FIRESIM_REMOTE_UARTLOG:-}"
# U280 Server
# REMOTE_SIM_DIR="${REMOTE_SIM_DIR:-/data/dwc06209/FIRESIM_RUNS_DIR}"
# REMOTE_SSH_OVERRIDE="${REMOTE_SSH:-dwc06209@165.132.140.144}"
# REMOTE_PORT_OVERRIDE="${REMOTE_PORT:-4121}"
# U250 Server
REMOTE_SIM_DIR="${REMOTE_SIM_DIR:-/home/dwc06209/FIRESIM_RUNS_DIR}"
REMOTE_SSH_OVERRIDE="${REMOTE_SSH:-dwc06209@165.132.142.249}"
REMOTE_PORT_OVERRIDE="${REMOTE_PORT:-22}"

PROMPT_TOKENS_CSV="${LLAMA_FIRESIM_REPL_SWEEP_PROMPT_TOKENS:-}"
PROMPT_MIN="${LLAMA_FIRESIM_REPL_SWEEP_PROMPT_MIN:-64}"
PROMPT_MAX="${LLAMA_FIRESIM_REPL_SWEEP_PROMPT_MAX:-256}"
PROMPT_STEP="${LLAMA_FIRESIM_REPL_SWEEP_PROMPT_STEP:-16}"
DECODE_TOKENS_CSV="${LLAMA_FIRESIM_REPL_SWEEP_DECODE_TOKENS:-${LLAMA_FIRESIM_N_PREDICT:-16}}"
HW_PROFILE="${LLAMA_FIRESIM_HW_PROFILE:-}"
if [[ -z "${HW_PROFILE}" && -f "${IMAGE_PROFILE_STAMP}" ]]; then
    HW_PROFILE=$(tr -d '[:space:]' < "${IMAGE_PROFILE_STAMP}")
fi
case "${HW_PROFILE}" in
    single)
        DEFAULT_MASKS_CSV="1"
        ;;
    multi)
        DEFAULT_MASKS_CSV="1,3,7,15"
        ;;
    *)
        echo "LLAMA_FIRESIM_HW_PROFILE must be single or multi (or build a stamped image first)." >&2
        exit 1
        ;;
esac
MASKS_CSV="${LLAMA_FIRESIM_REPL_SWEEP_MASKS:-${DEFAULT_MASKS_CSV}}"
MODES_CSV="${LLAMA_FIRESIM_REPL_SWEEP_MODES:-${HW_PROFILE}}"
POLL_SECONDS="${LLAMA_FIRESIM_REPL_SWEEP_POLL_SECONDS:-5}"
USE_TAGS="${LLAMA_FIRESIM_REPL_SWEEP_USE_TAGS:-0}"
KEEP_OPEN=0

usage() {
    cat >&2 <<EOF
usage: $0 [options]

Options:
  --prompt-tokens LIST   comma-separated exact prompt token counts
  --prompt-min N         first prompt token count when LIST is omitted
  --prompt-max N         last prompt token count when LIST is omitted
  --prompt-step N        prompt token count step when LIST is omitted
  --decode-tokens LIST   comma-separated decode token counts
  --masks LIST           comma-separated logical Gemmini masks
  --modes LIST           selected compiled profile only: ${HW_PROFILE}
  --uartlog PATH         explicit local uartlog to monitor
  --remote-uartlog PATH  explicit remote uartlog to monitor through SSH
  --use-tags             send per-case tags, requires a guest with TAG-aware :prompt-tokens
  --keep-open            leave the guest REPL open after the sweep

Environment defaults:
  LLAMA_FIRESIM_REPL_SWEEP_PROMPT_TOKENS
  LLAMA_FIRESIM_REPL_SWEEP_PROMPT_MIN=${PROMPT_MIN}
  LLAMA_FIRESIM_REPL_SWEEP_PROMPT_MAX=${PROMPT_MAX}
  LLAMA_FIRESIM_REPL_SWEEP_PROMPT_STEP=${PROMPT_STEP}
  LLAMA_FIRESIM_REPL_SWEEP_DECODE_TOKENS=${DECODE_TOKENS_CSV}
  LLAMA_FIRESIM_REPL_SWEEP_MASKS=${MASKS_CSV}
  LLAMA_FIRESIM_REPL_SWEEP_MODES=${MODES_CSV}
  LLAMA_FIRESIM_REPL_SWEEP_USE_TAGS=${USE_TAGS}
  REMOTE_SSH
  REMOTE_PORT
  REMOTE_SIM_DIR
  SCREEN_NAME
  LLAMA_FIRESIM_UARTLOG
  LLAMA_FIRESIM_REMOTE_UARTLOG
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-tokens)
            PROMPT_TOKENS_CSV="${2:?missing value for --prompt-tokens}"
            shift 2
            ;;
        --prompt-min)
            PROMPT_MIN="${2:?missing value for --prompt-min}"
            shift 2
            ;;
        --prompt-max)
            PROMPT_MAX="${2:?missing value for --prompt-max}"
            shift 2
            ;;
        --prompt-step)
            PROMPT_STEP="${2:?missing value for --prompt-step}"
            shift 2
            ;;
        --decode-tokens)
            DECODE_TOKENS_CSV="${2:?missing value for --decode-tokens}"
            shift 2
            ;;
        --masks)
            MASKS_CSV="${2:?missing value for --masks}"
            shift 2
            ;;
        --modes)
            MODES_CSV="${2:?missing value for --modes}"
            shift 2
            ;;
        --uartlog)
            UARTLOG_OVERRIDE="${2:?missing value for --uartlog}"
            shift 2
            ;;
        --remote-uartlog)
            REMOTE_UARTLOG="${2:?missing value for --remote-uartlog}"
            shift 2
            ;;
        --use-tags)
            USE_TAGS=1
            shift
            ;;
        --keep-open)
            KEEP_OPEN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

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

shell_quote() {
    printf '%q' "$1"
}

latest_uartlog() {
    find "${RESULTS_ROOT}" -path '*/llama-firesim0/uartlog' -print0 2>/dev/null | \
        xargs -0 ls -1t 2>/dev/null | head -n 1
}

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -p "${PORT}" "${HOST}" "$@"
}

refresh_uartlog() {
    if [[ -n "${REMOTE_UARTLOG}" ]]; then
        return
    fi

    if [[ -n "${UARTLOG_OVERRIDE}" ]]; then
        UARTLOG="${UARTLOG_OVERRIDE}"
        return
    fi

    local candidate
    candidate=$(latest_uartlog)
    if [[ -n "${candidate}" && "${candidate}" != "${UARTLOG:-}" ]]; then
        if [[ -n "${UARTLOG:-}" ]]; then
            echo "REPL-SWEEP-UARTLOG-SWITCH,old=${UARTLOG},new=${candidate}" >&2
        fi
        UARTLOG="${candidate}"
    fi
}

send_line() {
    local line="$1"
    local quoted
    quoted=$(screen_quote "${line}")
    ssh_cmd "screen -S $(shell_quote "${SESSION_NAME}") -X stuff ${quoted}"
}

split_csv() {
    local csv="$1"
    local item
    IFS=',' read -ra items <<< "${csv}"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        if [[ -n "${item}" ]]; then
            printf '%s\n' "${item}"
        fi
    done
}

is_decimal() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_mask() {
    [[ "$1" =~ ^(0[xX][0-9a-fA-F]+|[0-9]+)$ ]]
}

build_prompt_tokens() {
    if [[ -n "${PROMPT_TOKENS_CSV}" ]]; then
        split_csv "${PROMPT_TOKENS_CSV}"
        return
    fi

    if ! is_decimal "${PROMPT_MIN}" || ! is_decimal "${PROMPT_MAX}" || ! is_decimal "${PROMPT_STEP}"; then
        echo "prompt min/max/step must be decimal integers" >&2
        exit 1
    fi
    if (( PROMPT_MIN <= 0 || PROMPT_MAX < PROMPT_MIN || PROMPT_STEP <= 0 )); then
        echo "invalid prompt token range" >&2
        exit 1
    fi

    local value
    for ((value = PROMPT_MIN; value <= PROMPT_MAX; value += PROMPT_STEP)); do
        printf '%s\n' "${value}"
    done
}

current_log_offset() {
    if [[ -n "${REMOTE_UARTLOG}" ]]; then
        ssh_cmd "test -f $(shell_quote "${REMOTE_UARTLOG}") && wc -c < $(shell_quote "${REMOTE_UARTLOG}") || echo 0" | tr -d ' '
    else
        wc -c < "${UARTLOG}" | tr -d ' '
    fi
}

log_contains() {
    local pattern="$1"

    if [[ -n "${REMOTE_UARTLOG}" ]]; then
        ssh_cmd "test -f $(shell_quote "${REMOTE_UARTLOG}") && grep -aqF -- $(shell_quote "${pattern}") $(shell_quote "${REMOTE_UARTLOG}")"
    else
        [[ -n "${UARTLOG:-}" && -f "${UARTLOG}" ]] && grep -qF -- "${pattern}" "${UARTLOG}"
    fi
}

wait_for_ready() {
    while true; do
        refresh_uartlog
        if log_contains "llama-firesim ready"; then
            return
        fi
        sleep "${POLL_SECONDS}"
    done
}

verify_running_profile() {
    local expected="GEMMINI-HW-PROFILE,name=${HW_PROFILE},"

    if log_contains "${expected}"; then
        echo "REPL-SWEEP-HW-PROFILE,name=${HW_PROFILE}"
        return
    fi

    echo "Running guest did not report the selected ${HW_PROFILE} hardware profile." >&2
    echo "Expected UART marker: ${expected}" >&2
    exit 1
}

case_done_seen() {
    local chunk="$1"
    local prompt_tokens="$2"
    local decode_tokens="$3"
    local tag="$4"

    if [[ "${USE_TAGS}" == "1" ]] &&
            grep -F "PROMPT-TOKENS-DONE," <<< "${chunk}" | \
            grep -Fq "tag=${tag},prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}"; then
        return 0
    fi

    grep -q "PROMPT-TOKENS-DONE,.*prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}" <<< "${chunk}"
}

case_aborted_seen() {
    local chunk="$1"
    local prompt_tokens="$2"
    local decode_tokens="$3"
    local tag="$4"

    if [[ "${USE_TAGS}" == "1" ]] &&
            grep -F "PROMPT-TOKENS-ABORTED," <<< "${chunk}" | \
            grep -Fq "tag=${tag},prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}"; then
        return 0
    fi

    grep -q "PROMPT-TOKENS-ABORTED,.*prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}" <<< "${chunk}"
}

wait_for_case_end() {
    local offset="$1"
    local prompt_tokens="$2"
    local decode_tokens="$3"
    local tag="$4"
    local chunk

    while true; do
        if [[ -n "${REMOTE_UARTLOG}" ]]; then
            chunk=$(ssh_cmd "test -f $(shell_quote "${REMOTE_UARTLOG}") && tail -c +$((offset + 1)) $(shell_quote "${REMOTE_UARTLOG}") || true")
        else
            local old_uartlog="${UARTLOG:-}"
            refresh_uartlog
            if [[ "${UARTLOG:-}" != "${old_uartlog}" ]]; then
                offset=0
            fi
            chunk=$(tail -c +"$((offset + 1))" "${UARTLOG}" 2>/dev/null || true)
        fi
        if case_done_seen "${chunk}" "${prompt_tokens}" "${decode_tokens}" "${tag}"; then
            return 0
        fi
        if case_aborted_seen "${chunk}" "${prompt_tokens}" "${decode_tokens}" "${tag}"; then
            return 2
        fi
        if grep -q "PROMPT-TOKENS-ERROR" <<< "${chunk}"; then
            grep "PROMPT-TOKENS-ERROR" <<< "${chunk}" | tail -n 1 >&2
            return 1
        fi
        sleep "${POLL_SECONDS}"
    done
}

mapfile -t PROMPT_TOKENS < <(build_prompt_tokens)
mapfile -t DECODE_TOKENS < <(split_csv "${DECODE_TOKENS_CSV}")
mapfile -t MASKS < <(split_csv "${MASKS_CSV}")
mapfile -t MODES < <(split_csv "${MODES_CSV}")

if [[ "${#PROMPT_TOKENS[@]}" -eq 0 || "${#DECODE_TOKENS[@]}" -eq 0 || "${#MASKS[@]}" -eq 0 || "${#MODES[@]}" -eq 0 ]]; then
    echo "prompt tokens, decode tokens, masks, and modes must not be empty" >&2
    exit 1
fi

for value in "${PROMPT_TOKENS[@]}" "${DECODE_TOKENS[@]}"; do
    if ! is_decimal "${value}"; then
        echo "prompt and decode token counts must be decimal integers: ${value}" >&2
        exit 1
    fi
done

for value in "${MASKS[@]}"; do
    if ! is_mask "${value}"; then
        echo "masks must be decimal or hex integers: ${value}" >&2
        exit 1
    fi
done

for mode in "${MODES[@]}"; do
    if [[ "${mode}" != "multi" && "${mode}" != "single" ]]; then
        echo "modes must be multi or single: ${mode}" >&2
        exit 1
    fi
    if [[ "${mode}" != "${HW_PROFILE}" ]]; then
        echo "requested mode ${mode} does not match the built ${HW_PROFILE} profile" >&2
        exit 1
    fi
done

for value in "${MASKS[@]}"; do
    if [[ "${value}" == 0[xX]* ]]; then
        numeric_mask=$((16#${value:2}))
    else
        numeric_mask=$((10#${value}))
    fi
    if (( numeric_mask < 0 || numeric_mask > 15 )); then
        echo "logical masks must be in [0, 0xf]: ${value}" >&2
        exit 1
    fi
    if [[ "${HW_PROFILE}" == "single" && "${numeric_mask}" -gt 1 ]]; then
        echo "single profile accepts only logical masks 0x0 and 0x1: ${value}" >&2
        exit 1
    fi
done

echo "REPL-SWEEP-PLAN,modes=$(IFS=,; echo "${MODES[*]}"),masks=$(IFS=,; echo "${MASKS[*]}"),prompt_tokens=$(IFS=,; echo "${PROMPT_TOKENS[*]}"),decode_tokens=$(IFS=,; echo "${DECODE_TOKENS[*]}"),use_tags=${USE_TAGS}"

mapfile -t HOST_INFO < <(read_host_spec "${RUNTIME_CFG}")
HOST="${HOST_INFO[0]}"
PORT="${HOST_INFO[1]}"
if [[ -n "${REMOTE_SSH_OVERRIDE}" ]]; then
    HOST="${REMOTE_SSH_OVERRIDE}"
fi
if [[ -n "${REMOTE_PORT_OVERRIDE}" ]]; then
    PORT="${REMOTE_PORT_OVERRIDE}"
fi

load_ssh_agent
UARTLOG=""
if [[ -z "${REMOTE_UARTLOG}" && -z "${UARTLOG_OVERRIDE}" && -n "${REMOTE_SIM_DIR}" ]]; then
    REMOTE_UARTLOG="${REMOTE_SIM_DIR%/}/sim_slot_0/uartlog"
fi
if [[ -z "${REMOTE_UARTLOG}" ]]; then
    refresh_uartlog
    if [[ -z "${UARTLOG:-}" ]]; then
        echo "Unable to locate a running llama-firesim uartlog under ${RESULTS_ROOT}" >&2
        echo "If the uartlog is only on the remote host, pass --remote-uartlog PATH." >&2
        exit 1
    fi
else
    echo "REPL-SWEEP-REMOTE,host=${HOST},port=${PORT},screen=${SESSION_NAME},uartlog=${REMOTE_UARTLOG}"
fi

wait_for_ready
verify_running_profile

send_line ":backend gemmini"

case_count=0
for mode in "${MODES[@]}"; do
    send_line ":gemmini-mode ${mode}"

    for mask in "${MASKS[@]}"; do
        send_line ":active-mask ${mask}"

        for decode_tokens in "${DECODE_TOKENS[@]}"; do
            send_line ":n-predict ${decode_tokens}"

            for prompt_tokens in "${PROMPT_TOKENS[@]}"; do
                case_tag="case$((case_count + 1))_${mode}_mask${mask}_ptok${prompt_tokens}_dtok${decode_tokens}"
                case_tag="${case_tag//[^A-Za-z0-9_.-]/_}"
                offset=$(current_log_offset)
                echo "REPL-SWEEP-CASE-START,tag=${case_tag},mode=${mode},mask=${mask},prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}"
                if [[ "${USE_TAGS}" == "1" ]]; then
                    send_line ":prompt-tokens ${prompt_tokens} ${decode_tokens} ${case_tag}"
                else
                    send_line ":prompt-tokens ${prompt_tokens} ${decode_tokens}"
                fi
                if ! wait_for_case_end "${offset}" "${prompt_tokens}" "${decode_tokens}" "${case_tag}"; then
                    echo "REPL-SWEEP-CASE-FAILED,tag=${case_tag},mode=${mode},mask=${mask},prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}" >&2
                    exit 1
                fi
                case_count=$((case_count + 1))
                echo "REPL-SWEEP-CASE-DONE,tag=${case_tag},mode=${mode},mask=${mask},prompt_tokens=${prompt_tokens},n_predict=${decode_tokens}"
            done
        done
    done
done

if [[ "${KEEP_OPEN}" -eq 0 ]]; then
    send_line ":quit"
    until log_contains "firemarshal workload run/command done"; do
        sleep "${POLL_SECONDS}"
    done
fi

if [[ -n "${REMOTE_UARTLOG}" ]]; then
    RESULT_DIR="$(dirname "${REMOTE_UARTLOG}") on ${HOST}"
else
    RESULT_DIR=$(dirname "${UARTLOG}")
fi
echo "REPL sweep completed, cases=${case_count}."
echo "Artifacts:"
echo "  ${RESULT_DIR}/run_summary.csv"
echo "  ${RESULT_DIR}/token_trace.csv"
echo "  ${RESULT_DIR}/op_profile.csv"
echo "  ${RESULT_DIR}/stage_summary.csv"
echo "  ${RESULT_DIR}/matmul_summary.csv"
echo "  ${RESULT_DIR}/backend_summary.csv"
echo "  ${RESULT_DIR}/summary.md"
