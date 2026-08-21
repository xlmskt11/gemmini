#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
MARSHAL_DIR="${REPO_ROOT}/software/firemarshal"
WORKLOAD_CFG="${SCRIPT_DIR}/llama-firesim.yaml"
CONDA_ENV_DIR="${REPO_ROOT}/.conda-env"
RISCV_TOOLS_DIR="${CONDA_ENV_DIR}/riscv-tools/bin"
IMAGE_PATH="${MARSHAL_DIR}/images/firechip/llama-firesim/llama-firesim.img"
IMAGE_PROFILE_STAMP="${IMAGE_PATH}.hw-profile"
IMAGE_ABI_STAMP="${IMAGE_PATH}.hw-abi-sha256"
IMAGE_ARTIFACT_FORMAT_STAMP="${IMAGE_PATH}.artifact-format"
IMAGE_PAGE_PACKING_CONFIG_STAMP="${IMAGE_PATH}.page-packing-config"
WEIGHT_PACK_SOURCE="${SCRIPT_DIR}/generated/root/models/model.gguf.gemmini-pack"
WEIGHT_PACK_GUEST_PATH="/root/models/model.gguf.gemmini-pack"
MODEL_SOURCE="${SCRIPT_DIR}/generated/root/models/model.gguf"
MODEL_GUEST_PATH="/root/models/model.gguf"
GENERATED_PROFILE_STAMP="${SCRIPT_DIR}/generated/root/llama-firesim/llama-firesim-hw-profile.txt"
GENERATED_ABI_STAMP="${SCRIPT_DIR}/generated/root/llama-firesim/llama-firesim-hw-abi-sha256.txt"
GENERATED_ARTIFACT_FORMAT_STAMP="${SCRIPT_DIR}/generated/root/llama-firesim/llama-firesim-artifact-format.txt"
GENERATED_PAGE_PACKING_CONFIG_STAMP="${SCRIPT_DIR}/generated/root/llama-firesim/llama-firesim-page-packing-config.txt"
GEMMINI_SW_DIR="${REPO_ROOT}/sims/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests"
source "${SCRIPT_DIR}/llama-firesim-build-profile.sh"

llama_firesim_require_hw_profile
HW_PROFILE="${LLAMA_FIRESIM_SELECTED_PROFILE}"
llama_firesim_read_sw_abi "${GEMMINI_SW_DIR}"
SW_ABI_SHA256="${LLAMA_FIRESIM_SW_ABI_SHA256}"
export LLAMA_FIRESIM_HW_PROFILE="${HW_PROFILE}"
EXPECTED_ARTIFACT_FORMAT="gemmini-pack-v4-hybrid-${LLAMA_FIRESIM_HYBRID_SPARSE_GGUF:-1}-page-b-${GGML_GEMMINI_PAGE_PACKED_B:-1}"
EXPECTED_HYBRID_SPARSE_GGUF="${LLAMA_FIRESIM_HYBRID_SPARSE_GGUF:-1}"
EXPECTED_PAGE_PACKED_B="${GGML_GEMMINI_PAGE_PACKED_B:-1}"
EXPECTED_PAGE_PACKING_CONFIG="gemmini-a${GGML_GEMMINI_PAGE_PACKED_A:-0}-b${GGML_GEMMINI_PAGE_PACKED_B:-1}-c${GGML_GEMMINI_PAGE_PACKED_C:-0}-d${GGML_GEMMINI_PAGE_PACKED_D:-0}-flash-q${Flash_Q_PAGE_PACKED:-${FLASH_Q_PAGE_PACKED:-0}}-k${Flash_K_PAGE_PACKED:-${FLASH_K_PAGE_PACKED:-1}}-v${Flash_V_PAGE_PACKED:-${FLASH_V_PAGE_PACKED:-1}}"

read_profile_stamp() {
    local stamp_path=$1

    if [[ -f "${stamp_path}" ]]; then
        tr -d '[:space:]' < "${stamp_path}"
    fi
}

validate_generated_profile() {
    local generated_profile
    local generated_abi_sha256
    local generated_artifact_format
    local generated_page_packing_config
    local require_weight_pack=${1:-0}

    generated_profile=$(read_profile_stamp "${GENERATED_PROFILE_STAMP}")
    if [[ "${generated_profile}" != "${HW_PROFILE}" ]]; then
        echo "Generated llama-firesim artifacts do not match the selected build profile." >&2
        echo "  selected build profile: ${HW_PROFILE}" >&2
        echo "  generated profile: ${generated_profile:-missing}" >&2
        return 1
    fi
    generated_abi_sha256=$(read_profile_stamp "${GENERATED_ABI_STAMP}")
    if [[ "${generated_abi_sha256}" != "${SW_ABI_SHA256}" ]]; then
        echo "Generated llama-firesim artifacts do not match the current target-design software headers." >&2
        echo "  current include SHA-256: ${SW_ABI_SHA256}" >&2
        echo "  generated include SHA-256: ${generated_abi_sha256:-missing}" >&2
        return 1
    fi
    generated_artifact_format=$(read_profile_stamp "${GENERATED_ARTIFACT_FORMAT_STAMP}")
    if [[ "${generated_artifact_format}" != "${EXPECTED_ARTIFACT_FORMAT}" ]]; then
        echo "Generated llama-firesim artifacts do not match the required hybrid pack format." >&2
        echo "  expected artifact format: ${EXPECTED_ARTIFACT_FORMAT}" >&2
        echo "  generated artifact format: ${generated_artifact_format:-missing}" >&2
        return 1
    fi
    generated_page_packing_config=$(read_profile_stamp "${GENERATED_PAGE_PACKING_CONFIG_STAMP}")
    if [[ "${generated_page_packing_config}" != "${EXPECTED_PAGE_PACKING_CONFIG}" ]]; then
        echo "Generated llama-firesim defaults do not match the requested page-packing configuration." >&2
        echo "  expected: ${EXPECTED_PAGE_PACKING_CONFIG}" >&2
        echo "  generated: ${generated_page_packing_config:-missing}" >&2
        return 1
    fi
    if [[ "${require_weight_pack}" == "1" && ! -f "${WEIGHT_PACK_SOURCE}" ]]; then
        echo "Missing selected-profile weight pack: ${WEIGHT_PACK_SOURCE}" >&2
        return 1
    fi
    if [[ "${require_weight_pack}" == "1" && ! -f "${MODEL_SOURCE}" ]]; then
        echo "Missing generated model artifact: ${MODEL_SOURCE}" >&2
        return 1
    fi
    if [[ "${require_weight_pack}" == "1" ]]; then
        python3 "${SCRIPT_DIR}/verify-hybrid-artifacts.py" \
            --model "${MODEL_SOURCE}" \
            --pack "${WEIGHT_PACK_SOURCE}" \
            --hybrid "${EXPECTED_HYBRID_SPARSE_GGUF}" \
            --page-packed-b "${EXPECTED_PAGE_PACKED_B}"
    fi
}

validate_image_profile() {
    local image_profile
    local image_abi_sha256
    local image_artifact_format
    local image_page_packing_config

    image_profile=$(read_profile_stamp "${IMAGE_PROFILE_STAMP}")
    if [[ "${image_profile}" != "${HW_PROFILE}" ]]; then
        echo "FireMarshal image does not match the selected build profile." >&2
        echo "  selected build profile: ${HW_PROFILE}" >&2
        echo "  image profile: ${image_profile:-missing}" >&2
        return 1
    fi
    image_abi_sha256=$(read_profile_stamp "${IMAGE_ABI_STAMP}")
    if [[ "${image_abi_sha256}" != "${SW_ABI_SHA256}" ]]; then
        echo "FireMarshal image does not match the current target-design software headers." >&2
        echo "  current include SHA-256: ${SW_ABI_SHA256}" >&2
        echo "  image include SHA-256: ${image_abi_sha256:-missing}" >&2
        return 1
    fi
    image_artifact_format=$(read_profile_stamp "${IMAGE_ARTIFACT_FORMAT_STAMP}")
    if [[ "${image_artifact_format}" != "${EXPECTED_ARTIFACT_FORMAT}" ]]; then
        echo "FireMarshal image does not match the required hybrid pack format." >&2
        echo "  expected artifact format: ${EXPECTED_ARTIFACT_FORMAT}" >&2
        echo "  image artifact format: ${image_artifact_format:-missing}" >&2
        return 1
    fi
    image_page_packing_config=$(read_profile_stamp "${IMAGE_PAGE_PACKING_CONFIG_STAMP}")
    if [[ "${image_page_packing_config}" != "${EXPECTED_PAGE_PACKING_CONFIG}" ]]; then
        echo "FireMarshal image does not match the requested page-packing configuration." >&2
        echo "  expected: ${EXPECTED_PAGE_PACKING_CONFIG}" >&2
        echo "  image: ${image_page_packing_config:-missing}" >&2
        return 1
    fi
}

invalidate_image_for_profile_switch() {
    local image_profile
    local image_abi_sha256
    local image_artifact_format
    local image_page_packing_config

    if [[ ! -f "${IMAGE_PATH}" ]]; then
        return
    fi

    image_profile=$(read_profile_stamp "${IMAGE_PROFILE_STAMP}")
    image_abi_sha256=$(read_profile_stamp "${IMAGE_ABI_STAMP}")
    image_artifact_format=$(read_profile_stamp "${IMAGE_ARTIFACT_FORMAT_STAMP}")
    image_page_packing_config=$(read_profile_stamp "${IMAGE_PAGE_PACKING_CONFIG_STAMP}")
    if [[ "${image_profile}" != "${HW_PROFILE}" ||
          "${image_abi_sha256}" != "${SW_ABI_SHA256}" ||
          "${image_artifact_format}" != "${EXPECTED_ARTIFACT_FORMAT}" ||
          "${image_page_packing_config}" != "${EXPECTED_PAGE_PACKING_CONFIG}" ]]; then
        echo "Removing stale FireMarshal image for a profile, ABI, artifact, or page-packing change."
        rm -f "${IMAGE_PATH}" "${IMAGE_PROFILE_STAMP}" "${IMAGE_ABI_STAMP}" \
            "${IMAGE_ARTIFACT_FORMAT_STAMP}" "${IMAGE_PAGE_PACKING_CONFIG_STAMP}"
    fi
}

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

image_file_sha256() {
    local image_path=$1
    local guest_path=$2

    debugfs -R "cat ${guest_path}" "${image_path}" 2>/dev/null | sha256sum | awk '{print $1}'
}

image_file_allocated_bytes() {
    local image_path=$1
    local guest_path=$2

    debugfs -R "stat ${guest_path}" "${image_path}" 2>/dev/null |
        awk '/Blockcount:/ { for (i = 1; i <= NF; ++i) if ($i == "Blockcount:") { printf "%.0f\n", $(i + 1) * 512; exit } }'
}

verify_weight_pack_in_image() {
    local source_sha
    local image_sha
    local model_source_sha
    local model_image_sha
    local model_source_size
    local model_source_allocated
    local model_image_allocated

    source_sha=$(sha256sum "${WEIGHT_PACK_SOURCE}" | awk '{print $1}')
    image_sha=$(image_file_sha256 "${IMAGE_PATH}" "${WEIGHT_PACK_GUEST_PATH}")

    if [[ "${source_sha}" != "${image_sha}" ]]; then
        echo "Weight pack checksum mismatch in FireMarshal image." >&2
        echo "  source: ${source_sha}" >&2
        echo "  image:  ${image_sha}" >&2
        return 1
    fi

    model_source_sha=$(sha256sum "${MODEL_SOURCE}" | awk '{print $1}')
    model_image_sha=$(image_file_sha256 "${IMAGE_PATH}" "${MODEL_GUEST_PATH}")
    if [[ "${model_source_sha}" != "${model_image_sha}" ]]; then
        echo "Model artifact checksum mismatch in FireMarshal image." >&2
        echo "  source: ${model_source_sha}" >&2
        echo "  image:  ${model_image_sha}" >&2
        return 1
    fi

    if [[ "${EXPECTED_HYBRID_SPARSE_GGUF}" == "1" ]]; then
        model_source_size=$(stat -c '%s' "${MODEL_SOURCE}")
        model_source_allocated=$(( $(stat -c '%b' "${MODEL_SOURCE}") * 512 ))
        model_image_allocated=$(image_file_allocated_bytes "${IMAGE_PATH}" "${MODEL_GUEST_PATH}")
        if [[ -z "${model_image_allocated}" ]] ||
                (( model_source_size - model_source_allocated < 16 * 1024 * 1024 )); then
            echo "Generated hybrid model is not sparse or its image allocation could not be read." >&2
            return 1
        fi
        if (( model_image_allocated > model_source_allocated + 16 * 1024 * 1024 )); then
            echo "Hybrid model lost sparse holes while being copied into the FireMarshal image." >&2
            echo "  source allocated bytes: ${model_source_allocated}" >&2
            echo "  image allocated bytes:  ${model_image_allocated}" >&2
            return 1
        fi
    fi

    if ! e2fsck -fn "${IMAGE_PATH}" >/dev/null 2>&1; then
        echo "FireMarshal image filesystem check failed after copying the weight pack." >&2
        return 1
    fi

    echo "Verified hybrid model and Gemmini weight pack in FireMarshal image: model=${model_source_sha},pack=${source_sha}"
}

repair_image_filesystem() {
    local status=0

    e2fsck -fy "${IMAGE_PATH}" || status=$?
    if (( status >= 4 )); then
        echo "Unable to repair FireMarshal image; e2fsck status=${status}." >&2
        return "${status}"
    fi
}

rewrite_weight_pack_in_image() {
    local mount_dir
    local pid_file
    local mount_pid

    echo "Repairing FireMarshal image and rewriting Gemmini weight pack with fsync."
    repair_image_filesystem

    mount_dir=$(mktemp -d)
    pid_file=$(mktemp)

    cleanup_weight_pack_mount() {
        if mountpoint -q "${mount_dir}"; then
            guestunmount "${mount_dir}" || true
        fi
        rm -f "${pid_file}"
        rmdir "${mount_dir}" 2>/dev/null || true
    }
    trap cleanup_weight_pack_mount EXIT

    guestmount --pid-file "${pid_file}" -a "${IMAGE_PATH}" -m /dev/sda "${mount_dir}"
    mount_pid=$(cat "${pid_file}")

    mkdir -p "${mount_dir}$(dirname "${WEIGHT_PACK_GUEST_PATH}")"
    rm -f "${mount_dir}${MODEL_GUEST_PATH}.new" \
        "${mount_dir}${WEIGHT_PACK_GUEST_PATH}.new"
    cp --sparse=always "${MODEL_SOURCE}" "${mount_dir}${MODEL_GUEST_PATH}.new"
    chmod 0644 "${mount_dir}${MODEL_GUEST_PATH}.new"
    sync -f "${mount_dir}${MODEL_GUEST_PATH}.new"
    dd if="${WEIGHT_PACK_SOURCE}" \
       of="${mount_dir}${WEIGHT_PACK_GUEST_PATH}.new" \
       bs=16M conv=fsync status=progress
    chmod 0644 "${mount_dir}${WEIGHT_PACK_GUEST_PATH}.new"
    sync -f "${mount_dir}${WEIGHT_PACK_GUEST_PATH}.new"
    mv -f "${mount_dir}${MODEL_GUEST_PATH}.new" "${mount_dir}${MODEL_GUEST_PATH}"
    sync -f "${mount_dir}${MODEL_GUEST_PATH}"
    mv -f "${mount_dir}${WEIGHT_PACK_GUEST_PATH}.new" \
        "${mount_dir}${WEIGHT_PACK_GUEST_PATH}"
    sync -f "${mount_dir}${WEIGHT_PACK_GUEST_PATH}"

    guestunmount "${mount_dir}"
    while kill -0 "${mount_pid}" 2>/dev/null; do
        sleep 0.25
    done

    rm -f "${pid_file}"
    rmdir "${mount_dir}"
    trap - EXIT

    repair_image_filesystem
}

case "${1:-}" in
    --verify-image)
        validate_generated_profile 1
        validate_image_profile
        verify_weight_pack_in_image
        exit
        ;;
    --repair-image)
        validate_generated_profile 1
        validate_image_profile
        rm -f "${IMAGE_PROFILE_STAMP}" "${IMAGE_ABI_STAMP}" \
            "${IMAGE_ARTIFACT_FORMAT_STAMP}" "${IMAGE_PAGE_PACKING_CONFIG_STAMP}"
        rewrite_weight_pack_in_image
        verify_weight_pack_in_image
        printf '%s\n' "${HW_PROFILE}" > "${IMAGE_PROFILE_STAMP}"
        printf '%s\n' "${SW_ABI_SHA256}" > "${IMAGE_ABI_STAMP}"
        printf '%s\n' "${EXPECTED_ARTIFACT_FORMAT}" > "${IMAGE_ARTIFACT_FORMAT_STAMP}"
        printf '%s\n' "${EXPECTED_PAGE_PACKING_CONFIG}" > "${IMAGE_PAGE_PACKING_CONFIG_STAMP}"
        exit
        ;;
    "")
        ;;
    *)
        echo "usage: $0 [--verify-image|--repair-image]" >&2
        exit 2
        ;;
esac

invalidate_image_for_profile_switch
rm -f "${IMAGE_PROFILE_STAMP}" "${IMAGE_ABI_STAMP}" \
    "${IMAGE_ARTIFACT_FORMAT_STAMP}" "${IMAGE_PAGE_PACKING_CONFIG_STAMP}"
setup_build_env
cd "${MARSHAL_DIR}"
./marshal build "${WORKLOAD_CFG}"

validate_generated_profile 1

if ! verify_weight_pack_in_image; then
    rewrite_weight_pack_in_image
    verify_weight_pack_in_image
fi

printf '%s\n' "${HW_PROFILE}" > "${IMAGE_PROFILE_STAMP}"
printf '%s\n' "${SW_ABI_SHA256}" > "${IMAGE_ABI_STAMP}"
printf '%s\n' "${EXPECTED_ARTIFACT_FORMAT}" > "${IMAGE_ARTIFACT_FORMAT_STAMP}"
printf '%s\n' "${EXPECTED_PAGE_PACKING_CONFIG}" > "${IMAGE_PAGE_PACKING_CONFIG_STAMP}"

./marshal install "${WORKLOAD_CFG}"

echo "Installed FireMarshal workload and FireSim JSON for llama-firesim."
echo "FireSim runtime workload and bitstream selection remain separate from this software build."
