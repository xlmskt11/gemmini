#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
BUILD_ROOT="${SCRIPT_DIR}/build"
GENERATED_ROOT="${SCRIPT_DIR}/generated/root"
LLAMA_DIR="${BUILD_ROOT}/llama.cpp-b9060"
LLAMA_HOST_BUILD_DIR="${BUILD_ROOT}/llama.cpp-host-build"
LLAMA_BUILD_DIR="${BUILD_ROOT}/llama.cpp-build"
MODEL_DIR="${GENERATED_ROOT}/models"
GEMMINI_SW_DIR="${REPO_ROOT}/sims/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests"
MODEL_NAME="model.gguf"
MODEL_PACK_NAME="${MODEL_NAME}.gemmini-pack"
PAGE_PACKED_A="${GGML_GEMMINI_PAGE_PACKED_A:-0}"
PAGE_PACKED_B="${GGML_GEMMINI_PAGE_PACKED_B:-0}"
PAGE_PACKED_C="${GGML_GEMMINI_PAGE_PACKED_C:-0}"
PAGE_PACKED_D="${GGML_GEMMINI_PAGE_PACKED_D:-0}"
MODEL_SELECTOR="${LLAMA_FIRESIM_MODEL:-gemma-3-270m}"
MODEL_URL_OVERRIDE="${LLAMA_FIRESIM_MODEL_URL:-}"
LLAMA_TAG="${LLAMA_FIRESIM_LLAMA_TAG:-b9060}"

detect_compiler_prefix() {
    if command -v riscv64-unknown-linux-gnu-g++ >/dev/null 2>&1; then
        printf '%s\n' riscv64-unknown-linux-gnu
        return 0
    fi

    if command -v riscv64-linux-gnu-g++ >/dev/null 2>&1; then
        printf '%s\n' riscv64-linux-gnu
        return 0
    fi

    echo "Unable to find a RISC-V Linux cross compiler." >&2
    echo "Expected either riscv64-unknown-linux-gnu-g++ or riscv64-linux-gnu-g++ in PATH." >&2
    return 1
}

download_with_fallback() {
    local url="$1"
    local out="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "${out}" "${url}"
        return 0
    fi

    if command -v wget >/dev/null 2>&1; then
        wget -O "${out}" "${url}"
        return 0
    fi

    echo "Neither curl nor wget is available for downloading ${url}" >&2
    return 1
}

download_model() {
    local name="$1"
    local url="$2"
    local out="${MODEL_DIR}/${name}"
    local stamp="${MODEL_DIR}/.${name}.url"

    if [[ ! -f "${out}" || ! -f "${stamp}" || "$(cat "${stamp}")" != "${url}" ]]; then
        rm -f "${out}"
        download_with_fallback "${url}" "${out}"
        printf '%s\n' "${url}" > "${stamp}"
    fi
}

select_model() {
    case "${MODEL_SELECTOR}" in
        gemma|gemma3|gemma-3|gemma-3-270m|Gemma-3-270M)
            MODEL_ID="gemma-3-270m"
            MODEL_URL="https://huggingface.co/ggml-org/gemma-3-270m-GGUF/resolve/main/gemma-3-270m-Q8_0.gguf?download=true"
            ;;
        smollm|smollm2|smollm2-360m|SmolLM2-360M)
            MODEL_ID="smollm2-360m"
            MODEL_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf?download=true"
            ;;
        qwen|qwen3|qwen3-0.6b|Qwen3-0.6B)
            MODEL_ID="qwen3-0.6b"
            MODEL_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true"
            ;;
        qwen1.5|qwen1.5-1.8b|qwen1.5-1.8b-chat|Qwen1.5-1.8B-Chat)
            MODEL_ID="qwen1.5-1.8b-chat"
            MODEL_URL="https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q8_0.gguf?download=true"
            ;;
        qwen2.5|qwen2.5-1.5b|qwen2.5-1.5b-instruct|Qwen2.5-1.5B)
            MODEL_ID="qwen2.5-1.5b-instruct"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf?download=true"
            ;;
        llama|llama3.2|llama-3.2|llama-3.2-1b|Llama-3.2-1B)
            MODEL_ID="llama-3.2-1b"
            MODEL_URL="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true"
            ;;
        llama3.2-3b|llama-3.2-3b|Llama-3.2-3B)
            MODEL_ID="llama-3.2-3b-instruct"
            MODEL_URL="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true"
            ;;
        llama3.1|llama3.1-8b|llama-3.1|llama-3.1-8b|Llama-3.1-8B)
            MODEL_ID="llama-3.1-8b-instruct"
            MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf?download=true"
            ;;
        llama2|llama2-7b|llama-2|llama-2-7b|Llama-2-7B)
            MODEL_ID="llama-2-7b"
            MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf?download=true"
            ;;
        mistral|mistral-7b|Mistral-7B)
            MODEL_ID="mistral-7b"
            MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q8_0.gguf?download=true"
            ;;
        *)
            echo "unknown LLAMA_FIRESIM_MODEL=${MODEL_SELECTOR}" >&2
            echo "valid values: gemma-3-270m, smollm2-360m, qwen3-0.6b, qwen1.5-1.8b-chat, qwen2.5-1.5b, llama-3.2-1b, llama-3.2-3b, llama-3.1-8b, llama-2-7b, mistral-7b" >&2
            exit 1
            ;;
    esac

    if [[ -n "${MODEL_URL_OVERRIDE}" ]]; then
        MODEL_ID="${MODEL_SELECTOR}"
        MODEL_URL="${MODEL_URL_OVERRIDE}"
    fi
}

mkdir -p "${BUILD_ROOT}" "${GENERATED_ROOT}/llama-firesim" "${MODEL_DIR}"
select_model

echo "llama-firesim model: ${MODEL_ID}"
find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.gguf' ! -name "${MODEL_NAME}" -delete
download_model "${MODEL_NAME}" "${MODEL_URL}"

if [[ ! -f "${GEMMINI_SW_DIR}/include/gemmini_all.h" ||
      ! -f "${GEMMINI_SW_DIR}/include/gemmini_params.h" ||
      ! -f "${GEMMINI_SW_DIR}/include/gemmini_page_packed.h" ]]; then
    echo "Unable to find FireSim target-design Gemmini software headers under ${GEMMINI_SW_DIR}" >&2
    exit 1
fi

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    rm -rf "${LLAMA_DIR}"
    git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
else
    git -C "${LLAMA_DIR}" fetch --depth 1 origin "refs/tags/${LLAMA_TAG}:refs/tags/${LLAMA_TAG}"
    git -C "${LLAMA_DIR}" checkout --force "${LLAMA_TAG}"
    git -C "${LLAMA_DIR}" clean -fdx
fi

"${SCRIPT_DIR}/apply-local-patches.sh" "${LLAMA_DIR}"

if [[ "${LLAMA_FIRESIM_BUILD_WEIGHT_PACK:-1}" != "0" ]]; then
    rm -rf "${LLAMA_HOST_BUILD_DIR}"
    mkdir -p "${LLAMA_HOST_BUILD_DIR}"

    cmake -S "${LLAMA_DIR}" -B "${LLAMA_HOST_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU=ON \
        -DGGML_OPENMP=OFF \
        -DGGML_GEMMINI=OFF \
        -DGEMMINI_SW_DIR="${GEMMINI_SW_DIR}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DLLAMA_OPENSSL=OFF

    cmake --build "${LLAMA_HOST_BUILD_DIR}" --target llama-firesim-pack -j"$(nproc)"
    "${LLAMA_HOST_BUILD_DIR}/bin/llama-firesim-pack" \
        --model "${MODEL_DIR}/${MODEL_NAME}" \
        --output "${MODEL_DIR}/${MODEL_PACK_NAME}" \
        --page-packed-b "${PAGE_PACKED_B}"
else
    rm -f "${MODEL_DIR}/${MODEL_PACK_NAME}"
fi

COMPILER_PREFIX=$(detect_compiler_prefix)
export CC="${COMPILER_PREFIX}-gcc"
export CXX="${COMPILER_PREFIX}-g++"
export AR="${COMPILER_PREFIX}-ar"
export RANLIB="${COMPILER_PREFIX}-ranlib"
export STRIP="${COMPILER_PREFIX}-strip"

rm -rf "${LLAMA_BUILD_DIR}"
mkdir -p "${LLAMA_BUILD_DIR}"

cmake -S "${LLAMA_DIR}" -B "${LLAMA_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_GEMMINI=ON \
    -DGGML_RVV=OFF \
    -DGGML_RV_ZFH=OFF \
    -DGGML_RV_ZVFH=OFF \
    -DGGML_RV_ZICBOP=OFF \
    -DGGML_RV_ZIHINTPAUSE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGEMMINI_SW_DIR="${GEMMINI_SW_DIR}" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}"

cmake --build "${LLAMA_BUILD_DIR}" --target llama-firesim-cli -j"$(nproc)"

cp "${LLAMA_BUILD_DIR}/bin/llama-firesim-cli" "${GENERATED_ROOT}/llama-firesim/llama-firesim-cli"
cp "${SCRIPT_DIR}/guest/llama-firesim-launch.sh" "${GENERATED_ROOT}/llama-firesim/llama-firesim-launch.sh"
cp "${SCRIPT_DIR}/guest/llama-firesim-sweep.sh" "${GENERATED_ROOT}/llama-firesim/llama-firesim-sweep.sh"
cp "${SCRIPT_DIR}/guest/llama-firesim-page-packed-smoke.sh" "${GENERATED_ROOT}/llama-firesim/llama-firesim-page-packed-smoke.sh"
cat > "${GENERATED_ROOT}/llama-firesim/llama-firesim-defaults.env" <<EOF
: "\${LLAMA_FIRESIM_MODE:=${LLAMA_FIRESIM_MODE:-repl}}"
: "\${LLAMA_FIRESIM_SWEEP_PROMPT_MIN:=${LLAMA_FIRESIM_SWEEP_PROMPT_MIN:-64}}"
: "\${LLAMA_FIRESIM_SWEEP_PROMPT_MAX:=${LLAMA_FIRESIM_SWEEP_PROMPT_MAX:-256}}"
: "\${LLAMA_FIRESIM_SWEEP_PROMPT_STEP:=${LLAMA_FIRESIM_SWEEP_PROMPT_STEP:-16}}"
: "\${LLAMA_FIRESIM_SWEEP_MASKS:=${LLAMA_FIRESIM_SWEEP_MASKS:-1,3,7,15}}"
: "\${LLAMA_FIRESIM_N_PREDICT:=${LLAMA_FIRESIM_N_PREDICT:-16}}"
: "\${GGML_GEMMINI_PAGE_PACKED_A:=${PAGE_PACKED_A}}"
: "\${GGML_GEMMINI_PAGE_PACKED_B:=${PAGE_PACKED_B}}"
: "\${GGML_GEMMINI_PAGE_PACKED_C:=${PAGE_PACKED_C}}"
: "\${GGML_GEMMINI_PAGE_PACKED_D:=${PAGE_PACKED_D}}"
export LLAMA_FIRESIM_MODE
export LLAMA_FIRESIM_SWEEP_PROMPT_MIN
export LLAMA_FIRESIM_SWEEP_PROMPT_MAX
export LLAMA_FIRESIM_SWEEP_PROMPT_STEP
export LLAMA_FIRESIM_SWEEP_MASKS
export LLAMA_FIRESIM_N_PREDICT
export GGML_GEMMINI_PAGE_PACKED_A
export GGML_GEMMINI_PAGE_PACKED_B
export GGML_GEMMINI_PAGE_PACKED_C
export GGML_GEMMINI_PAGE_PACKED_D
EOF
chmod +x \
    "${GENERATED_ROOT}/llama-firesim/llama-firesim-cli" \
    "${GENERATED_ROOT}/llama-firesim/llama-firesim-launch.sh" \
    "${GENERATED_ROOT}/llama-firesim/llama-firesim-page-packed-smoke.sh" \
    "${GENERATED_ROOT}/llama-firesim/llama-firesim-sweep.sh"
