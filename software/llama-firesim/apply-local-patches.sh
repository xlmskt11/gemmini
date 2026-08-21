#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/llama.cpp" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLAMA_DIR=$(cd "$1" && pwd)

copy_tree_file() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "${dst}")"
    cp "${src}" "${dst}"
}

copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-bf16.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-bf16.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-pack.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-pack.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-pack-reader.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-pack-reader.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-page-packing.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-page-packing.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-profile.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-profile.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-weight-select.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-weight-select.h"
copy_tree_file "${SCRIPT_DIR}/src/cmake/llama-firesim-hw-profile.cmake" "${LLAMA_DIR}/cmake/llama-firesim-hw-profile.cmake"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/CMakeLists.txt" "${LLAMA_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini.cpp" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini.cpp"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini-flash-runtime-opt.h" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-flash-runtime-opt.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini-runtime-opt.h" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-runtime-opt.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini-pack-reader.cpp" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-pack-reader.cpp"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini-vpu.cpp" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-vpu.cpp"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini-vpu.h" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-vpu.h"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/CMakeLists.txt" "${LLAMA_DIR}/examples/firesim/CMakeLists.txt"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-cli.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-cli.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-pack.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-pack.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-bf16-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-bf16-test.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-flash-opt-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-flash-opt-test.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-matmul-opt-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-matmul-opt-test.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-page-packed-layout-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-page-packed-layout-test.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-profiler-append-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-profiler-append-test.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-vpu-admission-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-vpu-admission-test.cpp"

python3 - "${LLAMA_DIR}" <<'PY'
from pathlib import Path
import sys

llama_dir = Path(sys.argv[1])

def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"failed to patch {path}: marker not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

replace_once(
    llama_dir / "ggml" / "CMakeLists.txt",
    'option(GGML_CPU                             "ggml: enable CPU backend"                        ON)\n',
    'option(GGML_CPU                             "ggml: enable CPU backend"                        ON)\n'
    'option(GGML_GEMMINI                         "ggml: use Gemmini accelerator backend"           OFF)\n',
)

replace_once(
    llama_dir / "ggml" / "src" / "CMakeLists.txt",
    'ggml_add_backend(CPU)\n\n',
    'ggml_add_backend(CPU)\n'
    'ggml_add_backend(GEMMINI)\n\n',
)

replace_once(
    llama_dir / "ggml" / "src" / "ggml-backend-reg.cpp",
    '#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif\n',
    '#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif\n'
    '#ifdef GGML_USE_GEMMINI\n#include "ggml-gemmini.h"\n#endif\n',
)

replace_once(
    llama_dir / "ggml" / "src" / "ggml-backend-reg.cpp",
    '#ifdef GGML_USE_BLAS\n        register_backend(ggml_backend_blas_reg());\n#endif\n',
    '#ifdef GGML_USE_GEMMINI\n        register_backend(ggml_backend_gemmini_reg());\n#endif\n'
    '#ifdef GGML_USE_BLAS\n        register_backend(ggml_backend_blas_reg());\n#endif\n',
)

replace_once(
    llama_dir / "examples" / "CMakeLists.txt",
    '    add_subdirectory(simple-chat)\n',
    '    add_subdirectory(simple-chat)\n'
    '    add_subdirectory(firesim)\n',
)

# The Gemmini backend is an ACCEL device backed by ordinary host memory.  In
# AUTO mode llama.cpp otherwise disables Flash Attention solely because the KV
# cache owner is the CPU device, even though Gemmini can consume that buffer
# directly and the scheduler selected Gemmini for FLASH_ATTN_EXT.  Keep the
# mismatch guard for non-host-coherent device pairs.
replace_once(
    llama_dir / "src" / "llama-context.cpp",
    '''            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
''',
    '''            ggml_backend_buffer_type_t fa_host_buft =
                    ggml_backend_dev_host_buffer_type(device_fa);
            const bool host_coherent_accel_fa =
                    ggml_backend_dev_type(device_fa) == GGML_BACKEND_DEVICE_TYPE_ACCEL &&
                    ggml_backend_dev_type(device_kv) == GGML_BACKEND_DEVICE_TYPE_CPU &&
                    fa_host_buft != nullptr && ggml_backend_buft_is_host(fa_host_buft) &&
                    ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(device_kv));
            if (device_fa != device_kv && !host_coherent_accel_fa) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
''',
)

# Export the operator roles used to load each physical model tensor.  The
# FireSim packer uses this rather than guessing from tensor names; notably a
# tied token embedding is observed once as GET_ROWS and once as MUL_MAT.
replace_once(
    llama_dir / "src" / "llama-model.cpp",
    '''            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
''',
    '''            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }

            // Accelerator-specific model-weight buffer types must not be the
            // backend's default compute buffer.  Add them explicitly so the
            // weight selector can choose one without placing activations and
            // graph outputs there.
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
            if (get_extra_bufts) {
                ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts(dev);
                while (extra_bufts && *extra_bufts) {
                    buft_list.emplace_back(dev, *extra_bufts);
                    ++extra_bufts;
                }
            }
''',
)

replace_once(
    llama_dir / "src" / "llama-model-loader.h",
    '    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;\n',
    '    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;\n'
    '    std::unordered_map<std::string, uint8_t> gemmini_tensor_usage_by_name;\n',
)

replace_once(
    llama_dir / "src" / "llama-model-loader.cpp",
    '''        } else {
            op = info.op;
        }

        // sanity checks
''',
    '''        } else {
            op = info.op;
        }

        uint8_t & gemmini_usage = gemmini_tensor_usage_by_name[ggml_get_name(t_meta)];
        if (op == GGML_OP_MUL_MAT) {
            gemmini_usage |= 2u;
        } else {
            gemmini_usage |= 1u;
        }

        // sanity checks
''',
)

replace_once(
    llama_dir / "src" / "llama-model.h",
    '    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;\n',
    '    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;\n'
    '    std::unordered_map<std::string, uint8_t> gemmini_tensor_usage_by_name;\n',
)

replace_once(
    llama_dir / "src" / "llama-model.h",
    'const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);\n',
    'const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);\n'
    'uint8_t llama_internal_get_tensor_usage(const llama_model * model, const char * name);\n',
)

replace_once(
    llama_dir / "src" / "llama-model.cpp",
    '''    ml.done_getting_tensors();

    // populate tensors_by_name
''',
    '''    ml.done_getting_tensors();
    gemmini_tensor_usage_by_name = ml.gemmini_tensor_usage_by_name;

    // populate tensors_by_name
''',
)

replace_once(
    llama_dir / "src" / "llama-model.cpp",
    '''const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}
''',
    '''const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

uint8_t llama_internal_get_tensor_usage(const llama_model * model, const char * name) {
    if (model == nullptr || name == nullptr) {
        return 0;
    }
    const auto it = model->gemmini_tensor_usage_by_name.find(name);
    return it == model->gemmini_tensor_usage_by_name.end() ? 0 : it->second;
}
''',
)

replace_once(
    llama_dir / "src" / "llama-model.cpp",
    '#include <cstdint>\n',
    '#include <cstdint>\n#include <cstdlib>\n',
)

replace_once(
    llama_dir / "src" / "llama-model.cpp",
    '    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);\n',
    '    const bool gemmini_hybrid_sparse = std::getenv("LLAMA_FIRESIM_HYBRID_SPARSE_GGUF") != nullptr;\n'
    '    ml.init_mappings(!gemmini_hybrid_sparse, use_mlock ? &pimpl->mlock_mmaps : nullptr);\n',
)
PY
