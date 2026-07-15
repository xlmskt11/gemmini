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
copy_tree_file "${SCRIPT_DIR}/src/ggml/include/ggml-gemmini-pack.h" "${LLAMA_DIR}/ggml/include/ggml-gemmini-pack.h"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/CMakeLists.txt" "${LLAMA_DIR}/ggml/src/ggml-gemmini/CMakeLists.txt"
copy_tree_file "${SCRIPT_DIR}/src/ggml/src/ggml-gemmini/ggml-gemmini.cpp" "${LLAMA_DIR}/ggml/src/ggml-gemmini/ggml-gemmini.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/CMakeLists.txt" "${LLAMA_DIR}/examples/firesim/CMakeLists.txt"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-cli.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-cli.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-pack.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-pack.cpp"
copy_tree_file "${SCRIPT_DIR}/src/examples/firesim/llama-firesim-page-packed-layout-test.cpp" "${LLAMA_DIR}/examples/firesim/llama-firesim-page-packed-layout-test.cpp"

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
PY
