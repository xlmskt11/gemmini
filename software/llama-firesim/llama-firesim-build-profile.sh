#!/usr/bin/env bash

# The hardware profile is an explicit software-build input. Generated headers
# still provide DIM, opcodes, memory geometry, and VPU/kernel ABI, but they do
# not select single versus multi.
llama_firesim_require_hw_profile() {
    if [[ $# -ne 0 ]]; then
        echo "llama_firesim_require_hw_profile takes no arguments" >&2
        return 2
    fi

    case "${LLAMA_FIRESIM_HW_PROFILE:-}" in
        single|multi)
            LLAMA_FIRESIM_SELECTED_PROFILE="${LLAMA_FIRESIM_HW_PROFILE}"
            ;;
        "")
            echo "LLAMA_FIRESIM_HW_PROFILE is required; set it to single or multi." >&2
            return 1
            ;;
        *)
            echo "invalid LLAMA_FIRESIM_HW_PROFILE=${LLAMA_FIRESIM_HW_PROFILE}; expected single or multi." >&2
            return 1
            ;;
    esac
}

# Hash the generated include tree and RoCC instruction helper so the host packer
# and cross backend cannot silently consume different revisions in one build.
llama_firesim_compute_sw_abi_sha256() {
    if [[ $# -ne 1 ]]; then
        echo "llama_firesim_compute_sw_abi_sha256 requires GEMMINI_SW_DIR" >&2
        return 2
    fi

    python3 - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1])
include_root = root / "include"
xcustom = root / "rocc-software" / "src" / "xcustom.h"
if not include_root.is_dir():
    raise SystemExit(f"missing target-design include directory: {include_root}")
if not xcustom.is_file():
    raise SystemExit(f"missing target-design RoCC helper: {xcustom}")

digest = hashlib.sha256()
files = sorted(path for path in include_root.rglob("*") if path.is_file())
files.append(xcustom)
if not files:
    raise SystemExit(f"target-design include directory is empty: {include_root}")

for path in files:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(8, "little"))
    digest.update(relative)
    data = path.read_bytes()
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data)

print(digest.hexdigest())
PY
}

# Obtain a stable hash of the generated software ABI. This does not interpret
# the headers or compare them with the explicitly selected build profile.
llama_firesim_read_sw_abi() {
    if [[ $# -ne 1 ]]; then
        echo "llama_firesim_read_sw_abi requires GEMMINI_SW_DIR" >&2
        return 2
    fi

    local gemmini_sw_dir=$1
    local attempt
    local hash_before
    local hash_after

    for attempt in 1 2 3; do
        hash_before=$(llama_firesim_compute_sw_abi_sha256 "${gemmini_sw_dir}")
        hash_after=$(llama_firesim_compute_sw_abi_sha256 "${gemmini_sw_dir}")
        if [[ "${hash_before}" == "${hash_after}" ]]; then
            LLAMA_FIRESIM_SW_ABI_SHA256="${hash_after}"
            return 0
        fi
        echo "Target-design software headers changed while sampling ABI (attempt ${attempt}/3)." >&2
    done

    echo "Unable to obtain a stable target-design Gemmini/VPU software ABI." >&2
    return 1
}
