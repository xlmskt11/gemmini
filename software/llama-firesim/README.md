# FireSim `llama.cpp` + Gemmini

This workload clones `ggml-org/llama.cpp` at release tag `b9060`, applies the
local Gemmini backend and FireSim REPL patches from this directory, downloads
one GGUF model, and builds a
FireMarshal image that boots into a UART-driven REPL.

The software build takes its Gemmini and VPU ABI directly from the generated
headers under
`sims/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/include`.
It does not inspect `config_runtime.yaml` or `config_hwdb.yaml`. Select exactly
one software profile for each build with `LLAMA_FIRESIM_HW_PROFILE`:

| build profile | Gemmini | logical mask | physical RoCC |
| --- | --- | --- | --- |
| `single` | 1 x 16 | `0x1` | `custom3` |
| `multi` | 4 x 8 | `0x1..0xf` | `custom0..3` |

Elaborate or otherwise place the intended generated headers and kernels in that
target-design tree, then select the profile and build independently of the
FireSim runtime selection:

```bash
LLAMA_FIRESIM_HW_PROFILE=single ./llama-firesim-build.sh
# or
LLAMA_FIRESIM_HW_PROFILE=multi ./llama-firesim-build.sh
```

For example, a multi-Gemmini Llama 3.2 1B image with ordinary matmul B packed
and independently selected FlashAttention layouts is built with:

```bash
LLAMA_FIRESIM_MODEL=Llama-3.2-1B \
GGML_GEMMINI_PAGE_PACKED_A=0 GGML_GEMMINI_PAGE_PACKED_B=1 \
GGML_GEMMINI_PAGE_PACKED_C=0 GGML_GEMMINI_PAGE_PACKED_D=0 \
Flash_Q_PAGE_PACKED=0 Flash_K_PAGE_PACKED=1 Flash_V_PAGE_PACKED=1 \
LLAMA_FIRESIM_HW_PROFILE=multi ./llama-firesim-build.sh
```

The selected value fixes the compiled Gemmini count, logical mask, profile ID,
and v4 weight-pack profile metadata. DIM, RoCC opcode, memory geometry, and VPU
kernel ABI still come directly from the target-design generated headers. The
build deliberately does not infer or validate the profile from those headers;
placing matching headers and selecting the matching FireSim bitstream are the
user's responsibility. Changing either the selected profile or generated-header
hash invalidates the old CLI, BF16 weight pack, and FireMarshal image. The build
hashes the complete target-design `include/` tree plus
`rocc-software/src/xcustom.h`, and aborts if they change between the host packer
and RISC-V cross build.

The primary entry points are:

- `llama-firesim-build.sh`: build and install the FireMarshal workload
- `llama-firesim-attach.sh`: attach to the active FireSim UART `screen`
- `llama-firesim-benchmark.sh`: send a benchmark command through UART and wait
  for the run to complete
- `llama-firesim-repl-sweep.sh`: drive the UART REPL through `screen` for
  prompt-token, decode-token, and Gemmini-mask sweeps
- `llama-firesim-arrival-benchmark.sh`: send `:arrival-benchmark [N]` through
  UART to compare sequential 4-Gemmini serving with split one-Gemmini-per-request serving
- `CODE_STRUCTURE.md`: implementation structure and current Gemmini offload map
- `LLAMA_FIRESIM_RUNTIME_ARCHITECTURE.md`: main entry, CPU/Gemmini paths, and
  timing/profiling semantics

If your manager shell relies on `ssh-agent`, source `~/.ssh/AGENT_VARS` before
running `firesim infrasetup` or `firesim runworkload`. The attach and benchmark
helpers will source that file automatically when it exists.

The guest writes the following files under `/root/llama-results/`:

- `run_summary.csv`
- `token_trace.csv`
- `op_profile.csv`
- `stage_summary.csv`
- `matmul_summary.csv`
- `backend_summary.csv`
- `arrival_request_summary.csv`
- `arrival_mode_summary.csv`
- `arrival_stage_summary.csv`
- `summary.md`

`backend_summary.csv` keeps `gemmini_bf16`, `vpu`, and `flash_attention`
separate. In `op_profile.csv`, `accelerator_call_us/cycles` is the common
accelerator interval; the `gemmini_*` columns remain Gemmini-only. Gemmini host
work is reported as cache lookup/packing, activation staging/packing,
configuration/run, output unpack/store, and remaining host time. FlashAttention
reports its measurable validation/preparation, input packing, full fused run,
output unpack/store, and remaining host time; it does not infer a QK versus
online-softmax versus PV split. The run, operation, stage, matmul, and backend
summaries record the compiled `hardware_profile` so single and multi results
retain their ABI identity.

Bundled model selection is build-time only. By default the build emits a hybrid
pair: `/root/models/model.gguf` keeps the normal GGUF metadata and only the
CPU/VPU-required tensor payloads, while
`/root/models/model.gguf.gemmini-pack` is the page-packed backing for Gemmini
matrix weights. Gemmini-only GGUF payload ranges are sparse holes rather than a
second resident copy. Set `LLAMA_FIRESIM_HYBRID_SPARSE_GGUF=0` only when a full
canonical GGUF is explicitly required.

| `LLAMA_FIRESIM_MODEL` | GGUF source |
| --- | --- |
| `gemma-3-270m` | `gguf-org/gemma-3-270m-gguf`, `gemma-3-270m-bf16.gguf` |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`, `smollm2-360m-instruct-q8_0.gguf` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf` |
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B-Instruct-GGUF`, `Llama-3.2-1B-Instruct-BF16.gguf` |
| `llama-3.2-3b` | `unsloth/Llama-3.2-3B-Instruct-GGUF`, `Llama-3.2-3B-Instruct-BF16.gguf` |

Default is `gemma-3-270m`. Override `LLAMA_FIRESIM_MODEL_URL` at build time to
package a custom single GGUF. Runtime model selection is intentionally disabled
so the FireSim rootfs only carries one model.

The built-in Gemma 3 270M and Llama 3.2 1B/3B selectors use BF16 GGUF files.
Their Gemmini matrix weights take the packer's direct BF16 copy path, avoiding
Q8_0 dequantization and BF16 re-encoding. The offline pack is still generated
because Gemmini B requires a transposed row-major or page-packed layout that
differs from the GGUF tensor layout.

## REPL-Driven Prompt-Length/Gemmini-Count Sweep

Build and run the workload in the normal UART REPL mode, then drive the REPL
from the manager host:

```bash
LLAMA_FIRESIM_HW_PROFILE=multi ./llama-firesim-build.sh
# start the FireSim workload, wait for "llama-firesim ready"
./llama-firesim-repl-sweep.sh
```

The default sweep runs:

- prompt tokens: `64..256` in steps of `16`
- Gemmini group masks: `1,3,7,15` equivalent to 1, 2, 3, 4 Gemminis
- decode tokens: `16`
- Gemmini mode: `multi`

The script sends REPL commands through the active FireSim UART `screen`:
`:backend gemmini`, `:gemmini-mode`, `:active-mask`, `:n-predict`, and
`:prompt-tokens PROMPT_TOKENS`. Override defaults with flags:

```bash
./llama-firesim-repl-sweep.sh \
  --prompt-min 64 \
  --prompt-max 256 \
  --prompt-step 16 \
  --decode-tokens 1,16 \
  --masks 1,3,7,15 \
  --modes multi
```

Use `run_summary.csv` for one row per sweep case, and `backend_summary.csv` /
`op_profile.csv` for detailed split data.

The older guest-side non-interactive mode still exists as
`LLAMA_FIRESIM_MODE=sweep`, but the REPL-driven script avoids rebuilding the
workload just to change prompt-token, decode-token, or Gemmini-mask cases.

## Page-Packed Matmul and FlashAttention Layouts

Set A, B, C, and D independently before process startup:

```bash
export GGML_GEMMINI_PAGE_PACKED_A=1
export GGML_GEMMINI_PAGE_PACKED_B=1
export GGML_GEMMINI_PAGE_PACKED_C=1
export GGML_GEMMINI_PAGE_PACKED_D=0
```

The equivalent CLI options are
`--gemmini-page-packed-{a,b,c,d} 0|1`. B defaults to `1`; A/C/D default to `0`,
and the settings are fixed after startup. Current Llama `MUL_MAT` calls do not
provide D, so the D setting only affects the deterministic Gemmini smoke test.

The options are requests, not unconditional layout flags. For each matmul,
A/C/D remain row-major when `M <= 1`, and B remains row-major when `K <= 1`.
The trace and per-op profiler record these effective flags. Hybrid generation
fails if any Gemmini-only tensor cannot be represented in the selected pack;
the runtime never reconstructs a missing Gemmini-only entry from a sparse GGUF.

With C packing disabled, dense FP32 outputs are written directly from Gemmini
to the ggml destination. This removes the temporary FP32 accumulator and the
following host copy. Set `GGML_GEMMINI_DIRECT_C=0` to force the retained
workspace/copy fallback for A/B correctness comparisons. Packed C, non-FP32,
strided, overlapping, or insufficiently aligned outputs always use the
fallback regardless of this setting.

With A packing disabled, a dense row-major BF16 activation whose storage is
disjoint from the output is passed directly to Gemmini. This bypasses the
`encoded_a` staging copy. Set `GGML_GEMMINI_DIRECT_A=0` to force the staging
path; F32/F16, padded, packed-A, overlapping, or invalid-span inputs always
retain staging.

Fused FlashAttention has three independent, case-sensitive input-packing
requests. Their defaults are Q=`0`, K=`1`, and V=`1`:

```bash
export Flash_Q_PAGE_PACKED=0
export Flash_K_PAGE_PACKED=1
export Flash_V_PAGE_PACKED=1
```

The equivalent CLI overrides are `--flash-q-page-packed`,
`--flash-k-page-packed`, and `--flash-v-page-packed` with a `0|1` value.

The all-uppercase `FLASH_Q_PAGE_PACKED`, `FLASH_K_PAGE_PACKED`, and
`FLASH_V_PAGE_PACKED` spellings are accepted as aliases when the corresponding
`Flash_*` variable is unset; the exact `Flash_*` names above take precedence.

These settings are independent of `GGML_GEMMINI_PAGE_PACKED_{A,B,C,D}`. The
generic variables continue to control only ordinary `MUL_MAT`; changing them
does not change fused FlashAttention Q/K/V layouts. Flash Q uses the
page-packed A layout, while Flash K and V each use the page-packed B source
layout. The low-level kernel receives their choices independently through
`query_stride`, `key_stride`, and `value_stride`, so K and V need not use the
same layout. Packing is disabled for Q when `query_rows <= 1`, for K when
`q_dim <= 1`, and for V when `sequence <= 1`, even if the corresponding Flash
option is enabled. K is packed as `[sequence][q_dim]`; Gemmini QK applies its
transpose after reading that source layout.

FlashAttention has no output-packing option. The online accumulator remains
internal to VSRAM, and the VPU always H_STOREs each final FP32 row directly
into the ggml destination in row-major order. General `MUL_MAT` C/D packing
behavior is unchanged.

The offline weight pack is v4 BF16-RNE and records the selected profile, `DIM`,
B layout, page size, `MAX_BYTES`, model fingerprint, and per-tensor role. It
contains no quantization-scale array. Set `GGML_GEMMINI_PAGE_PACKED_B` while
running `host-init.sh` to generate the matching layout. In hybrid mode, an old
pack, fingerprint/role mismatch, missing slice, or unsupported Gemmini matrix
is a startup error; it cannot silently fall back to zero-filled sparse data.
CPU backend mode, active mask `0`, and limited `max-offloads` are likewise
rejected for a hybrid artifact. Use the original full GGUF for CPU baselines.

After installing the workload, run all A/B/C combinations plus packed D for
the profile compiled into the image with a non-DIM-multiple shape:

```bash
/root/llama-firesim/llama-firesim-page-packed-smoke.sh
```

FlashAttention defaults to `AUTO`. The backend uses VPU kernels only for exact
supported semantics and records CPU fallback reasons in `op_profile.csv`.
`GGML_GEMMINI_DISABLE_VPU=1` and
`GGML_GEMMINI_DISABLE_FLASH_ATTENTION=1` disable those paths independently for
diagnostics.
