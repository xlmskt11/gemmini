# FireSim `llama.cpp` + Gemmini

This workload clones `ggml-org/llama.cpp` at release tag `b9060`, applies the
local Gemmini backend and FireSim REPL patches from this directory, downloads
one GGUF model, and builds a
FireMarshal image that boots into a UART-driven REPL.

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
- `backend_summary.csv`
- `arrival_request_summary.csv`
- `arrival_mode_summary.csv`
- `arrival_stage_summary.csv`
- `summary.md`

Bundled model selection is build-time only. The selected GGUF is always copied
to the guest as `/root/models/model.gguf`.

| `LLAMA_FIRESIM_MODEL` | GGUF source |
| --- | --- |
| `gemma-3-270m` | `ggml-org/gemma-3-270m-GGUF`, `gemma-3-270m-Q8_0.gguf` |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`, `smollm2-360m-instruct-q8_0.gguf` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf` |
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B-Instruct-GGUF`, `Llama-3.2-1B-Instruct-Q8_0.gguf` |

Default is `gemma-3-270m`. Override `LLAMA_FIRESIM_MODEL_URL` at build time to
package a custom single GGUF. Runtime model selection is intentionally disabled
so the FireSim rootfs only carries one model.

## REPL-Driven Prompt-Length/Gemmini-Count Sweep

Build and run the workload in the normal UART REPL mode, then drive the REPL
from the manager host:

```bash
./llama-firesim-build.sh
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
  --modes multi,single
```

Use `run_summary.csv` for one row per sweep case, and `backend_summary.csv` /
`op_profile.csv` for detailed split data.

The older guest-side non-interactive mode still exists as
`LLAMA_FIRESIM_MODE=sweep`, but the REPL-driven script avoids rebuilding the
workload just to change prompt-token, decode-token, or Gemmini-mask cases.

## Page-Packed Matmul Layouts

Set A, B, C, and D independently before process startup:

```bash
export GGML_GEMMINI_PAGE_PACKED_A=1
export GGML_GEMMINI_PAGE_PACKED_B=1
export GGML_GEMMINI_PAGE_PACKED_C=1
export GGML_GEMMINI_PAGE_PACKED_D=0
```

The equivalent CLI options are
`--gemmini-page-packed-{a,b,c,d} 0|1`. All four defaults are `0`, and the
settings are fixed after startup. Current Llama `MUL_MAT` calls do not provide
D, so the D setting only affects the deterministic Gemmini smoke test.

The offline weight pack is v2 and records the B layout plus `DIM`, page size,
and `MAX_BYTES`. Set `GGML_GEMMINI_PAGE_PACKED_B` while running `host-init.sh`
to generate the matching layout. Existing v1 packs remain row-major and are
converted once during model load when packed B is requested. A v2 geometry
mismatch causes the runtime to ignore the pack and rebuild the cache from the
GGUF.

After installing the workload, run all A/B/C combinations plus packed D in
both single and multi modes with a non-DIM-multiple shape:

```bash
/root/llama-firesim/llama-firesim-page-packed-smoke.sh
```
