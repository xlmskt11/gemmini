# llama-firesim Runtime Architecture

이 문서는 현재 `llama-firesim` workload에서 `llama.cpp`가 어떻게 구성되어
있는지, main entry가 어디인지, CPU/Gemmini path가 어디서 갈라지는지, 그리고
현재 latency/profile 값이 어떻게 측정되는지 정리한다.

> Current ABI note: `single` (1x16/custom3) and `multi` (4x8/custom0..3) are
> separate build profiles. The backend now uses BF16-RNE inputs, FP32
> accumulation, exact VPU kernels, conditional fused FlashAttention, and
> reasoned CPU fallback. This document describes that current numerical and
> profiling contract.

## Scope

이 문서의 기준은 local source tree이다.

| 영역 | 파일 | 역할 |
| --- | --- | --- |
| guest launcher | `guest/llama-firesim-launch.sh` | guest boot 후 model/backend/env default를 설정하고 CLI 실행 |
| FireSim CLI main | `src/examples/firesim/llama-firesim-cli.cpp` | UART REPL, prompt 실행, benchmark, arrival benchmark |
| FireSim CLI CMake | `src/examples/firesim/CMakeLists.txt` | `llama-firesim-cli` executable 정의 |
| Gemmini backend API | `src/ggml/include/ggml-gemmini.h` | Gemmini backend/profiler/runtime API 선언 |
| Gemmini backend implementation | `src/ggml/src/ggml-gemmini/ggml-gemmini.cpp` | BF16 Gemmini, VPU/Flash admission, CPU fallback, profiling output |
| VPU/Flash adapter | `src/ggml/src/ggml-gemmini/ggml-gemmini-vpu.cpp` | nonlinear/RMSNorm/RoPE/FlashAttention staging과 exact-semantics 검증 |
| model flow doc | `MODEL_EXECUTION_FLOW.md` | Gemma 3 270M layer별 수식과 matmul dimension |
| old structure doc | `CODE_STRUCTURE.md` | workload/build/debug 중심 구조 설명 |

Generated paths are not the source of truth.

| 생성 경로 | 의미 |
| --- | --- |
| `build/llama.cpp-b9060/` | pinned upstream `llama.cpp` clone with local patches copied in |
| `build/llama.cpp-build/` | CMake build directory |
| `generated/root/` | FireMarshal rootfs copy-in staging area |

수정은 `src/`, `guest/`, `host-init.sh`, `apply-local-patches.sh`,
`llama-firesim.yaml` 같은 local source에 해야 한다. `build/llama.cpp-b9060/`
안의 파일은 build 과정에서 reset/copy될 수 있다.

## Build-Time Composition

`llama-firesim`은 upstream `llama.cpp` 전체를 fork처럼 직접 수정하는 방식이
아니라, pinned release tree에 local files를 복사하고 CMake/backend registry를
patch하는 구조이다.

1. `llama-firesim-build.sh`가 FireMarshal build/install을 실행한다.
2. FireMarshal이 `host-init.sh`를 실행한다.
3. `host-init.sh`가 `ggml-org/llama.cpp` release tag `b9060`을 clone/reset한다.
4. `apply-local-patches.sh`가 local `src/examples/firesim`과 `src/ggml` patch를 clone tree에 복사한다.
5. `apply-local-patches.sh`가 upstream CMake에 `GGML_GEMMINI` backend와 `examples/firesim`을 추가한다.
6. CMake가 RISC-V Linux용 `llama-firesim-cli`를 cross-compile한다.
7. guest launcher, CLI binary, model files가 `generated/root/`에 staged된다.
8. FireMarshal image가 `sims/firesim/deploy/workloads/llama-firesim.json`으로 install된다.

Gemmini software headers는 FireSim target-design 쪽을 써야 한다.

```text
sims/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/include
```

빌드는 이 경로의 `gemmini_all.h`, `gemmini_params.h`, `vpu_params.h`와 VPU
kernel을 직접 사용한다. software build는 `config_runtime.yaml` 또는 HWDB를
검사하지 않으며, 실행할 bitstream과 생성 헤더의 일치 여부는 사용자가
관리한다.

## Runtime Entry Points

### Guest Launcher

guest boot 후 FireMarshal workload command가 `guest/llama-firesim-launch.sh`를
실행한다. launcher는 환경변수 default를 잡고 CLI를 `exec`한다.

```bash
exec /root/llama-firesim/llama-firesim-cli \
    --backend "${BACKEND}" \
    --ctx-size "${CTX_SIZE}" \
    --n-predict "${N_PREDICT}" \
    --results-dir "${RESULTS_DIR}" \
    --interactive
```

현재 launcher default:

| env | default | 의미 |
| --- | --- | --- |
| `GGML_GEMMINI_ACTIVE_MASK` | single `0x1`, multi `0xf` | 선택 profile의 logical Gemmini port 사용 |
| `GGML_GEMMINI_MAX_OFFLOADS` | `-1` | eligible matmul offload 개수 제한 없음 |
| `GGML_GEMMINI_PREPACK_WEIGHTS` | `1` | model weight를 load 직후 BF16 transpose/page-pack cache로 선처리 |
| `GGML_GEMMINI_DIRECT_A` | `1` | dense row-major BF16 activation을 staging copy 없이 Gemmini A로 직접 사용 |
| `GGML_GEMMINI_DIRECT_C` | `1` | dense FP32/C-unpacked matmul 결과를 임시 accumulator 없이 destination에 직접 저장 |
| `GGML_GEMMINI_PAGE_PACKED_A` | `0` | 일반 `MUL_MAT`의 A packing 요청 |
| `GGML_GEMMINI_PAGE_PACKED_B` | `1` | 일반 `MUL_MAT`의 B packing 요청 및 offline weight-pack layout |
| `GGML_GEMMINI_PAGE_PACKED_C` | `0` | 일반 `MUL_MAT`의 C packing 요청 |
| `GGML_GEMMINI_PAGE_PACKED_D` | `0` | 일반 `MUL_MAT`의 D packing 요청 |
| `GGML_GEMMINI_TRACE` | `0` | Gemmini admission/dispatch trace 비활성 |
| `LLAMA_FIRESIM_BACKEND` | `gemmini` | hybrid Gemmini backend 사용 |
| `LLAMA_FIRESIM_CTX_SIZE` | `2048` | llama context size |
| `LLAMA_FIRESIM_N_PREDICT` | `16` | generated token count |
| `LLAMA_FIRESIM_RESULTS_DIR` | `/root/llama-results` | result output directory |

모델 선택은 guest runtime이 아니라 `host-init.sh`가 rootfs를 만들 때 수행한다.
`LLAMA_FIRESIM_MODEL`은 `gemma-3-270m`, `smollm2-360m`, `qwen3-0.6b`,
`llama-3.2-1b` 중 하나이며, guest path는 항상 `/root/models/model.gguf`이다.

### Main Function

FireSim workload의 host main은 upstream `llama-cli`가 아니라 custom executable이다.

```text
src/examples/firesim/llama-firesim-cli.cpp
```

main flow:

```text
main()
  -> parse_args()
  -> app app_(opts)
  -> app_.run()
```

`src/examples/firesim/CMakeLists.txt`가 executable을 정의한다.

```cmake
add_executable(llama-firesim-cli llama-firesim-cli.cpp)
target_link_libraries(llama-firesim-cli PRIVATE llama ggml Threads::Threads)
```

즉 upstream `examples/main`, `llama-cli`, `simple` example의 main은 이 workload에서
직접 쓰지 않는다. `llama-firesim-cli.cpp`가 libllama API를 호출해서 실행한다.

## CLI Object Lifetime

`app` constructor에서 최초 수행되는 작업:

1. signal handler 등록
2. `configure_memory_locking()`으로 `mlockall(MCL_CURRENT | MCL_FUTURE)` 시도
3. `ggml_backend_load_all()` 호출
4. results directory 생성
5. `llama_model_load_from_file()`로 GGUF model load
6. model load `us/cycles` 기록

각 request마다 새로 수행되는 작업:

1. prompt tokenize 또는 pre-tokenized prompt 준비
2. `llama_context_default_params()` 기반 fresh `llama_context` 생성
3. greedy sampler 생성
4. prefill `llama_decode()` 실행
5. sampling
6. 필요한 경우 decode `llama_decode()` 반복
7. profiler result 기록
8. sampler/context free

중요한 점은 model object는 `/root/models/model.gguf`에서 한 번 load되고,
request마다 `llama_context`와 KV cache는 항상 새로 만들어진다는 것이다.

## Request Execution Flow

일반 prompt는 `run_command()`로 들어간다.

```text
UART prompt
  -> run_command(prompt, n_predict, benchmark)
    -> ggml_gemmini_profiler_reset()
    -> ggml_gemmini_profiler_note_model_load()
    -> run_once(...)
    -> ggml_gemmini_profiler_write_results()
```

`run_once()`의 실행 순서:

```text
set backend mode or per-request Gemmini config
start profiler run label
create llama_context
create greedy sampler
record run_us_start and run_cycles_start

prefill:
  batch = all prompt tokens
  phase = "prefill"
  llama_decode(ctx, batch)

generation loop:
  sample one token from current logits
  record token and sampling time
  if generated >= n_predict: stop
  batch = sampled token
  phase = "decode"
  llama_decode(ctx, batch)

record run total
free sampler/context
clear request-local Gemmini config
```

`summary.md`에는 request-level latency metric도 기록된다.

| metric | 정의 |
| --- | --- |
| `TTFT` | `run_us_start`부터 첫 non-EOG token의 `llama_sampler_sample()` 종료 시점까지 |
| `TPOT` | 첫 token 이후 token sample timestamp 간 평균. `(last_token_us - first_token_us) / (generated_tokens - 1)` |

generated token이 0개이면 TTFT/TPOT는 `n/a`, 1개이면 TPOT만 `n/a`이다.

현재 loop semantics에서 `n_predict=1`은 prefill 후 token 하나를 sample하고 끝난다.
즉 heavy decode `llama_decode()`는 실행되지 않는다. decode compute까지 1회 포함하려면
현재 코드 기준으로 `n_predict=2`가 필요하다.

## CPU Path

CPU-only baseline은 `GGML_GEMMINI_DISABLE=1`로 구현되어 있다.

```text
:backend cpu
  -> opts_.backend = "cpu"
  -> run_command()
  -> run_once(..., cpu_baseline=true)
  -> set_backend_mode(true)
  -> setenv("GGML_GEMMINI_DISABLE", "1")
```

Gemmini backend의 `supports_op()`는 `GGML_GEMMINI_DISABLE`이 켜져 있으면 false를
반환한다. 그 결과 ggml scheduler는 모든 compute node를 CPU backend에 맡긴다.

CPU path에서 profiler label:

| label | 의미 |
| --- | --- |
| `cpu_baseline` | 전체 run이 CPU-only인 경우 |
| `cpu_fallback` | hybrid run 안에서 Gemmini가 처리하지 않은 op |

CPU fallback으로 남는 대표 작업:

| 작업 | 이유 |
| --- | --- |
| tokenizer | `llama_decode()` 전에 CPU string/token 처리 |
| embedding lookup | `GET_ROWS`, Gemmini backend op 아님 |
| 지원 조건 밖 RMSNorm/RoPE/nonlinear | VPU exact-semantics admission 실패 |
| KV cache view/write/read | `VIEW`, `SET_ROWS`, cache management |
| 임의 mask/SWA/ALiBi/softcap/sinks Flash attention | fused kernel 의미와 불일치 |
| mask/bias/scale 조건 밖 softmax | VPU softmax 의미와 불일치 |
| residual add, scale, GEGLU 일부 | elementwise op |
| sampling | sampler chain CPU code |

## Gemmini Path

Hybrid mode는 `GGML_GEMMINI_DISABLE`을 unset하고 active mask를 설정한다.

```text
:backend gemmini
  -> opts_.backend = "gemmini"
  -> run_command()
  -> run_once(..., cpu_baseline=false)
  -> set_backend_mode(false)
  -> unsetenv("GGML_GEMMINI_DISABLE")
  -> default GGML_GEMMINI_ACTIVE_MASK=0xf
```

실제 CPU/Gemmini 분기는 CLI에서 직접 하지 않는다. `llama_decode()`가 ggml graph를
만들고, llama.cpp backend scheduler가 각 backend의 `supports_op()`를 확인해서
node를 배치한다.

Gemmini backend가 지원한다고 광고하는 op:

| ggml op | 처리 |
| --- | --- |
| `GGML_OP_MUL_MAT` | 조건을 만족하면 Gemmini offload |
| `GGML_OP_NONE` | subgraph/view 허용 |
| `GGML_OP_RESHAPE` | subgraph/view 허용 |
| `GGML_OP_VIEW` | subgraph/view 허용 |
| `GGML_OP_PERMUTE` | subgraph/view 허용 |
| `GGML_OP_TRANSPOSE` | subgraph/view 허용 |
| 그 외 op | unsupported, CPU fallback |

`GGML_OP_MUL_MAT` offload 조건:

| 조건 | 의미 |
| --- | --- |
| `GGML_GEMMINI_DISABLE`이 false | CPU-only mode 제외 |
| `GGML_GEMMINI_ACTIVE_MASK != 0` | 사용할 custom accelerator 존재 |
| op가 `GGML_OP_MUL_MAT` | 현재 dense matmul만 실제 compute 구현 |
| `dst->type == GGML_TYPE_F32` 또는 `from_float_ref` 지원 | Gemmini FP32 accumulator를 ggml output type으로 저장 가능 |
| src/dst shape broadcast 조건 통과 | 현재 layout 제한 |
| innermost stride가 contiguous | row conversion/repack 가능 |
| `src0`, `src1`이 BF16/F32이거나 `to_float` 지원 | BF16 source는 직접 사용하고 나머지는 BF16-RNE encode 가능 |
| `GGML_GEMMINI_MAX_OFFLOADS` 제한 통과 | debugging용 앞 N개만 offload 가능 |

분기 call path:

```text
llama_decode()
  -> ggml backend scheduler
    -> ggml_backend_gemmini_device_supports_op()
      -> gemmini_can_offload()
      -> gemmini_admit_offload()
    -> ggml_backend_gemmini_graph_compute()
      -> ggml_backend_gemmini_mul_mat()
      -> run_gemmini_matmul()
```

## Gemmini Data Conversion

현재 Gemmini는 BF16 input과 FP32 accumulator path를 사용한다. 따라서 offload된
ggml `MUL_MAT`은 Gemmini에 들어가기 전에 필요한 layout과 BF16 encoding으로
staging된다.

```text
ggml src0/src1
  -> BF16 direct copy or tensor_row_to_float()
  -> BF16-RNE encode + transpose/page-pack
  -> run_gemmini_matmul()
  -> FP32 acc_t output
  -> layout conversion/writeback
```

Operand mapping:

| Gemmini operand | source | 처리 |
| --- | --- | --- |
| `A` | ggml `src1`, runtime activation | BF16-RNE encode, optional page-packed A |
| `B` | ggml `src0`, weight or intermediate | BF16 K x J transpose, optional page-packed B |
| `D` | bias input | `nullptr`, no fused bias |
| `C` | temporary `acc_t` output | Gemmini FP32 accumulation result |
| ggml `dst` | final F32 tensor | `C` layout conversion 후 저장 |

Page-packing 환경변수는 요청값이다. 각 matmul의 실제 layout은 A/C/D의
경우 `M > 1`, B의 경우 `K > 1`일 때만 page-packed가 되며, 그 이하에서는
옵션이 켜져 있어도 row-major를 사용한다. v3 packed-weight 파일에서는
전역 B layout과 섞이지 않도록 `K <= 1` entry를 생략하고 runtime GGUF
fallback이 row-major BF16 cache를 만든다.

fused FlashAttention은 Q/K/V page-packing 옵션을 제공하지 않는다. adapter는
Q `[query_rows][q_dim]`, K `[sequence][q_dim]`, V
`[sequence][value_dim]`를 linear BF16 입력으로 전달한다. physical row stride가
logical width와 일치하고 output과 겹치지 않는 BF16 K/V head slice는 직접 DMA로
읽으며, 나머지는 재사용 가능한 64-byte aligned linear BF16 workspace로
변환하거나 gather한다. `GGML_GEMMINI_PAGE_PACKED_{A,B,C,D}`는 이 경로에 영향을
주지 않는다. FireSim의 BF16 Flash KV cache는 ggml `[C,Tcap,Hkv,S]`, 즉
물리적으로 `[stream][KV head][token][channel]`에 저장한다. graph-facing view는
기존 `[C,Hkv,n_kv,S]` 계약을 유지하고 attention 직전 permute 뒤에는
`[C,n_kv,Hkv,S]`가 되어 K/V를 별도 repack 없이 직접 읽을 수 있다. adapter가
ggml의 양수 finite attention scale을
`vpu_flashattention_config_t::score_scale`로 전달하고 VPU가 FP32 QK score에 직접
적용한다. 0, 음수, NaN, Inf scale은 fused path admission에서 CPU fallback된다.
online accumulator는 VSRAM 내부 상태이고, VPU는 final
normalization 후 각 FP32 row를 최종 ggml destination 주소에 직접 기록한다.

Static model weight로 분류되는 `src0`는 `weight_cache`에 repack 결과를 저장한다.
runtime intermediate tensor는 stale data를 피하기 위해 매번 transient repack한다.

현재 static weight name patterns:

| stage | name patterns |
| --- | --- |
| q projection | `attn_q.weight`, `.wq` |
| k projection | `attn_k.weight`, `.wk` |
| v projection | `attn_v.weight`, `.wv` |
| o projection | `attn_output.weight`, `.wo` |
| ffn gate | `ffn_gate.weight` |
| ffn up | `ffn_up.weight` |
| ffn down | `ffn_down.weight` |
| LM head | `output.weight`, tied `token_embd.weight` used by `result_output` |

현재 all-eligible mode에서는 projection/FFN뿐 아니라 조건을 만족하는 다른
`MUL_MAT`도 Gemmini 후보가 될 수 있다. attention score/value 같은 F32 intermediate
matmul도 조건을 통과하면 BF16-RNE로 encode되어 Gemmini로 들어간다.

## Gemmini Matmul Helper Path

`run_gemmini_matmul()`이 실제 Gemmini helper를 호출한다.

공통 준비:

1. thread-local request config에서 `active_mask`, `gemmini_id`, `split_mode` 확인
2. active mask에 대해 최초 1회 `gemmini_flush()`
3. cycle start 기록

shared-multi path:

```text
shared_multi_matmul_job_t job
  -> job field 설정
  -> tiling 선택
  -> shared_multi_tiled_matmul_job_init(&job)
  -> while (!job.done) shared_multi_tiled_matmul_job_step(&job)
  -> gemmini_fence()
```

현재 group mode job 설정:

| field | value |
| --- | --- |
| `gemmini_list` | active mask, default `0xf` |
| `sp_addr_start_stack` | `0` |
| `sp_addr_end_stack` | `0` |
| `acc_addr_start_stack` | `0` |
| `sp_addr_range` | `TOTAL_SPAD_ROWS` |
| `acc_addr_range` | `TOTAL_ACC_ROWS` |
| tiling | `shared_multi_choose_tiling_factors()` |

현재 split mode job 설정:

| field | value |
| --- | --- |
| `gemmini_list` | request-specific `0x1`, `0x2`, `0x4`, or `0x8` |
| `sp_addr_start_stack` | `TOTAL_SPAD_ROWS * gemmini_id / 16` |
| `sp_addr_end_stack` | `TOTAL_SPAD_ROWS * gemmini_id / 16` |
| `acc_addr_start_stack` | `TOTAL_ACC_ROWS * gemmini_id / 8` |
| `sp_addr_range` | `TOTAL_SPAD_ROWS / 4` |
| `acc_addr_range` | `TOTAL_ACC_ROWS / 4` |
| tiling | partition-aware `choose_split_tiling_factors()` |

Split tiling budget:

| budget | value |
| --- | --- |
| `max_A_rows` | `TOTAL_SPAD_ROWS / 16` |
| `max_B_rows` | `TOTAL_SPAD_ROWS / 16` |
| `max_acc_rows` | `TOTAL_ACC_ROWS / 8` |

Split tile 후보 조건:

```text
tiled_matmul_A_spad_rows(tile_I, tile_J, tile_K) <= max_A_rows
tiled_matmul_B_spad_rows(tile_I, tile_J, tile_K) <= max_B_rows
tiled_matmul_total_acc_rows(tile_I, tile_J) <= max_acc_rows
```

`shared_multi_tiled_matmul_job_step()` 내부 double-buffer address formula는 바꾸지
않는다. split mode에서 바꾸는 것은 job 생성 시 stack/range 값과 tiling뿐이다.

## Arrival Benchmark Flow

REPL command:

```text
:arrival-benchmark [N]
```

Current flow:

1. synthetic 256-token prompt 생성
2. warm-up request 1회 실행
3. warm-up profile event reset, weight cache는 유지
4. `group_seq` 실행
5. `split_1gem` 실행
6. profiler CSV/Markdown write
7. arrival CSV/Markdown append

`group_seq`:

```text
request 0..3
arrival = base + request_id * 1s
single software thread
active_mask = 0xf
split_mode = false
```

`split_1gem`:

```text
4 software worker threads
worker i waits until base + i * 1s
request i active_mask = 1 << i
request i gemmini_id = i
split_mode = true
```

이 실험은 1 Rocket core에서 4 software worker가 time-slicing되는 contention까지
포함한다. 따라서 split mode는 Gemmini 병렬성만 보지 않고 guest software scheduling,
shared memory, backend mutex, weight cache contention도 함께 포함한다.

## Time And Cycle Measurement

현재 결과에는 두 종류의 시간이 있다.

| 값 | 측정 방식 | 의미 |
| --- | --- | --- |
| `*_us` | `ggml_time_us()` | guest Linux monotonic time |
| `*_cycles` | RISC-V `rdcycle` | target cycle count |

`ggml_time_us()`는 upstream ggml의 `clock_gettime(CLOCK_MONOTONIC)` 기반이다.
FireSim 안에서 실행되므로 host wall-clock이 아니라 guest/target 기준 시간으로
해석해야 한다.

FireSim host에서 사람이 기다린 시간은 `uartlog` 끝의 simulation summary에 있는
`Wallclock Time Elapsed`이다. `*_us`와 host wall-clock은 같지 않다.

### Model Load Time

최초 model load와 REPL model reload 모두 model load 전후를 잰다.

```text
load_us_start = ggml_time_us()
load_cycles_start = rdcycle
llama_model_load_from_file()
load_us_end = ggml_time_us()
load_cycles_end = rdcycle
```

결과는 `summary.md` 상단의 `Model load: ... us / ... cycles`에 기록된다.

### Run Total

`run_once()`에서 request 전체를 잰다.

```text
run_us_start = ggml_time_us()
run_cycles_start = rdcycle
prefill + sampling + optional decode
run_us_end = ggml_time_us()
run_cycles_end = rdcycle
```

`summary.md`의 각 run `total time (us)`와 `total cycles`는 이 값이다. 이 값은
단일 interval이므로 request latency 비교에 사용할 수 있다.

### Op/Profile Time

CLI는 `llama_context_params.cb_eval`에 `eval_callback()`을 설치한다.
callback은 각 ggml tensor evaluation 전후에 profiler hook을 호출한다.

```text
ask == true:
  ggml_gemmini_profiler_eval_begin(t, rdcycle, ggml_time_us)

ask == false:
  ggml_gemmini_profiler_eval_end(t, rdcycle, ggml_time_us, cpu_baseline)
```

`op_profile.csv`의 `wall_us/wall_cycles`는 이 callback 기반 op interval이다.
Gemmini로 실행된 op는 `last_gemmini` state와 tensor pointer가 일치하면 backend가
`gemmini_bf16`으로 바뀐다.

Gemmini `MUL_MAT` row는 `wall_us/wall_cycles` 안을 다시 나눈 host-side 컬럼을
가진다.

| column | 의미 |
| --- | --- |
| `weight_cache_lookup_pack_us/cycles` | weight cache lookup 및 cache miss/transient weight의 BF16 transpose/page-pack 구간 |
| `activation_stage_pack_us/cycles` | dense row-major BF16이면 `src1`을 direct A로 선택하고, 그 외에는 BF16-RNE staging 및 필요한 A-layout page-pack을 수행하는 구간 |
| `gemmini_call_us/cycles` | `run_gemmini_matmul()` 호출 구간. host-side BF16 staging/store를 제외한 accelerator dispatch/wait wall interval |
| `output_unpack_store_us/cycles` | FP32 accumulator layout을 풀어 `dst` type으로 저장하는 구간 |

따라서 Gemmini offload 자체 wall time을 볼 때는 `wall_us`가 아니라
`gemmini_call_us`를 우선 본다. `gemmini_total_cycles`와 하위 `gemmini_*_cycles`는
Gemmini counter 기반 세부 cycle이며, counter가 비활성화된 run에서는 0일 수 있다.

`summary.md`와 `backend_summary.csv`의 `Backend Split`은 `gemmini_bf16` op의 전체
`wall_us/wall_cycles`를 집계한다. 별도의 `Backend Gemmini Breakdown` /
`backend_gemmini_breakdown`이 위 host-side 구간과 Gemmini dispatch/run을 나누며,
`other_host`가 전체 Gemmini op wall과 계측된 구간 사이의 차이를 닫는다.

FlashAttention도 `mask_prepare`, `workspace_prepare`, `plan_preflight`,
`input_pack`, `fused_attention_run`, `output_unpack_store`, `other_host`로 같은
방식으로 표시한다. `input_validation` profiler 필드는 기존 CSV schema 호환을
위해 남아 있지만 새 run에서는 0이다. `fused_attention_run`은 전체 fused
QK/online-softmax/PV 실행 구간이며 내부 비율은 추정하지 않는다.

| subphase | 측정 범위 |
| --- | --- |
| `input_validation` | 호환용 0 값. fast path는 Q/K/V finite를 전제로 하며 전수 검사하지 않음 |
| `mask_prepare` | causal mask slice 검사와 query-base metadata 준비 |
| `workspace_prepare` | input staging/causal workspace 크기 계산과 할당 |
| `plan_preflight` | 각 batch/head의 FlashAttention plan 및 job preflight |
| `input_pack` | direct 조건을 만족하지 않는 Q/K/V의 linear BF16 변환 또는 gather. Q에는 host-side scale을 곱하지 않음 |
| `fused_attention_run` | `vpu_flashattention_auto()` 전체 실행 |
| `output_unpack_store` | 현재 direct output에서는 0; 호환용 profiler 항목 |
| `other_host` | backend op wall에서 위 실측 구간을 뺀 lock/control 등 나머지 시간 |

실패하여 CPU fallback된 FlashAttention 시도는 성공한 accelerator backend row가
아니므로 이 breakdown에 합산하지 않는다.

FlashAttention fast path는 Q/K/V가 finite라고 가정한다. NaN/Inf가 들어오면
전수 검사 기반 CPU fallback 없이 BF16 staging 또는 accelerator 결과로 전파될 수
있다.

FlashAttention score scale은 positive finite 값만 accelerator admission하며,
ggml op의 값을 `vpu_flashattention_config_t::score_scale`로 그대로 전달한다.
VPU가 QK FP32 score에 scale을 적용하므로 Q staging은 F32-to-BF16 변환만 하고
과거의 `scale * sqrt(d)` 선행 보정은 하지 않는다.

### Repack Time

Gemmini `src0` repack은 `populate_weight_cache_entry()`에서 별도 event로 기록된다.
기본값에서는 model load 직후 `GGML_GEMMINI_PREPACK_WEIGHTS=1`로 model weight를
미리 BF16 transpose/page-pack해서 cache에 넣는다. BF16 GGUF weight에는 수치
변환이 없다. 따라서 일반 prompt 실행 중에는
model weight repack이 다시 발생하지 않는다. `GGML_GEMMINI_PREPACK_WEIGHTS=0`이거나
non-model/transient tensor면 기존처럼 첫 사용 시 lazy repack될 수 있다.

```text
phase = "load/repack"
backend = "gemmini_bf16"
op = "weight_repack"
```

주의: lazy repack이 발생하는 경우에는 Gemmini `MUL_MAT` op 실행 중 발생하므로,
현재 summary 집계에서는 `load/repack` event와 해당 `MUL_MAT` op wall time이
겹칠 수 있다. prepack된 model weight의 repack 비용은 model load 시간으로 이동한다.

### Sampling Time

sampling은 `llama_sampler_sample()` 전후를 잰다.

```text
sample_us_start = ggml_time_us()
llama_sampler_sample()
sample_us_end = ggml_time_us()
```

sampling event는 `cpu_fallback` 또는 `cpu_baseline`으로 기록된다.

### Arrival Latency

arrival benchmark의 request timing:

```text
arrival_us = base_us + request_id * interval_us
wait_until_us(arrival_us)
start_us = ggml_time_us()
run_once(...)
finish_us = ggml_time_us()
```

CSV/summary 계산:

```text
e2e_us     = finish_us - arrival_us
queue_us   = start_us - arrival_us
service_us = finish_us - start_us
```

`group_seq`에서는 request가 순차 처리되므로 request 1..3의 `queue_us`가 커지는 것이
정상이다. `split_1gem`에서는 worker가 arrival 시간까지 sleep 후 바로 시작하므로
`queue_us`가 작고, `service_us`가 실제 실행 시간 대부분을 차지한다.

`arrival_cycles`는 `base_cycles + estimated_cycles_for_us(delta_us)`로 계산된다.
기본 `LLAMA_FIRESIM_CYCLES_PER_US=1000` 가정값을 사용하므로, arrival-relative
cycle은 근사치이다. 실제 실행 start/finish cycle은 `rdcycle`이다.

## Output Files

Guest result directory:

```text
/root/llama-results
```

FireMarshal copy-out 후 manager에서 보이는 위치:

```text
sims/firesim/deploy/results-workload/<run-name>/llama-firesim0/
```

Result files:

| file | 내용 |
| --- | --- |
| `run_summary.csv` | run-level prompt/generated tokens, total time, TTFT, TPOT |
| `token_trace.csv` | generated token id/piece |
| `op_profile.csv` | ggml op-level events, phase/backend/stage/dim/time |
| `stage_summary.csv` | stage별 aggregate |
| `backend_summary.csv` | backend/phase 및 Gemmini/FlashAttention subphase aggregate |
| `arrival_request_summary.csv` | request별 arrival/start/finish/e2e/queue/service |
| `arrival_mode_summary.csv` | group_seq vs split_1gem throughput/latency aggregate |
| `arrival_stage_summary.csv` | arrival request별 stage aggregate |
| `summary.md` | 사람이 읽는 aggregate report |

## How To Interpret Percentages

현재 `summary.md`의 phase/stage percent는 `op_profile.csv` event들을 집계한
runtime breakdown이다. `Backend Split`은 backend별 전체 op wall을 집계하고,
Gemmini와 FlashAttention의 측정 가능한 내부 구간은 각각 별도 breakdown으로
표시한다.

따라서 아래 현상은 현재 코드에서 가능하다.

```text
phase percent sum > 100%
stage percent > 100%
```

주요 이유:

1. `load/repack` event가 Gemmini `MUL_MAT` op wall time 안에서 발생하지만 별도 event로도 기록된다.
2. `stage_summary.csv`는 runtime op의 `wall_us`를 stage별로 합산한다.
3. `backend_summary.csv`의 `backend_gemmini_breakdown`과 `backend_flash_attention_breakdown`은 각 backend op wall을 실측 구간과 `other_host`로 나눈다.
4. `stage`는 backend 이름이 아니라 profiler의 op bucket이다. Projection matmul,
   ROPE, reshape/permute, KV cache, norm, residual 등으로 세분화하며,
   분류되지 않은 CPU op만 `other_cpu`로 남긴다.

해석 기준:

| 지표 | 현재 신뢰도 | 비고 |
| --- | --- | --- |
| `total time (us)` | 높음 | single request interval |
| `total cycles` | 높음 | rdcycle interval |
| arrival `e2e_us` | 높음 | arrival부터 finish까지 single interval |
| arrival `queue_us/service_us` | 높음 | arrival/start/finish interval |
| op-level `wall_us` | 참고용 | callback interval |
| phase/backend/stage `% of run` | 주의 필요 | inclusive aggregate, 100% 합산 보장 없음 |

exclusive percentage가 필요하면 `load/repack`을 별도 overhead로 두고 `MUL_MAT`
wall time에서 repack 시간을 빼거나, phase/backend/stage 집계를 non-overlap interval로
다시 정의해야 한다.

## Minimal Control Knobs

| env/command | 효과 |
| --- | --- |
| `:ctx-size [N|max]` | context size 조회/변경, current model max 확인 |
| `:n-predict [N|max]` | generated token 기본값 조회/변경, current context 기준 max 확인 |
| `:backend [gemmini|cpu]` | backend 조회 또는 CPU-only/hybrid run 선택 |
| `:gemmini-mode [multi|single]` | 조회 또는 compiled profile과 같은 값만 허용 |
| `:active-mask 0x1` | logical member 0; single은 physical custom3, multi는 custom0 |
| `:active-mask 0xf` | multi에서 logical member 0..3; single에서는 거부 |
| `:prompt-tokens P [N]` | 정확히 P개 token synthetic prompt를 N개 decode token으로 실행 |
| `:max-offloads N` | request마다 처음 N개 eligible matmul만 offload |
| `:trace 1` | `GEMMINI-ADMIT`, `GEMMINI-DISPATCH`, tiling trace 출력 |
| `:gemmini-smoke M N K` | llama graph 없이 direct Gemmini matmul smoke |
| `:arrival-benchmark N` | group_seq vs split_1gem arrival latency experiment |

## Debugging Path Map

CPU/Gemmini path를 확인할 때 볼 순서:

1. UART에서 `RUN-START,...backend=...` 확인
2. `:trace 1` 또는 `GGML_GEMMINI_TRACE=1`로 `GEMMINI-ADMIT` 출력 확인
3. `GEMMINI-DISPATCH`가 나오면 `ggml_backend_gemmini_mul_mat()`까지 도달한 것
4. `GEMMINI-RUN-BEGIN`이 나오면 `run_gemmini_matmul()` 진입
5. `GEMMINI-RUN-CALL`이 나오면 helper 호출 직전
6. `GEMMINI-RUN-END`가 나오면 helper loop와 `gemmini_fence()` 통과
7. `op_profile.csv`에 backend `gemmini_bf16` event가 있으면 callback 종료까지 완료

멈춤 위치별 의미:

| 마지막 trace | 가능성 |
| --- | --- |
| `GEMMINI-ADMIT`만 있음 | scheduler admission 후 graph compute 전 또는 첫 op 진입 전 |
| `GEMMINI-DISPATCH`까지 있음 | BF16 staging/packing 이후 helper 전후 문제 |
| `GEMMINI-RUN-CALL`까지 있음 | `shared_multi_tiled_matmul_job_init/step` 또는 `tiled_matmul_auto` 내부 장시간 실행/정지 |
| `GEMMINI-RUN-END` 없음 | Gemmini helper loop 또는 fence가 끝나지 않음 |
| `GEMMINI-RUN-END` 있음, op_profile 없음 | callback 종료 전후 runtime 문제 |
