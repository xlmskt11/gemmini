# llama-firesim 코드 구조

이 문서는 현재 FireSim `llama.cpp` + Gemmini 통합 코드의 전체 구조를
정리한다. 목적은 Gemmini mode에서 멈추는 경로를 좁혀서 디버깅하기 쉽게
만드는 것이다.

## Source Of Truth

수정해야 하는 원본 파일은 아래이다.

| 영역 | 경로 | 역할 |
| --- | --- | --- |
| FireMarshal workload | `llama-firesim.yaml` | rootfs 크기, guest command, copy-in 파일, copy-out 결과, FireSim output 정의 |
| build/install helper | `llama-firesim-build.sh` | `marshal build`와 `marshal install` 실행 |
| host build hook | `host-init.sh` | pinned `llama.cpp` clone, local patch 적용, cross compile, GGUF model download, guest 파일 staging |
| upstream patch copier | `apply-local-patches.sh` | local backend/CLI 파일을 clone된 `llama.cpp`에 복사하고 CMake/backend registry patch |
| guest launcher | `guest/llama-firesim-launch.sh` | guest에서 model/backend/env default 선택 후 CLI 실행 |
| UART CLI app | `src/examples/firesim/llama-firesim-cli.cpp` | REPL, benchmark mode, prefill/decode loop, token streaming, profiler 호출 |
| Gemmini backend header | `src/ggml/include/ggml-gemmini.h` | backend 등록 API와 profiler hook 선언 |
| Gemmini backend | `src/ggml/src/ggml-gemmini/ggml-gemmini.cpp` | ggml backend 등록, offload 조건, int8 repack/quantization, Gemmini matmul 호출, profiling output |
| Gemmini SW helper | `sims/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/gemmini_all.h` | 현재 FireSim bitstream과 같은 target-design Gemmini helper/header. backend가 직접 사용 |

아래 경로는 산출물이므로 지속적인 수정 위치가 아니다.

| 생성 경로 | 이유 |
| --- | --- |
| `build/llama.cpp-b9060/` | `host-init.sh`가 reset/clean한다. local `src/` 파일이 여기에 복사된다. |
| `build/llama.cpp-build/` | CMake build directory |
| `generated/root/` | FireMarshal rootfs에 들어갈 staging area. `host-init.sh`가 다시 만든다. |

## Build And Install Flow

manager 쪽 build 흐름은 다음과 같다.

1. `llama-firesim-build.sh`가 Chipyard/RISC-V tool PATH를 설정한다.
2. `software/firemarshal/marshal build llama-firesim.yaml`을 실행한다.
3. FireMarshal이 `host-init.sh`를 실행한다.
4. `host-init.sh`가 `ggml-org/llama.cpp` tag `b9060`을 clone/reset한다.
5. `apply-local-patches.sh`가 local `src/` 파일들을 clone된 upstream tree에 복사한다.
6. `apply-local-patches.sh`가 upstream CMake와 backend registry에 `GGML_GEMMINI`를 추가한다.
7. CMake가 target-design 쪽 `gemmini-rocc-tests` header를 include하도록 `GEMMINI_SW_DIR`를 설정한다.
8. CMake가 RISC-V Linux용 `llama-firesim-cli`를 `-DGGML_GEMMINI=ON`으로 cross-compile한다.
9. `host-init.sh`가 guest binary, launcher, GGUF model을 `generated/root/` 아래에 staging한다.
10. FireMarshal이 staging된 파일들을 rootfs image에 넣는다.
11. `marshal install`이 `sims/firesim/deploy/workloads/llama-firesim.json`을 생성/갱신한다.

`GEMMINI_SW_DIR`는 root 쪽 `generators/gemmini/software/gemmini-rocc-tests`가
아니라 FireSim target-design 쪽 경로를 사용한다. 이 경로의
`gemmini_params.h`가 실제 bitstream의 `DIM`, scratchpad, accumulator 설정과
맞아야 한다.

guest software나 rootfs에 들어가는 파일을 바꾸면 FireSim에서 다시
`infrasetup`을 해야 한다. 그렇지 않으면 FPGA host가 이전 rootfs를 계속
쓸 수 있다.

## Runtime Boot Flow

FireSim 실행 시 사용하는 주요 파일은 아래이다.

| Runtime file | 역할 |
| --- | --- |
| `sims/firesim/deploy/config_runtime.yaml` | `workload_name: llama-firesim.json` 선택 |
| `sims/firesim/deploy/workloads/llama-firesim.json` | FireMarshal bootbinary/rootfs와 output path를 FireSim에 전달 |
| guest `/etc/init.d/S99run` | FireMarshal workload command 실행 |
| guest `/root/llama-firesim/llama-firesim-launch.sh` | backend/runtime knob 설정 후 CLI exec |
| guest `/root/llama-firesim/llama-firesim-cli` | UART REPL 또는 benchmark 실행 |

현재 launcher default는 아래이다.

| 설정 | 기본값 |
| --- | --- |
| `LLAMA_FIRESIM_BACKEND` | `gemmini` |
| `LLAMA_FIRESIM_CTX_SIZE` | `2048` |
| `LLAMA_FIRESIM_N_PREDICT` | `16` |
| `LLAMA_FIRESIM_RESULTS_DIR` | `/root/llama-results` |

guest에는 단일 모델만 들어간다.

| Build-time selector | Packaged GGUF |
| --- | --- |
| `gemma-3-270m` | Gemma 3 270M Q8_0 |
| `smollm2-360m` | SmolLM2 360M Instruct Q8_0 |
| `qwen3-0.6b` | Qwen3 0.6B Q8_0 |
| `llama-3.2-1b` | Llama 3.2 1B Instruct Q8_0 |

선택은 `LLAMA_FIRESIM_MODEL`로 `host-init.sh` 실행 시점에만 수행되고,
guest path는 항상 `/root/models/model.gguf`이다. `LLAMA_FIRESIM_MODEL_URL`을
주면 selector 대신 custom single GGUF URL을 사용할 수 있다.

## UART CLI Flow

`llama-firesim-cli.cpp`가 사용자 interaction을 담당한다.

| CLI command | 동작 |
| --- | --- |
| 일반 prompt | 현재 `--backend` 설정으로 1회 실행 |
| `:backend [gemmini|cpu]` | 현재 backend 조회 또는 backend 변경 |
| `:ctx-size [N|max]` | 현재 context size/max context 조회 또는 context size 변경 |
| `:n-predict [N|max]` | 현재 generated-token 기본값/max 조회 또는 변경 |
| `:active-mask [MASK]` | `GGML_GEMMINI_ACTIVE_MASK` 조회 또는 변경 |
| `:max-offloads [N]` | `GGML_GEMMINI_MAX_OFFLOADS` 조회 또는 변경 |
| `:trace [0|1|on|off]` | `GGML_GEMMINI_TRACE` 조회 또는 변경 |
| `:benchmark [n_predict] PROMPT` | `hybrid` 실행 후 `cpu_baseline` 실행, CSV/Markdown 결과 출력 |
| `:quit` | guest poweroff |

각 run의 흐름은 다음과 같다.

1. model은 `app` constructor에서 `/root/models/model.gguf`를 한 번 load한다.
2. model load 전에 `mlockall(MCL_CURRENT | MCL_FUTURE)`를 시도한다.
3. `run_command()`가 profiler를 reset하고 model-load 시간을 기록한다.
4. `run_once()`가 fresh `llama_context`를 만든다.
5. prompt token 전체를 한 번의 `prefill` `llama_decode()`로 처리한다.
6. token을 greedy sampler로 하나 뽑는다.
7. 마지막 token이 아니면 방금 뽑은 token 하나를 다시 `decode` `llama_decode()`에 넣는다.
8. UART에는 ` -> piece` 형태로 token piece가 streaming된다.
9. run 또는 benchmark가 끝나면 profiler가 결과 파일을 쓴다.

CLI는 ggml eval callback을 설치한다. callback은 각 ggml tensor evaluation의
시작/종료 시각을 기록한다. 하지만 Gemmini custom instruction,
`gemmini_fence()`, 또는 helper step loop 안에서 멈추면 해당 node의 종료
callback까지 도달하지 못한다.

## Backend Registration

`apply-local-patches.sh`는 upstream `llama.cpp`를 아래처럼 patch한다.

| Upstream file | Patch |
| --- | --- |
| `ggml/CMakeLists.txt` | `option(GGML_GEMMINI ...)` 추가 |
| `ggml/src/CMakeLists.txt` | `ggml_add_backend(GEMMINI)` 추가 |
| `ggml/src/ggml-backend-reg.cpp` | `ggml-gemmini.h` include 및 `ggml_backend_gemmini_reg()` 등록 |
| `examples/CMakeLists.txt` | `examples/firesim` 추가 |

runtime에서 `llama-firesim-cli.cpp`가 `ggml_backend_load_all()`을 호출하면
upstream backend registry를 통해 Gemmini backend device가 등록된다.

## Gemmini Offload Rules

Gemmini backend가 지원한다고 광고하는 ggml op는 제한적이다.

| ggml op | Backend 동작 |
| --- | --- |
| `GGML_OP_MUL_MAT` | 조건을 만족할 때만 Gemmini offload |
| `GGML_OP_NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE` | supported subgraph 배치를 위한 metadata/view op로 허용 |
| 그 외 모든 op | CPU fallback |

`GGML_OP_MUL_MAT`은 아래 조건을 모두 만족해야 Gemmini로 간다.

| 조건 | 이유 |
| --- | --- |
| `GGML_GEMMINI_DISABLE`이 꺼져 있음 | CPU-only baseline 지원 |
| `GGML_GEMMINI_ACTIVE_MASK != 0` | 적어도 하나의 accelerator 필요 |
| `src0`, `src1`, destination shape가 backend 기대 layout과 맞음 | 현재 dense matmul layout만 구현 |
| `src0`, `src1`의 innermost dimension이 contiguous | row-wise float conversion과 int8 quantization을 위해 필요 |
| destination type이 `F32` | Gemmini 결과를 다시 F32로 rescale해서 ggml에 반환 |
| `src0`, `src1`이 F32이거나 `to_float` 변환 지원 type | Gemmini 입력은 실행 직전 int8로 quantize |
| activation row 수가 1 이상 | 비어 있는 matmul 제외 |
| `GGML_GEMMINI_MAX_OFFLOADS` 제한 이하 | 기본값 `-1`에서는 제한 없이 모든 eligible matmul offload |

현재는 model-weight 이름으로 offload 여부를 제한하지 않는다. 아래 이름 패턴은
static model weight cache 여부와 stage 분류에만 사용된다.

| Stage | Name patterns |
| --- | --- |
| Q projection | `attn_q.weight`, `.wq` |
| K projection | `attn_k.weight`, `.wk` |
| V projection | `attn_v.weight`, `.wv` |
| O projection | `attn_output.weight`, `.wo` |
| FFN gate | `ffn_gate.weight` |
| FFN up | `ffn_up.weight` |
| FFN down | `ffn_down.weight` |
| LM head | `output.weight`, tied `token_embd.weight` used by `result_output` |

아래 작업은 CPU에 남는다.

| 작업 | 이유 |
| --- | --- |
| norm, RoPE, softmax, KV cache, sampling | Gemmini backend 구현 없음 |
| output이 F32가 아니거나 innermost stride가 맞지 않는 matmul | 현재 backend output/writeback layout 밖 |

중요한 점은 projection/FFN activation인 `src1`은 runtime에서 보통 F32라는
것이다. 현재 all-GEMM mode에서는 F32 attention-score/value matmul도 Gemmini에
넣기 직전에 `src0`, `src1`을 int8로 quantize한다. model weight로 분류되는
`src0`만 repack cache를 재사용하고, runtime intermediate `src0`는 stale data를
피하기 위해 매 호출마다 다시 repack한다.

## Gemmini Data Path

offload된 `MUL_MAT(src0, src1) -> dst`는 Gemmini에 아래처럼 mapping된다.

| Gemmini operand | Source | 현재 format |
| --- | --- | --- |
| `A` | runtime activation `src1` | F32 또는 dequantized input에서 row-wise int8 quantize |
| `B` | matmul left operand `src0` | F32 또는 dequantized input에서 int8 repack, static model weight는 cache |
| `C` | temporary accumulator | `acc_t` / int32 |
| `dst` | ggml output tensor | rescale 후 F32 |

세부 흐름은 `ggml-gemmini.cpp` 기준으로 아래이다.

1. `tensor_row_to_float()`가 tensor row를 float로 변환한다. quantized ggml
   tensor는 ggml type trait의 `to_float`를 호출한다.
2. `quantize_row_to_i8()`가 row별 max-abs scale을 계산하고 int8 buffer를 만든다.
3. `populate_weight_cache_entry()`가 `src0` row를 float로 변환한 뒤 `K x J`
   int8 matrix로 repack한다.
4. static model weight로 분류되는 `src0`만 `get_or_create_weight_cache()`로
   repack/cache한다. runtime intermediate `src0`는 매 호출마다 다시 repack한다.
5. `ggml_backend_gemmini_mul_mat()`가 매 호출마다 activation row를 int8로 quantize한다.
6. A/B/C buffer는 `MAX_BYTES` alignment로 할당한다.
7. `run_gemmini_matmul()`이 Gemmini에 넘길 A/B/C buffer를 page 단위로 prefault한다.
8. `run_gemmini_matmul()`이 `shared_multi_tiled_matmul_job_step()` path를
   사용한다.
9. backend가 `acc_t` output을 `activation_scale[row] * weight_scale[col]`로
   rescale해서 ggml destination에 F32로 쓴다.

현재 Gemmini call에는 bias fuse가 없다.

| Job field | 현재 값 |
| --- | --- |
| `D` | `nullptr` |
| `repeating_bias` | `false` |
| `act` | `NO_ACTIVATION` |
| `scale` | `ACC_SCALE_IDENTITY` |
| `full_C` | `true`, 즉 `C`는 `acc_t` / int32 |
| `dataflow` | `WEIGHT_STATIONARY` |
| `a_transpose` | `false` |
| `b_transpose` | `false` |

## Multi-Gemmini Mapping

backend는 자체 low-level tiler를 구현하지 않는다. 대신
`shared_multi_matmul_job_t`를 채운 뒤 `gemmini_all.h` scheduling helper에 맡긴다.

현재 job mapping은 아래이다.

| `shared_multi_matmul_job_t` field | 값 |
| --- | --- |
| `gemmini_list` | `GGML_GEMMINI_ACTIVE_MASK`, default `0xf` |
| `dim_I` | activation row 수 |
| `dim_J` | output column 수 |
| `dim_K` | inner dimension |
| `A` | quantized activation buffer |
| `B` | repacked quantized weight buffer |
| `C` | int32 accumulator buffer |
| `stride_A` | `dim_K` |
| `stride_B` | `dim_J` |
| `stride_C` | `dim_J` |
| `sp_addr_range` | `TOTAL_SPAD_ROWS` |
| `acc_addr_range` | `TOTAL_ACC_ROWS` |

`run_gemmini_matmul()` 실행 순서는 아래이다.

1. active accelerator별 Gemmini counter configure/reset
2. active mask에 포함된 Gemmini를 process lifetime 기준 한 번 flush
3. `shared_multi_choose_tiling_factors(&job)` 호출
4. `shared_multi_tiled_matmul_job_init(&job)` 호출
5. `job.done`이 될 때까지 `shared_multi_tiled_matmul_job_step(&job)` 반복
6. `gemmini_fence()` 호출
7. load/store/execute/preload/wait counter read

`shared_multi_tiled_matmul_job_step()` 내부에서 helper가 `gemmini_list`를
보고 active Gemmini를 선택한다. 각 tile을 선택된 accelerator들에 나눠서
`shared_multi_sp_tiled_matmul_ws` 또는 `shared_multi_sp_tiled_matmul_os`를
호출한다. 현재 backend는 항상 weight-stationary를 선택한다.

## Profiling Outputs

guest는 `/root/llama-results/` 아래에 결과를 쓴다. workload가 종료되면
FireSim이 이 파일들을 copy-out한다.

| 파일 | 내용 |
| --- | --- |
| `run_summary.csv` | run별 prompt/generated tokens, total time, TTFT, TPOT |
| `token_trace.csv` | run label, generated token index, token id, decoded piece |
| `op_profile.csv` | ggml op별 timing, backend/stage label, shape, Gemmini host-side subphase, Gemmini counter |
| `stage_summary.csv` | stage별 aggregate와 run 대비 비율 |
| `backend_summary.csv` | backend/phase/Gemmini host-side subphase/Gemmini counter subphase aggregate |
| `summary.md` | 사람이 읽기 위한 request summary, TTFT/TPOT, aggregate breakdown |

`op_profile.csv`의 raw backend label 의미는 아래이다.

| Label | 의미 |
| --- | --- |
| `gemmini` | hybrid run에서 Gemmini로 완료된 op |
| `cpu_fallback` | hybrid run에서 CPU backend로 실행된 op |
| `cpu_baseline` | CPU-only benchmark pass |

`summary.md`와 `backend_summary.csv`의 `Backend Split`은 Gemmini op를 다시 나눈다.
여기서 `gemmini`는 `run_gemmini_matmul()` call interval만 의미하고,
`gemmini_host_overhead`는 weight 준비, activation quantization, output
dequant/store를 포함한 나머지 Gemmini backend-side overhead이다.

stage label은 ggml node name, source tensor name, op name에서 추론한다. stage
label은 aggregation/debugging용이며 별도의 scheduler는 아니다. Projection
matmul은 `q_proj/k_proj/v_proj/o_proj/ffn_*`로 남기고, ROPE, reshape/permute,
KV cache read/write, norm, residual, embedding은 별도 bucket으로
분리해 projection 시간이 부풀려지지 않게 한다. 분류되지 않은 CPU op만
`other_cpu`로 집계한다.

## Current Debugging Implications

현재 문제는 CPU path가 아니라 Gemmini execution path에서 발생한다. 코드
구조상 가장 좁은 critical section은 아래이다.

1. ggml scheduler가 eligible `MUL_MAT`을 `GEMMINI`에 배치한다.
2. `ggml_backend_gemmini_mul_mat()`가 input repack/quantization을 수행한다.
3. `run_gemmini_matmul()`이 `shared_multi_matmul_job_t`를 만든다.
4. `shared_multi_tiled_matmul_job_step()`이 하나 이상의 custom accelerator에 RoCC command를 발행한다.
5. guest가 step loop, custom instruction, DMA, 또는 `gemmini_fence()` 안에서 멈출 수 있다.

isolation에 유용한 knob는 아래이다.

| Knob | 효과 |
| --- | --- |
| `LLAMA_FIRESIM_BACKEND=cpu` | 전체 CPU sanity path |
| `GGML_GEMMINI_DISABLE=1` | backend가 등록돼도 CPU fallback 강제 |
| `GGML_GEMMINI_ACTIVE_MASK=0x1` | `custom0`만 사용 |
| `GGML_GEMMINI_ACTIVE_MASK=0x2` | `custom1`만 사용 |
| `GGML_GEMMINI_ACTIVE_MASK=0x4` | `custom2`만 사용 |
| `GGML_GEMMINI_ACTIVE_MASK=0x8` | `custom3`만 사용 |
| `GGML_GEMMINI_MAX_OFFLOADS=N` | run마다 최대 N개의 unique eligible matmul만 Gemmini offload. 기본값은 `-1`로 제한 없음, `0`은 offload 없음 |
| `GGML_GEMMINI_PREPACK_WEIGHTS=0` | model load 직후 weight int8/repack 선처리를 끄고 첫 사용 시 lazy repack |
| `GGML_GEMMINI_COUNTERS=1` | Gemmini performance counter 설정/읽기 활성화. 기본값은 `0`으로 RoCC counter path를 우회 |
| `GGML_GEMMINI_PROFILE=1` | 완료된 op에 대한 profiler output 활성화 |
| `GGML_GEMMINI_TRACE=1` | Gemmini dispatch 직전에 stage/src0/shape/mask를 UART에 출력 |
| `LLAMA_FIRESIM_MLOCK=0` | process-wide `mlockall()` 비활성화 |
| `LLAMA_FIRESIM_MLOCK_STRICT=1` | `mlockall()` 실패 시 guest CLI를 즉시 실패 처리 |
| `GGML_GEMMINI_PREFAULT=0` | Gemmini A/B/C buffer prefault 비활성화 |

UART가 `PREFILL-START` 또는 `DECODE-START` 뒤에 멈추면 Ctrl-C가 안 먹을 수
있다. core가 RoCC/Gemmini path 안에서 stuck되면 CLI signal handler까지
돌아오지 못하기 때문이다. 기존 eval callback은 완료된 op 이후에만 progress를
찍으므로, 하나의 offloaded op 안에서 멈추면 그 op의 완료 row는
`op_profile.csv`에 남지 않는다.

다음으로 가장 실용적인 debug step은 `run_gemmini_matmul()` 직전에 UART log를
추가해서 tensor name, stage, dimension, active mask, slice index를 찍고,
`GGML_GEMMINI_ACTIVE_MASK=0x1`로 재현하는 것이다. 그러면 처음 멈추는 정확한
Gemmini matmul을 확인할 수 있다.
