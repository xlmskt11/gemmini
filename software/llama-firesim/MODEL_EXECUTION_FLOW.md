# llama-firesim Model Execution Flow

This document describes the current default model execution order in the
`llama-firesim` workload, including equations, matrix dimensions, and operation
dependencies. It is written for the default model selected by
`host-init.sh`.

> Backend update: the current accelerator path is profile-locked BF16
> Gemmini/FP32 accumulation plus exact VPU and conditional FlashAttention. The
> sections below describe that current numerical contract.

## Current Default Model

`host-init.sh` defaults to:

| item | value |
| --- | --- |
| packaged model | default Gemma 3 270M BF16 |
| GGUF path in guest | `/root/models/model.gguf` |
| architecture | `gemma3` |
| model name | `Gemma 3 270m` |
| runtime context | `LLAMA_FIRESIM_CTX_SIZE=2048` |
| default backend | `LLAMA_FIRESIM_BACKEND=gemmini` |
| default generated tokens | `LLAMA_FIRESIM_N_PREDICT=16` |

Other supported build-time selectors are `smollm2-360m`, `qwen3-0.6b`, and
`llama-3.2-1b`. Those models still appear in the guest as
`/root/models/model.gguf`; the operation details below are for the default
Gemma model.

The GGUF metadata for the bundled default model is:

| parameter | symbol | value |
| --- | --- | --- |
| layers | `L` | 18 |
| hidden size | `H` | 640 |
| FFN hidden size | `F` | 2048 |
| query heads | `N_h` | 4 |
| KV heads | `N_kv` | 1 |
| head dimension | `D` | 256 |
| query projection size | `H_q = N_h * D` | 1024 |
| KV projection size | `H_kv = N_kv * D` | 256 |
| vocabulary size | `V` | 262144 |
| train context length | | 32768 |
| sliding window | | 512 |
| RoPE base | | 1000000 |
| RMSNorm epsilon | | `1e-6` |

Gemma 3 uses interleaved sliding-window attention. The GGUF does not provide a
custom pattern, so llama.cpp uses the default pattern `6`: layers whose
`layer_id % 6 < 5` are SWA layers and every sixth layer is dense/full-attention.

| layer ids | attention range |
| --- | --- |
| `0-4`, `6-10`, `12-16` | sliding window, up to 512 previous positions |
| `5`, `11`, `17` | dense/full KV history |

## Runtime Shape Convention

Use these symbols throughout the document:

| symbol | meaning |
| --- | --- |
| `T` | number of tokens in the current `llama_decode()` batch |
| `C` | current KV-cache sequence length after writing the current batch |
| `S_l` | visible KV length for layer `l`; `min(C, 512)` for SWA layers, `C` for dense layers |
| `H` | hidden size, 640 |
| `D` | per-head dimension, 256 |
| `N_h` | query head count, 4 |
| `N_kv` | KV head count, 1 |
| `H_q` | query/output attention width, 1024 |
| `F` | FFN hidden size, 2048 |
| `V` | vocabulary size, 262144 |

llama.cpp stores dense weights as `ggml` tensors with shape `[K, N]`, and
computes:

```text
Y = W^T X
W shape: [K, N]
X shape: [K, T]
Y shape: [N, T]
```

In the current Gemmini wrapper, this maps to:

```text
Gemmini dim_I = T
Gemmini dim_J = N
Gemmini dim_K = K
```

During prefill, `T = prompt_tokens`. During decode, `T = 1` for each generated
token decode step.

## Top-Level Request Flow

The CLI execution order is:

1. Load GGUF model once.
2. For each prompt, create a fresh `llama_context`.
3. Tokenize prompt.
4. Run one prefill `llama_decode()` over all prompt tokens.
5. Greedy-sample one token from logits.
6. For each remaining generated token, run one decode `llama_decode()` with the previous sampled token.
7. Write token trace and profiling CSV/Markdown.

For one `llama_decode()`, the graph flow is:

```text
tokens
  -> token embedding lookup
  -> embedding scale
  -> layer 0
  -> layer 1
  -> ...
  -> layer 17
  -> final RMSNorm
  -> LM head
  -> logits
```

## Per-Layer Equations

Let `X_l in R^{H x T}` be the input hidden states to layer `l`.

### 1. Attention Input Norm

```text
A_l = RMSNorm(X_l) * gamma_attn_l
```

Shapes:

| tensor | shape |
| --- | --- |
| `X_l` | `[640, T]` |
| `gamma_attn_l` | `[640]` |
| `A_l` | `[640, T]` |

This is CPU fallback today because RMSNorm and elementwise multiply are not
Gemmini backend ops.

### 2. Q/K/V Projections

```text
Q_raw_l = Wq_l^T A_l
K_raw_l = Wk_l^T A_l
V_l     = Wv_l^T A_l
```

Shapes:

| op | weight shape `[K,N]` | input shape | output shape | Gemmini dims `(I,J,K)` |
| --- | --- | --- | --- | --- |
| Q projection | `[640, 1024]` | `[640, T]` | `[1024, T]` | `(T, 1024, 640)` |
| K projection | `[640, 256]` | `[640, T]` | `[256, T]` | `(T, 256, 640)` |
| V projection | `[640, 256]` | `[640, T]` | `[256, T]` | `(T, 256, 640)` |

Then llama.cpp reshapes:

```text
Q_raw_l -> [D, N_h, T]  = [256, 4, T]
K_raw_l -> [D, N_kv, T] = [256, 1, T]
V_l     -> [D, N_kv, T] = [256, 1, T]
```

These projection matmuls are eligible for Gemmini offload. The current Gemmini
backend stages operands as BF16-RNE, runs BF16 matmul with FP32 accumulation,
and unpacks/stores the FP32 result in the destination tensor layout.

### 3. Q/K Norm, RoPE, And Attention Scaling

Gemma 3 applies per-head RMSNorm to Q and K, then RoPE:

```text
Q_norm_l = RMSNorm(Q_raw_l) * gamma_q_l
K_norm_l = RMSNorm(K_raw_l) * gamma_k_l

Q_rope_l = RoPE(Q_norm_l, position)
K_rope_l = RoPE(K_norm_l, position)

score_l = attention_scale * (Q_rope_l * K_rope_l^T)
```

For this model:

```text
attention_scale = 1 / sqrt(D) = 1 / sqrt(256) = 1/16
```

Shapes:

| tensor | shape |
| --- | --- |
| `gamma_q_l` | `[256]` |
| `gamma_k_l` | `[256]` |
| `Q_l` | `[256, 4, T]` |
| `K_rope_l` | `[256, 1, T]` |

RMSNorm and RoPE are VPU-eligible when their runtime contracts are satisfied.
The fused FlashAttention path stages Q without multiplying it and applies
`attention_scale` directly to the FP32 QK score in the VPU kernel.

### 4. KV Cache Store

The current batch K/V is written into the live KV cache:

```text
cache_K_l[position : position + T] = K_rope_l
cache_V_l[position : position + T] = V_l
```

In ggml this appears as `SET_ROWS` nodes. The upstream default is F16, while
`llama-firesim-cli` explicitly selects BF16 for both caches:

| cache | dtype |
| --- | --- |
| K cache | `BF16` |
| V cache | `BF16` |

With FlashAttention and BF16 K/V, the local cache patch allocates each layer
as ggml `[C, Tcap, Hkv, S]`, physically
`[stream][KV head][token][channel]`. The token index tensor is broadcast over
the head and stream dimensions by one `SET_ROWS`, so the projection result is
written directly into HTC without a separate repack. Reads retain llama.cpp's
logical `[C, Hkv, n_kv, S]` contract; the existing `(0,2,1,3)` permute then
gives FlashAttention `[C, n_kv, Hkv, S]` with a dense `C*sizeof(BF16)` token
stride. Other dtype or non-Flash cache configurations retain the upstream
layout. New HTC state payloads use an explicit layout marker and bulk
head-major ranges; the reader still accepts legacy cell-major,
non-transposed cache state files.

This is CPU fallback today. It is a cache write/copy, not a Gemmini mvin into
scratchpad.

### 5. Attention Core

For each query head `h`, Gemma 3 has grouped-query attention with one KV head,
so all four query heads share the same K/V head.

Conceptually:

```text
score_l[h, t, j] = dot(Q_l[:, h, t], cache_K_l[:, 0, j]) + mask_l[t, j]
P_l[h, t, :]     = softmax(score_l[h, t, :])
O_l[:, h, t]     = sum_j P_l[h, t, j] * cache_V_l[:, 0, j]
```

where `j` ranges over visible KV positions:

```text
j in [0, S_l)
S_l = min(C, 512) for SWA layers
S_l = C for dense layers
```

The conceptual matmul dimensions are:

| operation | per-head input | output | conceptual dims |
| --- | --- | --- | --- |
| attention score `Q K^T` | `Q [T, 256]`, `K [S_l, 256]` | `[T, S_l]` | `(I=T, J=S_l, K=256)` |
| attention value `P V` | `P [T, S_l]`, `V [S_l, 256]` | `[T, 256]` | `(I=T, J=256, K=S_l)` |

Current graph behavior:

| mode | current behavior |
| --- | --- |
| default `flash_attn=auto` | attention core is a fused `FLASH_ATTN_EXT` op |
| backend placement | exact prefix-causal/no-bias/no-softcap cases use the fused VPU+Gemmini kernel; other semantics fall back to CPU |
| Gemmini visibility | the fused kernel internally issues `QK^T` and `P*V` Gemmini jobs |

Fused FlashAttention does not reuse the ordinary matmul
`GGML_GEMMINI_PAGE_PACKED_{A,B,C,D}` settings and exposes no Q/K/V
page-packing option. The adapter supplies linear BF16 Q, K, and V matrices.
Dense BF16 K/V head slices can be consumed directly; incompatible views are
gathered into reusable linear staging workspaces. After final normalization
the VPU writes FP32 rows directly to the final ggml destination in row-major
order. The online accumulator stays in VSRAM and has no host page layout.

If FlashAttention is disabled, llama.cpp can build explicit `kq = mul_mat(k,q)`,
`softmax`, and `kqv = mul_mat(v,kq)` nodes. In the current default path,
however, these are fused inside `FLASH_ATTN_EXT`.

After attention, heads are concatenated:

```text
O_l shape = [1024, T]
```

### 6. Output Projection

```text
SA_proj_l = Wo_l^T O_l
```

Shapes:

| op | weight shape `[K,N]` | input shape | output shape | Gemmini dims `(I,J,K)` |
| --- | --- | --- | --- | --- |
| O projection | `[1024, 640]` | `[1024, T]` | `[640, T]` | `(T, 640, 1024)` |

This matmul is eligible for Gemmini offload.

### 7. Attention Post-Norm And Residual

Gemma 3 normalizes the attention projection output before adding it back:

```text
SA_norm_l = RMSNorm(SA_proj_l) * gamma_attn_post_l
SA_l      = X_l + SA_norm_l
```

Shapes:

| tensor | shape |
| --- | --- |
| `gamma_attn_post_l` | `[640]` |
| `SA_l` | `[640, T]` |

This stage is CPU fallback today.

### 8. FFN Input Norm

```text
F_in_l = RMSNorm(SA_l) * gamma_ffn_l
```

Shapes:

| tensor | shape |
| --- | --- |
| `gamma_ffn_l` | `[640]` |
| `F_in_l` | `[640, T]` |

This is CPU fallback.

### 9. Gated FFN

Gemma 3 270M uses a parallel GEGLU-style FFN:

```text
U_l = Wup_l^T   F_in_l
G_l = Wgate_l^T F_in_l
M_l = GELU(G_l) * U_l
D_l = Wdown_l^T M_l
```

Shapes:

| op | weight shape `[K,N]` | input shape | output shape | Gemmini dims `(I,J,K)` |
| --- | --- | --- | --- | --- |
| FFN up | `[640, 2048]` | `[640, T]` | `[2048, T]` | `(T, 2048, 640)` |
| FFN gate | `[640, 2048]` | `[640, T]` | `[2048, T]` | `(T, 2048, 640)` |
| GEGLU | elementwise | `[2048, T]`, `[2048, T]` | `[2048, T]` | not matmul |
| FFN down | `[2048, 640]` | `[2048, T]` | `[640, T]` | `(T, 640, 2048)` |

`FFN up`, `FFN gate`, and `FFN down` are eligible for Gemmini offload.
`GELU` and elementwise multiply are CPU fallback.

### 10. FFN Post-Norm And Residual

```text
F_norm_l = RMSNorm(D_l) * gamma_ffn_post_l
X_{l+1}  = SA_l + F_norm_l
```

Shapes:

| tensor | shape |
| --- | --- |
| `gamma_ffn_post_l` | `[640]` |
| `X_{l+1}` | `[640, T]` |

This stage is CPU fallback.

## Final Output

After layer 17:

```text
Y = RMSNorm(X_18) * gamma_output
logits = W_out^T Y
```

Gemma 3 270M in this GGUF ties `W_out` to the token embedding tensor if no
separate output tensor exists.

Shapes:

| op | weight shape `[K,N]` | input shape | output shape | Gemmini dims `(I,J,K)` |
| --- | --- | --- | --- | --- |
| final LM head | `[640, 262144]` | `[640, T_out]` | `[262144, T_out]` | `(T_out, 262144, 640)` |

`T_out` is normally the number of requested output rows. In this CLI path, the
sampler only needs the most recent logits row, so this is usually effectively
one token during decode.

The LM head is a `MUL_MAT` and is eligible for Gemmini offload if it satisfies
the current backend shape/layout checks. In practice this is a very wide output
matrix because `V = 262144`.

## Dependency Graph

Per layer, the dependency graph is:

```text
X_l
  -> RMSNorm + attn_norm
       -> Q projection -> Q RMSNorm -> RoPE
       -> K projection -> K RMSNorm -> RoPE -> KV cache store
       -> V projection ----------------------> KV cache store
  -> attention(Q, cache_K, cache_V, mask)
  -> O projection
  -> RMSNorm + attn_post_norm
  -> residual add with X_l = SA_l
  -> RMSNorm + ffn_norm
       -> FFN up ----\
       -> FFN gate --> GEGLU -> FFN down
  -> RMSNorm + ffn_post_norm
  -> residual add with SA_l = X_{l+1}
```

The main parallelism inside one layer is:

| independent once dependency is ready | common dependency |
| --- | --- |
| Q, K, V projection | `A_l = RMSNorm(X_l) * gamma_attn_l` |
| FFN up and FFN gate | `F_in_l = RMSNorm(SA_l) * gamma_ffn_l` |

Hard serialization points are:

| serialization point | why |
| --- | --- |
| attention waits for Q, K, V and KV cache writes | attention reads the live cache |
| O projection waits for attention output | dense projection input is attention result |
| FFN waits for attention residual `SA_l` | transformer block order |
| FFN down waits for both FFN up and gate | GEGLU requires both tensors |
| next layer waits for `X_{l+1}` | sequential transformer layers |
| final LM head waits for all 18 layers | logits need final hidden state |

## Current Backend Mapping

The registered accelerator backend sends supported `GGML_OP_MUL_MAT` nodes to
Gemmini, exact supported vector nodes to the VPU, and admitted
`GGML_OP_FLASH_ATTN_EXT` nodes to fused FlashAttention. Other nodes fall back
to the CPU.

| stage | current default op type | backend today |
| --- | --- | --- |
| token embedding lookup | `GET_ROWS` | CPU fallback |
| embedding scale | `SCALE` | CPU fallback |
| RMSNorms | `RMS_NORM` + `MUL` | VPU eligible, CPU fallback otherwise |
| Q/K/V projections | `MUL_MAT` | Gemmini eligible |
| Q/K RoPE | `ROPE` | VPU eligible, CPU fallback otherwise |
| KV cache write | `SET_ROWS` | CPU fallback |
| attention core | `FLASH_ATTN_EXT` | fused Gemmini+VPU eligible, CPU fallback otherwise |
| O projection | `MUL_MAT` | Gemmini eligible |
| FFN up/gate/down | `MUL_MAT` | Gemmini eligible |
| GEGLU | `GEGLU` | VPU eligible, CPU fallback otherwise |
| residual adds | `ADD` | VPU eligible, CPU fallback otherwise |
| final LM head | `MUL_MAT` | Gemmini eligible |
| sampling | sampler code | CPU |

For offloaded matmuls, the current Gemmini path uses:

```text
F32, BF16, or supported GGUF tensor
  -> decode/encode as BF16-RNE where needed
  -> transpose/page-pack the weight and stage/page-pack the activation
  -> Gemmini BF16 matmul, FP32 accumulation
  -> unpack/store the FP32 ggml output
```

Model weights such as `attn_q.weight`, `attn_k.weight`, `attn_v.weight`,
`attn_output.weight`, `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`, and
`output.weight` are cached after repacking. Runtime intermediates are repacked
per call to avoid stale view data.

## Per-Layer Matmul Summary

For each of the 18 layers, the dense matmuls are:

| order | stage | Gemmini dims prefill `(I,J,K)` | Gemmini dims decode `(I,J,K)` | dependency |
| --- | --- | --- | --- | --- |
| 1 | Q projection | `(T, 1024, 640)` | `(1, 1024, 640)` | attention norm |
| 2 | K projection | `(T, 256, 640)` | `(1, 256, 640)` | attention norm |
| 3 | V projection | `(T, 256, 640)` | `(1, 256, 640)` | attention norm |
| 4 | attention score, conceptual only | `(T, S_l, 256)` | `(1, S_l, 256)` | Q and K cache |
| 5 | attention value, conceptual only | `(T, 256, S_l)` | `(1, 256, S_l)` | softmax scores and V cache |
| 6 | O projection | `(T, 640, 1024)` | `(1, 640, 1024)` | attention output |
| 7 | FFN gate | `(T, 2048, 640)` | `(1, 2048, 640)` | FFN norm |
| 8 | FFN up | `(T, 2048, 640)` | `(1, 2048, 640)` | FFN norm |
| 9 | FFN down | `(T, 640, 2048)` | `(1, 640, 2048)` | GEGLU output |

Rows 4 and 5 are not Gemmini matmuls in the current default graph because they
are fused into `FLASH_ATTN_EXT`.

After all layers:

| stage | Gemmini dims prefill/decode `(I,J,K)` | dependency |
| --- | --- | --- |
| LM head | `(T_out, 262144, 640)` | final RMSNorm |

## Source Locations

| file | relevant content |
| --- | --- |
| `guest/llama-firesim-launch.sh` | default model/backend/runtime settings |
| `src/models/gemma3.cpp` in the pinned llama.cpp tree | Gemma 3 graph order and equations |
| `src/llama-graph.cpp` in the pinned llama.cpp tree | QKV, attention, FFN helper implementations |
| `src/llama-context.cpp` in the pinned llama.cpp tree | default context params, KV dtype, FlashAttention auto |
| `src/ggml/src/ggml-gemmini/ggml-gemmini.cpp` | Gemmini offload conditions, BF16 staging/packing, profiling |
