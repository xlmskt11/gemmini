#ifndef GGML_GEMMINI_VPU_H
#define GGML_GEMMINI_VPU_H

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_GEMMINI_VPU_REASON_CAPACITY 256

enum ggml_gemmini_vpu_status {
    GGML_GEMMINI_VPU_STATUS_OK = 0,
    GGML_GEMMINI_VPU_STATUS_UNSUPPORTED,
    GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT,
    GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED,
    GGML_GEMMINI_VPU_STATUS_HARDWARE_ERROR,
};

struct ggml_gemmini_flash_breakdown {
    // Retained for profiler schema compatibility. The fast path does not
    // rescan Q/K/V values, so successful executions leave these fields zero.
    int64_t  input_validation_us;
    uint64_t input_validation_cycles;
    int64_t  mask_prepare_us;
    uint64_t mask_prepare_cycles;
    int64_t  workspace_prepare_us;
    uint64_t workspace_prepare_cycles;
    int64_t  plan_preflight_us;
    uint64_t plan_preflight_cycles;
    int64_t  input_pack_us;
    uint64_t input_pack_cycles;
    int64_t  fused_attention_run_us;
    uint64_t fused_attention_run_cycles;
    int64_t  output_unpack_store_us;
    uint64_t output_unpack_store_cycles;
};

struct ggml_gemmini_vpu_result {
    enum ggml_gemmini_vpu_status status;
    uint64_t                     vpu_status;
    struct ggml_gemmini_flash_breakdown flash;
    // False when a direct in-place accelerator attempt may have modified a
    // source tensor before reporting a hardware error. In that case CPU
    // fallback is not safe because its input may no longer be intact.
    bool                         fallback_safe;
    char                         reason[GGML_GEMMINI_VPU_REASON_CAPACITY];
};

// Structural admission only. Data-dependent restrictions, including finite
// standalone-softmax inputs and an exact prefix-causal FlashAttention mask,
// are checked again by ggml_gemmini_vpu_compute() before any VPU command is
// issued. FlashAttention assumes finite Q/K/V and deliberately avoids a full
// input scan. logical_gemmini_mask names VPU matrix ports, not physical custom
// opcodes (the single-Gemmini profile therefore passes bit 0, not custom3).
// requested_flash_page_packing_mask has a FlashAttention-only namespace:
// bit 0 selects Q, bit 1 selects K, and bit 2 selects V. Generic Gemmini
// A/B/C/D settings never enter this interface. The VPU always H_STOREs final
// FP32 rows directly to dst; its online accumulator remains internal VSRAM
// state. The adapter applies the shared M/K <= 1 policy.
bool ggml_gemmini_vpu_can_compute(
        const struct ggml_tensor * op,
        unsigned                   logical_gemmini_mask,
        uint8_t                    requested_flash_page_packing_mask,
        char *                     reason,
        size_t                     reason_capacity);

// Execute one admitted node. This function never aborts on an unsupported
// tensor: it returns a status and leaves CPU fallback policy to the caller.
// Contiguous tensors use direct DMA. A failed non-aliasing operation remains
// safe to recompute on CPU; an in-place hardware failure reports
// fallback_safe=false and must be surfaced instead of retried.
struct ggml_gemmini_vpu_result ggml_gemmini_vpu_compute(
        struct ggml_tensor * op,
        unsigned             logical_gemmini_mask,
        uint8_t              requested_flash_page_packing_mask);

// The public VPU RMSNorm primitive includes the learned weight multiply while
// ggml normally represents RMS_NORM and MUL as two graph nodes. These helpers
// let the backend fuse that exact pair. A successful call writes weighted_dst;
// it intentionally does not materialize rms_norm_node itself.
bool ggml_gemmini_vpu_can_compute_weighted_rms_norm(
        const struct ggml_tensor * rms_norm_node,
        const struct ggml_tensor * weight,
        const struct ggml_tensor * weighted_dst,
        char *                     reason,
        size_t                     reason_capacity);

struct ggml_gemmini_vpu_result ggml_gemmini_vpu_compute_weighted_rms_norm(
        const struct ggml_tensor * rms_norm_node,
        const struct ggml_tensor * weight,
        struct ggml_tensor *       weighted_dst);

const char * ggml_gemmini_vpu_status_name(enum ggml_gemmini_vpu_status status);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GGML_GEMMINI_VPU_H
