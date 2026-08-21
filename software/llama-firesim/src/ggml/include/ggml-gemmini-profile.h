#ifndef GGML_GEMMINI_PROFILE_H
#define GGML_GEMMINI_PROFILE_H

/*
 * The build-selected profile defines the software scheduling and weight-pack
 * identity.  Hardware parameters and kernel ABIs always come directly from
 * the currently elaborated target-design headers. GEMMINI_SW_DIR is added to
 * every consuming target before CMake force-includes this header.
 */
#include "include/gemmini_params.h"
#include "include/vpu_params.h"

/* Stable IDs are part of the weight-pack ABI. Do not renumber them. */
#define GGML_GEMMINI_PROFILE_SINGLE_ID 1u
#define GGML_GEMMINI_PROFILE_MULTI_ID 2u

#if defined(LLAMA_FIRESIM_HW_PROFILE_SINGLE) && \
    defined(LLAMA_FIRESIM_HW_PROFILE_MULTI)
#error "only one llama-firesim hardware profile may be selected"
#elif defined(LLAMA_FIRESIM_HW_PROFILE_SINGLE)
#define GGML_GEMMINI_PROFILE_ID GGML_GEMMINI_PROFILE_SINGLE_ID
#define GGML_GEMMINI_PROFILE_NAME "single"
#define GGML_GEMMINI_PROFILE_GEMMINI_COUNT 1u
#define GGML_GEMMINI_PROFILE_LOGICAL_MASK 0x1u
#elif defined(LLAMA_FIRESIM_HW_PROFILE_MULTI)
#define GGML_GEMMINI_PROFILE_ID GGML_GEMMINI_PROFILE_MULTI_ID
#define GGML_GEMMINI_PROFILE_NAME "multi"
#define GGML_GEMMINI_PROFILE_GEMMINI_COUNT 4u
#define GGML_GEMMINI_PROFILE_LOGICAL_MASK 0xFu
#else
#error "select LLAMA_FIRESIM_HW_PROFILE=single or multi at CMake configure time"
#endif

/* Derive the physical Gemmini opcode window from the generated base opcode. */
#define GGML_GEMMINI_PROFILE_ROCC_OPCODE_MASK \
    ((((1u << GGML_GEMMINI_PROFILE_GEMMINI_COUNT) - 1u)) << XCUSTOM_ACC)

/*
 * Public masks are always logical matrix-port masks.  Only the generic
 * Gemmini command path translates them to physical RoCC opcode bits; the VPU
 * FlashAttention API must receive the logical mask unchanged.
 */
static inline unsigned ggml_gemmini_profile_physical_mask(unsigned logical_mask) {
    return (logical_mask & GGML_GEMMINI_PROFILE_LOGICAL_MASK) << XCUSTOM_ACC;
}

/* Validate the public, logical request tuple before any physical translation. */
static inline int ggml_gemmini_profile_request_is_valid(
        int active_mask, int gemmini_id, int split_mode) {
    if (active_mask < 0 ||
            ((unsigned) active_mask & ~GGML_GEMMINI_PROFILE_LOGICAL_MASK) != 0u) {
        return 0;
    }
    if (!split_mode) {
        return gemmini_id == -1;
    }
#if GGML_GEMMINI_PROFILE_ID == GGML_GEMMINI_PROFILE_SINGLE_ID
    return 0;
#else
    return gemmini_id >= 0 &&
        (unsigned) gemmini_id < GGML_GEMMINI_PROFILE_GEMMINI_COUNT &&
        (unsigned) active_mask == (1u << (unsigned) gemmini_id);
#endif
}

#if defined(__cplusplus)
#define GGML_GEMMINI_PROFILE_STATIC_ASSERT(condition_, message_) \
    static_assert((condition_), message_)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define GGML_GEMMINI_PROFILE_STATIC_ASSERT(condition_, message_) \
    _Static_assert((condition_), message_)
#else
#define GGML_GEMMINI_PROFILE_STATIC_ASSERT_GLUE_(lhs_, rhs_) lhs_##rhs_
#define GGML_GEMMINI_PROFILE_STATIC_ASSERT_GLUE(lhs_, rhs_) \
    GGML_GEMMINI_PROFILE_STATIC_ASSERT_GLUE_(lhs_, rhs_)
#define GGML_GEMMINI_PROFILE_STATIC_ASSERT(condition_, message_) \
    typedef char GGML_GEMMINI_PROFILE_STATIC_ASSERT_GLUE( \
        ggml_gemmini_profile_static_assert_, __LINE__)[(condition_) ? 1 : -1]
#endif

#if !defined(ELEM_T_IS_LOWPREC_FLOAT) || !defined(ELEM_T_EXP_BITS) || \
    !defined(ELEM_T_SIG_BITS) || !defined(ACC_T_EXP_BITS) || \
    !defined(ACC_T_SIG_BITS)
#error "llama-firesim requires generated BF16/FP32 Gemmini type metadata"
#endif

#if !defined(VPU_HAS_GENERATED_PARAMS) || VPU_HAS_GENERATED_PARAMS != 1
#error "llama-firesim requires target-design generated VPU parameters"
#endif

GGML_GEMMINI_PROFILE_STATIC_ASSERT(MAX_BYTES == 64,
    "llama-firesim requires 64-byte Gemmini DMA transactions");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(BANK_NUM == 4,
    "llama-firesim requires four Gemmini scratchpad banks");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(ELEM_T_EXP_BITS == 8 && ELEM_T_SIG_BITS == 8,
    "llama-firesim Gemmini operands must use BF16 encoding");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(ACC_T_EXP_BITS == 8 && ACC_T_SIG_BITS == 24,
    "llama-firesim Gemmini accumulators must use FP32 encoding");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(sizeof(elem_t) == 2,
    "llama-firesim Gemmini operands must be BF16 storage");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(sizeof(acc_t) == 4,
    "llama-firesim Gemmini accumulators must be FP32");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_VLEN == 128u,
    "llama-firesim VPU VLEN must be 128 elements");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_NLANES == 8u,
    "llama-firesim VPU must expose eight FP32 lanes");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_VSPAD_KIB == 256u,
    "llama-firesim VPU scratchpad must be 256 KiB");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_DMA_BUS_BITS == 256u,
    "llama-firesim VPU DMA bus must be 256 bits");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_STORAGE_KIND == VPU_STORAGE_FP32,
    "llama-firesim VPU storage must be FP32");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_COMPUTE_KIND == VPU_STORAGE_FP32,
    "llama-firesim VPU compute must be FP32");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_SHARED_DEPS == 1u,
    "llama-firesim fusion requires shared dependencies");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_GROUPED_COMMANDS == 1u,
    "llama-firesim fusion requires grouped commands");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(VPU_MATRIX_ROW_ELEMENTS == DIM,
    "VPU matrix row width must equal the live Gemmini DIM");
GGML_GEMMINI_PROFILE_STATIC_ASSERT(
    VPU_MATRIX_WORDS_PER_ROW * VPU_NLANES == VPU_MATRIX_ROW_ELEMENTS,
    "VPU matrix row packing is inconsistent with the live lane geometry");

#endif  // GGML_GEMMINI_PROFILE_H
