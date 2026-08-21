#include "ggml-gemmini-profile.h"

#include "ggml-gemmini-flash-runtime-opt.h"
#include "ggml-gemmini-page-packing.h"
#include "ggml-gemmini-vpu.h"
#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
#include "ggml-impl.h"
#endif

extern "C" {
#include "include/gemmini_page_packed.h"
#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#include "include/vpu_kernels.h"
#endif
}

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

static_assert(VPU_STORAGE_KIND == VPU_STORAGE_FP32,
              "llama.cpp VPU staging requires FP32 VPU storage");
#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static_assert(sizeof(vpu_storage_t) == sizeof(float),
              "llama.cpp VPU staging requires 32-bit VPU elements");
#endif
static_assert(sizeof(elem_t) == sizeof(uint16_t),
              "FlashAttention staging requires BF16 Gemmini elements");

namespace {

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
constexpr size_t k_workspace_alignment = 64;
constexpr size_t k_page_packed_alignment = GEMMINI_PAGE_PACKED_PAGE_BYTES;
static_assert((k_page_packed_alignment & (k_page_packed_alignment - 1)) == 0,
              "Gemmini page size must be a power-of-two allocation alignment");

std::mutex & vpu_mutex() {
    static std::mutex mutex;
    return mutex;
}
#endif

static bool reject(char * reason, size_t capacity, const char * fmt, ...) {
    if (reason != nullptr && capacity != 0) {
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(reason, capacity, fmt, args);
        va_end(args);
        reason[capacity - 1] = '\0';
    }
    return false;
}

static void clear_reason(char * reason, size_t capacity) {
    if (reason != nullptr && capacity != 0) {
        reason[0] = '\0';
    }
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static uint64_t read_cycles_local() {
#if defined(__riscv)
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
#else
    return 0;
#endif
}

struct flash_phase_timer {
    uint64_t cycles;
    int64_t us;

    flash_phase_timer()
        : cycles(read_cycles_local()), us(ggml_time_us()) {
    }

    void finish(int64_t * elapsed_us, uint64_t * elapsed_cycles) const {
        const int64_t end_us = ggml_time_us();
        const uint64_t end_cycles = read_cycles_local();
        *elapsed_us += end_us - us;
        *elapsed_cycles += end_cycles - cycles;
    }
};

static ggml_gemmini_vpu_result make_result(
        ggml_gemmini_vpu_status status,
        uint64_t                vpu_status,
        const char *            fmt,
        ...) {
    ggml_gemmini_vpu_result result{};
    result.status = status;
    result.vpu_status = vpu_status;
    result.fallback_safe = true;
    if (fmt != nullptr) {
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(result.reason, sizeof(result.reason), fmt, args);
        va_end(args);
        result.reason[sizeof(result.reason) - 1] = '\0';
    }
    return result;
}

static ggml_gemmini_vpu_result make_ok(uint64_t vpu_status = 0) {
    return make_result(GGML_GEMMINI_VPU_STATUS_OK, vpu_status, nullptr);
}
#endif

static bool checked_mul(size_t lhs, size_t rhs, size_t * out) {
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

static bool checked_add(size_t lhs, size_t rhs, size_t * out) {
    if (out == nullptr || rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

static bool checked_ceil_div(size_t numerator, size_t denominator, size_t * out) {
    if (out == nullptr || denominator == 0) {
        return false;
    }
    *out = numerator / denominator + (numerator % denominator != 0);
    return true;
}

enum class flash_storage_layout {
    linear,
    page_packed_a,
    page_packed_b,
};

// Return the allocation size in elements without relying on the unchecked
// addition/multiplication in the generated convenience page-count helpers.
// GGML shapes are adversarial input, so page rounding must not be allowed to
// wrap after the ordinary logical element-count checks have succeeded.
static bool flash_storage_elements(
        size_t               rows,
        size_t               columns,
        flash_storage_layout layout,
        size_t               element_size,
        size_t *             elements) {
    if (elements == nullptr || rows == 0 || columns == 0 || element_size == 0) {
        return false;
    }
    if (layout == flash_storage_layout::linear) {
        return checked_mul(rows, columns, elements);
    }

    size_t row_blocks = 0;
    size_t column_blocks = 0;
    if (!checked_ceil_div(rows, DIM, &row_blocks) ||
        !checked_ceil_div(columns, DIM, &column_blocks)) {
        return false;
    }

    size_t pages = 0;
    switch (layout) {
        case flash_storage_layout::page_packed_a: {
            const size_t row_blocks_per_page = gemmini_page_packed_a_i_blocks_per_page();
            const size_t column_blocks_per_page = gemmini_page_packed_a_k_blocks_per_page();
            size_t row_pages = 0;
            size_t column_pages = 0;
            if (!checked_ceil_div(row_blocks, row_blocks_per_page, &row_pages) ||
                !checked_ceil_div(column_blocks, column_blocks_per_page, &column_pages) ||
                !checked_mul(row_pages, column_pages, &pages)) {
                return false;
            }
            break;
        }
        case flash_storage_layout::page_packed_b: {
            const size_t column_blocks_per_page = gemmini_page_packed_b_j_blocks_per_page();
            size_t column_pages = 0;
            if (!checked_ceil_div(column_blocks, column_blocks_per_page, &column_pages) ||
                !checked_mul(row_blocks, column_pages, &pages)) {
                return false;
            }
            break;
        }
        case flash_storage_layout::linear:
            return false;
    }

    size_t bytes = 0;
    if (!checked_mul(pages, GEMMINI_PAGE_PACKED_PAGE_BYTES, &bytes) ||
        bytes % element_size != 0) {
        return false;
    }
    *elements = bytes / element_size;
    return true;
}

static ggml_gemmini_flash_page_packing_config flash_page_packing_for_shape(
        uint8_t requested_mask,
        size_t  query_rows,
        size_t  q_dim,
    size_t  sequence) {
    return ggml_gemmini_effective_flash_page_packing(
        ggml_gemmini_flash_page_packing_from_mask(requested_mask),
        query_rows, q_dim, sequence);
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
template <typename T>
class aligned_buffer {
public:
    aligned_buffer() = default;
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;

    ~aligned_buffer() {
        std::free(data_);
    }

    bool allocate(
            size_t count,
            size_t alignment = k_workspace_alignment,
            bool   zero_initialize = false) {
        if (count == 0) {
            count_ = 0;
            return true;
        }

        size_t bytes = 0;
        if (!checked_mul(count, sizeof(T), &bytes)) {
            return false;
        }
        if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0) {
            return false;
        }
        if (data_ == nullptr || capacity_ < count || alignment_ < alignment) {
            void * ptr = nullptr;
            if (posix_memalign(&ptr, alignment, bytes) != 0) {
                return false;
            }
            std::free(data_);
            data_ = static_cast<T *>(ptr);
            capacity_ = count;
            alignment_ = alignment;
        }
        if (zero_initialize) {
            std::memset(data_, 0, bytes);
        }
        count_ = count;
        return true;
    }

    T * data() {
        return data_;
    }

    const T * data() const {
        return data_;
    }

    size_t size() const {
        return count_;
    }

    size_t capacity() const {
        return capacity_;
    }

    size_t alignment() const {
        return alignment_;
    }

    T & operator[](size_t index) {
        return data_[index];
    }

    const T & operator[](size_t index) const {
        return data_[index];
    }

private:
    T *    data_  = nullptr;
    size_t count_ = 0;
    size_t capacity_ = 0;
    size_t alignment_ = 0;
};

template <typename T>
static bool allocate_overwritten_flash_workspace(
        aligned_buffer<T> * buffer,
        size_t              count,
        size_t              alignment = k_workspace_alignment) {
    if (buffer == nullptr) {
        return false;
    }
    const ggml_gemmini_flash_workspace_plan plan =
        ggml_gemmini_flash_make_workspace_plan(
            buffer->capacity(), buffer->alignment(), count, alignment, true);
    return plan.valid && buffer->allocate(
        count, alignment, plan.zero_initialize);
}

struct flash_workspace {
    aligned_buffer<size_t> query_bases;
    aligned_buffer<size_t> unique_mask_bases;
    aligned_buffer<elem_t> staged_q;
    aligned_buffer<elem_t> staged_k;
    aligned_buffer<elem_t> staged_v;
    aligned_buffer<float> causal_mask;
};

static flash_workspace & get_flash_workspace() {
    static thread_local flash_workspace workspace;
    return workspace;
}
#endif

static bool valid_tensor_shape(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return false;
        }
    }
    return true;
}

static bool tensor_counts(const ggml_tensor * tensor, size_t * rows, size_t * elements) {
    if (!valid_tensor_shape(tensor) || rows == nullptr || elements == nullptr) {
        return false;
    }

    size_t value = 1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (!checked_mul(value, static_cast<size_t>(tensor->ne[i]), &value)) {
            return false;
        }
    }
    *rows = value;
    return checked_mul(value, static_cast<size_t>(tensor->ne[0]), elements);
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static bool tensor_f32_dense(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->type == GGML_TYPE_F32 &&
        tensor->data != nullptr && ggml_is_contiguous(tensor);
}

static bool tensor_f32_rows_element_aligned(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || tensor->data == nullptr ||
        (reinterpret_cast<uintptr_t>(tensor->data) % sizeof(float)) != 0) {
        return false;
    }
    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] > 1 && tensor->nb[dimension] % sizeof(float) != 0) {
            return false;
        }
    }
    return true;
}

// The VPU reservation station protects VSRAM hazards, not two DMA stores to
// overlapping host rows.  Batched direct output therefore requires every
// logical row to occupy a distinct byte range.  This conservative nested-
// stride check accepts ordinary contiguous/padded ggml tensors and routes
// overlapping or exotic views through the linear staging path.
static bool tensor_f32_rows_non_overlapping(const ggml_tensor * tensor) {
    if (!valid_tensor_shape(tensor) || tensor->type != GGML_TYPE_F32 ||
        tensor->data == nullptr) {
        return false;
    }
    size_t lower_extent = 0;
    if (!checked_mul(static_cast<size_t>(tensor->ne[0]), sizeof(float), &lower_extent)) {
        return false;
    }
    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] <= 1) {
            continue;
        }
        if (tensor->nb[dimension] < lower_extent) {
            return false;
        }
        size_t added_extent = 0;
        if (!checked_mul(static_cast<size_t>(tensor->ne[dimension] - 1),
                         tensor->nb[dimension], &added_extent) ||
            !checked_add(lower_extent, added_extent, &lower_extent)) {
            return false;
        }
    }
    return true;
}

static bool tensor_byte_span(const ggml_tensor * tensor, size_t * span) {
    if (tensor == nullptr || span == nullptr) {
        return false;
    }
    size_t last_offset = 0;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        size_t dimension_offset = 0;
        if (!checked_mul(static_cast<size_t>(tensor->ne[dimension] - 1),
                         tensor->nb[dimension], &dimension_offset) ||
            !checked_add(last_offset, dimension_offset, &last_offset)) {
            return false;
        }
    }
    return checked_add(last_offset, ggml_type_size(tensor->type), span);
}

static bool tensor_storage_overlaps(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    if (lhs == nullptr || rhs == nullptr || lhs->data == nullptr || rhs->data == nullptr) {
        return false;
    }
    size_t lhs_span = 0;
    size_t rhs_span = 0;
    if (!tensor_byte_span(lhs, &lhs_span) || !tensor_byte_span(rhs, &rhs_span)) {
        return true;
    }
    const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs->data);
    const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs->data);
    if (lhs_span > std::numeric_limits<uintptr_t>::max() - lhs_begin ||
        rhs_span > std::numeric_limits<uintptr_t>::max() - rhs_begin) {
        return true;
    }
    return lhs_begin < rhs_begin + rhs_span && rhs_begin < lhs_begin + lhs_span;
}

static bool tensor_exact_alias(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    if (lhs == nullptr || rhs == nullptr || lhs->data != rhs->data || lhs->type != rhs->type) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (lhs->ne[dimension] != rhs->ne[dimension] ||
            lhs->nb[dimension] != rhs->nb[dimension]) {
            return false;
        }
    }
    return true;
}

static bool tensor_direct_alias_supported(
        const ggml_tensor * destination,
        const ggml_tensor * source) {
    return !tensor_storage_overlaps(destination, source) ||
        tensor_exact_alias(destination, source);
}
#endif

static bool same_shape(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs->ne[i] != rhs->ne[i]) {
            return false;
        }
    }
    return true;
}

static bool validate_tensor_address_span(
        const ggml_tensor * tensor,
        size_t              scalar_bytes,
        const char *        name,
        char *              reason,
        size_t              reason_capacity) {
    size_t last_offset = 0;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        size_t dimension_offset = 0;
        if (!checked_mul(static_cast<size_t>(tensor->ne[dimension] - 1),
                         tensor->nb[dimension], &dimension_offset) ||
            !checked_add(last_offset, dimension_offset, &last_offset)) {
            return reject(reason, reason_capacity,
                          "%s stride span overflows size_t", name);
        }
    }

    size_t byte_span = 0;
    if (scalar_bytes == 0 || !checked_add(last_offset, scalar_bytes, &byte_span)) {
        return reject(reason, reason_capacity,
                      "%s byte span overflows size_t", name);
    }
    if (tensor->data != nullptr) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(tensor->data);
        if (byte_span - 1 > std::numeric_limits<uintptr_t>::max() - base) {
            return reject(reason, reason_capacity,
                          "%s address span wraps uintptr_t", name);
        }
    }
    return true;
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static const uint8_t * tensor_row(const ggml_tensor * tensor, size_t row) {
    size_t i1 = row % static_cast<size_t>(tensor->ne[1]);
    row /= static_cast<size_t>(tensor->ne[1]);
    const size_t i2 = row % static_cast<size_t>(tensor->ne[2]);
    const size_t i3 = row / static_cast<size_t>(tensor->ne[2]);
    return static_cast<const uint8_t *>(tensor->data) +
           i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
}

static uint8_t * tensor_row(ggml_tensor * tensor, size_t row) {
    return const_cast<uint8_t *>(tensor_row(
        static_cast<const ggml_tensor *>(tensor), row));
}
#endif

static bool validate_f32_row_tensor(
        const ggml_tensor * tensor,
        const char *        name,
        char *              reason,
        size_t              reason_capacity) {
    if (!valid_tensor_shape(tensor)) {
        return reject(reason, reason_capacity, "%s is null or has an invalid shape", name);
    }
    if (tensor->type != GGML_TYPE_F32) {
        return reject(reason, reason_capacity, "%s must be F32", name);
    }
    if (tensor->nb[0] != sizeof(float)) {
        return reject(reason, reason_capacity, "%s innermost dimension is not contiguous", name);
    }
    if (!validate_tensor_address_span(
            tensor, sizeof(float), name, reason, reason_capacity)) {
        return false;
    }
    size_t rows = 0;
    size_t elements = 0;
    if (!tensor_counts(tensor, &rows, &elements)) {
        return reject(reason, reason_capacity, "%s element count overflows size_t", name);
    }
    (void) rows;
    (void) elements;
    return true;
}

static bool is_staging_float_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static size_t staging_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return sizeof(float);
        case GGML_TYPE_F16:  return sizeof(ggml_fp16_t);
        case GGML_TYPE_BF16: return sizeof(ggml_bf16_t);
        default:             return 0;
    }
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static float load_staging_float(ggml_type type, const void * address) {
    switch (type) {
        case GGML_TYPE_F32: {
            float value = 0.0f;
            std::memcpy(&value, address, sizeof(value));
            return value;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t value = 0;
            std::memcpy(&value, address, sizeof(value));
            return ggml_fp16_to_fp32(value);
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t value{};
            std::memcpy(&value, address, sizeof(value));
            return ggml_bf16_to_fp32(value);
        }
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}
#endif

static bool validate_staging_float_tensor(
        const ggml_tensor * tensor,
        const char *        name,
        char *              reason,
        size_t              reason_capacity) {
    if (!valid_tensor_shape(tensor)) {
        return reject(reason, reason_capacity, "%s is null or has an invalid shape", name);
    }
    if (!is_staging_float_type(tensor->type)) {
        return reject(reason, reason_capacity, "%s must be F32, F16, or BF16", name);
    }
    if (tensor->nb[0] != staging_type_size(tensor->type)) {
        return reject(reason, reason_capacity, "%s innermost dimension is not contiguous", name);
    }
    if (!validate_tensor_address_span(
            tensor, staging_type_size(tensor->type), name, reason, reason_capacity)) {
        return false;
    }
    size_t rows = 0;
    size_t elements = 0;
    if (!tensor_counts(tensor, &rows, &elements)) {
        return reject(reason, reason_capacity, "%s element count overflows size_t", name);
    }
    return true;
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static void copy_f32_to_linear(const ggml_tensor * source, float * destination) {
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(source, &rows, &elements);
    const size_t width = static_cast<size_t>(source->ne[0]);
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(destination + row * width, tensor_row(source, row), width * sizeof(float));
    }
}

static void scatter_f32_from_linear(const float * source, ggml_tensor * destination) {
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(destination, &rows, &elements);
    const size_t width = static_cast<size_t>(destination->ne[0]);
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(tensor_row(destination, row), source + row * width, width * sizeof(float));
    }
}
#endif

static float op_param_f32(const ggml_tensor * tensor, size_t index) {
#if defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
    float value = 0.0f;
    std::memcpy(&value, &tensor->op_params[index], sizeof(value));
    return value;
#else
    return ggml_get_op_params_f32(tensor, static_cast<uint32_t>(index));
#endif
}

static int32_t op_param_i32(const ggml_tensor * tensor, size_t index) {
#if defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
    int32_t value = 0;
    std::memcpy(&value, &tensor->op_params[index], sizeof(value));
    return value;
#else
    return ggml_get_op_params_i32(tensor, static_cast<uint32_t>(index));
#endif
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static bool vpu_status_failed(uint64_t status) {
    return (status & VPU_STATUS_ERROR_MASK) != 0;
}

static ggml_gemmini_vpu_result hardware_failure(
        const char * operation,
        uint64_t     status,
        bool         fallback_safe = true) {
    ggml_gemmini_vpu_result result = make_result(
        GGML_GEMMINI_VPU_STATUS_HARDWARE_ERROR,
        status,
        "%s returned VPU error status 0x%llx",
        operation,
        static_cast<unsigned long long>(status));
    result.fallback_safe = fallback_safe;
    return result;
}
#endif

static bool can_compute_unary(const ggml_tensor * op, char * reason, size_t capacity) {
    if (op->src[0] == nullptr) {
        return reject(reason, capacity, "unary source is null");
    }
    if (!validate_f32_row_tensor(op->src[0], "unary source", reason, capacity) ||
        !validate_f32_row_tensor(op, "unary destination", reason, capacity)) {
        return false;
    }
    if (!same_shape(op->src[0], op)) {
        return reject(reason, capacity, "unary source and destination shapes differ");
    }

    const int32_t unary = op_param_i32(op, 0);
    if (unary != GGML_UNARY_OP_SILU && unary != GGML_UNARY_OP_RELU &&
        unary != GGML_UNARY_OP_SIGMOID && unary != GGML_UNARY_OP_TANH &&
        unary != GGML_UNARY_OP_GELU_QUICK) {
        return reject(reason, capacity, "unary subtype %d has no exact VPU kernel", unary);
    }

    size_t rows = 0;
    size_t elements = 0;
    if (!tensor_counts(op, &rows, &elements) || elements > UINT32_MAX) {
        return reject(reason, capacity, "unary element count exceeds the VPU 32-bit interface");
    }
    return true;
}

static bool can_compute_swiglu(const ggml_tensor * op, char * reason, size_t capacity) {
    const ggml_tensor * gate_source = op->src[0];
    const ggml_tensor * split_up = op->src[1];
    if (gate_source == nullptr) {
        return reject(reason, capacity, "SwiGLU source is null");
    }
    const int32_t glu = op_param_i32(op, 0);
    if (glu != GGML_GLU_OP_SWIGLU) {
        return reject(reason, capacity, "GLU subtype %d is not exact SwiGLU", glu);
    }
    if (!validate_f32_row_tensor(gate_source, "SwiGLU gate source", reason, capacity) ||
        !validate_f32_row_tensor(op, "SwiGLU destination", reason, capacity)) {
        return false;
    }

    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        if (gate_source->ne[dimension] != op->ne[dimension]) {
            return reject(reason, capacity, "SwiGLU outer dimensions differ");
        }
    }

    if (split_up != nullptr) {
        if (!validate_f32_row_tensor(split_up, "SwiGLU up source", reason, capacity) ||
            !same_shape(gate_source, split_up) || !same_shape(gate_source, op)) {
            return reject(reason, capacity, "split SwiGLU inputs and destination must have the same F32 shape");
        }
    } else {
        if ((gate_source->ne[0] & 1) != 0 || gate_source->ne[0] / 2 != op->ne[0]) {
            return reject(reason, capacity, "packed SwiGLU source width must be twice the destination width");
        }
    }

    size_t rows = 0;
    size_t elements = 0;
    if (!tensor_counts(op, &rows, &elements) || elements > UINT32_MAX) {
        return reject(reason, capacity, "SwiGLU output count exceeds the VPU 32-bit interface");
    }
    return true;
}

static bool can_compute_binary_elementwise(
        const ggml_tensor * op,
        char *              reason,
        size_t              capacity) {
    if (op == nullptr || (op->op != GGML_OP_ADD && op->op != GGML_OP_MUL)) {
        return reject(reason, capacity, "operation is not VPU ADD/MUL");
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) {
        return reject(reason, capacity, "VPU ADD/MUL source is null");
    }
    if (!validate_f32_row_tensor(op->src[0], "VPU binary source 0", reason, capacity) ||
        !validate_f32_row_tensor(op->src[1], "VPU binary source 1", reason, capacity) ||
        !validate_f32_row_tensor(op, "VPU binary destination", reason, capacity)) {
        return false;
    }
    if (!same_shape(op->src[0], op->src[1]) || !same_shape(op->src[0], op)) {
        return reject(reason, capacity,
                      "VPU ADD/MUL currently requires equal, non-broadcast F32 shapes");
    }
    size_t rows = 0;
    size_t elements = 0;
    if (!tensor_counts(op, &rows, &elements) || elements > UINT32_MAX) {
        return reject(reason, capacity, "VPU ADD/MUL element count exceeds the 32-bit interface");
    }
    return true;
}

static bool can_compute_softmax(const ggml_tensor * op, char * reason, size_t capacity) {
    if (op->src[0] == nullptr) {
        return reject(reason, capacity, "softmax source is null");
    }
    if (!validate_f32_row_tensor(op->src[0], "softmax source", reason, capacity) ||
        !validate_f32_row_tensor(op, "softmax destination", reason, capacity) ||
        !same_shape(op->src[0], op)) {
        return reject(reason, capacity, "softmax requires equal F32 source/destination shapes");
    }
    if (op->src[1] != nullptr) {
        return reject(reason, capacity, "standalone VPU softmax does not support an additive mask");
    }
    if (op->src[2] != nullptr) {
        return reject(reason, capacity, "standalone VPU softmax does not support attention sinks");
    }
    const float scale = op_param_f32(op, 0);
    const float max_bias = op_param_f32(op, 1);
    if (scale != 1.0f || max_bias != 0.0f) {
        return reject(reason, capacity, "standalone VPU softmax requires scale=1 and max_bias=0");
    }
    if (op->ne[0] > UINT32_MAX) {
        return reject(reason, capacity, "softmax row width exceeds the VPU 32-bit interface");
    }
    return true;
}

static bool can_compute_rms_norm(const ggml_tensor * op, char * reason, size_t capacity) {
    if (op->src[0] == nullptr) {
        return reject(reason, capacity, "RMSNorm source is null");
    }
    if (!validate_f32_row_tensor(op->src[0], "RMSNorm source", reason, capacity) ||
        !validate_f32_row_tensor(op, "RMSNorm destination", reason, capacity) ||
        !same_shape(op->src[0], op)) {
        return reject(reason, capacity, "RMSNorm requires equal F32 source/destination shapes");
    }
    const float epsilon = op_param_f32(op, 0);
    if (!std::isfinite(epsilon) || epsilon < 0.0f) {
        return reject(reason, capacity, "RMSNorm epsilon must be finite and non-negative");
    }
    if (op->ne[0] > UINT32_MAX) {
        return reject(reason, capacity, "RMSNorm row width exceeds the VPU 32-bit interface");
    }
    return true;
}

static bool can_compute_weighted_rms_norm_impl(
        const ggml_tensor * rms_norm_node,
        const ggml_tensor * weight,
        const ggml_tensor * weighted_dst,
        char *              reason,
        size_t              capacity) {
    if (rms_norm_node == nullptr || rms_norm_node->op != GGML_OP_RMS_NORM) {
        return reject(reason, capacity, "weighted RMSNorm requires an RMS_NORM node");
    }
    if (weighted_dst == nullptr || weighted_dst->op != GGML_OP_MUL ||
        !((weighted_dst->src[0] == rms_norm_node && weighted_dst->src[1] == weight) ||
          (weighted_dst->src[1] == rms_norm_node && weighted_dst->src[0] == weight))) {
        return reject(reason, capacity,
                      "weighted RMSNorm destination must be the MUL of this RMS_NORM and weight");
    }
    if (!can_compute_rms_norm(rms_norm_node, reason, capacity)) {
        return false;
    }
    if (!validate_f32_row_tensor(weight, "RMSNorm weight", reason, capacity) ||
        !validate_f32_row_tensor(weighted_dst, "weighted RMSNorm destination", reason, capacity)) {
        return false;
    }
    if (!same_shape(rms_norm_node, weighted_dst)) {
        return reject(reason, capacity, "weighted RMSNorm destination shape differs from RMS_NORM");
    }
    if (weight->ne[0] != rms_norm_node->ne[0]) {
        return reject(reason, capacity, "RMSNorm weight width differs from the normalized row width");
    }
    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        if (weight->ne[dimension] != 1 && weight->ne[dimension] != rms_norm_node->ne[dimension]) {
            return reject(reason, capacity, "RMSNorm weight is not broadcastable in dimension %d", dimension);
        }
    }
    return true;
}

struct rope_parameters {
    int32_t n_dims = 0;
    int32_t mode = 0;
    int32_t n_ctx_orig = 0;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 0.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
};

static rope_parameters read_rope_parameters(const ggml_tensor * op) {
    rope_parameters parameters;
    parameters.n_dims = op_param_i32(op, 1);
    parameters.mode = op_param_i32(op, 2);
    parameters.n_ctx_orig = op_param_i32(op, 4);
    parameters.freq_base = op_param_f32(op, 5);
    parameters.freq_scale = op_param_f32(op, 6);
    parameters.ext_factor = op_param_f32(op, 7);
    parameters.attn_factor = op_param_f32(op, 8);
    parameters.beta_fast = op_param_f32(op, 9);
    parameters.beta_slow = op_param_f32(op, 10);
    return parameters;
}

static bool can_compute_rope(const ggml_tensor * op, char * reason, size_t capacity) {
    const ggml_tensor * source = op->src[0];
    const ggml_tensor * positions = op->src[1];
    const ggml_tensor * factors = op->src[2];
    if (source == nullptr || positions == nullptr) {
        return reject(reason, capacity, "RoPE source or position tensor is null");
    }
    if (!validate_f32_row_tensor(source, "RoPE source", reason, capacity) ||
        !validate_f32_row_tensor(op, "RoPE destination", reason, capacity) ||
        !same_shape(source, op)) {
        return reject(reason, capacity, "RoPE requires equal F32 source/destination shapes");
    }
    if (!valid_tensor_shape(positions) || positions->type != GGML_TYPE_I32 ||
        positions->nb[0] != sizeof(int32_t) ||
        positions->ne[0] != source->ne[2]) {
        return reject(reason, capacity, "RoPE positions must be a contiguous I32 vector of token count");
    }
    if (!validate_tensor_address_span(
            positions, sizeof(int32_t), "RoPE positions", reason, capacity)) {
        return false;
    }
    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        if (positions->ne[dimension] != 1) {
            return reject(reason, capacity, "RoPE positions must be a one-dimensional I32 vector");
        }
    }

    const rope_parameters parameters = read_rope_parameters(op);
    if (parameters.mode != GGML_ROPE_TYPE_NORMAL && parameters.mode != GGML_ROPE_TYPE_NEOX) {
        return reject(reason, capacity, "RoPE mode %d is not normal or NeoX", parameters.mode);
    }
    if (parameters.n_dims <= 0 || (parameters.n_dims & 1) != 0 ||
        parameters.n_dims > source->ne[0] ||
        static_cast<uint32_t>(parameters.n_dims) > VPU_VLEN) {
        return reject(reason, capacity, "RoPE rotary dimension must be even, positive, <= row width, and <= VPU_VLEN");
    }
    if (!std::isfinite(parameters.freq_base) || parameters.freq_base <= 0.0f ||
        !std::isfinite(parameters.freq_scale) || parameters.freq_scale <= 0.0f ||
        !std::isfinite(parameters.ext_factor) || !std::isfinite(parameters.attn_factor) ||
        !std::isfinite(parameters.beta_fast) || !std::isfinite(parameters.beta_slow)) {
        return reject(reason, capacity, "RoPE floating parameters are outside the supported finite range");
    }
    if (parameters.ext_factor != 0.0f &&
        (parameters.n_ctx_orig <= 0 || parameters.beta_fast <= 0.0f || parameters.beta_slow <= 0.0f)) {
        return reject(reason, capacity, "YaRN RoPE requires positive n_ctx_orig, beta_fast, and beta_slow");
    }
    if (factors != nullptr) {
        if (!valid_tensor_shape(factors) || factors->type != GGML_TYPE_F32 ||
            factors->nb[0] != sizeof(float) ||
            factors->ne[0] < parameters.n_dims / 2) {
            return reject(reason, capacity, "RoPE frequency factors must be a contiguous F32 vector");
        }
        if (!validate_tensor_address_span(
                factors, sizeof(float), "RoPE frequency factors", reason, capacity)) {
            return false;
        }
        for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
            if (factors->ne[dimension] != 1) {
                return reject(reason, capacity,
                              "RoPE frequency factors must be a one-dimensional F32 vector");
            }
        }
    }

    size_t packed_rows = 0;
    if (!checked_mul(static_cast<size_t>(source->ne[1]),
                     static_cast<size_t>(parameters.n_dims), &packed_rows) ||
        packed_rows > UINT32_MAX) {
        return reject(reason, capacity, "RoPE staged head group exceeds the VPU 32-bit interface");
    }
    return true;
}

static bool valid_logical_gemmini_mask(unsigned mask) {
    const unsigned valid = GGML_GEMMINI_PROFILE_LOGICAL_MASK;
    return mask != 0u && (mask & ~valid) == 0u;
}

static bool can_compute_flash_attention(
        const ggml_tensor * op,
        unsigned            logical_mask,
        uint8_t             requested_flash_page_packing_mask,
        char *              reason,
        size_t              capacity) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
        return reject(reason, capacity, "FlashAttention requires Q, K, V, and an explicit causal mask");
    }
    // GGML's FLASH_ATTN_EXT CPU contract reads Q as F32. Keeping the same
    // admission contract guarantees that a late mask/planner rejection can be
    // recomputed safely by the CPU fallback backend.
    if (q->type != GGML_TYPE_F32) {
        return reject(reason, capacity, "FlashAttention Q must be F32 for safe CPU fallback");
    }
    if (!validate_staging_float_tensor(q, "FlashAttention Q", reason, capacity) ||
        !validate_staging_float_tensor(k, "FlashAttention K", reason, capacity) ||
        !validate_staging_float_tensor(v, "FlashAttention V", reason, capacity) ||
        !validate_f32_row_tensor(op, "FlashAttention destination", reason, capacity)) {
        return false;
    }
    if (!valid_tensor_shape(mask) || mask->type != GGML_TYPE_F16 ||
        mask->nb[0] != sizeof(ggml_fp16_t)) {
        return reject(reason, capacity, "FlashAttention mask must be row-contiguous F16");
    }
    if (!validate_tensor_address_span(
            mask, sizeof(ggml_fp16_t), "FlashAttention mask", reason, capacity)) {
        return false;
    }
    size_t mask_rows = 0;
    size_t mask_elements = 0;
    if (!tensor_counts(mask, &mask_rows, &mask_elements)) {
        return reject(reason, capacity, "FlashAttention mask element count overflows size_t");
    }
    if (op->src[4] != nullptr) {
        return reject(reason, capacity, "VPU FlashAttention does not support attention sinks");
    }
    if (!valid_logical_gemmini_mask(logical_mask)) {
        return reject(reason, capacity,
                      "logical Gemmini mask 0x%x exceeds selected %s profile mask 0x%x",
                      logical_mask, GGML_GEMMINI_PROFILE_NAME,
                      static_cast<unsigned>(GGML_GEMMINI_PROFILE_LOGICAL_MASK));
    }
    if ((requested_flash_page_packing_mask & UINT8_C(0xf8)) != 0) {
        return reject(reason, capacity,
                      "FlashAttention page-packing mask 0x%x has unknown bits",
                      static_cast<unsigned>(requested_flash_page_packing_mask));
    }

    const int64_t d = q->ne[0];
    const int64_t queries = q->ne[1];
    const int64_t heads = q->ne[2];
    const int64_t batches = q->ne[3];
    const int64_t sequence = k->ne[1];
    const int64_t kv_heads = k->ne[2];
    const int64_t kv_batches = k->ne[3];
    const int64_t value_dim = v->ne[0];

    if (k->ne[0] != d || v->ne[1] != sequence || v->ne[2] != kv_heads ||
        v->ne[3] != kv_batches) {
        return reject(reason, capacity, "FlashAttention K/V dimensions do not form one shared KV cache");
    }
    if (queries > sequence || heads % kv_heads != 0 || batches % kv_batches != 0) {
        return reject(reason, capacity, "FlashAttention query dimensions are not broadcast-compatible with KV");
    }
    if (op->ne[0] != value_dim || op->ne[1] != heads ||
        op->ne[2] != queries || op->ne[3] != batches) {
        return reject(reason, capacity, "FlashAttention destination must be [DV, H, N, B]");
    }
    if (mask->ne[0] != sequence || mask->ne[1] < queries ||
        heads % mask->ne[2] != 0 || batches % mask->ne[3] != 0) {
        return reject(reason, capacity, "FlashAttention mask dimensions are not broadcast-compatible");
    }

    const float scale = op_param_f32(op, 0);
    const float max_bias = op_param_f32(op, 1);
    const float softcap = op_param_f32(op, 2);
    const int32_t precision = op_param_i32(op, 3);
    if (!std::isfinite(scale) || scale <= 0.0f) {
        return reject(reason, capacity,
                      "VPU FlashAttention scale must be finite and positive");
    }
    if (max_bias != 0.0f) {
        return reject(reason, capacity, "VPU FlashAttention does not support ALiBi/max_bias");
    }
    if (softcap != 0.0f) {
        return reject(reason, capacity, "VPU FlashAttention does not support logit softcapping");
    }
    if (precision != GGML_PREC_DEFAULT && precision != GGML_PREC_F32) {
        return reject(reason, capacity,
                      "VPU FlashAttention supports only default or FP32-accumulator precision, not mode %d",
                      precision);
    }

    for (int64_t value : {d, queries, sequence, value_dim}) {
        if (value <= 0 || static_cast<uint64_t>(value) > UINT32_MAX) {
            return reject(reason, capacity, "FlashAttention dimensions exceed the 32-bit fusion ABI");
        }
    }

    const auto packing = flash_page_packing_for_shape(
        requested_flash_page_packing_mask,
        static_cast<size_t>(queries),
        static_cast<size_t>(d),
        static_cast<size_t>(sequence));
    // Bit 31 tags a page-packed stride even when page packing was not
    // requested, so every encoded logical row width is limited to the
    // remaining payload bits.
    if (static_cast<uint64_t>(d) > GEMMINI_PAGE_PACKED_STRIDE_MASK ||
        static_cast<uint64_t>(value_dim) > GEMMINI_PAGE_PACKED_STRIDE_MASK) {
        return reject(reason, capacity,
                      "FlashAttention row width exceeds the 31-bit stride payload");
    }

    size_t ignored = 0;
    if (!flash_storage_elements(
            static_cast<size_t>(queries), static_cast<size_t>(d),
            packing.queries ? flash_storage_layout::page_packed_a
                            : flash_storage_layout::linear,
            sizeof(elem_t), &ignored) ||
        !flash_storage_elements(
            static_cast<size_t>(sequence), static_cast<size_t>(d),
            packing.keys ? flash_storage_layout::page_packed_b
                         : flash_storage_layout::linear,
            sizeof(elem_t), &ignored) ||
        !flash_storage_elements(
            static_cast<size_t>(sequence), static_cast<size_t>(value_dim),
            packing.values ? flash_storage_layout::page_packed_b
                           : flash_storage_layout::linear,
            sizeof(elem_t), &ignored)) {
        return reject(reason, capacity, "FlashAttention staging size overflows size_t");
    }
    return true;
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
static uint64_t run_unary_kernel(
        const ggml_tensor * op,
        const float *       input,
        float *             output,
        size_t              elements) {
    switch (op_param_i32(op, 0)) {
        case GGML_UNARY_OP_SILU:
            return vpu_silu_auto(input, output, elements);
        case GGML_UNARY_OP_RELU:
            return vpu_relu_auto(input, output, elements);
        case GGML_UNARY_OP_SIGMOID:
            return vpu_sigmoid_auto(input, output, elements);
        case GGML_UNARY_OP_TANH:
            return vpu_tanh_auto(input, output, elements);
        case GGML_UNARY_OP_GELU_QUICK:
            return vpu_gelu_auto(input, output, elements);
        default:
            return VPU_STATUS_ILLEGAL_COMMAND;
    }
}

static uint64_t enqueue_unary_kernel(
        const ggml_tensor * op,
        const float *       input,
        float *             output,
        size_t              elements) {
    switch (op_param_i32(op, 0)) {
        case GGML_UNARY_OP_SILU:
            return vpu_silu_enqueue(input, output, elements);
        case GGML_UNARY_OP_RELU:
            return vpu_relu_enqueue(input, output, elements);
        case GGML_UNARY_OP_SIGMOID:
            return vpu_sigmoid_enqueue(input, output, elements);
        case GGML_UNARY_OP_TANH:
            return vpu_tanh_enqueue(input, output, elements);
        case GGML_UNARY_OP_GELU_QUICK:
            return vpu_gelu_enqueue(input, output, elements);
        default:
            return VPU_STATUS_ILLEGAL_COMMAND;
    }
}

static ggml_gemmini_vpu_result compute_unary(ggml_tensor * op) {
    const ggml_tensor * source = op->src[0];
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(op, &rows, &elements);

    if (tensor_f32_rows_element_aligned(source) &&
            tensor_f32_rows_element_aligned(op) &&
            tensor_f32_rows_non_overlapping(op) &&
            tensor_direct_alias_supported(op, source)) {
        const bool fallback_safe = !tensor_storage_overlaps(op, source);
        const bool dense = tensor_f32_dense(source) && tensor_f32_dense(op);
        uint64_t combined_status = 0;
        {
            std::lock_guard<std::mutex> lock(vpu_mutex());
            vpu_clear_status(VPU_CLEAR_ALL);
            if (dense) {
                combined_status = run_unary_kernel(
                    op,
                    static_cast<const float *>(source->data),
                    static_cast<float *>(op->data),
                    elements);
            } else {
                const size_t width = static_cast<size_t>(op->ne[0]);
                vpu_batch_begin();
                for (size_t row = 0; row < rows; ++row) {
                    const uint64_t status = enqueue_unary_kernel(
                        op,
                        reinterpret_cast<const float *>(tensor_row(source, row)),
                        reinterpret_cast<float *>(tensor_row(op, row)),
                        width);
                    combined_status |= status;
                    if (vpu_status_failed(status)) {
                        break;
                    }
                }
                combined_status |= vpu_batch_finish();
            }
        }
        if (vpu_status_failed(combined_status)) {
            return hardware_failure("VPU unary", combined_status, fallback_safe);
        }
        return make_ok(combined_status);
    }

    static thread_local aligned_buffer<float> input;
    static thread_local aligned_buffer<float> output;
    if (!input.allocate(elements) || !output.allocate(elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "unable to allocate 64-byte-aligned unary staging");
    }
    copy_f32_to_linear(source, input.data());

    uint64_t status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        vpu_clear_status(VPU_CLEAR_ALL);
        status = run_unary_kernel(op, input.data(), output.data(), elements);
    }
    if (vpu_status_failed(status)) {
        return hardware_failure("VPU unary", status);
    }
    scatter_f32_from_linear(output.data(), op);
    return make_ok(status);
}

static ggml_gemmini_vpu_result compute_binary_elementwise(ggml_tensor * op) {
    const ggml_tensor * lhs = op->src[0];
    const ggml_tensor * rhs = op->src[1];
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(op, &rows, &elements);

    const bool direct_rows = tensor_f32_rows_element_aligned(lhs) &&
        tensor_f32_rows_element_aligned(rhs) &&
        tensor_f32_rows_element_aligned(op) &&
        tensor_f32_rows_non_overlapping(op) && tensor_direct_alias_supported(op, lhs) &&
        tensor_direct_alias_supported(op, rhs);
    if (direct_rows) {
        const bool dense = tensor_f32_dense(lhs) && tensor_f32_dense(rhs) &&
            tensor_f32_dense(op);
        const bool fallback_safe = !tensor_storage_overlaps(op, lhs) &&
            !tensor_storage_overlaps(op, rhs);
        uint64_t combined_status = 0;
        {
            std::lock_guard<std::mutex> lock(vpu_mutex());
            vpu_clear_status(VPU_CLEAR_ALL);
            if (dense) {
                combined_status = op->op == GGML_OP_ADD
                    ? vpu_add_auto(
                        static_cast<const float *>(lhs->data),
                        static_cast<const float *>(rhs->data),
                        static_cast<float *>(op->data), elements)
                    : vpu_mul_auto(
                        static_cast<const float *>(lhs->data),
                        static_cast<const float *>(rhs->data),
                        static_cast<float *>(op->data), elements);
            } else {
                const size_t width = static_cast<size_t>(op->ne[0]);
                vpu_batch_begin();
                for (size_t row = 0; row < rows; ++row) {
                    const uint64_t status = op->op == GGML_OP_ADD
                        ? vpu_add_enqueue(
                            reinterpret_cast<const float *>(tensor_row(lhs, row)),
                            reinterpret_cast<const float *>(tensor_row(rhs, row)),
                            reinterpret_cast<float *>(tensor_row(op, row)), width)
                        : vpu_mul_enqueue(
                            reinterpret_cast<const float *>(tensor_row(lhs, row)),
                            reinterpret_cast<const float *>(tensor_row(rhs, row)),
                            reinterpret_cast<float *>(tensor_row(op, row)), width);
                    combined_status |= status;
                    if (vpu_status_failed(status)) {
                        break;
                    }
                }
                combined_status |= vpu_batch_finish();
            }
        }
        if (vpu_status_failed(combined_status)) {
            return hardware_failure(
                op->op == GGML_OP_ADD ? "VPU ADD" : "VPU MUL",
                combined_status, fallback_safe);
        }
        return make_ok(combined_status);
    }

    static thread_local aligned_buffer<float> staged_lhs;
    static thread_local aligned_buffer<float> staged_rhs;
    static thread_local aligned_buffer<float> staged_output;
    if (!staged_lhs.allocate(elements) || !staged_rhs.allocate(elements) ||
        !staged_output.allocate(elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "unable to allocate persistent VPU ADD/MUL staging");
    }
    copy_f32_to_linear(lhs, staged_lhs.data());
    copy_f32_to_linear(rhs, staged_rhs.data());

    uint64_t status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        vpu_clear_status(VPU_CLEAR_ALL);
        status = op->op == GGML_OP_ADD
            ? vpu_add_auto(staged_lhs.data(), staged_rhs.data(), staged_output.data(), elements)
            : vpu_mul_auto(staged_lhs.data(), staged_rhs.data(), staged_output.data(), elements);
    }
    if (vpu_status_failed(status)) {
        return hardware_failure(
            op->op == GGML_OP_ADD ? "VPU ADD" : "VPU MUL", status);
    }
    scatter_f32_from_linear(staged_output.data(), op);
    return make_ok(status);
}

static ggml_gemmini_vpu_result compute_swiglu(ggml_tensor * op) {
    const ggml_tensor * source = op->src[0];
    const ggml_tensor * split_up = op->src[1];
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(op, &rows, &elements);
    const size_t width = static_cast<size_t>(op->ne[0]);
    const bool swapped = op_param_i32(op, 1) != 0;

    const bool split_direct_rows = split_up != nullptr &&
        tensor_f32_rows_element_aligned(source) &&
        tensor_f32_rows_element_aligned(split_up) &&
        tensor_f32_rows_element_aligned(op) &&
        tensor_f32_rows_non_overlapping(op) &&
        tensor_direct_alias_supported(op, source) &&
        tensor_direct_alias_supported(op, split_up);
    const bool packed_direct_rows = split_up == nullptr &&
        tensor_f32_rows_element_aligned(source) &&
        tensor_f32_rows_element_aligned(op) &&
        tensor_f32_rows_non_overlapping(op) &&
        !tensor_storage_overlaps(op, source);
    if (split_direct_rows || packed_direct_rows) {
        const bool fallback_safe = !tensor_storage_overlaps(op, source) &&
            !tensor_storage_overlaps(op, split_up);
        const bool dense_split = split_up != nullptr && tensor_f32_dense(source) &&
            tensor_f32_dense(split_up) && tensor_f32_dense(op);
        uint64_t combined_status = 0;
        {
            std::lock_guard<std::mutex> lock(vpu_mutex());
            vpu_clear_status(VPU_CLEAR_ALL);
            if (dense_split) {
                combined_status = vpu_swiglu_auto(
                    static_cast<const float *>(source->data),
                    static_cast<const float *>(split_up->data),
                    static_cast<float *>(op->data),
                    elements);
            } else {
                vpu_batch_begin();
                for (size_t row = 0; row < rows; ++row) {
                    const float * packed = reinterpret_cast<const float *>(
                        tensor_row(source, row));
                    const float * gate_row = packed;
                    const float * up_row = nullptr;
                    if (split_up != nullptr) {
                        up_row = reinterpret_cast<const float *>(tensor_row(split_up, row));
                    } else {
                        gate_row = packed + (swapped ? width : 0);
                        up_row = packed + (swapped ? 0 : width);
                    }
                    const uint64_t status = vpu_swiglu_enqueue(
                        gate_row, up_row,
                        reinterpret_cast<float *>(tensor_row(op, row)), width);
                    combined_status |= status;
                    if (vpu_status_failed(status)) {
                        break;
                    }
                }
                combined_status |= vpu_batch_finish();
            }
        }
        if (vpu_status_failed(combined_status)) {
            return hardware_failure("VPU SwiGLU", combined_status, fallback_safe);
        }
        return make_ok(combined_status);
    }

    static thread_local aligned_buffer<float> gate;
    static thread_local aligned_buffer<float> up;
    static thread_local aligned_buffer<float> output;
    if (!gate.allocate(elements) || !up.allocate(elements) || !output.allocate(elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "unable to allocate 64-byte-aligned SwiGLU staging");
    }

    for (size_t row = 0; row < rows; ++row) {
        const float * packed = reinterpret_cast<const float *>(tensor_row(source, row));
        const float * gate_row = packed;
        const float * up_row = nullptr;
        if (split_up != nullptr) {
            up_row = reinterpret_cast<const float *>(tensor_row(split_up, row));
        } else {
            gate_row = packed + (swapped ? width : 0);
            up_row = packed + (swapped ? 0 : width);
        }
        std::memcpy(gate.data() + row * width, gate_row, width * sizeof(float));
        std::memcpy(up.data() + row * width, up_row, width * sizeof(float));
    }

    uint64_t status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        vpu_clear_status(VPU_CLEAR_ALL);
        status = vpu_swiglu_auto(gate.data(), up.data(), output.data(), elements);
    }
    if (vpu_status_failed(status)) {
        return hardware_failure("VPU SwiGLU", status);
    }
    scatter_f32_from_linear(output.data(), op);
    return make_ok(status);
}

static ggml_gemmini_vpu_result compute_softmax(ggml_tensor * op) {
    const ggml_tensor * source = op->src[0];
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(op, &rows, &elements);
    const size_t width = static_cast<size_t>(op->ne[0]);

    for (size_t row = 0; row < rows; ++row) {
        const uint8_t * source_row = tensor_row(source, row);
        for (size_t column = 0; column < width; ++column) {
            float value = 0.0f;
            std::memcpy(&value, source_row + column * sizeof(float), sizeof(value));
            if (!std::isfinite(value)) {
                return make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
                                   "standalone VPU softmax requires finite logits (row=%zu, column=%zu)",
                                   row, column);
            }
        }
    }

    if (tensor_f32_rows_element_aligned(source) &&
            tensor_f32_rows_element_aligned(op) &&
            tensor_f32_rows_non_overlapping(op) &&
            tensor_direct_alias_supported(op, source)) {
        const bool fallback_safe = !tensor_storage_overlaps(op, source);
        uint64_t combined_status = 0;
        {
            std::lock_guard<std::mutex> lock(vpu_mutex());
            vpu_clear_status(VPU_CLEAR_ALL);
            vpu_batch_begin();
            for (size_t row = 0; row < rows; ++row) {
                const uint64_t status = vpu_softmax_enqueue(
                    reinterpret_cast<const float *>(tensor_row(source, row)),
                    reinterpret_cast<float *>(tensor_row(op, row)),
                    width);
                combined_status |= status;
                if (vpu_status_failed(status)) {
                    break;
                }
            }
            combined_status |= vpu_batch_finish();
        }
        if (vpu_status_failed(combined_status)) {
            return hardware_failure("VPU softmax", combined_status, fallback_safe);
        }
        return make_ok(combined_status);
    }

    static thread_local aligned_buffer<float> input;
    static thread_local aligned_buffer<float> output;
    if (!input.allocate(elements) || !output.allocate(elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "unable to allocate 64-byte-aligned softmax staging");
    }
    copy_f32_to_linear(source, input.data());

    uint64_t combined_status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        vpu_clear_status(VPU_CLEAR_ALL);
        vpu_batch_begin();
        for (size_t row = 0; row < rows; ++row) {
            const uint64_t status = vpu_softmax_enqueue(
                input.data() + row * width,
                output.data() + row * width,
                width);
            combined_status |= status;
            if (vpu_status_failed(status)) {
                break;
            }
        }
        combined_status |= vpu_batch_finish();
    }
    if (vpu_status_failed(combined_status)) {
        return hardware_failure("VPU softmax", combined_status);
    }

    scatter_f32_from_linear(output.data(), op);
    return make_ok(combined_status);
}

static const float * broadcast_weight_row(
        const ggml_tensor * weight,
        const ggml_tensor * normalized,
        size_t              row) {
    const size_t i1 = row % static_cast<size_t>(normalized->ne[1]);
    row /= static_cast<size_t>(normalized->ne[1]);
    const size_t i2 = row % static_cast<size_t>(normalized->ne[2]);
    const size_t i3 = row / static_cast<size_t>(normalized->ne[2]);
    const size_t w1 = weight->ne[1] == 1 ? 0 : i1;
    const size_t w2 = weight->ne[2] == 1 ? 0 : i2;
    const size_t w3 = weight->ne[3] == 1 ? 0 : i3;
    return reinterpret_cast<const float *>(
        static_cast<const uint8_t *>(weight->data) +
        w1 * weight->nb[1] + w2 * weight->nb[2] + w3 * weight->nb[3]);
}

static ggml_gemmini_vpu_result compute_rms_norm_impl(
        const ggml_tensor * source,
        const ggml_tensor * weight,
        ggml_tensor *       destination,
        float               epsilon) {
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(destination, &rows, &elements);
    const size_t width = static_cast<size_t>(destination->ne[0]);

    static thread_local aligned_buffer<float> unit_weight;
    static thread_local size_t unit_weight_initialized = 0;
    if (weight == nullptr) {
        if (!unit_weight.allocate(width)) {
            return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                               "unable to allocate RMSNorm unit-weight workspace");
        }
        if (unit_weight_initialized < width) {
            std::fill(unit_weight.data(), unit_weight.data() + width, 1.0f);
            unit_weight_initialized = width;
        }
    }

    const bool source_alias_supported =
        tensor_direct_alias_supported(destination, source);
    const bool weight_alias_supported = weight == nullptr ||
        tensor_direct_alias_supported(destination, weight);
    const bool row_addresses_aligned =
        tensor_f32_rows_element_aligned(source) &&
        tensor_f32_rows_element_aligned(destination) &&
        tensor_f32_rows_non_overlapping(destination) &&
        (weight == nullptr || tensor_f32_rows_element_aligned(weight));
    if (source_alias_supported && weight_alias_supported && row_addresses_aligned) {
        const bool fallback_safe = !tensor_storage_overlaps(destination, source) &&
            (weight == nullptr || !tensor_storage_overlaps(destination, weight));
        uint64_t combined_status = 0;
        {
            std::lock_guard<std::mutex> lock(vpu_mutex());
            vpu_clear_status(VPU_CLEAR_ALL);
            vpu_batch_begin();
            for (size_t row = 0; row < rows; ++row) {
                const float * weight_data = weight == nullptr
                    ? unit_weight.data()
                    : broadcast_weight_row(weight, destination, row);
                const uint64_t status = vpu_rmsnorm_enqueue(
                    reinterpret_cast<const float *>(tensor_row(source, row)),
                    weight_data,
                    reinterpret_cast<float *>(tensor_row(destination, row)),
                    width, epsilon);
                combined_status |= status;
                if (vpu_status_failed(status)) {
                    break;
                }
            }
            combined_status |= vpu_batch_finish();
        }
        if (vpu_status_failed(combined_status)) {
            return hardware_failure("VPU RMSNorm", combined_status, fallback_safe);
        }
        return make_ok(combined_status);
    }

    static thread_local aligned_buffer<float> input;
    static thread_local aligned_buffer<float> weight_rows;
    static thread_local aligned_buffer<float> output;
    if (!input.allocate(elements) || !output.allocate(elements) ||
        (weight != nullptr && !weight_rows.allocate(elements))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "unable to allocate 64-byte-aligned RMSNorm staging");
    }
    copy_f32_to_linear(source, input.data());
    if (weight != nullptr) {
        for (size_t row = 0; row < rows; ++row) {
            std::memcpy(
                weight_rows.data() + row * width,
                broadcast_weight_row(weight, destination, row),
                width * sizeof(float));
        }
    }

    uint64_t combined_status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        vpu_clear_status(VPU_CLEAR_ALL);
        vpu_batch_begin();
        for (size_t row = 0; row < rows; ++row) {
            const uint64_t status = vpu_rmsnorm_enqueue(
                input.data() + row * width,
                weight == nullptr ? unit_weight.data() : weight_rows.data() + row * width,
                output.data() + row * width,
                width, epsilon);
            combined_status |= status;
            if (vpu_status_failed(status)) {
                break;
            }
        }
        combined_status |= vpu_batch_finish();
    }
    if (vpu_status_failed(combined_status)) {
        return hardware_failure("VPU RMSNorm", combined_status);
    }

    scatter_f32_from_linear(output.data(), destination);
    return make_ok(combined_status);
}

static float rope_yarn_ramp(float low, float high, int32_t element) {
    const float y = (element / 2.0f - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

static void rope_yarn(
        float theta_extrap,
        float freq_scale,
        const float correction[2],
        int32_t element,
        float ext_factor,
        float magnitude_scale,
        float * cosine,
        float * sine) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(correction[0], correction[1], element) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        magnitude_scale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    }
    *cosine = std::cos(theta) * magnitude_scale;
    *sine = std::sin(theta) * magnitude_scale;
}

static bool build_rope_tables(
        const ggml_tensor * factors,
        const rope_parameters & parameters,
        int32_t position,
        float correction[2],
        float * cosine,
        float * sine,
        char * reason,
        size_t capacity) {
    const size_t rotary_dim = static_cast<size_t>(parameters.n_dims);
    const size_t half = rotary_dim / 2;
    const float theta_scale = std::pow(parameters.freq_base, -2.0f / parameters.n_dims);
    float theta = static_cast<float>(position);

    for (size_t pair = 0; pair < half; ++pair) {
        float factor = 1.0f;
        if (factors != nullptr) {
            std::memcpy(&factor,
                        static_cast<const uint8_t *>(factors->data) + pair * factors->nb[0],
                        sizeof(factor));
        }
        if (!std::isfinite(factor) || factor == 0.0f) {
            return reject(reason, capacity, "RoPE frequency factor %zu is zero or non-finite", pair);
        }

        float pair_cosine = 0.0f;
        float pair_sine = 0.0f;
        rope_yarn(theta / factor, parameters.freq_scale, correction,
                  static_cast<int32_t>(pair * 2), parameters.ext_factor,
                  parameters.attn_factor, &pair_cosine, &pair_sine);
        if (!std::isfinite(pair_cosine) || !std::isfinite(pair_sine)) {
            return reject(reason, capacity, "RoPE generated a non-finite angle table");
        }

        if (parameters.mode == GGML_ROPE_TYPE_NORMAL) {
            cosine[pair * 2] = pair_cosine;
            cosine[pair * 2 + 1] = pair_cosine;
            sine[pair * 2] = pair_sine;
            sine[pair * 2 + 1] = pair_sine;
        } else {
            cosine[pair] = pair_cosine;
            cosine[pair + half] = pair_cosine;
            sine[pair] = pair_sine;
            sine[pair + half] = pair_sine;
        }
        theta *= theta_scale;
    }
    return true;
}

static bool same_rope_parameters(
        const rope_parameters & lhs,
        const rope_parameters & rhs) {
    return lhs.n_dims == rhs.n_dims && lhs.mode == rhs.mode &&
        lhs.n_ctx_orig == rhs.n_ctx_orig && lhs.freq_base == rhs.freq_base &&
        lhs.freq_scale == rhs.freq_scale && lhs.ext_factor == rhs.ext_factor &&
        lhs.attn_factor == rhs.attn_factor && lhs.beta_fast == rhs.beta_fast &&
        lhs.beta_slow == rhs.beta_slow;
}

struct rope_table_cache {
    bool valid = false;
    rope_parameters parameters{};
    size_t token_count = 0;
    aligned_buffer<int32_t> positions;
    aligned_buffer<float> cosine;
    aligned_buffer<float> sine;
};

static ggml_gemmini_vpu_result compute_rope(ggml_tensor * op) {
    const ggml_tensor * source = op->src[0];
    const ggml_tensor * positions = op->src[1];
    const ggml_tensor * factors = op->src[2];
    const rope_parameters parameters = read_rope_parameters(op);
    const size_t width = static_cast<size_t>(source->ne[0]);
    const size_t heads = static_cast<size_t>(source->ne[1]);
    const size_t tokens = static_cast<size_t>(source->ne[2]);
    const size_t batches = static_cast<size_t>(source->ne[3]);
    const size_t rotary_dim = static_cast<size_t>(parameters.n_dims);
    size_t packed_elements = 0;
    size_t rows = 0;
    size_t elements = 0;
    (void) tensor_counts(source, &rows, &elements);
    if (!checked_mul(heads, rotary_dim, &packed_elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "RoPE per-token staging size overflows size_t");
    }

    float correction[2] = {0.0f, static_cast<float>(parameters.n_dims - 1)};
    if (parameters.ext_factor != 0.0f) {
        const float denominator = 2.0f * std::log(parameters.freq_base);
        const float fast = parameters.n_dims *
            std::log(parameters.n_ctx_orig /
                     (parameters.beta_fast * 2.0f * static_cast<float>(M_PI))) / denominator;
        const float slow = parameters.n_dims *
            std::log(parameters.n_ctx_orig /
                     (parameters.beta_slow * 2.0f * static_cast<float>(M_PI))) / denominator;
        correction[0] = std::max(0.0f, std::floor(fast));
        correction[1] = std::min(static_cast<float>(parameters.n_dims - 1), std::ceil(slow));
    }

    static thread_local rope_table_cache table_cache;
    static thread_local aligned_buffer<float> cosine_scratch;
    static thread_local aligned_buffer<float> sine_scratch;
    const bool cacheable_tables = factors == nullptr;
    size_t table_elements = 0;
    if (!checked_mul(tokens, rotary_dim, &table_elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "RoPE table size overflows size_t");
    }

    if (cacheable_tables) {
        if (!table_cache.positions.allocate(tokens) ||
            !table_cache.cosine.allocate(table_elements) ||
            !table_cache.sine.allocate(table_elements)) {
            table_cache.valid = false;
            return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                               "unable to allocate persistent RoPE table cache");
        }

        bool cache_hit = table_cache.valid && table_cache.token_count == tokens &&
            same_rope_parameters(table_cache.parameters, parameters);
        for (size_t token = 0; token < tokens; ++token) {
            int32_t position = 0;
            std::memcpy(&position,
                        static_cast<const uint8_t *>(positions->data) + token * positions->nb[0],
                        sizeof(position));
            if (!cache_hit || table_cache.positions[token] != position) {
                cache_hit = false;
            }
            table_cache.positions[token] = position;
        }

        if (!cache_hit) {
            table_cache.valid = false;
            for (size_t token = 0; token < tokens; ++token) {
                char table_reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
                if (!build_rope_tables(
                        nullptr, parameters, table_cache.positions[token], correction,
                        table_cache.cosine.data() + token * rotary_dim,
                        table_cache.sine.data() + token * rotary_dim,
                        table_reason, sizeof(table_reason))) {
                    return make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
                                       "%s", table_reason);
                }
            }
            table_cache.parameters = parameters;
            table_cache.token_count = tokens;
            table_cache.valid = true;
        }
    } else {
        if (!cosine_scratch.allocate(table_elements) ||
            !sine_scratch.allocate(table_elements)) {
            return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                               "unable to allocate persistent RoPE table workspace");
        }
        for (size_t token = 0; token < tokens; ++token) {
            int32_t position = 0;
            std::memcpy(&position,
                        static_cast<const uint8_t *>(positions->data) + token * positions->nb[0],
                        sizeof(position));
            char table_reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
            if (!build_rope_tables(
                    factors, parameters, position, correction,
                    cosine_scratch.data() + token * rotary_dim,
                    sine_scratch.data() + token * rotary_dim,
                    table_reason, sizeof(table_reason))) {
                return make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
                                   "%s", table_reason);
            }
        }
    }

    const bool direct_rows = width == rotary_dim &&
        tensor_f32_rows_element_aligned(source) &&
        tensor_f32_rows_element_aligned(op) &&
        tensor_f32_rows_non_overlapping(op) &&
        source->nb[1] == rotary_dim * sizeof(float) &&
        op->nb[1] == rotary_dim * sizeof(float) &&
        tensor_direct_alias_supported(op, source) &&
        !tensor_storage_overlaps(op, positions) &&
        (factors == nullptr || !tensor_storage_overlaps(op, factors));
    const bool fallback_safe = !tensor_storage_overlaps(op, source);

    static thread_local aligned_buffer<float> packed_input;
    static thread_local aligned_buffer<float> packed_output;
    static thread_local aligned_buffer<float> output;
    size_t token_batches = 0;
    size_t packed_total_elements = 0;
    if (!checked_mul(tokens, batches, &token_batches) ||
        !checked_mul(token_batches, packed_elements, &packed_total_elements)) {
        return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                           "RoPE packed staging size overflows size_t");
    }
    if (!direct_rows) {
        if (!packed_input.allocate(packed_total_elements) ||
            !packed_output.allocate(packed_total_elements) || !output.allocate(elements)) {
            return make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                               "unable to allocate persistent RoPE staging");
        }
        copy_f32_to_linear(source, output.data());
        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t token = 0; token < tokens; ++token) {
                const size_t token_batch = token + tokens * batch;
                const size_t first_tensor_row = heads * token_batch;
                float * packed_token =
                    packed_input.data() + token_batch * packed_elements;
                for (size_t head = 0; head < heads; ++head) {
                    std::memcpy(
                        packed_token + head * rotary_dim,
                        tensor_row(source, first_tensor_row + head),
                        rotary_dim * sizeof(float));
                }
            }
        }
    }

    uint64_t combined_status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        vpu_clear_status(VPU_CLEAR_ALL);
        vpu_batch_begin();
        bool enqueue_failed = false;
        for (size_t batch = 0; batch < batches && !enqueue_failed; ++batch) {
            for (size_t token = 0; token < tokens; ++token) {
                const float * cosine = cacheable_tables
                    ? table_cache.cosine.data() + token * rotary_dim
                    : cosine_scratch.data() + token * rotary_dim;
                const float * sine = cacheable_tables
                    ? table_cache.sine.data() + token * rotary_dim
                    : sine_scratch.data() + token * rotary_dim;

                const size_t token_batch = token + tokens * batch;
                const size_t first_tensor_row = heads * token_batch;
                const float * kernel_input = nullptr;
                float * kernel_output = nullptr;
                if (direct_rows) {
                    kernel_input = reinterpret_cast<const float *>(
                        tensor_row(source, first_tensor_row));
                    kernel_output = reinterpret_cast<float *>(
                        tensor_row(op, first_tensor_row));
                } else {
                    kernel_input =
                        packed_input.data() + token_batch * packed_elements;
                    kernel_output =
                        packed_output.data() + token_batch * packed_elements;
                }

                const uint64_t status = vpu_rope_enqueue(
                    kernel_input, cosine, sine, kernel_output,
                    heads, rotary_dim,
                    parameters.mode == GGML_ROPE_TYPE_NORMAL
                        ? VPU_ROPE_INTERLEAVED : VPU_ROPE_NEOX);
                combined_status |= status;
                if (vpu_status_failed(status)) {
                    enqueue_failed = true;
                    break;
                }
            }
        }
        combined_status |= vpu_batch_finish();
    }
    if (vpu_status_failed(combined_status)) {
        return hardware_failure(
            "VPU RoPE", combined_status, direct_rows ? fallback_safe : true);
    }

    if (!direct_rows) {
        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t token = 0; token < tokens; ++token) {
                const size_t token_batch = token + tokens * batch;
                const size_t first_tensor_row = heads * token_batch;
                const float * packed_token =
                    packed_output.data() + token_batch * packed_elements;
                for (size_t head = 0; head < heads; ++head) {
                    std::memcpy(
                        output.data() + (first_tensor_row + head) * width,
                        packed_token + head * rotary_dim,
                        rotary_dim * sizeof(float));
                }
            }
        }
        scatter_f32_from_linear(output.data(), op);
    }
    return make_ok(combined_status);
}

static const uint8_t * tensor_element_address(
        const ggml_tensor * tensor,
        size_t i0,
        size_t i1,
        size_t i2,
        size_t i3) {
    return static_cast<const uint8_t *>(tensor->data) +
           i0 * tensor->nb[0] + i1 * tensor->nb[1] +
           i2 * tensor->nb[2] + i3 * tensor->nb[3];
}

static bool validate_flash_mask_slice(
        const ggml_tensor * mask,
        size_t query_rows,
        size_t sequence,
        size_t mask_head,
        size_t mask_batch,
        size_t * query_base,
        char * reason,
        size_t capacity) {
    size_t base = 0;
    for (size_t query = 0; query < query_rows; ++query) {
        size_t visible = 0;
        bool hidden_started = false;
        for (size_t key = 0; key < sequence; ++key) {
            uint16_t encoded = 0;
            std::memcpy(&encoded,
                        tensor_element_address(mask, key, query, mask_head, mask_batch),
                        sizeof(encoded));
            const bool is_zero =
                (encoded & UINT16_C(0x7fff)) == UINT16_C(0);
            const bool is_negative_infinity = encoded == UINT16_C(0xfc00);
            if (is_zero && !hidden_started) {
                ++visible;
            } else if (is_negative_infinity) {
                hidden_started = true;
            } else if (is_zero) {
                return reject(reason, capacity,
                              "FlashAttention mask is not prefix-causal at query=%zu key=%zu",
                              query, key);
            } else {
                return reject(reason, capacity,
                              "FlashAttention mask has an unsupported additive value at query=%zu key=%zu",
                              query, key);
            }
        }

        if (visible == 0) {
            return reject(reason, capacity, "FlashAttention mask hides every key for query %zu", query);
        }
        if (query == 0) {
            base = visible - 1;
        } else if (visible != base + query + 1) {
            return reject(reason, capacity,
                          "FlashAttention mask cutoff does not advance by one key per query");
        }
    }
    if (base > sequence || query_rows > sequence - base) {
        return reject(reason, capacity, "FlashAttention causal query range exceeds the KV sequence");
    }
    *query_base = base;
    return true;
}

static const char * flash_status_name(vpu_flashattention_status_t status) {
    switch (status) {
        case VPU_FLASHATTENTION_OK:                          return "ok";
        case VPU_FLASHATTENTION_INVALID_ARGUMENT:            return "invalid argument";
        case VPU_FLASHATTENTION_INVALID_SHAPE:               return "invalid shape";
        case VPU_FLASHATTENTION_INVALID_STRIDE:              return "invalid stride";
        case VPU_FLASHATTENTION_INVALID_GEMMINI_MASK:        return "invalid Gemmini mask";
        case VPU_FLASHATTENTION_INSUFFICIENT_MASK_WORKSPACE: return "insufficient mask workspace";
        case VPU_FLASHATTENTION_ADDRESS_RANGE_OVERFLOW:      return "address range overflow";
        case VPU_FLASHATTENTION_NO_TILING:                   return "no legal tiling";
        case VPU_FLASHATTENTION_QK_JOB_UNSUPPORTED:          return "QK job unsupported";
        case VPU_FLASHATTENTION_PV_JOB_UNSUPPORTED:          return "PV job unsupported";
        case VPU_FLASHATTENTION_VPU_ERROR:                   return "VPU error";
    }
    return "unknown FlashAttention status";
}

static void encode_flash_input_span(
        const uint8_t * source,
        ggml_type       type,
        elem_t *        destination,
        size_t          elements) {
    if (type == GGML_TYPE_BF16) {
        static_assert(sizeof(ggml_bf16_t) == sizeof(elem_t),
                      "FlashAttention BF16 input must match Gemmini elements");
        std::memcpy(destination, source, elements * sizeof(elem_t));
        return;
    }

    const size_t source_element_bytes = staging_type_size(type);
    for (size_t element = 0; element < elements; ++element) {
        const float value = load_staging_float(
            type, source + element * source_element_bytes);
        destination[element] =
            static_cast<elem_t>(vpu_float_to_bf16(value));
    }
}

static elem_t * flash_staging_row_span(
        elem_t *             base,
        size_t               row,
        size_t               column,
        size_t               columns,
        flash_storage_layout layout) {
    switch (layout) {
        case flash_storage_layout::linear:
            return base + row * columns + column;
        case flash_storage_layout::page_packed_a: {
            elem_t * block = gemmini_page_packed_a_block_addr_mut(
                base, row / DIM, column / DIM, columns);
            return block + (row % DIM) *
                gemmini_page_packed_a_k_blocks_per_page() * DIM + column % DIM;
        }
        case flash_storage_layout::page_packed_b: {
            elem_t * block = gemmini_page_packed_b_block_addr_mut(
                base, row / DIM, column / DIM, columns);
            return block + (row % DIM) *
                gemmini_page_packed_b_j_blocks_per_page() * DIM + column % DIM;
        }
    }
    return nullptr;
}

static ggml_gemmini_flash_tensor_view flash_tensor_view(
        const ggml_tensor * tensor,
        size_t              element_bytes) {
    ggml_gemmini_flash_tensor_view view;
    if (tensor == nullptr) {
        return view;
    }
    view.base = reinterpret_cast<uintptr_t>(tensor->data);
    view.element_bytes = element_bytes;
    for (size_t dimension = 0; dimension < 4; ++dimension) {
        view.ne[dimension] = static_cast<size_t>(tensor->ne[dimension]);
        view.nb[dimension] = tensor->nb[dimension];
    }
    return view;
}

// A linear BF16 host slice can be consumed in place only when its physical row
// stride exactly matches the stride encoded in the fusion config.  Keep the
// overlap check because FlashAttention always writes directly to its final
// destination while the corresponding K/V slice remains live.
static bool flash_linear_bf16_zero_copy_eligible(
        const ggml_tensor * source,
        size_t              rows,
        size_t              columns,
        flash_storage_layout layout,
        const ggml_tensor * destination) {
    if (source == nullptr ||
        rows != static_cast<size_t>(source->ne[1]) ||
        columns != static_cast<size_t>(source->ne[0])) {
        return false;
    }
    const ggml_gemmini_flash_direct_slice_plan plan =
        ggml_gemmini_flash_make_direct_bf16_input_slice(
            true, layout != flash_storage_layout::linear,
            source->type == GGML_TYPE_BF16, true,
            !tensor_storage_overlaps(source, destination),
            flash_tensor_view(source, sizeof(ggml_bf16_t)), 0, 0);
    return plan.direct;
}

static const elem_t * flash_bf16_slice(
        const ggml_tensor * source,
        size_t              head,
        size_t              batch) {
    static_assert(sizeof(ggml_bf16_t) == sizeof(elem_t),
                  "FlashAttention BF16 input must match Gemmini elements");
    const ggml_gemmini_flash_direct_slice_plan plan =
        ggml_gemmini_flash_make_direct_bf16_input_slice(
            true, false, source->type == GGML_TYPE_BF16, true, true,
            flash_tensor_view(source, sizeof(ggml_bf16_t)), head, batch);
    GGML_ASSERT(plan.direct);
    return reinterpret_cast<const elem_t *>(plan.address);
}

static const elem_t * stage_flash_matrix(
        const ggml_tensor * source,
        size_t rows,
        size_t columns,
        size_t head,
        size_t batch,
        flash_storage_layout layout,
        elem_t * destination) {
    const size_t source_element_bytes = staging_type_size(source->type);
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t * source_row =
            tensor_element_address(source, 0, row, head, batch);
        if (layout == flash_storage_layout::linear) {
            encode_flash_input_span(
                source_row, source->type,
                destination + row * columns, columns);
            continue;
        }

        for (size_t column = 0; column < columns; column += DIM) {
            const size_t valid_columns =
                std::min(static_cast<size_t>(DIM), columns - column);
            elem_t * destination_span = flash_staging_row_span(
                destination, row, column, columns, layout);
            GGML_ASSERT(destination_span != nullptr);
            encode_flash_input_span(
                source_row + column * source_element_bytes,
                source->type, destination_span, valid_columns);
        }
    }
    return destination;
}

static bool flash_direct_output_layout(
        const ggml_tensor * destination,
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * mask,
        size_t              query_rows,
        size_t              value_dim,
        size_t *            output_stride) {
    if (output_stride == nullptr || destination == nullptr ||
        query_rows != static_cast<size_t>(destination->ne[2]) ||
        value_dim != static_cast<size_t>(destination->ne[0])) {
        return false;
    }
    const bool disjoint =
        !tensor_storage_overlaps(destination, q) &&
        !tensor_storage_overlaps(destination, k) &&
        !tensor_storage_overlaps(destination, v) &&
        !tensor_storage_overlaps(destination, mask);
    const ggml_gemmini_flash_direct_slice_plan plan =
        ggml_gemmini_flash_make_direct_output_slice(
            true, destination->type == GGML_TYPE_F32, disjoint,
            flash_tensor_view(destination, sizeof(float)), 0, 0);
    if (!plan.direct) {
        return false;
    }
    *output_stride = plan.row_stride_elements;
    return true;
}

static float * flash_direct_output_base(
        ggml_tensor * destination,
        size_t        batch,
        size_t        head) {
    const ggml_gemmini_flash_direct_slice_plan plan =
        ggml_gemmini_flash_make_direct_output_slice(
            true, destination->type == GGML_TYPE_F32, true,
            flash_tensor_view(destination, sizeof(float)), head, batch);
    GGML_ASSERT(plan.direct);
    return reinterpret_cast<float *>(plan.address);
}

static size_t flash_config_stride(size_t logical_width, bool page_packed) {
    return page_packed ? GEMMINI_PAGE_PACKED_STRIDE(logical_width)
                       : logical_width;
}

static vpu_flashattention_config_t make_flash_config(
        const elem_t * queries,
        const elem_t * keys,
        const elem_t * values,
        float * output,
        size_t query_rows,
        size_t sequence,
        size_t q_dim,
        size_t value_dim,
        float score_scale,
        size_t output_stride,
        size_t query_base,
        unsigned logical_mask,
        const ggml_gemmini_flash_page_packing_config & packing,
        float * causal_workspace) {
    vpu_flashattention_config_t config{};
    config.queries = queries;
    config.keys = keys;
    config.values = values;
    config.output = output;
    config.query_rows = query_rows;
    config.sequence = sequence;
    config.q_dim = q_dim;
    config.k_dim = q_dim;
    config.value_dim = value_dim;
    config.score_scale = score_scale;
    config.query_base = query_base;
    config.query_stride = flash_config_stride(q_dim, packing.queries);
    config.key_stride = flash_config_stride(q_dim, packing.keys);
    config.value_stride = flash_config_stride(value_dim, packing.values);
    config.output_stride = output_stride;
    config.gemmini_mask = logical_mask;
    config.causal_mask_workspace = causal_workspace;
    config.causal_mask_workspace_elements = VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS;
    return config;
}

static ggml_gemmini_vpu_result compute_flash_attention(
        ggml_tensor * op,
        unsigned      logical_mask,
        uint8_t       requested_flash_page_packing_mask) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    const size_t d = static_cast<size_t>(q->ne[0]);
    const size_t query_rows = static_cast<size_t>(q->ne[1]);
    const size_t heads = static_cast<size_t>(q->ne[2]);
    const size_t batches = static_cast<size_t>(q->ne[3]);
    const size_t sequence = static_cast<size_t>(k->ne[1]);
    const size_t kv_heads = static_cast<size_t>(k->ne[2]);
    const size_t kv_batches = static_cast<size_t>(k->ne[3]);
    const size_t value_dim = static_cast<size_t>(v->ne[0]);
    const float scale = op_param_f32(op, 0);
    const auto packing = flash_page_packing_for_shape(
        requested_flash_page_packing_mask, query_rows, d, sequence);
    const flash_storage_layout q_layout = packing.queries
        ? flash_storage_layout::page_packed_a
        : flash_storage_layout::linear;
    const flash_storage_layout k_layout = packing.keys
        ? flash_storage_layout::page_packed_b
        : flash_storage_layout::linear;
    const flash_storage_layout v_layout = packing.values
        ? flash_storage_layout::page_packed_b
        : flash_storage_layout::linear;
    const bool direct_k_input = flash_linear_bf16_zero_copy_eligible(
        k, sequence, d, k_layout, op);
    const bool direct_v_input = flash_linear_bf16_zero_copy_eligible(
        v, sequence, value_dim, v_layout, op);
    size_t direct_output_stride = 0;
    const bool direct_output = flash_direct_output_layout(
        op, q, k, v, mask, query_rows, value_dim, &direct_output_stride);
    ggml_gemmini_flash_breakdown breakdown{};
    const auto finish = [&breakdown](ggml_gemmini_vpu_result result) {
        result.flash = breakdown;
        return result;
    };
    flash_workspace & workspace = get_flash_workspace();

    // FlashAttention output packing is intentionally ignored. Every hardware
    // execution must H_STORE its final FP32 rows directly into the ggml dst.
    // Exotic/overlapping destinations stay on the CPU fallback path instead
    // of reintroducing an output staging buffer.
    if (!direct_output) {
        return finish(make_result(
            GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
            "FlashAttention destination is not a direct-safe row-major F32 view"));
    }
    const size_t config_output_stride = direct_output_stride;

    flash_phase_timer mask_prepare_timer;
    size_t invocation_count = 0;
    if (!checked_mul(batches, heads, &invocation_count)) {
        mask_prepare_timer.finish(
            &breakdown.mask_prepare_us, &breakdown.mask_prepare_cycles);
        return finish(make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0,
                                  "FlashAttention invocation count overflows size_t"));
    }
    size_t unique_mask_count = 0;
    const size_t mask_heads = static_cast<size_t>(mask->ne[2]);
    const size_t mask_batches = static_cast<size_t>(mask->ne[3]);
    if (!checked_mul(mask_heads, mask_batches, &unique_mask_count) ||
        !allocate_overwritten_flash_workspace(
            &workspace.query_bases, invocation_count) ||
        !allocate_overwritten_flash_workspace(
            &workspace.unique_mask_bases, unique_mask_count)) {
        mask_prepare_timer.finish(
            &breakdown.mask_prepare_us, &breakdown.mask_prepare_cycles);
        return finish(make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                                  "unable to allocate causal-mask metadata"));
    }
    // A broadcast mask slice is identical for every query head/batch mapped
    // to it. Validate each physical slice once instead of rescanning the same
    // N-by-M mask for every FlashAttention invocation.
    for (size_t mask_batch = 0; mask_batch < mask_batches; ++mask_batch) {
        for (size_t mask_head = 0; mask_head < mask_heads; ++mask_head) {
            const size_t unique_index = mask_batch * mask_heads + mask_head;
            char mask_reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
            if (!validate_flash_mask_slice(mask, query_rows, sequence,
                                           mask_head, mask_batch,
                                           &workspace.unique_mask_bases[unique_index],
                                           mask_reason, sizeof(mask_reason))) {
                mask_prepare_timer.finish(
                    &breakdown.mask_prepare_us, &breakdown.mask_prepare_cycles);
                return finish(make_result(
                    GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
                    "%s (mask batch=%zu, head=%zu)",
                    mask_reason, mask_batch, mask_head));
            }
        }
    }
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t head = 0; head < heads; ++head) {
            const size_t unique_index =
                (batch % mask_batches) * mask_heads + (head % mask_heads);
            workspace.query_bases[batch * heads + head] =
                workspace.unique_mask_bases[unique_index];
        }
    }
    mask_prepare_timer.finish(
        &breakdown.mask_prepare_us, &breakdown.mask_prepare_cycles);

    flash_phase_timer workspace_prepare_timer;
    size_t q_storage_elements = 0;
    size_t k_storage_elements = 0;
    size_t v_storage_elements = 0;
    if (!flash_storage_elements(
            query_rows, d, q_layout, sizeof(elem_t), &q_storage_elements) ||
        !flash_storage_elements(
            sequence, d, k_layout, sizeof(elem_t), &k_storage_elements) ||
        !flash_storage_elements(
            sequence, value_dim, v_layout, sizeof(elem_t), &v_storage_elements)) {
        workspace_prepare_timer.finish(
            &breakdown.workspace_prepare_us,
            &breakdown.workspace_prepare_cycles);
        return finish(make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0,
                                  "FlashAttention staging size overflows size_t"));
    }

    if (!allocate_overwritten_flash_workspace(
            &workspace.staged_q, q_storage_elements,
            packing.queries ? k_page_packed_alignment : k_workspace_alignment) ||
        (!direct_k_input && !allocate_overwritten_flash_workspace(
            &workspace.staged_k, k_storage_elements,
            packing.keys ? k_page_packed_alignment : k_workspace_alignment)) ||
        (!direct_v_input && !allocate_overwritten_flash_workspace(
            &workspace.staged_v, v_storage_elements,
            packing.values ? k_page_packed_alignment : k_workspace_alignment)) ||
        !allocate_overwritten_flash_workspace(
            &workspace.causal_mask, VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS)) {
        workspace_prepare_timer.finish(
            &breakdown.workspace_prepare_us,
            &breakdown.workspace_prepare_cycles);
        return finish(make_result(GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED, 0,
                                  "unable to allocate aligned FlashAttention workspace"));
    }
    workspace_prepare_timer.finish(
        &breakdown.workspace_prepare_us,
        &breakdown.workspace_prepare_cycles);

    const elem_t * const preflight_k_input = direct_k_input
        ? flash_bf16_slice(k, 0, 0)
        : workspace.staged_k.data();
    const elem_t * const preflight_v_input = direct_v_input
        ? flash_bf16_slice(v, 0, 0)
        : workspace.staged_v.data();

    const auto invocation_output = [op](
            size_t batch, size_t head) -> float * {
        return flash_direct_output_base(op, batch, head);
    };

    // Tiling and job legality depend on the causal query base, not on the
    // broadcast head that uses it. Direct input/output helpers already prove
    // every slice address and stride, so preflight each distinct base once
    // before issuing the first hardware command.
    flash_phase_timer plan_preflight_timer;
    for (size_t unique_index = 0; unique_index < unique_mask_count;
         ++unique_index) {
        const size_t query_base = workspace.unique_mask_bases[unique_index];
        bool already_preflighted = false;
        for (size_t previous = 0; previous < unique_index; ++previous) {
            if (workspace.unique_mask_bases[previous] == query_base) {
                already_preflighted = true;
                break;
            }
        }
        if (already_preflighted) {
            continue;
        }
        const vpu_flashattention_config_t config = make_flash_config(
            workspace.staged_q.data(), preflight_k_input, preflight_v_input,
            invocation_output(0, 0),
            query_rows, sequence, d, value_dim, scale, config_output_stride,
            query_base,
            logical_mask, packing, workspace.causal_mask.data());
        vpu_flashattention_plan_t plan{};
        vpu_flashattention_status_t status = vpu_flashattention_make_plan(&config, &plan);
        if (status == VPU_FLASHATTENTION_OK) {
            status = vpu_fa_preflight_jobs(&config, &plan);
        }
        if (status != VPU_FLASHATTENTION_OK) {
            plan_preflight_timer.finish(
                &breakdown.plan_preflight_us,
                &breakdown.plan_preflight_cycles);
            return finish(make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
                                      "FlashAttention preflight failed: %s",
                                      flash_status_name(status)));
        }
    }
    plan_preflight_timer.finish(
        &breakdown.plan_preflight_us,
        &breakdown.plan_preflight_cycles);

    uint64_t combined_status = 0;
    {
        std::lock_guard<std::mutex> lock(vpu_mutex());
        size_t staged_k_head = std::numeric_limits<size_t>::max();
        size_t staged_k_batch = std::numeric_limits<size_t>::max();
        const elem_t * staged_k_input = preflight_k_input;
        const elem_t * staged_v_input = preflight_v_input;
        for (size_t batch = 0; batch < batches; ++batch) {
            const size_t kv_batch = batch / (batches / kv_batches);
            for (size_t head = 0; head < heads; ++head) {
                const size_t kv_head = head / (heads / kv_heads);
                if (kv_head != staged_k_head || kv_batch != staged_k_batch) {
                    flash_phase_timer input_pack_timer;
                    staged_k_input = direct_k_input
                        ? flash_bf16_slice(k, kv_head, kv_batch)
                        : stage_flash_matrix(
                              k, sequence, d, kv_head, kv_batch,
                              k_layout, workspace.staged_k.data());
                    staged_v_input = direct_v_input
                        ? flash_bf16_slice(v, kv_head, kv_batch)
                        : stage_flash_matrix(
                              v, sequence, value_dim, kv_head, kv_batch,
                              v_layout, workspace.staged_v.data());
                    input_pack_timer.finish(
                        &breakdown.input_pack_us,
                        &breakdown.input_pack_cycles);
                    staged_k_head = kv_head;
                    staged_k_batch = kv_batch;
                }
                flash_phase_timer input_pack_timer;
                const elem_t * staged_q_input = stage_flash_matrix(
                    q, query_rows, d, head, batch, q_layout,
                    workspace.staged_q.data());
                input_pack_timer.finish(
                    &breakdown.input_pack_us,
                    &breakdown.input_pack_cycles);

                const vpu_flashattention_config_t config = make_flash_config(
                    staged_q_input, staged_k_input, staged_v_input,
                    invocation_output(batch, head),
                    query_rows, sequence, d, value_dim, scale, config_output_stride,
                    workspace.query_bases[batch * heads + head], logical_mask,
                    packing, workspace.causal_mask.data());
                flash_phase_timer fused_attention_run_timer;
                const vpu_flashattention_result_t flash = vpu_flashattention_auto(&config);
                fused_attention_run_timer.finish(
                    &breakdown.fused_attention_run_us,
                    &breakdown.fused_attention_run_cycles);
                combined_status |= flash.vpu_status;
                if (flash.status != VPU_FLASHATTENTION_OK) {
                    const ggml_gemmini_vpu_status result_status =
                        flash.status == VPU_FLASHATTENTION_VPU_ERROR
                            ? GGML_GEMMINI_VPU_STATUS_HARDWARE_ERROR
                            : GGML_GEMMINI_VPU_STATUS_UNSUPPORTED;
                    return finish(make_result(
                        result_status, flash.vpu_status,
                        "FlashAttention execution failed: %s",
                        flash_status_name(flash.status)));
                }

            }
        }
    }
    return finish(make_ok(combined_status));
}

static bool require_compute_data(
        const ggml_tensor * tensor,
        const char * name,
        char * reason,
        size_t capacity) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return reject(reason, capacity, "%s data is not allocated", name);
    }
    return true;
}
#endif

} // namespace

extern "C" bool ggml_gemmini_vpu_can_compute(
        const ggml_tensor * op,
        unsigned            logical_gemmini_mask,
        uint8_t             requested_flash_page_packing_mask,
        char *              reason,
        size_t              reason_capacity) {
    clear_reason(reason, reason_capacity);
    if (op == nullptr) {
        return reject(reason, reason_capacity, "operation is null");
    }

    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return can_compute_binary_elementwise(op, reason, reason_capacity);
        case GGML_OP_UNARY:
            return can_compute_unary(op, reason, reason_capacity);
        case GGML_OP_GLU:
            return can_compute_swiglu(op, reason, reason_capacity);
        case GGML_OP_SOFT_MAX:
            return can_compute_softmax(op, reason, reason_capacity);
        case GGML_OP_RMS_NORM:
            return can_compute_rms_norm(op, reason, reason_capacity);
        case GGML_OP_ROPE:
            return can_compute_rope(op, reason, reason_capacity);
        case GGML_OP_FLASH_ATTN_EXT:
            return can_compute_flash_attention(
                op, logical_gemmini_mask, requested_flash_page_packing_mask,
                reason, reason_capacity);
        default:
            return reject(reason, reason_capacity,
                          "ggml op %d has no VPU adapter", static_cast<int>(op->op));
    }
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
extern "C" ggml_gemmini_vpu_result ggml_gemmini_vpu_compute(
        ggml_tensor * op,
        unsigned      logical_gemmini_mask,
        uint8_t       requested_flash_page_packing_mask) {
    char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
    if (!ggml_gemmini_vpu_can_compute(
            op, logical_gemmini_mask, requested_flash_page_packing_mask,
            reason, sizeof(reason))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0, "%s", reason);
    }

    if (!require_compute_data(op, "destination", reason, sizeof(reason)) ||
        !require_compute_data(op->src[0], "source 0", reason, sizeof(reason))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0, "%s", reason);
    }
    if (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL ||
        (op->op == GGML_OP_GLU && op->src[1] != nullptr) ||
        op->op == GGML_OP_FLASH_ATTN_EXT) {
        if (!require_compute_data(op->src[1], "source 1", reason, sizeof(reason))) {
            return make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0, "%s", reason);
        }
    }
    if (op->op == GGML_OP_ROPE &&
        !require_compute_data(op->src[1], "RoPE positions", reason, sizeof(reason))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0, "%s", reason);
    }
    if (op->op == GGML_OP_ROPE && op->src[2] != nullptr &&
        !require_compute_data(op->src[2], "RoPE frequency factors", reason, sizeof(reason))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0, "%s", reason);
    }
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        for (int source = 2; source <= 3; ++source) {
            if (!require_compute_data(op->src[source], source == 2 ? "FlashAttention V" : "FlashAttention mask",
                                      reason, sizeof(reason))) {
                return make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0, "%s", reason);
            }
        }
    }

    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:           return compute_binary_elementwise(op);
        case GGML_OP_UNARY:         return compute_unary(op);
        case GGML_OP_GLU:           return compute_swiglu(op);
        case GGML_OP_SOFT_MAX:      return compute_softmax(op);
        case GGML_OP_RMS_NORM:
            return compute_rms_norm_impl(op->src[0], nullptr, op, op_param_f32(op, 0));
        case GGML_OP_ROPE:          return compute_rope(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return compute_flash_attention(
                op, logical_gemmini_mask, requested_flash_page_packing_mask);
        default:
            return make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0,
                               "operation changed after admission");
    }
}
#endif

extern "C" bool ggml_gemmini_vpu_can_compute_weighted_rms_norm(
        const ggml_tensor * rms_norm_node,
        const ggml_tensor * weight,
        const ggml_tensor * weighted_dst,
        char *              reason,
        size_t              reason_capacity) {
    clear_reason(reason, reason_capacity);
    return can_compute_weighted_rms_norm_impl(
        rms_norm_node, weight, weighted_dst, reason, reason_capacity);
}

#if !defined(GGML_GEMMINI_VPU_ADMISSION_ONLY)
extern "C" ggml_gemmini_vpu_result ggml_gemmini_vpu_compute_weighted_rms_norm(
        const ggml_tensor * rms_norm_node,
        const ggml_tensor * weight,
        ggml_tensor *       weighted_dst) {
    char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
    if (!can_compute_weighted_rms_norm_impl(
            rms_norm_node, weight, weighted_dst, reason, sizeof(reason))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_UNSUPPORTED, 0, "%s", reason);
    }
    if (!require_compute_data(rms_norm_node->src[0], "RMSNorm source", reason, sizeof(reason)) ||
        !require_compute_data(weight, "RMSNorm weight", reason, sizeof(reason)) ||
        !require_compute_data(weighted_dst, "weighted RMSNorm destination", reason, sizeof(reason))) {
        return make_result(GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT, 0, "%s", reason);
    }
    return compute_rms_norm_impl(
        rms_norm_node->src[0], weight, weighted_dst, op_param_f32(rms_norm_node, 0));
}
#endif

extern "C" const char * ggml_gemmini_vpu_status_name(ggml_gemmini_vpu_status status) {
    switch (status) {
        case GGML_GEMMINI_VPU_STATUS_OK:                return "ok";
        case GGML_GEMMINI_VPU_STATUS_UNSUPPORTED:       return "unsupported";
        case GGML_GEMMINI_VPU_STATUS_INVALID_ARGUMENT:  return "invalid argument";
        case GGML_GEMMINI_VPU_STATUS_ALLOCATION_FAILED: return "allocation failed";
        case GGML_GEMMINI_VPU_STATUS_HARDWARE_ERROR:    return "hardware error";
    }
    return "unknown";
}
