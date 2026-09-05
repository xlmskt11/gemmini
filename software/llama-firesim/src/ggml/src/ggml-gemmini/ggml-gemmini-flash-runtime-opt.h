#ifndef GGML_GEMMINI_FLASH_RUNTIME_OPT_H
#define GGML_GEMMINI_FLASH_RUNTIME_OPT_H

#include <cstddef>
#include <cstdint>
#include <limits>

// A dependency-free description of a ggml four-dimensional tensor view.
// ne[] is in elements and nb[] is in bytes.  Keeping ggml types out of this
// header lets host-only regression tests exercise the exact runtime policy.
struct ggml_gemmini_flash_tensor_view {
    uintptr_t base = 0;
    size_t ne[4] = {};
    size_t nb[4] = {};
    size_t element_bytes = 0;
};

struct ggml_gemmini_flash_direct_slice_plan {
    bool direct = false;
    uintptr_t address = 0;
    size_t row_stride_elements = 0;
};

struct ggml_gemmini_flash_workspace_plan {
    bool valid = false;
    bool reuse = false;
    bool allocate = false;
    bool zero_initialize = false;
    size_t active_count = 0;
    size_t retained_capacity = 0;
    size_t required_alignment = 0;
};

namespace ggml_gemmini_flash_runtime_detail {

constexpr size_t linear_stride_max =
    static_cast<size_t>(std::numeric_limits<uint32_t>::max() >> 1);

static inline bool checked_add(size_t lhs, size_t rhs, size_t * result) {
    if (result == nullptr || rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

static inline bool checked_mul(size_t lhs, size_t rhs, size_t * result) {
    if (result == nullptr ||
            (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static inline bool valid_nonzero_shape(
        const ggml_gemmini_flash_tensor_view & view) {
    for (size_t dimension = 0; dimension < 4; ++dimension) {
        if (view.ne[dimension] == 0) {
            return false;
        }
    }
    return true;
}

static inline bool checked_slice_address(
        const ggml_gemmini_flash_tensor_view & view,
        size_t head,
        size_t batch,
        uintptr_t * address) {
    size_t head_offset = 0;
    size_t batch_offset = 0;
    size_t offset = 0;
    if (address == nullptr || head >= view.ne[2] || batch >= view.ne[3] ||
            !checked_mul(head, view.nb[2], &head_offset) ||
            !checked_mul(batch, view.nb[3], &batch_offset) ||
            !checked_add(head_offset, batch_offset, &offset) ||
            offset > std::numeric_limits<uintptr_t>::max() - view.base) {
        return false;
    }
    *address = view.base + offset;
    return true;
}

static inline bool power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace ggml_gemmini_flash_runtime_detail

// Plan one direct FlashAttention output invocation.  Output views use ggml's
// [DV, H, N, B] order, whereas one hardware invocation writes [N, DV].  A
// direct store is safe only for an ordinary nested, non-overlapping F32 view.
// FlashAttention has no packed-output mode: its C request bit is ignored and
// the VPU always emits ordinary row-major H_STORE operations.
static inline ggml_gemmini_flash_direct_slice_plan
ggml_gemmini_flash_make_direct_output_slice(
        bool enabled,
        bool fp32_destination,
        bool destination_disjoint_from_inputs,
        const ggml_gemmini_flash_tensor_view & view,
        size_t head,
        size_t batch) {
    ggml_gemmini_flash_direct_slice_plan plan;
    constexpr size_t fp32_bytes = sizeof(float);
    using namespace ggml_gemmini_flash_runtime_detail;

    if (!enabled || !fp32_destination ||
            !destination_disjoint_from_inputs || view.base == 0 ||
            view.element_bytes != fp32_bytes || view.nb[0] != fp32_bytes ||
            !valid_nonzero_shape(view) || head >= view.ne[1] ||
            batch >= view.ne[3] || view.base % alignof(float) != 0) {
        return plan;
    }

    // Prove that all logical destination elements occupy distinct bytes.  It
    // is not enough to check rows inside one invocation: later head/batch
    // invocations must not overwrite an earlier invocation's result.
    size_t byte_extent = 0;
    if (!checked_mul(view.ne[0], fp32_bytes, &byte_extent)) {
        return plan;
    }
    for (size_t dimension = 1; dimension < 4; ++dimension) {
        if (view.ne[dimension] <= 1) {
            continue;
        }
        if (view.nb[dimension] < byte_extent ||
                view.nb[dimension] % fp32_bytes != 0) {
            return plan;
        }
        size_t added_extent = 0;
        if (!checked_mul(view.ne[dimension] - 1, view.nb[dimension],
                         &added_extent) ||
                !checked_add(byte_extent, added_extent, &byte_extent)) {
            return plan;
        }
    }
    if (byte_extent == 0 ||
            byte_extent - 1 >
                std::numeric_limits<uintptr_t>::max() - view.base) {
        return plan;
    }

    const size_t output_stride = view.ne[2] > 1
        ? view.nb[2] / fp32_bytes
        : view.ne[0];
    if (output_stride < view.ne[0] || output_stride > linear_stride_max) {
        return plan;
    }

    size_t last_row_offset = 0;
    size_t last_element_offset = 0;
    if (!checked_mul(view.ne[2] - 1, output_stride, &last_row_offset) ||
            !checked_add(last_row_offset, view.ne[0] - 1,
                         &last_element_offset) ||
            last_element_offset > std::numeric_limits<uint32_t>::max()) {
        return plan;
    }

    // For an output view, head is dimension 1 and query is dimension 2.
    size_t head_offset = 0;
    size_t batch_offset = 0;
    size_t slice_offset = 0;
    if (!checked_mul(head, view.nb[1], &head_offset) ||
            !checked_mul(batch, view.nb[3], &batch_offset) ||
            !checked_add(head_offset, batch_offset, &slice_offset) ||
            slice_offset > std::numeric_limits<uintptr_t>::max() - view.base) {
        return plan;
    }
    const uintptr_t address = view.base + slice_offset;
    if (address % alignof(float) != 0) {
        return plan;
    }

    plan.direct = true;
    plan.address = address;
    plan.row_stride_elements = output_stride;
    return plan;
}

// Plan a direct K/V input slice. Input views use [width, rows, heads,
// batches]. The runtime may bypass BF16 conversion only when no scaling or
// other transform is required. The current fusion config encodes logical
// width for a linear source, so padded/strided rows retain staging. Read-only
// source slices may overlap one another (broadcast views),
// but they must not overlap the selected direct output.
static inline ggml_gemmini_flash_direct_slice_plan
ggml_gemmini_flash_make_direct_bf16_input_slice(
        bool enabled,
        bool bf16_source,
        bool identity_transform,
        bool source_disjoint_from_output,
        const ggml_gemmini_flash_tensor_view & view,
        size_t head,
        size_t batch) {
    ggml_gemmini_flash_direct_slice_plan plan;
    constexpr size_t bf16_bytes = sizeof(uint16_t);
    using namespace ggml_gemmini_flash_runtime_detail;

    if (!enabled || !bf16_source || !identity_transform ||
            !source_disjoint_from_output || view.base == 0 ||
            view.element_bytes != bf16_bytes || view.nb[0] != bf16_bytes ||
            !valid_nonzero_shape(view) || head >= view.ne[2] ||
            batch >= view.ne[3]) {
        return plan;
    }

    size_t row_bytes = 0;
    if (!checked_mul(view.ne[0], bf16_bytes, &row_bytes) ||
            view.nb[1] != row_bytes ||
            (view.ne[2] > 1 && view.nb[2] % alignof(uint16_t) != 0) ||
            (view.ne[3] > 1 && view.nb[3] % alignof(uint16_t) != 0)) {
        return plan;
    }
    const size_t row_stride = view.ne[0];
    if (row_stride > linear_stride_max) {
        return plan;
    }

    uintptr_t address = 0;
    if (!checked_slice_address(view, head, batch, &address) ||
            address % alignof(uint16_t) != 0) {
        return plan;
    }

    // Mirror vpu_fa_element_extent_overflows() and also prove that adding the
    // byte span to this particular head/batch base cannot wrap uintptr_t.
    size_t last_row = 0;
    size_t last_element = 0;
    size_t span_elements = 0;
    size_t span_bytes = 0;
    if (!checked_mul(view.ne[1] - 1, row_stride, &last_row) ||
            !checked_add(last_row, view.ne[0] - 1, &last_element) ||
            last_element >
                std::numeric_limits<size_t>::max() / bf16_bytes ||
            !checked_add(last_element, 1, &span_elements) ||
            !checked_mul(span_elements, bf16_bytes, &span_bytes) ||
            span_bytes - 1 >
                std::numeric_limits<uintptr_t>::max() - address) {
        return plan;
    }

    plan.direct = true;
    plan.address = address;
    plan.row_stride_elements = row_stride;
    return plan;
}

// Pure high-water allocation policy shared by production and host tests.
// zero_initialize is false only when every element that may be read is known
// to be overwritten first; unused page padding need not be cleared.
static inline ggml_gemmini_flash_workspace_plan
ggml_gemmini_flash_make_workspace_plan(
        size_t current_capacity,
        size_t current_alignment,
        size_t required_count,
        size_t required_alignment,
        bool all_readable_elements_overwritten) {
    ggml_gemmini_flash_workspace_plan plan;
    using namespace ggml_gemmini_flash_runtime_detail;
    plan.active_count = required_count;
    plan.retained_capacity = current_capacity;
    plan.required_alignment = required_alignment;

    if (!power_of_two(required_alignment) ||
            (current_capacity != 0 && !power_of_two(current_alignment))) {
        return plan;
    }
    plan.valid = true;
    if (required_count == 0) {
        return plan;
    }

    plan.reuse = current_capacity >= required_count &&
        current_alignment >= required_alignment;
    plan.allocate = !plan.reuse;
    if (plan.allocate) {
        plan.retained_capacity = required_count;
    }
    plan.zero_initialize = !all_readable_elements_overwritten;
    return plan;
}

#endif
