#ifndef GGML_GEMMINI_RUNTIME_OPT_H
#define GGML_GEMMINI_RUNTIME_OPT_H

#include "ggml-gemmini-bf16.h"

#include <cstddef>
#include <cstdint>
#include <limits>

static inline void ggml_gemmini_encode_f32_row_to_bf16(
        const float * source,
        uint16_t * destination,
        size_t elements) {
    for (size_t i = 0; i < elements; ++i) {
        destination[i] = ggml_gemmini_float_to_bf16(source[i]);
    }
}

struct ggml_gemmini_direct_a_plan {
    bool direct = false;
    uintptr_t address = 0;
    size_t span_bytes = 0;
};

namespace ggml_gemmini_runtime_opt_detail {

static inline bool checked_mul(size_t lhs, size_t rhs, size_t * result) {
    if (result == nullptr ||
            (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static inline bool valid_address_span(uintptr_t address, size_t bytes) {
    return address != 0 && bytes != 0 &&
        bytes - 1 <= std::numeric_limits<uintptr_t>::max() - address;
}

static inline bool address_spans_overlap(
        uintptr_t lhs,
        size_t lhs_bytes,
        uintptr_t rhs,
        size_t rhs_bytes) {
    // Callers validate both inclusive end addresses before reaching here.
    const uintptr_t lhs_last = lhs + lhs_bytes - 1;
    const uintptr_t rhs_last = rhs + rhs_bytes - 1;
    return lhs <= rhs_last && rhs <= lhs_last;
}

} // namespace ggml_gemmini_runtime_opt_detail

// Select a ggml BF16 activation slice as Gemmini's ordinary row-major A
// input.  Both tensors must have contiguous ggml storage; their full spans
// then prove that the selected matrix is contained in its source tensor and
// that no destination write can alias any of its bytes.
static inline ggml_gemmini_direct_a_plan
ggml_gemmini_make_direct_bf16_a_plan(
        bool enabled,
        bool page_packed_a,
        bool bf16_source,
        bool source_contiguous,
        bool destination_contiguous,
        uintptr_t source_base,
        size_t source_span_bytes,
        uintptr_t slice_address,
        size_t row_stride_bytes,
        size_t rows,
        size_t cols,
        uintptr_t destination_base,
        size_t destination_span_bytes) {
    ggml_gemmini_direct_a_plan plan;
    constexpr size_t bf16_bytes = sizeof(uint16_t);
    using namespace ggml_gemmini_runtime_opt_detail;

    if (!enabled || page_packed_a || !bf16_source ||
            !source_contiguous || !destination_contiguous ||
            rows == 0 || cols == 0 ||
            source_base % alignof(uint16_t) != 0 ||
            slice_address % alignof(uint16_t) != 0) {
        return plan;
    }

    size_t dense_row_bytes = 0;
    size_t matrix_span_bytes = 0;
    if (!checked_mul(cols, bf16_bytes, &dense_row_bytes) ||
            row_stride_bytes != dense_row_bytes ||
            !checked_mul(rows, dense_row_bytes, &matrix_span_bytes) ||
            !valid_address_span(source_base, source_span_bytes) ||
            !valid_address_span(slice_address, matrix_span_bytes) ||
            !valid_address_span(destination_base, destination_span_bytes) ||
            slice_address < source_base) {
        return plan;
    }

    const size_t slice_offset = static_cast<size_t>(slice_address - source_base);
    if (slice_offset > source_span_bytes ||
            matrix_span_bytes > source_span_bytes - slice_offset ||
            address_spans_overlap(
                slice_address, matrix_span_bytes,
                destination_base, destination_span_bytes)) {
        return plan;
    }

    plan.direct = true;
    plan.address = slice_address;
    plan.span_bytes = matrix_span_bytes;
    return plan;
}

struct ggml_gemmini_activation_cache_state {
    int family = 0;
    unsigned members = 0;
    bool valid = false;
    bool epoch_active = false;
};

static inline void ggml_gemmini_activation_cache_invalidate(
        ggml_gemmini_activation_cache_state & state) {
    state.family = 0;
    state.members = 0;
    state.valid = false;
}

static inline void ggml_gemmini_activation_cache_begin_epoch(
        ggml_gemmini_activation_cache_state & state) {
    state = {};
    state.epoch_active = true;
}

static inline void ggml_gemmini_activation_cache_end_epoch(
        ggml_gemmini_activation_cache_state & state) {
    state = {};
}

static inline bool ggml_gemmini_activation_cache_can_hit(
        const ggml_gemmini_activation_cache_state & state,
        int family,
        unsigned member,
        bool key_matches) {
    return state.epoch_active && state.valid && key_matches && state.family == family &&
        member != 0 && (state.members & member) == 0;
}

static inline void ggml_gemmini_activation_cache_record_hit(
        ggml_gemmini_activation_cache_state & state,
        unsigned member) {
    state.members |= member;
}

static inline void ggml_gemmini_activation_cache_record_miss(
        ggml_gemmini_activation_cache_state & state,
        int family,
        unsigned member) {
    // Q and gate are member zero in their respective families and are the
    // only safe seeds.  A partial graph starting at K/V/up must never retain
    // an input which a later graph could mistake for its own activation.
    const bool seeds_cache = state.epoch_active && family != 0 && member == (1u << 0);
    state.family = seeds_cache ? family : 0;
    state.members = seeds_cache ? member : 0;
    state.valid = seeds_cache;
}

static inline void ggml_gemmini_activation_cache_record_direct_input(
        ggml_gemmini_activation_cache_state & state) {
    // A direct source does not populate encoded_a. Preserve the enclosing
    // evaluation epoch, but prevent sibling projections from treating stale
    // encoded storage as a cache hit.
    ggml_gemmini_activation_cache_invalidate(state);
}

static inline bool ggml_gemmini_direct_c_layout_eligible(
        bool enabled,
        bool page_packed_c,
        bool fp32_destination,
        uintptr_t destination,
        size_t destination_alignment,
        size_t row_stride_bytes,
        size_t cols_out,
        size_t accumulator_bytes) {
    if (!enabled || page_packed_c || !fp32_destination || destination == 0 ||
            destination_alignment == 0 || destination % destination_alignment != 0 ||
            accumulator_bytes == 0 ||
            cols_out > std::numeric_limits<size_t>::max() / accumulator_bytes) {
        return false;
    }
    return row_stride_bytes == cols_out * accumulator_bytes;
}

#endif
