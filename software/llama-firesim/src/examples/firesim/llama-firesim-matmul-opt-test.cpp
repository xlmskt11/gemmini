#include "ggml-gemmini-runtime-opt.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

static bool test_direct_c_layout() {
    constexpr uintptr_t aligned = 0x1000;
    constexpr size_t cols = 8192;
    constexpr size_t row_bytes = cols * sizeof(float);

    return ggml_gemmini_direct_c_layout_eligible(
               true, false, true, aligned, alignof(float), row_bytes, cols, sizeof(float)) &&
        !ggml_gemmini_direct_c_layout_eligible(
               false, false, true, aligned, alignof(float), row_bytes, cols, sizeof(float)) &&
        !ggml_gemmini_direct_c_layout_eligible(
               true, true, true, aligned, alignof(float), row_bytes, cols, sizeof(float)) &&
        !ggml_gemmini_direct_c_layout_eligible(
               true, false, false, aligned, alignof(float), row_bytes, cols, sizeof(float)) &&
        !ggml_gemmini_direct_c_layout_eligible(
               true, false, true, aligned + 1, alignof(float), row_bytes, cols, sizeof(float)) &&
        !ggml_gemmini_direct_c_layout_eligible(
               true, false, true, aligned, alignof(float), row_bytes + sizeof(float), cols, sizeof(float)) &&
        !ggml_gemmini_direct_c_layout_eligible(
               true, false, true, aligned, alignof(float), 0,
               std::numeric_limits<size_t>::max(), sizeof(float));
}

static bool test_direct_bf16_a_layout() {
    constexpr uintptr_t source_base = 0x1000;
    constexpr size_t source_span = 0x400;
    constexpr uintptr_t slice = source_base + 0x100;
    constexpr size_t rows = 4;
    constexpr size_t cols = 8;
    constexpr size_t row_bytes = cols * sizeof(uint16_t);
    constexpr uintptr_t destination_base = 0x2000;
    constexpr size_t destination_span = 0x400;

    const ggml_gemmini_direct_a_plan direct =
        ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, true,
            source_base, source_span, slice, row_bytes, rows, cols,
            destination_base, destination_span);
    if (!direct.direct || direct.address != slice ||
            direct.span_bytes != rows * row_bytes) {
        return false;
    }

    // A padded row cannot use Gemmini's fixed ordinary-A stride.
    if (ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, true,
            source_base, source_span, slice, row_bytes + sizeof(uint16_t), rows, cols,
            destination_base, destination_span).direct) {
        return false;
    }

    // Packed A and non-BF16 sources both require the staging encoder.
    if (ggml_gemmini_make_direct_bf16_a_plan(
            true, true, true,
            true, true,
            source_base, source_span, slice, row_bytes, rows, cols,
            destination_base, destination_span).direct ||
        ggml_gemmini_make_direct_bf16_a_plan(
            true, false, false,
            true, true,
            source_base, source_span, slice, row_bytes, rows, cols,
            destination_base, destination_span).direct) {
        return false;
    }

    // Outer-strided tensor views keep the staging path even when the
    // selected 2D slice itself happens to look dense.
    if (ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            false, true,
            source_base, source_span, slice, row_bytes, rows, cols,
            destination_base, destination_span).direct ||
        ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, false,
            source_base, source_span, slice, row_bytes, rows, cols,
            destination_base, destination_span).direct) {
        return false;
    }

    // DMA inputs must be element-aligned and every inclusive address span
    // must be representable without uintptr_t wraparound.
    if (ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, true,
            source_base + 1, source_span, slice + 1, row_bytes, rows, cols,
            destination_base, destination_span).direct ||
        ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, true,
            std::numeric_limits<uintptr_t>::max() - 7, 16,
            std::numeric_limits<uintptr_t>::max() - 7,
            row_bytes, 1, cols,
            destination_base, destination_span).direct) {
        return false;
    }

    // Reject both a slice which extends beyond the declared source span and
    // any source/destination overlap, even if the current output row itself
    // would happen not to touch the input.
    if (ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, true,
            source_base, source_span, source_base + source_span - row_bytes,
            row_bytes, rows, cols,
            destination_base, destination_span).direct ||
        ggml_gemmini_make_direct_bf16_a_plan(
            true, false, true,
            true, true,
            source_base, source_span, slice, row_bytes, rows, cols,
            slice + row_bytes, rows * row_bytes).direct) {
        return false;
    }

    return true;
}

static bool test_activation_cache_sequence() {
    ggml_gemmini_activation_cache_state state;

    // Generic backend callers have no explicit evaluation epoch and therefore
    // never retain an activation across independent graph_compute calls.
    ggml_gemmini_activation_cache_record_miss(state, 1, 1u << 0);
    if (state.valid || ggml_gemmini_activation_cache_can_hit(state, 1, 1u << 1, true)) {
        return false;
    }

    ggml_gemmini_activation_cache_begin_epoch(state);

    // Q seeds the cache; V and K may consume the same activation once each.
    ggml_gemmini_activation_cache_record_miss(state, 1, 1u << 0);
    if (!ggml_gemmini_activation_cache_can_hit(state, 1, 1u << 2, true)) {
        return false;
    }
    ggml_gemmini_activation_cache_record_hit(state, 1u << 2);
    if (!ggml_gemmini_activation_cache_can_hit(state, 1, 1u << 1, true)) {
        return false;
    }
    ggml_gemmini_activation_cache_record_hit(state, 1u << 1);
    if (ggml_gemmini_activation_cache_can_hit(state, 1, 1u << 1, true) ||
            ggml_gemmini_activation_cache_can_hit(state, 1, 1u << 0, true) ||
            ggml_gemmini_activation_cache_can_hit(state, 1, 1u << 3, false)) {
        return false;
    }

    // Gate/up is a separate family. A partial graph beginning at up cannot
    // seed a cache entry which might escape into the next evaluation.
    ggml_gemmini_activation_cache_record_miss(state, 2, 1u << 1);
    if (state.valid) {
        return false;
    }
    ggml_gemmini_activation_cache_record_miss(state, 2, 1u << 0);
    if (!ggml_gemmini_activation_cache_can_hit(state, 2, 1u << 1, true)) {
        return false;
    }

    // A zero-copy input never updates encoded_a. It must invalidate the
    // staged cache while keeping the evaluation epoch active.
    ggml_gemmini_activation_cache_record_direct_input(state);
    if (!state.epoch_active || state.valid || state.family != 0 ||
            state.members != 0 ||
            ggml_gemmini_activation_cache_can_hit(state, 2, 1u << 1, true)) {
        return false;
    }

    ggml_gemmini_activation_cache_end_epoch(state);
    return !state.epoch_active && !state.valid && state.family == 0 && state.members == 0;
}

static bool test_direct_activation_encoding() {
    const float source[] = {
        0.0f,
        1.0f,
        -2.5f,
        3.14159265f,
        std::numeric_limits<float>::infinity(),
    };
    uint16_t encoded[sizeof(source) / sizeof(source[0])] = {};
    ggml_gemmini_encode_f32_row_to_bf16(
        source, encoded, sizeof(source) / sizeof(source[0]));
    for (size_t i = 0; i < sizeof(source) / sizeof(source[0]); ++i) {
        if (encoded[i] != ggml_gemmini_float_to_bf16(source[i])) {
            return false;
        }
    }
    return true;
}

// Keep an independent oracle for the generated multi-profile packed-A
// geometry.  The production writer uses the same page/block equations from
// gemmini_page_packed_a_block_addr_mut(), while this test deliberately does
// not call that writer: an addressing regression must not make both the
// implementation and its expected result wrong in the same way.
constexpr size_t packed_a_dim = 8;
constexpr size_t packed_a_element_bytes = sizeof(uint16_t);
constexpr size_t packed_a_page_bytes = 4096;
constexpr size_t packed_a_max_bytes = 64;
constexpr size_t packed_a_blocks_per_page =
    packed_a_page_bytes /
    (packed_a_dim * packed_a_dim * packed_a_element_bytes);
constexpr size_t packed_a_k_blocks_per_page =
    packed_a_max_bytes / (packed_a_dim * packed_a_element_bytes);
constexpr size_t packed_a_i_blocks_per_page =
    packed_a_blocks_per_page / packed_a_k_blocks_per_page;

static size_t ceil_div(size_t numerator, size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

static size_t packed_a_storage_elements(size_t rows, size_t cols) {
    const size_t i_pages = ceil_div(
        ceil_div(rows, packed_a_dim), packed_a_i_blocks_per_page);
    const size_t k_pages = ceil_div(
        ceil_div(cols, packed_a_dim), packed_a_k_blocks_per_page);
    return i_pages * k_pages * packed_a_page_bytes / packed_a_element_bytes;
}

static size_t packed_a_element_index(
        size_t row,
        size_t col,
        size_t cols) {
    const size_t i_block = row / packed_a_dim;
    const size_t k_block = col / packed_a_dim;
    const size_t k_blocks = ceil_div(cols, packed_a_dim);
    const size_t k_pages = ceil_div(k_blocks, packed_a_k_blocks_per_page);
    const size_t i_page = i_block / packed_a_i_blocks_per_page;
    const size_t k_page = k_block / packed_a_k_blocks_per_page;
    const size_t i_in_page = i_block % packed_a_i_blocks_per_page;
    const size_t k_in_page = k_block % packed_a_k_blocks_per_page;
    const size_t page_index = i_page * k_pages + k_page;
    const size_t page_cols = packed_a_k_blocks_per_page * packed_a_dim;
    const size_t page_elements = packed_a_page_bytes / packed_a_element_bytes;
    return page_index * page_elements +
        i_in_page * packed_a_dim * page_cols +
        k_in_page * packed_a_dim +
        (row % packed_a_dim) * page_cols +
        col % packed_a_dim;
}

static void encode_packed_a_direct(
        const std::vector<float> & source,
        size_t rows,
        size_t cols,
        std::vector<uint16_t> & workspace) {
    const size_t required = packed_a_storage_elements(rows, cols);
    if (workspace.size() < required) {
        workspace.resize(required, UINT16_C(0xa55a));
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            workspace[packed_a_element_index(row, col, cols)] =
                ggml_gemmini_float_to_bf16(source[row * cols + col]);
        }
    }
}

static float packed_a_input(size_t row, size_t col, unsigned generation) {
    const int value = static_cast<int>(
        (row * 37 + col * 19 + generation * 23) % 251) - 125;
    return static_cast<float>(value) / 16.0f;
}

static std::vector<float> make_packed_a_input(
        size_t rows,
        size_t cols,
        unsigned generation) {
    std::vector<float> source(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            source[row * cols + col] = packed_a_input(row, col, generation);
        }
    }
    return source;
}

static bool packed_a_valid_entries_match(
        const std::vector<float> & source,
        size_t rows,
        size_t cols,
        const std::vector<uint16_t> & workspace) {
    const size_t required = packed_a_storage_elements(rows, cols);
    std::vector<uint8_t> occupied(required, 0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const size_t index = packed_a_element_index(row, col, cols);
            if (index >= required || occupied[index] != 0 ||
                    workspace[index] != ggml_gemmini_float_to_bf16(
                        source[row * cols + col])) {
                return false;
            }
            occupied[index] = 1;
        }
    }
    return true;
}

static bool test_packed_a_direct_workspace_reuse() {
    static_assert(packed_a_blocks_per_page == 32, "unexpected packed-A page geometry");
    static_assert(packed_a_k_blocks_per_page == 4, "unexpected packed-A row geometry");
    static_assert(packed_a_i_blocks_per_page == 8, "unexpected packed-A slab geometry");

    // Cross both the I-page and K-page boundaries and end in partial blocks.
    constexpr size_t large_rows = packed_a_dim * packed_a_i_blocks_per_page + 3;
    constexpr size_t large_cols = packed_a_dim * packed_a_k_blocks_per_page + 5;
    const std::vector<float> large = make_packed_a_input(large_rows, large_cols, 1);
    std::vector<uint16_t> workspace;
    encode_packed_a_direct(large, large_rows, large_cols, workspace);
    if (!packed_a_valid_entries_match(
            large, large_rows, large_cols, workspace)) {
        return false;
    }

    // Direct conversion must not clear every packed page.  At least one
    // padding entry in the active large layout remains the allocation poison.
    const size_t large_required = packed_a_storage_elements(large_rows, large_cols);
    std::vector<uint8_t> large_occupied(large_required, 0);
    for (size_t row = 0; row < large_rows; ++row) {
        for (size_t col = 0; col < large_cols; ++col) {
            large_occupied[packed_a_element_index(row, col, large_cols)] = 1;
        }
    }
    bool found_untouched_padding = false;
    for (size_t index = 0; index < large_required; ++index) {
        if (large_occupied[index] == 0 && workspace[index] == UINT16_C(0xa55a)) {
            found_untouched_padding = true;
            break;
        }
    }
    if (!found_untouched_padding) {
        return false;
    }

    // Reuse the same high-water workspace for a smaller matrix whose row and
    // column tails are both partial.  The logical-to-physical mapping changes
    // because K now occupies fewer pages, so every valid entry must be
    // overwritten with the new generation rather than retaining large-shape
    // data.
    constexpr size_t small_rows = packed_a_dim + 3;
    constexpr size_t small_cols = packed_a_dim + 5;
    const std::vector<float> small = make_packed_a_input(small_rows, small_cols, 2);
    const uint16_t * const retained_address = workspace.data();
    const size_t retained_size = workspace.size();
    const uint16_t retained_tail = workspace.back();
    encode_packed_a_direct(small, small_rows, small_cols, workspace);
    if (workspace.data() != retained_address || workspace.size() != retained_size ||
            workspace.back() != retained_tail ||
            !packed_a_valid_entries_match(
                small, small_rows, small_cols, workspace)) {
        return false;
    }

    // A packed activation cache hit must retain the already encoded storage
    // byte-for-byte and may be consumed only once by each sibling projection.
    ggml_gemmini_activation_cache_state cache;
    ggml_gemmini_activation_cache_begin_epoch(cache);
    ggml_gemmini_activation_cache_record_miss(cache, 1, 1u << 0);
    const std::vector<uint16_t> cached = workspace;
    if (!ggml_gemmini_activation_cache_can_hit(cache, 1, 1u << 2, true)) {
        return false;
    }
    ggml_gemmini_activation_cache_record_hit(cache, 1u << 2);
    if (workspace != cached ||
            !ggml_gemmini_activation_cache_can_hit(cache, 1, 1u << 1, true)) {
        return false;
    }
    ggml_gemmini_activation_cache_record_hit(cache, 1u << 1);
    return workspace == cached &&
        !ggml_gemmini_activation_cache_can_hit(cache, 1, 1u << 2, true);
}

} // namespace

int main() {
    if (!test_direct_c_layout()) {
        std::cerr << "GEMMINI-MATMUL-OPT-TEST-FAIL,direct-c-layout\n";
        return 1;
    }
    if (!test_direct_bf16_a_layout()) {
        std::cerr << "GEMMINI-MATMUL-OPT-TEST-FAIL,direct-bf16-a-layout\n";
        return 1;
    }
    if (!test_activation_cache_sequence()) {
        std::cerr << "GEMMINI-MATMUL-OPT-TEST-FAIL,activation-cache\n";
        return 1;
    }
    if (!test_direct_activation_encoding()) {
        std::cerr << "GEMMINI-MATMUL-OPT-TEST-FAIL,activation-encoding\n";
        return 1;
    }
    if (!test_packed_a_direct_workspace_reuse()) {
        std::cerr << "GEMMINI-MATMUL-OPT-TEST-FAIL,packed-a-direct-reuse\n";
        return 1;
    }
    std::cout << "GEMMINI-MATMUL-OPT-TEST-PASS\n";
    return 0;
}
