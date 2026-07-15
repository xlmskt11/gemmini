#include "include/gemmini_params.h"
#include "include/gemmini_page_packed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("page-packed layout check failed");
    }
}

static elem_t * a_element(elem_t * base, size_t row, size_t col, size_t stride) {
    const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
    elem_t * block = gemmini_page_packed_a_block_addr_mut(
        base, row / DIM, col / DIM, stride);
    return block + (row % DIM) * page_k_blocks * DIM + col % DIM;
}

static elem_t * b_element(elem_t * base, size_t row, size_t col, size_t stride) {
    elem_t * block = gemmini_page_packed_b_block_addr_mut(
        base, row / DIM, col / DIM, stride);
    return block + (row % DIM) * gemmini_page_packed_b_j_blocks_per_page() * DIM + col % DIM;
}

static acc_t * acc_element(acc_t * base, size_t row, size_t col, size_t stride) {
    const size_t page_j_blocks =
        gemmini_page_packed_acc_j_blocks_per_page(sizeof(acc_t));
    acc_t * block = static_cast<acc_t *>(gemmini_page_packed_acc_block_addr_mut(
        base, row / DIM, col / DIM, stride, sizeof(acc_t)));
    return block + (row % DIM) * page_j_blocks * DIM + col % DIM;
}

template <typename T, typename ElementFn>
static void check_unique_round_trip(
        size_t rows,
        size_t cols,
        size_t bytes,
        ElementFn element) {
    std::vector<T> storage(bytes / sizeof(T), 0);
    std::vector<uint8_t> occupied(bytes / sizeof(T), 0);

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            T * ptr = element(storage.data(), row, col, cols);
            const size_t offset = static_cast<size_t>(ptr - storage.data());
            require(offset < storage.size());
            require(occupied[offset] == 0);
            occupied[offset] = 1;
            *ptr = static_cast<T>((row * 17 + col * 5) % 113);
        }
    }

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const T expected = static_cast<T>((row * 17 + col * 5) % 113);
            require(*element(storage.data(), row, col, cols) == expected);
        }
    }
}

int main() {
    const size_t rows = DIM + 3;
    const size_t a_cols = MAX_BLOCK_LEN * DIM + 5;
    const size_t b_cols = gemmini_page_packed_input_blocks_per_page() * DIM + 7;
    const size_t acc_cols = DIM + 11;

    const size_t a_pages = gemmini_page_packed_a_page_count(rows, a_cols);
    const size_t b_pages = gemmini_page_packed_b_page_count(a_cols, b_cols);
    const size_t acc_pages = gemmini_page_packed_acc_page_count(rows, acc_cols, sizeof(acc_t));

    require(a_pages ==
        gemmini_ceil_div_size(gemmini_ceil_div_size(rows, DIM),
            gemmini_page_packed_input_blocks_per_page() /
                std::min(static_cast<size_t>(MAX_BLOCK_LEN), gemmini_page_packed_input_blocks_per_page())) *
        gemmini_ceil_div_size(gemmini_ceil_div_size(a_cols, DIM),
            std::min(static_cast<size_t>(MAX_BLOCK_LEN), gemmini_page_packed_input_blocks_per_page())));
    require(b_pages ==
        gemmini_ceil_div_size(a_cols, DIM) *
        gemmini_ceil_div_size(
            gemmini_ceil_div_size(b_cols, DIM), gemmini_page_packed_input_blocks_per_page()));
    require(a_pages > 0 && b_pages > 0 && acc_pages > 0);

    check_unique_round_trip<elem_t>(
        rows, a_cols, a_pages * GEMMINI_PAGE_PACKED_PAGE_BYTES, a_element);
    check_unique_round_trip<elem_t>(
        a_cols, b_cols, b_pages * GEMMINI_PAGE_PACKED_PAGE_BYTES, b_element);
    check_unique_round_trip<acc_t>(
        rows, acc_cols, acc_pages * GEMMINI_PAGE_PACKED_PAGE_BYTES, acc_element);

    const size_t encoded = GEMMINI_PAGE_PACKED_STRIDE(a_cols);
    require(gemmini_page_packed_stride_is_packed(encoded));
    require(gemmini_page_packed_stride_payload(encoded) == a_cols);
    require(gemmini_page_packed_a_dma_stride_bytes(a_cols) == a_cols * sizeof(elem_t));
    require(gemmini_page_packed_b_dma_stride_bytes(b_cols) == b_cols * sizeof(elem_t));
    require(gemmini_page_packed_acc_dma_stride_bytes(acc_cols, sizeof(acc_t)) ==
           acc_cols * sizeof(acc_t));
    require(gemmini_page_packed_a_dma_stride_bytes(encoded) % MAX_BYTES == 0);

    std::cout << "GEMMINI-PAGE-PACKED-LAYOUT-TEST-PASS"
              << ",dim=" << DIM
              << ",a_pages=" << a_pages
              << ",b_pages=" << b_pages
              << ",acc_pages=" << acc_pages << "\n";
    return 0;
}
