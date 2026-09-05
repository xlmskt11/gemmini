#include "ggml-gemmini-flash-runtime-opt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct htc_layout {
    size_t channels;
    size_t token_capacity;
    size_t heads;
    size_t streams;
    size_t element_bytes;

    size_t row_count() const {
        return streams * heads * token_capacity;
    }

    size_t element_count() const {
        return row_count() * channels;
    }

    size_t destination_row(size_t stream, size_t head, size_t token) const {
        require(stream < streams, "HTC stream index is out of range");
        require(head < heads, "HTC head index is out of range");
        require(token < token_capacity, "HTC token index is out of range");
        return (stream * heads + head) * token_capacity + token;
    }

    size_t element_offset(
            size_t stream,
            size_t head,
            size_t token,
            size_t channel) const {
        require(channel < channels, "HTC channel index is out of range");
        return destination_row(stream, head, token) * channels + channel;
    }

    ggml_gemmini_flash_tensor_view pre_permute_view(
            uintptr_t base,
            size_t n_kv) const {
        require(n_kv > 0 && n_kv <= token_capacity,
                "visible KV length is outside the HTC capacity");

        // The physical allocation is contiguous [C, Tcap, H, S].  Preserve
        // llama.cpp's logical get_k/get_v contract [C, H, n_kv, S] by
        // swapping only the head and token strides.  ggml_permute(0,2,1,3)
        // then exposes a dense per-head [C, n_kv] matrix to FlashAttention.
        ggml_gemmini_flash_tensor_view view;
        view.base = base;
        view.element_bytes = element_bytes;
        view.ne[0] = channels;
        view.ne[1] = heads;
        view.ne[2] = n_kv;
        view.ne[3] = streams;
        view.nb[0] = element_bytes;
        view.nb[1] = token_capacity * channels * element_bytes;
        view.nb[2] = channels * element_bytes;
        view.nb[3] = heads * token_capacity * channels * element_bytes;
        return view;
    }
};

static ggml_gemmini_flash_tensor_view permute_0_2_1_3(
        const ggml_gemmini_flash_tensor_view & source) {
    ggml_gemmini_flash_tensor_view result;
    result.base = source.base;
    result.element_bytes = source.element_bytes;

    constexpr std::array<size_t, 4> axes = { 0, 2, 1, 3 };
    for (size_t dimension = 0; dimension < axes.size(); ++dimension) {
        result.ne[dimension] = source.ne[axes[dimension]];
        result.nb[dimension] = source.nb[axes[dimension]];
    }
    return result;
}

static size_t view_byte_offset(
        const ggml_gemmini_flash_tensor_view & view,
        size_t i0,
        size_t i1,
        size_t i2,
        size_t i3) {
    require(i0 < view.ne[0] && i1 < view.ne[1] &&
            i2 < view.ne[2] && i3 < view.ne[3],
            "tensor view index is out of range");
    return i0 * view.nb[0] + i1 * view.nb[1] +
        i2 * view.nb[2] + i3 * view.nb[3];
}

static void require_view(
        const ggml_gemmini_flash_tensor_view & view,
        const std::array<size_t, 4> & expected_ne,
        const std::array<size_t, 4> & expected_nb,
        const std::string & name) {
    for (size_t dimension = 0; dimension < 4; ++dimension) {
        require(view.ne[dimension] == expected_ne[dimension],
                name + " extent mismatch at dimension " +
                    std::to_string(dimension));
        require(view.nb[dimension] == expected_nb[dimension],
                name + " stride mismatch at dimension " +
                    std::to_string(dimension));
    }
}

static void test_pre_post_permute_contract(const htc_layout & layout) {
    constexpr uintptr_t base = static_cast<uintptr_t>(0x100000);
    constexpr size_t n_kv = 6;

    const auto pre = layout.pre_permute_view(base, n_kv);
    require_view(
        pre,
        { layout.channels, layout.heads, n_kv, layout.streams },
        {
            layout.element_bytes,
            layout.token_capacity * layout.channels * layout.element_bytes,
            layout.channels * layout.element_bytes,
            layout.heads * layout.token_capacity * layout.channels *
                layout.element_bytes,
        },
        "pre-permute HTC view");

    const auto post = permute_0_2_1_3(pre);
    require_view(
        post,
        { layout.channels, n_kv, layout.heads, layout.streams },
        {
            layout.element_bytes,
            layout.channels * layout.element_bytes,
            layout.token_capacity * layout.channels * layout.element_bytes,
            layout.heads * layout.token_capacity * layout.channels *
                layout.element_bytes,
        },
        "post-permute HTC view");

    // n_kv is deliberately smaller than Tcap.  Head and stream strides must
    // continue to use the allocation capacity, not the visible graph length.
    require(post.nb[2] != n_kv * post.nb[1],
            "HTC head stride accidentally used n_kv instead of Tcap");

    std::vector<uint8_t> occupied(layout.element_count(), 0);
    for (size_t stream = 0; stream < layout.streams; ++stream) {
        for (size_t head = 0; head < layout.heads; ++head) {
            for (size_t token = 0; token < n_kv; ++token) {
                for (size_t channel = 0; channel < layout.channels; ++channel) {
                    const size_t physical = layout.element_offset(
                        stream, head, token, channel);
                    const size_t pre_offset = view_byte_offset(
                        pre, channel, head, token, stream);
                    const size_t post_offset = view_byte_offset(
                        post, channel, token, head, stream);

                    require(pre_offset == physical * layout.element_bytes,
                            "pre-permute address does not map to HTC storage");
                    require(post_offset == physical * layout.element_bytes,
                            "post-permute address does not map to HTC storage");
                    require(occupied[physical] == 0,
                            "two visible HTC elements alias each other");
                    occupied[physical] = 1;
                }
            }
        }
    }

    // The pre-permute view is not a Flash input matrix: its logical row axis
    // is heads and consequently has a capacity-sized stride.
    const auto rejected_pre = ggml_gemmini_flash_make_direct_bf16_input_slice(
        true, true, true, true, pre, 0, 0);
    require(!rejected_pre.direct,
            "pre-permute HTC view was incorrectly accepted as direct input");

    for (size_t stream = 0; stream < layout.streams; ++stream) {
        for (size_t head = 0; head < layout.heads; ++head) {
            const auto plan = ggml_gemmini_flash_make_direct_bf16_input_slice(
                true, true, true, true, post, head, stream);
            const uintptr_t expected_address = base +
                layout.element_offset(stream, head, 0, 0) *
                    layout.element_bytes;

            require(plan.direct,
                    "post-permute HTC view was not accepted as direct input");
            require(plan.address == expected_address,
                    "direct HTC head/stream base address is incorrect");
            require(plan.row_stride_elements == layout.channels,
                    "direct HTC token-row stride is not the head dimension");
        }
    }

    require(!ggml_gemmini_flash_make_direct_bf16_input_slice(
                true, true, true, true, post, layout.heads, 0).direct,
            "out-of-range HTC head was accepted");
    require(!ggml_gemmini_flash_make_direct_bf16_input_slice(
                true, true, true, true, post, 0, layout.streams).direct,
            "out-of-range HTC stream was accepted");
}

static uint16_t source_value(uint16_t seed, size_t source_index) {
    const size_t value = static_cast<size_t>(seed) + source_index + 1;
    require(value < UINT16_MAX, "test BF16 payload overflowed uint16_t");
    return static_cast<uint16_t>(value);
}

static void test_scatter_round_trip(
        const htc_layout & layout,
        const std::vector<std::vector<size_t>> & cells,
        uint16_t seed,
        const std::string & name) {
    constexpr uint16_t poison = UINT16_C(0xffff);

    require(layout.element_bytes == sizeof(uint16_t),
            "scatter test requires BF16-sized elements");
    require(cells.size() == layout.streams,
            "scatter cell map does not cover every stream");
    require(!cells.empty() && !cells.front().empty(),
            "scatter cell map is empty");

    const size_t tokens_per_stream = cells.front().size();
    for (const auto & stream_cells : cells) {
        require(stream_cells.size() == tokens_per_stream,
                "scatter streams have different token counts");
    }

    const size_t source_rows =
        layout.streams * tokens_per_stream * layout.heads;
    std::vector<uint16_t> source(source_rows * layout.channels);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = source_value(seed, index);
    }

    std::vector<uint16_t> cache(layout.element_count(), poison);
    std::vector<uint8_t> touched(layout.row_count(), 0);

    // This is the row order produced by flattening a projection result with
    // logical shape [C,H,T]: head varies inside token, and stream batches are
    // contiguous groups of tokens.  It models one ggml_set_rows scatter into
    // the flattened contiguous [C,Tcap,H,S] allocation.
    for (size_t stream = 0; stream < layout.streams; ++stream) {
        for (size_t token_index = 0;
                token_index < tokens_per_stream; ++token_index) {
            const size_t cell = cells[stream][token_index];
            require(cell < layout.token_capacity,
                    name + " scatter cell exceeds Tcap");

            for (size_t head = 0; head < layout.heads; ++head) {
                const size_t source_row =
                    (stream * tokens_per_stream + token_index) *
                        layout.heads + head;
                const size_t destination_row =
                    layout.destination_row(stream, head, cell);

                require(touched[destination_row] == 0,
                        name + " scatter destination row collision");
                touched[destination_row] = 1;

                std::memcpy(
                    cache.data() + destination_row * layout.channels,
                    source.data() + source_row * layout.channels,
                    layout.channels * sizeof(uint16_t));
            }
        }
    }

    const auto full_view = permute_0_2_1_3(
        layout.pre_permute_view(
            reinterpret_cast<uintptr_t>(cache.data()),
            layout.token_capacity));

    for (size_t stream = 0; stream < layout.streams; ++stream) {
        for (size_t token_index = 0;
                token_index < tokens_per_stream; ++token_index) {
            const size_t cell = cells[stream][token_index];
            for (size_t head = 0; head < layout.heads; ++head) {
                const size_t source_row =
                    (stream * tokens_per_stream + token_index) *
                        layout.heads + head;
                for (size_t channel = 0;
                        channel < layout.channels; ++channel) {
                    const size_t physical = layout.element_offset(
                        stream, head, cell, channel);
                    const size_t view_offset = view_byte_offset(
                        full_view, channel, cell, head, stream) /
                            layout.element_bytes;
                    const uint16_t expected = source[
                        source_row * layout.channels + channel];

                    require(view_offset == physical,
                            name + " Flash view address disagrees with scatter");
                    require(cache[physical] == expected,
                            name + " scatter/read round-trip mismatch");
                }
            }
        }
    }

    for (size_t row = 0; row < layout.row_count(); ++row) {
        if (touched[row]) {
            continue;
        }
        for (size_t channel = 0; channel < layout.channels; ++channel) {
            require(cache[row * layout.channels + channel] == poison,
                    name + " scatter modified an unselected cache cell");
        }
    }
}

} // namespace

int main() {
    try {
        // Non-power-of-two test dimensions make accidental interchange of C,
        // Tcap, H, or S visible in both addresses and strides.
        const htc_layout layout{
            /* channels       = */ 3,
            /* token_capacity = */ 7,
            /* heads          = */ 2,
            /* streams        = */ 2,
            /* element_bytes  = */ sizeof(uint16_t),
        };

        test_pre_post_permute_contract(layout);

        // Each stream uses a different, deliberately non-contiguous cell map.
        // The maximum selected cell is below Tcap so the allocation retains a
        // capacity-only tail that must not affect the visible mapping.
        const std::vector<std::vector<size_t>> cells = {
            { 4, 1, 3 },
            { 5, 2, 0 },
        };
        test_scatter_round_trip(layout, cells, UINT16_C(0x0100), "K");
        test_scatter_round_trip(layout, cells, UINT16_C(0x4000), "V");

        std::cout << "GEMMINI-KV-HTC-LAYOUT-TEST-PASS"
                  << ",c=" << layout.channels
                  << ",tcap=" << layout.token_capacity
                  << ",h=" << layout.heads
                  << ",s=" << layout.streams << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "GEMMINI-KV-HTC-LAYOUT-TEST-FAIL,"
                  << error.what() << "\n";
        return 1;
    }
}
