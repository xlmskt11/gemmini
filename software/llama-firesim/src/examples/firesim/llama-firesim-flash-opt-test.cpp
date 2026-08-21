#include "ggml-gemmini-flash-runtime-opt.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

static ggml_gemmini_flash_tensor_view make_contiguous_view(
        uintptr_t base,
        size_t element_bytes,
        size_t ne0,
        size_t ne1,
        size_t ne2,
        size_t ne3) {
    ggml_gemmini_flash_tensor_view view;
    view.base = base;
    view.element_bytes = element_bytes;
    view.ne[0] = ne0;
    view.ne[1] = ne1;
    view.ne[2] = ne2;
    view.ne[3] = ne3;
    view.nb[0] = element_bytes;
    view.nb[1] = ne0 * view.nb[0];
    view.nb[2] = ne1 * view.nb[1];
    view.nb[3] = ne2 * view.nb[2];
    return view;
}

static bool test_direct_output_layout_base_and_stride() {
    constexpr uintptr_t base = static_cast<uintptr_t>(0x100000);
    constexpr size_t value_dim = 64;
    constexpr size_t heads = 4;
    constexpr size_t queries = 112;
    constexpr size_t batches = 2;
    auto view = make_contiguous_view(
        base, sizeof(float), value_dim, heads, queries, batches);

    auto plan = ggml_gemmini_flash_make_direct_output_slice(
        true, true, true, view, 3, 1);
    const uintptr_t expected = base + 3 * view.nb[1] + view.nb[3];
    if (!plan.direct || plan.address != expected ||
            plan.row_stride_elements != value_dim * heads) {
        return false;
    }

    // Padding between heads and query rows is supported when the complete
    // [DV,H,N,B] layout remains nested and non-overlapping.
    view.nb[1] = value_dim * sizeof(float) + 16;
    view.nb[2] = heads * view.nb[1] + 32;
    view.nb[3] = queries * view.nb[2] + 64;
    plan = ggml_gemmini_flash_make_direct_output_slice(
        true, true, true, view, 2, 1);
    if (!plan.direct ||
            plan.address != base + 2 * view.nb[1] + view.nb[3] ||
            plan.row_stride_elements != view.nb[2] / sizeof(float)) {
        return false;
    }

    // A single query does not consume nb[2]; use a legal contiguous hardware
    // stride even if the unused ggml stride contains a sentinel value.
    auto one_query = make_contiguous_view(
        base, sizeof(float), value_dim, heads, 1, batches);
    one_query.nb[2] = 1;
    one_query.nb[3] = heads * one_query.nb[1] + 32;
    plan = ggml_gemmini_flash_make_direct_output_slice(
        true, true, true, one_query, 1, 1);
    return plan.direct && plan.row_stride_elements == value_dim;
}

static bool test_output_fallback_policy() {
    constexpr uintptr_t base = static_cast<uintptr_t>(0x200000);
    auto good = make_contiguous_view(base, sizeof(float), 64, 4, 112, 2);
    const auto direct = [&](bool enabled, bool fp32, bool disjoint,
                            const ggml_gemmini_flash_tensor_view & view) {
        return ggml_gemmini_flash_make_direct_output_slice(
            enabled, fp32, disjoint, view, 0, 0).direct;
    };
    if (!direct(true, true, true, good) ||
            direct(false, true, true, good) ||
            direct(true, false, true, good) ||
            direct(true, true, false, good)) {
        return false;
    }

    auto bad = good;
    bad.base += 1;
    if (direct(true, true, true, bad)) {
        return false;
    }
    bad = good;
    bad.nb[1] = 63 * sizeof(float); // heads overlap
    if (direct(true, true, true, bad)) {
        return false;
    }
    bad = good;
    bad.nb[2] -= sizeof(float); // query rows overlap the last head
    if (direct(true, true, true, bad)) {
        return false;
    }
    bad = good;
    bad.nb[3] -= sizeof(float); // batches overlap the last query
    if (direct(true, true, true, bad)) {
        return false;
    }
    bad = good;
    bad.nb[2] += 1; // hardware stride is not in F32 elements
    bad.nb[3] = bad.ne[2] * bad.nb[2];
    if (direct(true, true, true, bad)) {
        return false;
    }

    // Bit 31 is the packed-layout flag, so a linear stride must fit in the
    // low 31-bit payload even though host element offsets are 32-bit.
    bad = make_contiguous_view(base, sizeof(float), 64, 1, 2, 1);
    bad.nb[2] = (static_cast<size_t>(std::numeric_limits<uint32_t>::max() >> 1) + 1) *
        sizeof(float);
    if (direct(true, true, true, bad)) {
        return false;
    }
    bad.nb[2] = static_cast<size_t>(std::numeric_limits<uint32_t>::max() >> 1) *
        sizeof(float);
    return direct(true, true, true, bad);
}

static bool test_direct_bf16_input_and_staging_policy() {
    constexpr uintptr_t base = static_cast<uintptr_t>(0x300000);
    constexpr size_t width = 64;
    constexpr size_t rows = 2048;
    constexpr size_t heads = 8;
    constexpr size_t batches = 2;
    auto view = make_contiguous_view(
        base, sizeof(uint16_t), width, rows, heads, batches);

    auto plan = ggml_gemmini_flash_make_direct_bf16_input_slice(
        true, false, true, true, true, view, 3, 1);
    if (!plan.direct ||
            plan.address != base + 3 * view.nb[2] + view.nb[3] ||
            plan.row_stride_elements != width) {
        return false;
    }
    const auto contiguous = view;

    // The current fusion config encodes the logical width for linear K/V.
    // A padded physical row must therefore retain the staging path.
    view.nb[1] += 16;
    view.nb[2] = rows * view.nb[1];
    view.nb[3] = heads * view.nb[2];
    plan = ggml_gemmini_flash_make_direct_bf16_input_slice(
        true, false, true, true, true, view, 7, 1);
    if (plan.direct) {
        return false;
    }

    const auto direct = [&](bool enabled, bool packed, bool bf16,
                            bool identity, bool disjoint,
                            const ggml_gemmini_flash_tensor_view & candidate) {
        return ggml_gemmini_flash_make_direct_bf16_input_slice(
            enabled, packed, bf16, identity, disjoint,
            candidate, 0, 0).direct;
    };
    if (direct(false, false, true, true, true, contiguous) ||
            direct(true, true, true, true, true, contiguous) ||
            direct(true, false, false, true, true, contiguous) ||
            direct(true, false, true, false, true, contiguous) ||
            direct(true, false, true, true, false, contiguous)) {
        return false;
    }

    auto bad = contiguous;
    bad.base += 1;
    if (direct(true, false, true, true, true, bad)) {
        return false;
    }
    bad = contiguous;
    bad.nb[1] = width * sizeof(uint16_t) - sizeof(uint16_t);
    if (direct(true, false, true, true, true, bad)) {
        return false;
    }
    bad = contiguous;
    bad.nb[1] += 1;
    if (direct(true, false, true, true, true, bad)) {
        return false;
    }
    bad = make_contiguous_view(
        base, sizeof(uint16_t),
        static_cast<size_t>(std::numeric_limits<uint32_t>::max() >> 1) + 1,
        1, 1, 1);
    if (direct(true, false, true, true, true, bad)) {
        return false;
    }
    bad = make_contiguous_view(
        base, sizeof(uint16_t),
        static_cast<size_t>(std::numeric_limits<uint32_t>::max() >> 1),
        1, 1, 1);
    if (!direct(true, false, true, true, true, bad)) {
        return false;
    }
    return !ggml_gemmini_flash_make_direct_bf16_input_slice(
        true, false, true, true, true, contiguous, heads, 0).direct;
}

static bool test_workspace_high_water_and_no_clear() {
    constexpr size_t page_alignment = 4096;
    auto plan = ggml_gemmini_flash_make_workspace_plan(
        0, 0, 8192, page_alignment, true);
    if (!plan.valid || plan.reuse || !plan.allocate || plan.zero_initialize ||
            plan.retained_capacity != 8192) {
        return false;
    }

    std::vector<uint8_t> model(plan.retained_capacity, UINT8_C(0xa5));
    uint8_t * const retained_address = model.data();
    const std::vector<uint8_t> poison = model;

    // A smaller request reuses the allocation at its high-water capacity and
    // does not clear valid entries or unused page padding before producers
    // overwrite the entries the kernel can read.
    plan = ggml_gemmini_flash_make_workspace_plan(
        model.size(), page_alignment, 257, 64, true);
    if (!plan.valid || !plan.reuse || plan.allocate || plan.zero_initialize ||
            plan.retained_capacity != model.size() ||
            model.data() != retained_address || model != poison) {
        return false;
    }

    // If a future workspace has observable entries not covered by its writer,
    // the same policy explicitly restores zero initialization.
    plan = ggml_gemmini_flash_make_workspace_plan(
        model.size(), page_alignment, 257, 64, false);
    if (!plan.valid || !plan.reuse || !plan.zero_initialize) {
        return false;
    }

    plan = ggml_gemmini_flash_make_workspace_plan(
        model.size(), page_alignment, model.size() + 1, page_alignment, true);
    if (!plan.valid || plan.reuse || !plan.allocate || plan.zero_initialize ||
            plan.retained_capacity != model.size() + 1) {
        return false;
    }
    plan = ggml_gemmini_flash_make_workspace_plan(
        model.size(), 64, 128, page_alignment, true);
    if (!plan.valid || plan.reuse || !plan.allocate) {
        return false;
    }
    plan = ggml_gemmini_flash_make_workspace_plan(
        model.size(), page_alignment, 128, 96, true);
    if (plan.valid) {
        return false;
    }
    plan = ggml_gemmini_flash_make_workspace_plan(
        model.size(), page_alignment, 0, 64, true);
    return plan.valid && !plan.reuse && !plan.allocate &&
        !plan.zero_initialize && plan.retained_capacity == model.size();
}

} // namespace

int main() {
    if (!test_direct_output_layout_base_and_stride()) {
        std::cerr << "GEMMINI-FLASH-OPT-TEST-FAIL,direct-output-layout\n";
        return 1;
    }
    if (!test_output_fallback_policy()) {
        std::cerr << "GEMMINI-FLASH-OPT-TEST-FAIL,output-fallback\n";
        return 1;
    }
    if (!test_direct_bf16_input_and_staging_policy()) {
        std::cerr << "GEMMINI-FLASH-OPT-TEST-FAIL,input-direct-staging\n";
        return 1;
    }
    if (!test_workspace_high_water_and_no_clear()) {
        std::cerr << "GEMMINI-FLASH-OPT-TEST-FAIL,workspace-reuse-clear\n";
        return 1;
    }
    std::cout << "GEMMINI-FLASH-OPT-TEST-PASS\n";
    return 0;
}
