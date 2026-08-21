#pragma once

#include <cstdint>

// Page packing is useful only when the matrix dimension distributed across
// pages has more than one element.  Keep this policy independent of the
// generated Gemmini geometry so the host packer and the RISC-V backend make
// exactly the same admission decision.
struct ggml_gemmini_page_packing_config {
    bool a = false;
    bool b = false;
    bool c = false;
    bool d = false;

    constexpr std::uint8_t mask() const {
        return static_cast<std::uint8_t>((a ? 1u : 0u) |
                                         (b ? 2u : 0u) |
                                         (c ? 4u : 0u) |
                                         (d ? 8u : 0u));
    }
};

// FlashAttention performs two Gemmini matmuls with different logical
// operands:
//
//   Q * K^T: Q is A, K is the (pre-transpose) B source
//   P * V:   P is held in VSRAM and V is B
//
// The final FP32 result is emitted by the VPU H_STORE path in linear row-major
// order. It never uses the Gemmini C page layout. D is an internal VSRAM
// online accumulator and therefore also has no host page layout. Keep these
// decisions in one header so admission, execution, and host tests cannot
// drift apart.
struct ggml_gemmini_flash_page_packing_config {
    bool queries = false;
    bool keys = false;
    bool values = false;

    // FlashAttention has an independent Q/K/V request namespace. Do not map
    // these bits back onto generic Gemmini A/B/C/D: K and V can be selected
    // independently even though both are B operands in different matmuls.
    constexpr std::uint8_t mask() const {
        return static_cast<std::uint8_t>((queries ? 1u : 0u) |
                                         (keys ? 2u : 0u) |
                                         (values ? 4u : 0u));
    }
};

constexpr ggml_gemmini_page_packing_config ggml_gemmini_page_packing_from_mask(
        std::uint8_t mask) {
    return {
        (mask & 0x1u) != 0,
        (mask & 0x2u) != 0,
        (mask & 0x4u) != 0,
        (mask & 0x8u) != 0,
    };
}

constexpr ggml_gemmini_flash_page_packing_config
ggml_gemmini_flash_page_packing_from_mask(std::uint8_t mask) {
    return {
        (mask & 0x1u) != 0,
        (mask & 0x2u) != 0,
        (mask & 0x4u) != 0,
    };
}

constexpr bool ggml_gemmini_page_pack_m_operand(bool requested, std::uint64_t m) {
    return requested && m > 1;
}

constexpr bool ggml_gemmini_page_pack_b_operand(bool requested, std::uint64_t k) {
    return requested && k > 1;
}

constexpr ggml_gemmini_page_packing_config ggml_gemmini_effective_page_packing(
        const ggml_gemmini_page_packing_config & requested,
        std::uint64_t m,
        std::uint64_t k,
        bool has_d) {
    return {
        ggml_gemmini_page_pack_m_operand(requested.a, m),
        ggml_gemmini_page_pack_b_operand(requested.b, k),
        ggml_gemmini_page_pack_m_operand(requested.c, m),
        has_d && ggml_gemmini_page_pack_m_operand(requested.d, m),
    };
}

constexpr ggml_gemmini_flash_page_packing_config
ggml_gemmini_effective_flash_page_packing(
        const ggml_gemmini_flash_page_packing_config & requested,
        std::uint64_t query_rows,
        std::uint64_t q_dim,
        std::uint64_t sequence) {
    return {
        // Q is the QK matmul's A, so its M dimension is query_rows.
        ggml_gemmini_page_pack_m_operand(requested.queries, query_rows),
        // K is the physical/source B matrix before QK's B transpose.  The
        // logical contraction dimension is q_dim.
        ggml_gemmini_page_pack_b_operand(requested.keys, q_dim),
        // V is the PV matmul's B and contracts across the KV sequence.
        ggml_gemmini_page_pack_b_operand(requested.values, sequence),
    };
}

// Weight-pack v3 records one B layout for the whole file.  When that layout
// is page-packed, omit K <= 1 entries instead of mixing row-major payloads
// into the file.  The runtime prepack fallback recreates those entries from
// the GGUF using the effective row-major policy above.
constexpr bool ggml_gemmini_weight_pack_should_emit_entry(
        bool file_page_packed_b,
        std::uint64_t k) {
    return k > 0 && (!file_page_packed_b || k > 1);
}
