#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

static_assert(sizeof(float) == sizeof(uint32_t), "Gemmini BF16 conversion requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559, "Gemmini BF16 conversion requires IEEE-754 float");

// Convert an IEEE-754 binary32 value to BF16 using round-to-nearest, ties-to-even.
// NaNs remain NaNs and are quieted so truncating a low-only payload cannot produce
// infinity. Infinities, signed zeroes, and finite subnormals retain IEEE semantics.
static inline uint16_t ggml_gemmini_float_to_bf16(float value) noexcept {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    constexpr uint32_t f32_exponent_mask = UINT32_C(0x7f800000);
    constexpr uint32_t f32_fraction_mask = UINT32_C(0x007fffff);
    constexpr uint16_t bf16_quiet_nan_bit = UINT16_C(0x0040);

    if ((bits & f32_exponent_mask) == f32_exponent_mask) {
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        if ((bits & f32_fraction_mask) != 0) {
            upper = static_cast<uint16_t>(upper | bf16_quiet_nan_bit);
        }
        return upper;
    }

    const uint32_t retained_lsb = (bits >> 16) & UINT32_C(1);
    bits += UINT32_C(0x7fff) + retained_lsb;
    return static_cast<uint16_t>(bits >> 16);
}

static inline float ggml_gemmini_bf16_to_float(uint16_t value) noexcept {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
