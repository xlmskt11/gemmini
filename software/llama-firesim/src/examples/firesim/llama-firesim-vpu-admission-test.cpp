#include "ggml-gemmini-profile.h"
#include "ggml-gemmini-vpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>

namespace {

static size_t scalar_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return sizeof(float);
        case GGML_TYPE_F16:  return sizeof(ggml_fp16_t);
        case GGML_TYPE_BF16: return sizeof(ggml_bf16_t);
        case GGML_TYPE_I32:  return sizeof(int32_t);
        default:             return 1;
    }
}

struct test_tensor {
    ggml_tensor value = {};

    test_tensor(ggml_type type, std::initializer_list<int64_t> dimensions) {
        value.type = type;
        std::fill(std::begin(value.ne), std::end(value.ne), INT64_C(1));

        size_t dimension = 0;
        for (const int64_t extent : dimensions) {
            value.ne[dimension++] = extent;
        }

        value.nb[0] = scalar_size(type);
        for (size_t index = 1; index < GGML_MAX_DIMS; ++index) {
            value.nb[index] = value.nb[index - 1] * static_cast<size_t>(value.ne[index - 1]);
        }
    }

    ggml_tensor * get() { return &value; }
};

static void set_i32(ggml_tensor & tensor, size_t index, int32_t value) {
    std::memcpy(&tensor.op_params[index], &value, sizeof(value));
}

static void set_f32(ggml_tensor & tensor, size_t index, float value) {
    std::memcpy(&tensor.op_params[index], &value, sizeof(value));
}

class checks {
public:
    void admitted(
            const std::string & name,
            const ggml_tensor & op,
            unsigned            mask = 1u,
            uint8_t             page_packing_mask = 0u) {
        char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
        if (!ggml_gemmini_vpu_can_compute(
                &op, mask, page_packing_mask, reason, sizeof(reason))) {
            std::cerr << "expected admission: " << name << ": " << reason << "\n";
            ++failures_;
        } else if (reason[0] != '\0') {
            std::cerr << "admitted operation retained a reason: " << name << ": " << reason << "\n";
            ++failures_;
        }
    }

    void rejected(
            const std::string & name,
            const ggml_tensor & op,
            unsigned            mask = 1u,
            uint8_t             page_packing_mask = 0u) {
        char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
        if (ggml_gemmini_vpu_can_compute(
                &op, mask, page_packing_mask, reason, sizeof(reason))) {
            std::cerr << "expected CPU fallback: " << name << "\n";
            ++failures_;
        } else if (reason[0] == '\0') {
            std::cerr << "fallback did not report a reason: " << name << "\n";
            ++failures_;
        }
    }

    void admitted_weighted(
            const std::string & name,
            const ggml_tensor & norm,
            const ggml_tensor & weight,
            const ggml_tensor & dst) {
        char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
        if (!ggml_gemmini_vpu_can_compute_weighted_rms_norm(
                &norm, &weight, &dst, reason, sizeof(reason))) {
            std::cerr << "expected weighted RMSNorm admission: " << name << ": " << reason << "\n";
            ++failures_;
        }
    }

    void rejected_weighted(
            const std::string & name,
            const ggml_tensor & norm,
            const ggml_tensor & weight,
            const ggml_tensor & dst) {
        char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
        if (ggml_gemmini_vpu_can_compute_weighted_rms_norm(
                &norm, &weight, &dst, reason, sizeof(reason))) {
            std::cerr << "expected weighted RMSNorm fallback: " << name << "\n";
            ++failures_;
        } else if (reason[0] == '\0') {
            std::cerr << "weighted RMSNorm fallback did not report a reason: " << name << "\n";
            ++failures_;
        }
    }

    int failures() const { return failures_; }

private:
    int failures_ = 0;
};

static void test_unary(checks & check) {
    test_tensor source(GGML_TYPE_F32, {128, 3});
    test_tensor destination(GGML_TYPE_F32, {128, 3});
    destination.value.op = GGML_OP_UNARY;
    destination.value.src[0] = source.get();

    const std::array<int32_t, 5> admitted = {
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_GELU_QUICK,
    };
    for (const int32_t unary : admitted) {
        set_i32(destination.value, 0, unary);
        check.admitted("supported F32 unary " + std::to_string(unary), destination.value);
    }

    set_i32(destination.value, 0, GGML_UNARY_OP_GELU);
    check.rejected("ordinary GELU", destination.value);
    source.value.type = GGML_TYPE_F16;
    source.value.nb[0] = sizeof(ggml_fp16_t);
    check.rejected("non-F32 unary", destination.value);
}

static void test_glu(checks & check) {
    test_tensor packed(GGML_TYPE_F32, {256, 2});
    test_tensor split_gate(GGML_TYPE_F32, {128, 2});
    test_tensor split_up(GGML_TYPE_F32, {128, 2});
    test_tensor destination(GGML_TYPE_F32, {128, 2});
    destination.value.op = GGML_OP_GLU;
    destination.value.src[0] = packed.get();
    set_i32(destination.value, 0, GGML_GLU_OP_SWIGLU);
    check.admitted("packed SwiGLU", destination.value);

    destination.value.src[0] = split_gate.get();
    destination.value.src[1] = split_up.get();
    check.admitted("split SwiGLU", destination.value);

    set_i32(destination.value, 0, GGML_GLU_OP_GEGLU);
    check.rejected("GEGLU", destination.value);
    set_i32(destination.value, 0, GGML_GLU_OP_SWIGLU_OAI);
    check.rejected("SwiGLU-OAI", destination.value);

    set_i32(destination.value, 0, GGML_GLU_OP_SWIGLU);
    destination.value.src[0] = packed.get();
    destination.value.src[1] = nullptr;
    packed.value.ne[0] = 255;
    check.rejected("malformed packed SwiGLU width", destination.value);
}

static void test_binary_elementwise(checks & check) {
    test_tensor lhs(GGML_TYPE_F32, {128, 3});
    test_tensor rhs(GGML_TYPE_F32, {128, 3});
    test_tensor destination(GGML_TYPE_F32, {128, 3});
    destination.value.src[0] = lhs.get();
    destination.value.src[1] = rhs.get();

    destination.value.op = GGML_OP_ADD;
    check.admitted("equal-shape F32 ADD", destination.value);
    destination.value.op = GGML_OP_MUL;
    check.admitted("equal-shape F32 MUL", destination.value);

    rhs.value.ne[1] = 1;
    check.rejected("broadcast VPU MUL", destination.value);
    rhs.value.ne[1] = 3;
    rhs.value.type = GGML_TYPE_BF16;
    rhs.value.nb[0] = sizeof(ggml_bf16_t);
    check.rejected("non-F32 VPU MUL", destination.value);
}

static void test_softmax(checks & check) {
    test_tensor source(GGML_TYPE_F32, {64, 4});
    test_tensor destination(GGML_TYPE_F32, {64, 4});
    test_tensor additive_mask(GGML_TYPE_F32, {64, 4});
    test_tensor sinks(GGML_TYPE_F32, {1});
    destination.value.op = GGML_OP_SOFT_MAX;
    destination.value.src[0] = source.get();
    set_f32(destination.value, 0, 1.0f);
    set_f32(destination.value, 1, 0.0f);
    check.admitted("unmasked scale-one F32 softmax", destination.value);

    set_f32(destination.value, 0, 0.5f);
    check.rejected("scaled softmax", destination.value);
    set_f32(destination.value, 0, 1.0f);
    set_f32(destination.value, 1, 0.25f);
    check.rejected("biased softmax", destination.value);
    set_f32(destination.value, 1, 0.0f);
    destination.value.src[1] = additive_mask.get();
    check.rejected("masked standalone softmax", destination.value);
    destination.value.src[1] = nullptr;
    destination.value.src[2] = sinks.get();
    check.rejected("softmax attention sinks", destination.value);
}

static void test_rms_norm(checks & check) {
    test_tensor source(GGML_TYPE_F32, {128, 3});
    test_tensor norm(GGML_TYPE_F32, {128, 3});
    test_tensor weight(GGML_TYPE_F32, {128});
    test_tensor weighted(GGML_TYPE_F32, {128, 3});
    test_tensor unrelated(GGML_TYPE_F32, {128, 3});

    norm.value.op = GGML_OP_RMS_NORM;
    norm.value.src[0] = source.get();
    set_f32(norm.value, 0, 1.0e-5f);
    check.admitted("standalone RMSNorm", norm.value);

    weighted.value.op = GGML_OP_MUL;
    weighted.value.src[0] = norm.get();
    weighted.value.src[1] = weight.get();
    check.admitted_weighted("RMSNorm + learned weight", norm.value, weight.value, weighted.value);

    weighted.value.src[0] = unrelated.get();
    check.rejected_weighted("unrelated MUL", norm.value, weight.value, weighted.value);
    weighted.value.src[0] = norm.get();
    weight.value.type = GGML_TYPE_F16;
    weight.value.nb[0] = sizeof(ggml_fp16_t);
    check.rejected_weighted("non-F32 RMSNorm weight", norm.value, weight.value, weighted.value);
    weight.value.type = GGML_TYPE_F32;
    weight.value.nb[0] = sizeof(float);

    set_f32(norm.value, 0, -1.0f);
    check.rejected("negative RMSNorm epsilon", norm.value);
    set_f32(norm.value, 0, std::numeric_limits<float>::quiet_NaN());
    check.rejected("NaN RMSNorm epsilon", norm.value);
}

static void set_valid_rope_parameters(ggml_tensor & rope, int32_t mode) {
    set_i32(rope, 1, 128);
    set_i32(rope, 2, mode);
    set_i32(rope, 4, 4096);
    set_f32(rope, 5, 10000.0f);
    set_f32(rope, 6, 1.0f);
    set_f32(rope, 7, 0.0f);
    set_f32(rope, 8, 1.0f);
    set_f32(rope, 9, 32.0f);
    set_f32(rope, 10, 1.0f);
}

static void test_rope(checks & check) {
    test_tensor source(GGML_TYPE_F32, {128, 4, 2, 1});
    test_tensor positions(GGML_TYPE_I32, {2});
    test_tensor factors(GGML_TYPE_F32, {64});
    test_tensor destination(GGML_TYPE_F32, {128, 4, 2, 1});
    destination.value.op = GGML_OP_ROPE;
    destination.value.src[0] = source.get();
    destination.value.src[1] = positions.get();

    set_valid_rope_parameters(destination.value, GGML_ROPE_TYPE_NORMAL);
    check.admitted("normal RoPE", destination.value);
    set_valid_rope_parameters(destination.value, GGML_ROPE_TYPE_NEOX);
    check.admitted("NeoX RoPE", destination.value);
    destination.value.src[2] = factors.get();
    check.admitted("RoPE with frequency factors", destination.value);
    destination.value.src[2] = nullptr;

    set_i32(destination.value, 1, 127);
    check.rejected("odd RoPE rotary dimension", destination.value);
    set_valid_rope_parameters(destination.value, GGML_ROPE_TYPE_MROPE);
    check.rejected("MRoPE", destination.value);
    set_valid_rope_parameters(destination.value, GGML_ROPE_TYPE_VISION);
    check.rejected("vision RoPE", destination.value);
    set_valid_rope_parameters(destination.value, GGML_ROPE_TYPE_NORMAL);
    positions.value.ne[0] = 1;
    check.rejected("RoPE position count mismatch", destination.value);
}

struct flash_fixture {
    test_tensor q{GGML_TYPE_F32, {64, 2, 4, 1}};
    test_tensor k{GGML_TYPE_F16, {64, 4, 2, 1}};
    test_tensor v{GGML_TYPE_F32, {64, 4, 2, 1}};
    test_tensor mask{GGML_TYPE_F16, {4, 2, 1, 1}};
    test_tensor dst{GGML_TYPE_F32, {64, 4, 2, 1}};
    test_tensor sinks{GGML_TYPE_F32, {1}};

    flash_fixture() {
        dst.value.op = GGML_OP_FLASH_ATTN_EXT;
        dst.value.src[0] = q.get();
        dst.value.src[1] = k.get();
        dst.value.src[2] = v.get();
        dst.value.src[3] = mask.get();
        set_f32(dst.value, 0, 0.125f);
        set_f32(dst.value, 1, 0.0f);
        set_f32(dst.value, 2, 0.0f);
        set_i32(dst.value, 3, GGML_PREC_DEFAULT);
    }
};

static void test_flash_attention(checks & check) {
    flash_fixture flash;

#if GGML_GEMMINI_PROFILE_ID == GGML_GEMMINI_PROFILE_SINGLE_ID
    check.admitted("single logical FlashAttention mask 0x1", flash.dst.value, 0x1u);
    check.rejected("single logical mask 0x0", flash.dst.value, 0x0u);
    check.rejected("single physical custom3 bit passed as logical mask", flash.dst.value, 0x8u);
    check.rejected("single out-of-range logical mask", flash.dst.value, 0x2u);
#else
    for (const unsigned mask : {0x1u, 0x3u, 0x7u, 0xfu}) {
        check.admitted("multi logical FlashAttention mask " + std::to_string(mask), flash.dst.value, mask);
    }
    check.rejected("multi logical mask 0x0", flash.dst.value, 0x0u);
    check.rejected("multi out-of-range logical mask", flash.dst.value, 0x10u);
#endif

    check.admitted(
        "FlashAttention with requested Q page packing",
        flash.dst.value, 0x1u, 0x01u);
    check.admitted(
        "FlashAttention with requested K page packing",
        flash.dst.value, 0x1u, 0x02u);
    check.admitted(
        "FlashAttention with requested V page packing",
        flash.dst.value, 0x1u, 0x04u);
    check.admitted(
        "FlashAttention with requested Q/K/V page packing",
        flash.dst.value, 0x1u, 0x07u);
    check.rejected(
        "FlashAttention with legacy generic D page-packing bit",
        flash.dst.value, 0x1u, 0x08u);
    check.rejected(
        "FlashAttention with a higher unknown page-packing bit",
        flash.dst.value, 0x1u, 0x10u);

    flash.dst.value.src[3] = nullptr;
    check.rejected("FlashAttention without explicit mask", flash.dst.value);
    flash.dst.value.src[3] = flash.mask.get();
    flash.mask.value.type = GGML_TYPE_F32;
    flash.mask.value.nb[0] = sizeof(float);
    check.rejected("non-F16 FlashAttention mask", flash.dst.value);
    flash.mask.value.type = GGML_TYPE_F16;
    flash.mask.value.nb[0] = sizeof(ggml_fp16_t);

    set_f32(flash.dst.value, 1, 0.25f);
    check.rejected("FlashAttention ALiBi/max_bias", flash.dst.value);
    set_f32(flash.dst.value, 1, 0.0f);
    set_f32(flash.dst.value, 2, 2.0f);
    check.rejected("FlashAttention softcap", flash.dst.value);
    set_f32(flash.dst.value, 2, 0.0f);
    set_i32(flash.dst.value, 3, GGML_PREC_F32);
    check.admitted("FlashAttention F32 accumulator precision", flash.dst.value);
    set_i32(flash.dst.value, 3, GGML_PREC_DEFAULT);
    flash.dst.value.src[4] = flash.sinks.get();
    check.rejected("FlashAttention sinks", flash.dst.value);
    flash.dst.value.src[4] = nullptr;

    flash.q.value.ne[2] = 3;
    check.rejected("FlashAttention incompatible GQA heads", flash.dst.value);
    flash.q.value.ne[2] = 4;
    flash.q.value.type = GGML_TYPE_Q8_0;
    flash.q.value.nb[0] = 1;
    check.rejected("quantized FlashAttention Q", flash.dst.value);
    flash.q.value.type = GGML_TYPE_BF16;
    flash.q.value.nb[0] = sizeof(ggml_bf16_t);
    check.rejected("BF16 FlashAttention Q cannot safely use CPU fallback", flash.dst.value);
    flash.q.value.type = GGML_TYPE_F32;
    flash.q.value.nb[0] = sizeof(float);
    set_f32(flash.dst.value, 0, 0.15625f);
    check.admitted("arbitrary positive FlashAttention scale", flash.dst.value);
    set_f32(flash.dst.value, 0, 0.0f);
    check.rejected("zero FlashAttention scale", flash.dst.value);
    set_f32(flash.dst.value, 0, -0.125f);
    check.rejected("negative FlashAttention scale", flash.dst.value);
    set_f32(flash.dst.value, 0, std::numeric_limits<float>::quiet_NaN());
    check.rejected("non-finite FlashAttention scale", flash.dst.value);
}

static void test_metadata_fallback(checks & check) {
    test_tensor source(GGML_TYPE_F32, {16, 16});
    test_tensor permute(GGML_TYPE_F32, {16, 16});
    permute.value.op = GGML_OP_PERMUTE;
    permute.value.src[0] = source.get();
    check.rejected("PERMUTE metadata operation", permute.value);

    test_tensor bad_stride_source(GGML_TYPE_F32, {2, 2});
    test_tensor unary_destination(GGML_TYPE_F32, {2, 2});
    unary_destination.value.op = GGML_OP_UNARY;
    unary_destination.value.src[0] = bad_stride_source.get();
    set_i32(unary_destination.value, 0, GGML_UNARY_OP_RELU);
    bad_stride_source.value.nb[1] = std::numeric_limits<size_t>::max();
    check.rejected("overflowing tensor stride span", unary_destination.value);
}

} // namespace

int main() {
    checks check;
    test_unary(check);
    test_glu(check);
    test_binary_elementwise(check);
    test_softmax(check);
    test_rms_norm(check);
    test_rope(check);
    test_flash_attention(check);
    test_metadata_fallback(check);

    if (check.failures() != 0) {
        std::cerr << "VPU admission tests failed for " << GGML_GEMMINI_PROFILE_NAME
                  << ": " << check.failures() << " failure(s)\n";
        return 1;
    }

    std::cout << "VPU admission tests passed for " << GGML_GEMMINI_PROFILE_NAME
              << " (logical mask 0x" << std::hex << GGML_GEMMINI_PROFILE_LOGICAL_MASK
              << std::dec << ")\n";
    return 0;
}
