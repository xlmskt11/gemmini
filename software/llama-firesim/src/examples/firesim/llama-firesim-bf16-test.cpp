#include "ggml-gemmini-bf16.h"
#include "ggml-gemmini-pack.h"
#include "ggml-gemmini-pack-reader.h"
#include "ggml-gemmini-page-packing.h"
#include "ggml-gemmini-profile.h"
#include "ggml-gemmini-weight-select.h"
#include "include/gemmini_page_packed.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static_assert(sizeof(ggml_gemmini_weight_pack_file_header) == 32,
    "weight-pack file header ABI changed");
static_assert(sizeof(ggml_gemmini_weight_pack_geometry_v2) == 16,
    "weight-pack v2 geometry ABI changed");
static_assert(sizeof(ggml_gemmini_weight_pack_geometry_v3) == 24,
    "weight-pack v3 geometry ABI changed");
static_assert(sizeof(ggml_gemmini_weight_pack_geometry_v4) == 40,
    "weight-pack v4 geometry ABI changed");
static_assert(sizeof(ggml_gemmini_hybrid_footer) == 16,
    "hybrid footer ABI changed");
static_assert(sizeof(ggml_gemmini_weight_pack_entry_header) == 56,
    "weight-pack v1/v2 entry ABI changed");
static_assert(sizeof(ggml_gemmini_weight_pack_entry_header_v3) == 48,
    "weight-pack v3 entry ABI changed");

namespace {

static float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool expect_encoding(uint32_t f32_bits, uint16_t expected) {
    const uint16_t actual = ggml_gemmini_float_to_bf16(float_from_bits(f32_bits));
    if (actual == expected) {
        return true;
    }

    std::cerr << "encode mismatch: f32_bits=0x" << std::hex << f32_bits
              << " expected=0x" << expected << " actual=0x" << actual << std::dec << "\n";
    return false;
}

template <typename T>
static void append_bytes(std::vector<uint8_t> & bytes, const T & value) {
    const auto * begin = reinterpret_cast<const uint8_t *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

struct captured_pack_entry {
    bool allocated = false;
    ggml_gemmini_weight_pack_v3_entry_descriptor descriptor = {};
    std::vector<uint16_t> data;
};

static uint16_t * allocate_captured_entry(
        void * context,
        const ggml_gemmini_weight_pack_v3_entry_descriptor & descriptor,
        std::string & reason) {
    auto * captured = static_cast<captured_pack_entry *>(context);
    if (captured == nullptr || captured->allocated) {
        reason = "test reader received duplicate entry";
        return nullptr;
    }
    captured->allocated = true;
    captured->descriptor = descriptor;
    captured->data.resize(descriptor.data_count);
    return captured->data.data();
}

static ggml_gemmini_weight_pack_expected_geometry expected_geometry(uint32_t profile_id) {
    return {
        sizeof(uint16_t),
        GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE,
        profile_id,
        DIM,
        GEMMINI_PAGE_PACKED_PAGE_BYTES,
        MAX_BYTES,
    };
}

static ggml_gemmini_weight_pack_read_result read_pack_bytes(
        const std::vector<uint8_t> & bytes,
        const ggml_gemmini_weight_pack_expected_geometry & expected,
        captured_pack_entry * captured) {
    const std::string serialized(
        reinterpret_cast<const char *>(bytes.data()), bytes.size());
    std::istringstream input(serialized, std::ios::in | std::ios::binary);
    return ggml_gemmini_weight_pack_read_v3(
        input, expected, allocate_captured_entry, captured);
}

static bool test_pack_v3_round_trip() {
    const std::string name = "test.weight";
    const size_t payload_count =
        gemmini_page_packed_b_page_count(3, 1) *
        GEMMINI_PAGE_PACKED_PAGE_BYTES / sizeof(uint16_t);
    std::vector<uint16_t> payload(payload_count, 0);
    payload[0] = UINT16_C(0x3f80);
    payload[1] = UINT16_C(0xc020);
    payload[2] = UINT16_C(0x0001);

    ggml_gemmini_weight_pack_file_header header = {};
    std::memcpy(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC,
        GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE);
    header.version = GGML_GEMMINI_WEIGHT_PACK_VERSION_V3;
    header.endian = GGML_GEMMINI_WEIGHT_PACK_ENDIAN;
    header.elem_size = sizeof(uint16_t);
    header.scale_size = 0;
    header.entry_count = 1;

    const ggml_gemmini_weight_pack_geometry_v3 geometry = {
        GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE,
        GGML_GEMMINI_PROFILE_ID,
        DIM,
        GEMMINI_PAGE_PACKED_PAGE_BYTES,
        MAX_BYTES,
        GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B,
    };
    ggml_gemmini_weight_pack_entry_header_v3 entry = {};
    entry.name_size = static_cast<uint32_t>(name.size());
    entry.cols_in = 3;
    entry.cols_out = 1;
    entry.data_count = payload.size();

    std::vector<uint8_t> bytes;
    append_bytes(bytes, header);
    append_bytes(bytes, geometry);
    append_bytes(bytes, entry);
    bytes.insert(bytes.end(), name.begin(), name.end());
    bytes.insert(bytes.end(), reinterpret_cast<const uint8_t *>(payload.data()),
        reinterpret_cast<const uint8_t *>(payload.data()) + payload.size() * sizeof(uint16_t));

    captured_pack_entry captured;
    const ggml_gemmini_weight_pack_read_result result = read_pack_bytes(
        bytes, expected_geometry(GGML_GEMMINI_PROFILE_ID), &captured);
    if (result.status != GGML_GEMMINI_WEIGHT_PACK_READ_OK ||
            result.entry_count != 1 || !result.page_packed_b ||
            !captured.allocated || captured.descriptor.name != name ||
            captured.descriptor.slice_i2 != 0 || captured.descriptor.slice_i3 != 0 ||
            captured.descriptor.cols_in != 3 || captured.descriptor.cols_out != 1 ||
            captured.descriptor.data_count != payload.size() ||
            !captured.descriptor.page_packed_b || captured.data != payload) {
        return false;
    }

    const uint32_t opposite_profile = GGML_GEMMINI_PROFILE_ID == GGML_GEMMINI_PROFILE_SINGLE_ID
        ? GGML_GEMMINI_PROFILE_MULTI_ID
        : GGML_GEMMINI_PROFILE_SINGLE_ID;
    captured_pack_entry opposite_capture;
    const ggml_gemmini_weight_pack_read_result opposite = read_pack_bytes(
        bytes, expected_geometry(opposite_profile), &opposite_capture);
    if (opposite.status != GGML_GEMMINI_WEIGHT_PACK_READ_PROFILE_OR_GEOMETRY_MISMATCH ||
            opposite_capture.allocated) {
        return false;
    }

    for (uint32_t legacy_version : {
            GGML_GEMMINI_WEIGHT_PACK_VERSION_V1,
            GGML_GEMMINI_WEIGHT_PACK_VERSION_V2}) {
        ggml_gemmini_weight_pack_file_header legacy = header;
        legacy.version = legacy_version;
        std::vector<uint8_t> legacy_bytes;
        append_bytes(legacy_bytes, legacy);
        captured_pack_entry legacy_capture;
        const ggml_gemmini_weight_pack_read_result legacy_result = read_pack_bytes(
            legacy_bytes, expected_geometry(GGML_GEMMINI_PROFILE_ID), &legacy_capture);
        if (legacy_result.status != GGML_GEMMINI_WEIGHT_PACK_READ_LEGACY_VERSION ||
                legacy_result.version != legacy_version || legacy_capture.allocated) {
            return false;
        }
    }

    return true;
}

static bool test_pack_v4_hybrid_role() {
    const std::string name = "blk.0.attn_q.weight";
    const size_t payload_count =
        gemmini_page_packed_b_page_count(8, 8) *
        GEMMINI_PAGE_PACKED_PAGE_BYTES / sizeof(uint16_t);
    std::vector<uint16_t> payload(payload_count, UINT16_C(0x3f80));

    ggml_gemmini_weight_pack_file_header header = {};
    std::memcpy(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC,
        GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE);
    header.version = GGML_GEMMINI_WEIGHT_PACK_VERSION_V4;
    header.endian = GGML_GEMMINI_WEIGHT_PACK_ENDIAN;
    header.elem_size = sizeof(uint16_t);
    header.entry_count = 1;

    ggml_gemmini_weight_pack_geometry_v4 geometry = {};
    geometry.numeric_format = GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE;
    geometry.profile_id = GGML_GEMMINI_PROFILE_ID;
    geometry.dim = DIM;
    geometry.page_size = GEMMINI_PAGE_PACKED_PAGE_BYTES;
    geometry.max_bytes = MAX_BYTES;
    geometry.b_layout = GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B;
    geometry.flags = GGML_GEMMINI_WEIGHT_PACK_FLAG_HYBRID_SPARSE_GGUF;
    geometry.model_fingerprint = UINT64_C(0x123456789abcdef0);

    ggml_gemmini_weight_pack_entry_header_v3 entry = {};
    entry.name_size = static_cast<uint32_t>(name.size());
    entry.reserved = GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY;
    entry.cols_in = 8;
    entry.cols_out = 8;
    entry.data_count = payload.size();

    std::vector<uint8_t> bytes;
    append_bytes(bytes, header);
    append_bytes(bytes, geometry);
    append_bytes(bytes, entry);
    bytes.insert(bytes.end(), name.begin(), name.end());
    bytes.insert(bytes.end(), reinterpret_cast<const uint8_t *>(payload.data()),
        reinterpret_cast<const uint8_t *>(payload.data()) + payload.size() * sizeof(uint16_t));

    captured_pack_entry captured;
    const auto result = read_pack_bytes(
        bytes, expected_geometry(GGML_GEMMINI_PROFILE_ID), &captured);
    return result.status == GGML_GEMMINI_WEIGHT_PACK_READ_OK &&
        result.version == GGML_GEMMINI_WEIGHT_PACK_VERSION_V4 &&
        result.hybrid_sparse_gguf &&
        result.model_fingerprint == geometry.model_fingerprint &&
        captured.descriptor.role == GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY &&
        captured.data == payload;
}

static bool test_weight_name_selection() {
    return ggml_gemmini_is_model_weight_name("blk.0.attn_qkv.weight") &&
        ggml_gemmini_is_model_weight_name("blk.0.ffn_gate_exps.weight") &&
        ggml_gemmini_is_model_weight_name("blk.0.attn_norm.weight") &&
        ggml_gemmini_is_model_weight_name("layers.0.attention.wq") &&
        !ggml_gemmini_is_model_weight_name("blk.0.attn_q.bias");
}

static bool test_weight_role_selection() {
    return ggml_gemmini_classify_weight_usage(GGML_GEMMINI_WEIGHT_USAGE_ORIGINAL) ==
            GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY &&
        ggml_gemmini_classify_weight_usage(GGML_GEMMINI_WEIGHT_USAGE_MUL_MAT) ==
            GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY &&
        ggml_gemmini_classify_weight_usage(
            GGML_GEMMINI_WEIGHT_USAGE_ORIGINAL | GGML_GEMMINI_WEIGHT_USAGE_MUL_MAT) ==
            GGML_GEMMINI_WEIGHT_ROLE_SHARED;
}

static bool test_packed_source_admission() {
    return ggml_gemmini_packed_source_mask_allowed(true, 1u) &&
        !ggml_gemmini_packed_source_mask_allowed(false, 1u) &&
        !ggml_gemmini_packed_source_mask_allowed(true, 2u) &&
        !ggml_gemmini_packed_source_mask_allowed(true, 3u) &&
        !ggml_gemmini_packed_source_mask_allowed(true, 0u);
}

static bool test_pack_size_overflow_rejected() {
    const std::string name = "overflow.weight";
    for (uint32_t layout : {
            GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR,
            GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B}) {
        ggml_gemmini_weight_pack_file_header header = {};
        std::memcpy(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC,
            GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE);
        header.version = GGML_GEMMINI_WEIGHT_PACK_VERSION_V3;
        header.endian = GGML_GEMMINI_WEIGHT_PACK_ENDIAN;
        header.elem_size = sizeof(uint16_t);
        header.entry_count = 1;

        const ggml_gemmini_weight_pack_geometry_v3 geometry = {
            GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE,
            GGML_GEMMINI_PROFILE_ID,
            DIM,
            GEMMINI_PAGE_PACKED_PAGE_BYTES,
            MAX_BYTES,
            layout,
        };
        ggml_gemmini_weight_pack_entry_header_v3 entry = {};
        entry.name_size = static_cast<uint32_t>(name.size());
        entry.cols_in = std::numeric_limits<int64_t>::max();
        entry.cols_out = std::numeric_limits<int64_t>::max();
        entry.data_count = 0;

        std::vector<uint8_t> bytes;
        append_bytes(bytes, header);
        append_bytes(bytes, geometry);
        append_bytes(bytes, entry);
        bytes.insert(bytes.end(), name.begin(), name.end());

        captured_pack_entry captured;
        const ggml_gemmini_weight_pack_read_result result = read_pack_bytes(
            bytes, expected_geometry(GGML_GEMMINI_PROFILE_ID), &captured);
        if (result.status != GGML_GEMMINI_WEIGHT_PACK_READ_SIZE_OVERFLOW ||
                captured.allocated) {
            return false;
        }
    }
    return true;
}

static bool test_logical_to_physical_mask() {
#if GGML_GEMMINI_PROFILE_ID == GGML_GEMMINI_PROFILE_SINGLE_ID
    const unsigned expected_physical_mask = 1u << XCUSTOM_ACC;
    return ggml_gemmini_profile_physical_mask(0u) == 0u &&
        ggml_gemmini_profile_physical_mask(0x1u) == expected_physical_mask &&
        ggml_gemmini_profile_physical_mask(0x3u) == expected_physical_mask &&
        ggml_gemmini_profile_request_is_valid(0x0, -1, 0) &&
        ggml_gemmini_profile_request_is_valid(0x1, -1, 0) &&
        !ggml_gemmini_profile_request_is_valid(0x2, -1, 0) &&
        !ggml_gemmini_profile_request_is_valid(0x1, 0, 1);
#else
    const unsigned physical_base = 1u << XCUSTOM_ACC;
    return ggml_gemmini_profile_physical_mask(0u) == 0u &&
        ggml_gemmini_profile_physical_mask(0x1u) == 0x1u * physical_base &&
        ggml_gemmini_profile_physical_mask(0x7u) == 0x7u * physical_base &&
        ggml_gemmini_profile_physical_mask(0x1fu) == 0xfu * physical_base &&
        ggml_gemmini_profile_request_is_valid(0xf, -1, 0) &&
        ggml_gemmini_profile_request_is_valid(0x4, 2, 1) &&
        !ggml_gemmini_profile_request_is_valid(0x10, -1, 0) &&
        !ggml_gemmini_profile_request_is_valid(0x3, 1, 1) &&
        !ggml_gemmini_profile_request_is_valid(0x4, -1, 1);
#endif
}

} // namespace

int main() {
    bool ok = true;

    ok &= expect_encoding(UINT32_C(0x00000000), UINT16_C(0x0000)); // +0
    ok &= expect_encoding(UINT32_C(0x80000000), UINT16_C(0x8000)); // -0
    ok &= expect_encoding(UINT32_C(0x3f800000), UINT16_C(0x3f80)); // 1
    ok &= expect_encoding(UINT32_C(0xc0200000), UINT16_C(0xc020)); // -2.5
    ok &= expect_encoding(UINT32_C(0x7f800000), UINT16_C(0x7f80)); // +inf
    ok &= expect_encoding(UINT32_C(0xff800000), UINT16_C(0xff80)); // -inf

    // RNE halfway cases: retain an even low bit and increment an odd low bit.
    ok &= expect_encoding(UINT32_C(0x3f808000), UINT16_C(0x3f80));
    ok &= expect_encoding(UINT32_C(0x3f818000), UINT16_C(0x3f82));
    ok &= expect_encoding(UINT32_C(0xbf808000), UINT16_C(0xbf80));
    ok &= expect_encoding(UINT32_C(0xbf818000), UINT16_C(0xbf82));

    // Float subnormal values around BF16 halfway points.
    ok &= expect_encoding(UINT32_C(0x00008000), UINT16_C(0x0000));
    ok &= expect_encoding(UINT32_C(0x00008001), UINT16_C(0x0001));
    ok &= expect_encoding(UINT32_C(0x00018000), UINT16_C(0x0002));
    ok &= expect_encoding(UINT32_C(0x00028000), UINT16_C(0x0002));
    ok &= expect_encoding(UINT32_C(0x80008000), UINT16_C(0x8000));
    ok &= expect_encoding(UINT32_C(0x80008001), UINT16_C(0x8001));

    // A NaN whose payload exists only in discarded bits must not become infinity.
    const uint16_t positive_nan = ggml_gemmini_float_to_bf16(
        float_from_bits(UINT32_C(0x7f800001)));
    const uint16_t negative_nan = ggml_gemmini_float_to_bf16(
        float_from_bits(UINT32_C(0xff812345)));
    ok &= (positive_nan & UINT16_C(0x7fff)) > UINT16_C(0x7f80);
    ok &= (negative_nan & UINT16_C(0x7fff)) > UINT16_C(0x7f80);
    ok &= (positive_nan & UINT16_C(0x0040)) != 0;
    ok &= (negative_nan & UINT16_C(0x0040)) != 0;

    // Every representable non-NaN BF16 value round-trips exactly through float.
    for (uint32_t raw = 0; raw <= UINT16_MAX; ++raw) {
        const uint16_t bf16 = static_cast<uint16_t>(raw);
        const bool is_nan = (bf16 & UINT16_C(0x7f80)) == UINT16_C(0x7f80) &&
            (bf16 & UINT16_C(0x007f)) != 0;
        const uint16_t round_trip = ggml_gemmini_float_to_bf16(
            ggml_gemmini_bf16_to_float(bf16));
        if ((!is_nan && round_trip != bf16) ||
            (is_nan && (round_trip & UINT16_C(0x7fff)) <= UINT16_C(0x7f80))) {
            std::cerr << "round-trip mismatch: bf16=0x" << std::hex << bf16
                      << " result=0x" << round_trip << std::dec << "\n";
            ok = false;
            break;
        }
    }

    if (!test_pack_v3_round_trip()) {
        std::cerr << "pack v3 round-trip/profile mismatch test failed\n";
        ok = false;
    }
    if (!test_pack_v4_hybrid_role()) {
        std::cerr << "pack v4 hybrid-role test failed\n";
        ok = false;
    }
    if (!test_logical_to_physical_mask()) {
        std::cerr << "logical-to-physical mask test failed\n";
        ok = false;
    }
    if (!test_weight_name_selection()) {
        std::cerr << "weight-name selection test failed\n";
        ok = false;
    }
    if (!test_weight_role_selection()) {
        std::cerr << "weight-role selection test failed\n";
        ok = false;
    }
    if (!test_packed_source_admission()) {
        std::cerr << "packed-source admission test failed\n";
        ok = false;
    }
    if (!test_pack_size_overflow_rejected()) {
        std::cerr << "weight-pack size overflow rejection test failed\n";
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    std::cout << "GEMMINI-BF16-TEST-PASS\n";
    return 0;
}
