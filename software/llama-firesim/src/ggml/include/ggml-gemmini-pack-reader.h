#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>

struct ggml_gemmini_weight_pack_expected_geometry {
    uint32_t elem_size;
    uint32_t numeric_format;
    uint32_t profile_id;
    uint32_t dim;
    uint32_t page_size;
    uint32_t max_bytes;
};

struct ggml_gemmini_weight_pack_v3_entry_descriptor {
    std::string name;
    int64_t slice_i2;
    int64_t slice_i3;
    int64_t cols_in;
    int64_t cols_out;
    size_t data_count;
    bool page_packed_b;
    uint32_t role;
};

enum ggml_gemmini_weight_pack_read_status {
    GGML_GEMMINI_WEIGHT_PACK_READ_OK = 0,
    GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR,
    GGML_GEMMINI_WEIGHT_PACK_READ_BAD_HEADER,
    GGML_GEMMINI_WEIGHT_PACK_READ_LEGACY_VERSION,
    GGML_GEMMINI_WEIGHT_PACK_READ_UNSUPPORTED_VERSION,
    GGML_GEMMINI_WEIGHT_PACK_READ_PROFILE_OR_GEOMETRY_MISMATCH,
    GGML_GEMMINI_WEIGHT_PACK_READ_INVALID_ENTRY,
    GGML_GEMMINI_WEIGHT_PACK_READ_SIZE_OVERFLOW,
    GGML_GEMMINI_WEIGHT_PACK_READ_ENTRY_REJECTED,
    GGML_GEMMINI_WEIGHT_PACK_READ_TRAILING_DATA,
    GGML_GEMMINI_WEIGHT_PACK_READ_RESOURCE_ERROR,
};

struct ggml_gemmini_weight_pack_read_result {
    enum ggml_gemmini_weight_pack_read_status status = GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR;
    uint32_t version = 0;
    uint64_t entry_count = 0;
    bool page_packed_b = false;
    bool hybrid_sparse_gguf = false;
    uint64_t model_fingerprint = 0;
    std::string reason;
};

// The callback allocates the final storage for one entry.  The reader writes
// validated BF16 payload bytes directly into that storage, avoiding a second
// full-pack copy in the production loader.  Returning nullptr rejects the
// entry; reason should describe why.
using ggml_gemmini_weight_pack_allocate_entry_fn = uint16_t * (*)(
    void * context,
    const ggml_gemmini_weight_pack_v3_entry_descriptor & descriptor,
    std::string & reason);

ggml_gemmini_weight_pack_read_result ggml_gemmini_weight_pack_read_v3(
    std::istream & input,
    const ggml_gemmini_weight_pack_expected_geometry & expected,
    ggml_gemmini_weight_pack_allocate_entry_fn allocate_entry,
    void * context);

const char * ggml_gemmini_weight_pack_read_status_name(
    enum ggml_gemmini_weight_pack_read_status status);
