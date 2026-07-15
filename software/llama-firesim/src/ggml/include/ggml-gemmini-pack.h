#pragma once

#include <stdint.h>

#define GGML_GEMMINI_WEIGHT_PACK_MAGIC "GMINIPK1"
#define GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE 8
#define GGML_GEMMINI_WEIGHT_PACK_VERSION_V1 1u
#define GGML_GEMMINI_WEIGHT_PACK_VERSION 2u
#define GGML_GEMMINI_WEIGHT_PACK_ENDIAN 0x01020304u

#define GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR 0u
#define GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B 1u

struct ggml_gemmini_weight_pack_file_header {
    char magic[GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE];
    uint32_t version;
    uint32_t endian;
    uint32_t elem_size;
    uint32_t scale_size;
    uint64_t entry_count;
};

struct ggml_gemmini_weight_pack_geometry_v2 {
    uint32_t dim;
    uint32_t page_size;
    uint32_t max_bytes;
    uint32_t b_layout;
};

struct ggml_gemmini_weight_pack_entry_header {
    uint32_t name_size;
    uint32_t reserved;
    int64_t slice_i2;
    int64_t slice_i3;
    int64_t cols_in;
    int64_t cols_out;
    uint64_t scale_count;
    uint64_t data_count;
};
