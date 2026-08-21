#pragma once

#include <stdint.h>

#define GGML_GEMMINI_WEIGHT_PACK_MAGIC "GMINIPK1"
#define GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE 8
#define GGML_GEMMINI_WEIGHT_PACK_VERSION_V1 1u
#define GGML_GEMMINI_WEIGHT_PACK_VERSION_V2 2u
#define GGML_GEMMINI_WEIGHT_PACK_VERSION_V3 3u
#define GGML_GEMMINI_WEIGHT_PACK_VERSION_V4 4u
#define GGML_GEMMINI_WEIGHT_PACK_VERSION GGML_GEMMINI_WEIGHT_PACK_VERSION_V4
#define GGML_GEMMINI_WEIGHT_PACK_ENDIAN 0x01020304u

#define GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR 0u
#define GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B 1u

#define GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_UNSPECIFIED 0u
#define GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE 1u

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

struct ggml_gemmini_weight_pack_geometry_v3 {
    uint32_t numeric_format;
    uint32_t profile_id;
    uint32_t dim;
    uint32_t page_size;
    uint32_t max_bytes;
    uint32_t b_layout;
};

#define GGML_GEMMINI_WEIGHT_PACK_FLAG_HYBRID_SPARSE_GGUF (1u << 0)

struct ggml_gemmini_weight_pack_geometry_v4 {
    uint32_t numeric_format;
    uint32_t profile_id;
    uint32_t dim;
    uint32_t page_size;
    uint32_t max_bytes;
    uint32_t b_layout;
    uint32_t flags;
    uint32_t reserved;
    uint64_t model_fingerprint;
};

#define GGML_GEMMINI_HYBRID_FOOTER_MAGIC "GMINIHY1"
#define GGML_GEMMINI_HYBRID_FOOTER_MAGIC_SIZE 8

struct ggml_gemmini_hybrid_footer {
    char magic[GGML_GEMMINI_HYBRID_FOOTER_MAGIC_SIZE];
    uint64_t model_fingerprint;
};

#define GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_MASK 0x3u
#define GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY 1u
#define GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_SHARED 2u

// Legacy v1/v2 entries contain one float scale per output column before data.
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

// BF16 v3 entries contain the tensor name followed immediately by data_count
// raw BF16 values. There is no scale payload.
struct ggml_gemmini_weight_pack_entry_header_v3 {
    uint32_t name_size;
    uint32_t reserved;
    int64_t slice_i2;
    int64_t slice_i3;
    int64_t cols_in;
    int64_t cols_out;
    uint64_t data_count;
};

#if defined(__cplusplus)
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
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(struct ggml_gemmini_weight_pack_file_header) == 32,
    "weight-pack file header ABI changed");
_Static_assert(sizeof(struct ggml_gemmini_weight_pack_geometry_v2) == 16,
    "weight-pack v2 geometry ABI changed");
_Static_assert(sizeof(struct ggml_gemmini_weight_pack_geometry_v3) == 24,
    "weight-pack v3 geometry ABI changed");
_Static_assert(sizeof(struct ggml_gemmini_weight_pack_geometry_v4) == 40,
    "weight-pack v4 geometry ABI changed");
_Static_assert(sizeof(struct ggml_gemmini_hybrid_footer) == 16,
    "hybrid footer ABI changed");
_Static_assert(sizeof(struct ggml_gemmini_weight_pack_entry_header) == 56,
    "weight-pack v1/v2 entry ABI changed");
_Static_assert(sizeof(struct ggml_gemmini_weight_pack_entry_header_v3) == 48,
    "weight-pack v3 entry ABI changed");
#endif
