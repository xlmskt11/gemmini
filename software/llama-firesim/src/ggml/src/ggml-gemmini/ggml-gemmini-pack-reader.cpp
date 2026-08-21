#include "ggml-gemmini-pack-reader.h"

#include "ggml-gemmini-pack.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

template <typename T>
static bool read_pod(std::istream & input, T & value) {
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return static_cast<bool>(input);
}

static bool read_exact(std::istream & input, void * data, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    input.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(input);
}

static bool checked_mul(size_t lhs, size_t rhs, size_t * result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static bool checked_ceil_div(size_t numerator, size_t denominator, size_t * result) {
    if (denominator == 0) {
        return false;
    }
    *result = numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
    return true;
}

static bool checked_b_data_count(
        int64_t rows_i64,
        int64_t cols_i64,
        bool page_packed_b,
        const ggml_gemmini_weight_pack_expected_geometry & expected,
        uint64_t serialized_count,
        size_t * count_out) {
    if (rows_i64 <= 0 || cols_i64 <= 0 || expected.elem_size == 0 ||
            expected.dim == 0 || expected.page_size == 0) {
        return false;
    }
    if (static_cast<uint64_t>(rows_i64) > std::numeric_limits<size_t>::max() ||
            static_cast<uint64_t>(cols_i64) > std::numeric_limits<size_t>::max()) {
        return false;
    }

    const size_t rows = static_cast<size_t>(rows_i64);
    const size_t cols = static_cast<size_t>(cols_i64);
    size_t count = 0;
    if (!page_packed_b) {
        if (!checked_mul(rows, cols, &count)) {
            return false;
        }
    } else {
        size_t block_elements = 0;
        size_t block_bytes = 0;
        if (!checked_mul(expected.dim, expected.dim, &block_elements) ||
                !checked_mul(block_elements, expected.elem_size, &block_bytes) ||
                block_bytes == 0 || expected.page_size % block_bytes != 0) {
            return false;
        }
        const size_t blocks_per_page = expected.page_size / block_bytes;
        if (blocks_per_page == 0 || expected.page_size % expected.elem_size != 0) {
            return false;
        }

        size_t k_blocks = 0;
        size_t j_blocks = 0;
        size_t j_pages = 0;
        size_t pages = 0;
        size_t elements_per_page = expected.page_size / expected.elem_size;
        if (!checked_ceil_div(rows, expected.dim, &k_blocks) ||
                !checked_ceil_div(cols, expected.dim, &j_blocks) ||
                !checked_ceil_div(j_blocks, blocks_per_page, &j_pages) ||
                !checked_mul(k_blocks, j_pages, &pages) ||
                !checked_mul(pages, elements_per_page, &count)) {
            return false;
        }
    }

    if (serialized_count != static_cast<uint64_t>(count)) {
        return false;
    }
    *count_out = count;
    return true;
}

static ggml_gemmini_weight_pack_read_result fail(
        enum ggml_gemmini_weight_pack_read_status status,
        uint32_t version,
        const char * reason) {
    ggml_gemmini_weight_pack_read_result result;
    result.status = status;
    result.version = version;
    result.reason = reason;
    return result;
}

} // namespace

ggml_gemmini_weight_pack_read_result ggml_gemmini_weight_pack_read_v3(
        std::istream & input,
        const ggml_gemmini_weight_pack_expected_geometry & expected,
        ggml_gemmini_weight_pack_allocate_entry_fn allocate_entry,
        void * context) {
    static constexpr uint64_t kMaxPackEntries = UINT64_C(1) << 20;
    static constexpr uint32_t kMaxPackNameBytes = UINT32_C(1) << 20;

    try {
        if (expected.elem_size != sizeof(uint16_t)) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_BAD_HEADER, 0,
                "v3 reader requires 16-bit BF16 elements");
        }
        ggml_gemmini_weight_pack_file_header header = {};
        if (!read_pod(input, header)) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, 0,
                "truncated weight-pack header");
        }
        if (std::memcmp(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC,
                        GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE) != 0 ||
                header.endian != GGML_GEMMINI_WEIGHT_PACK_ENDIAN) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_BAD_HEADER, header.version,
                "weight-pack magic or endian marker mismatch");
        }
        if (header.version == GGML_GEMMINI_WEIGHT_PACK_VERSION_V1 ||
                header.version == GGML_GEMMINI_WEIGHT_PACK_VERSION_V2) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_LEGACY_VERSION, header.version,
                "legacy INT8 weight pack");
        }
        if (header.version != GGML_GEMMINI_WEIGHT_PACK_VERSION_V3 &&
                header.version != GGML_GEMMINI_WEIGHT_PACK_VERSION_V4) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_UNSUPPORTED_VERSION, header.version,
                "unsupported weight-pack version");
        }
        if (header.elem_size != expected.elem_size || header.scale_size != 0 ||
                header.entry_count > kMaxPackEntries) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_BAD_HEADER, header.version,
                "invalid v3 header geometry or entry count");
        }

        uint32_t numeric_format = 0;
        uint32_t profile_id = 0;
        uint32_t dim = 0;
        uint32_t page_size = 0;
        uint32_t max_bytes = 0;
        uint32_t b_layout = 0;
        uint32_t pack_flags = 0;
        uint64_t model_fingerprint = 0;
        if (header.version == GGML_GEMMINI_WEIGHT_PACK_VERSION_V4) {
            ggml_gemmini_weight_pack_geometry_v4 geometry = {};
            if (!read_pod(input, geometry)) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, header.version,
                    "truncated v4 geometry");
            }
            if (geometry.reserved != 0 ||
                    (geometry.flags & ~GGML_GEMMINI_WEIGHT_PACK_FLAG_HYBRID_SPARSE_GGUF) != 0) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_BAD_HEADER, header.version,
                    "invalid v4 flags or reserved field");
            }
            numeric_format = geometry.numeric_format;
            profile_id = geometry.profile_id;
            dim = geometry.dim;
            page_size = geometry.page_size;
            max_bytes = geometry.max_bytes;
            b_layout = geometry.b_layout;
            pack_flags = geometry.flags;
            model_fingerprint = geometry.model_fingerprint;
        } else {
            ggml_gemmini_weight_pack_geometry_v3 geometry = {};
            if (!read_pod(input, geometry)) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, header.version,
                    "truncated v3 geometry");
            }
            numeric_format = geometry.numeric_format;
            profile_id = geometry.profile_id;
            dim = geometry.dim;
            page_size = geometry.page_size;
            max_bytes = geometry.max_bytes;
            b_layout = geometry.b_layout;
        }
        if (numeric_format != expected.numeric_format ||
                profile_id != expected.profile_id ||
                dim != expected.dim ||
                page_size != expected.page_size ||
                max_bytes != expected.max_bytes ||
                (b_layout != GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR &&
                 b_layout != GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B)) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_PROFILE_OR_GEOMETRY_MISMATCH,
                header.version, "profile or geometry mismatch");
        }
        const bool page_packed_b =
            b_layout == GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B;

        for (uint64_t index = 0; index < header.entry_count; ++index) {
            ggml_gemmini_weight_pack_entry_header_v3 entry = {};
            if (!read_pod(input, entry)) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, header.version,
                    "truncated v3 entry header");
            }
            const uint32_t role = entry.reserved & GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_MASK;
            const bool valid_reserved = header.version == GGML_GEMMINI_WEIGHT_PACK_VERSION_V4
                ? (entry.reserved == GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY ||
                   entry.reserved == GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_SHARED)
                : entry.reserved == 0;
            if (entry.name_size == 0 || entry.name_size > kMaxPackNameBytes ||
                    !valid_reserved || entry.slice_i2 < 0 || entry.slice_i3 < 0 ||
                    entry.cols_in <= 0 || entry.cols_out <= 0) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_INVALID_ENTRY, header.version,
                    "invalid v3 entry metadata");
            }

            size_t data_count = 0;
            if (!checked_b_data_count(entry.cols_in, entry.cols_out, page_packed_b,
                                      expected, entry.data_count, &data_count)) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_SIZE_OVERFLOW, header.version,
                    "invalid or overflowing v3 B storage size");
            }

            std::string name(entry.name_size, '\0');
            if (!read_exact(input, name.data(), name.size())) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, header.version,
                    "truncated v3 entry name");
            }

            ggml_gemmini_weight_pack_v3_entry_descriptor descriptor{
                std::move(name),
                entry.slice_i2,
                entry.slice_i3,
                entry.cols_in,
                entry.cols_out,
                data_count,
                page_packed_b,
                role,
            };
            std::string callback_reason;
            uint16_t * destination = allocate_entry == nullptr ? nullptr :
                allocate_entry(context, descriptor, callback_reason);
            if (destination == nullptr && data_count != 0) {
                ggml_gemmini_weight_pack_read_result result = fail(
                    GGML_GEMMINI_WEIGHT_PACK_READ_ENTRY_REJECTED, header.version,
                    "entry allocator rejected v3 entry");
                if (!callback_reason.empty()) {
                    result.reason = std::move(callback_reason);
                }
                return result;
            }

            size_t data_bytes = 0;
            if (!checked_mul(data_count, expected.elem_size, &data_bytes)) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_SIZE_OVERFLOW, header.version,
                    "v3 payload byte size overflow");
            }
            if (!read_exact(input, destination, data_bytes)) {
                return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, header.version,
                    "truncated v3 BF16 payload");
            }
        }

        char extra = 0;
        if (input.read(&extra, 1)) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_TRAILING_DATA, header.version,
                "trailing data after v3 entries");
        }
        if (!input.eof()) {
            return fail(GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR, header.version,
                "I/O error while checking v3 end of file");
        }

        ggml_gemmini_weight_pack_read_result result;
        result.status = GGML_GEMMINI_WEIGHT_PACK_READ_OK;
        result.version = header.version;
        result.entry_count = header.entry_count;
        result.page_packed_b = page_packed_b;
        result.hybrid_sparse_gguf =
            (pack_flags & GGML_GEMMINI_WEIGHT_PACK_FLAG_HYBRID_SPARSE_GGUF) != 0;
        result.model_fingerprint = model_fingerprint;
        return result;
    } catch (const std::bad_alloc &) {
        return fail(GGML_GEMMINI_WEIGHT_PACK_READ_RESOURCE_ERROR, 0,
            "memory allocation failed while reading weight pack");
    } catch (const std::length_error &) {
        return fail(GGML_GEMMINI_WEIGHT_PACK_READ_SIZE_OVERFLOW, 0,
            "container size overflow while reading weight pack");
    } catch (const std::exception & error) {
        ggml_gemmini_weight_pack_read_result result = fail(
            GGML_GEMMINI_WEIGHT_PACK_READ_RESOURCE_ERROR, 0,
            "exception while reading weight pack");
        result.reason = error.what();
        return result;
    }
}

const char * ggml_gemmini_weight_pack_read_status_name(
        enum ggml_gemmini_weight_pack_read_status status) {
    switch (status) {
        case GGML_GEMMINI_WEIGHT_PACK_READ_OK: return "ok";
        case GGML_GEMMINI_WEIGHT_PACK_READ_IO_ERROR: return "io_error";
        case GGML_GEMMINI_WEIGHT_PACK_READ_BAD_HEADER: return "bad_header";
        case GGML_GEMMINI_WEIGHT_PACK_READ_LEGACY_VERSION: return "legacy_version";
        case GGML_GEMMINI_WEIGHT_PACK_READ_UNSUPPORTED_VERSION: return "unsupported_version";
        case GGML_GEMMINI_WEIGHT_PACK_READ_PROFILE_OR_GEOMETRY_MISMATCH: return "profile_or_geometry_mismatch";
        case GGML_GEMMINI_WEIGHT_PACK_READ_INVALID_ENTRY: return "invalid_entry";
        case GGML_GEMMINI_WEIGHT_PACK_READ_SIZE_OVERFLOW: return "size_overflow";
        case GGML_GEMMINI_WEIGHT_PACK_READ_ENTRY_REJECTED: return "entry_rejected";
        case GGML_GEMMINI_WEIGHT_PACK_READ_TRAILING_DATA: return "trailing_data";
        case GGML_GEMMINI_WEIGHT_PACK_READ_RESOURCE_ERROR: return "resource_error";
    }
    return "unknown";
}
