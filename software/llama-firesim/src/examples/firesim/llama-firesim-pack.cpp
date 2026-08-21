#include "ggml-gemmini-bf16.h"
#include "ggml-gemmini-pack.h"
#include "ggml-gemmini-page-packing.h"
#include "ggml-gemmini-profile.h"
#include "ggml-gemmini-weight-select.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "include/gemmini_page_packed.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
uint8_t llama_internal_get_tensor_usage(const llama_model * model, const char * name);

namespace fs = std::filesystem;

namespace {

static std::string tensor_name(const ggml_tensor * t);

struct options {
    std::string model_path;
    std::string output_path;
    bool page_packed_b = false;
    bool hybrid_sparse_gguf = false;
    std::string thin_model_output_path;
};

static fs::path normalized_path(const std::string & path) {
    return fs::weakly_canonical(fs::absolute(fs::path(path)));
}

static bool paths_refer_to_same_file(
        const std::string & lhs,
        const std::string & rhs) {
    if (normalized_path(lhs) == normalized_path(rhs)) {
        return true;
    }
    std::error_code ec;
    return fs::exists(lhs, ec) && !ec && fs::exists(rhs, ec) && !ec &&
        fs::equivalent(lhs, rhs, ec) && !ec;
}

static std::string make_temporary_path(const std::string & final_path) {
    const fs::path final = fs::absolute(fs::path(final_path));
    fs::create_directories(final.parent_path());
    std::string pattern =
        (final.parent_path() / ("." + final.filename().string() + ".tmp.XXXXXX")).string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int fd = ::mkstemp(writable.data());
    if (fd < 0) {
        throw std::runtime_error("failed to create temporary artifact beside: " + final_path);
    }
    if (::close(fd) != 0) {
        const std::string tmp(writable.data());
        fs::remove(tmp);
        throw std::runtime_error("failed to close temporary artifact: " + tmp);
    }
    return std::string(writable.data());
}

static void fsync_path(const std::string & path, bool directory = false) {
    const int flags = O_RDONLY
#if defined(O_DIRECTORY)
        | (directory ? O_DIRECTORY : 0)
#endif
        ;
    const int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        throw std::runtime_error("failed to open artifact for fsync: " + path);
    }
    const int sync_status = ::fsync(fd);
    const int close_status = ::close(fd);
    if (sync_status != 0 || close_status != 0) {
        throw std::runtime_error("failed to fsync artifact: " + path);
    }
}

static void publish_artifact(
        const std::string & temporary_path,
        const std::string & final_path) {
    fsync_path(temporary_path);
    // Both paths are deliberately created in the same directory.  POSIX
    // rename replaces an old artifact atomically; never unlink the last known
    // good file before the replacement is durable.
    fs::rename(temporary_path, final_path);
    const fs::path parent = fs::absolute(fs::path(final_path)).parent_path();
    fsync_path(parent.string(), true);
}

static uint64_t allocated_file_bytes(const std::string & path) {
    struct stat status = {};
    if (::stat(path.c_str(), &status) != 0 || status.st_blocks < 0) {
        throw std::runtime_error("failed to stat artifact allocation: " + path);
    }
    return static_cast<uint64_t>(status.st_blocks) * UINT64_C(512);
}

static enum ggml_gemmini_weight_role tensor_role(
        const llama_model * model,
        const ggml_tensor * tensor) {
    const std::string name = tensor_name(tensor);
    const unsigned usage = llama_internal_get_tensor_usage(model, name.c_str());
    if (usage == 0 || (usage & ~(GGML_GEMMINI_WEIGHT_USAGE_ORIGINAL |
                                GGML_GEMMINI_WEIGHT_USAGE_MUL_MAT)) != 0) {
        throw std::runtime_error("missing or invalid operator-role metadata for tensor: " + name);
    }
    return ggml_gemmini_classify_weight_usage(usage);
}

static bool parse_bool_value(const std::string & value) {
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON") {
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off" || value == "OFF") {
        return false;
    }
    throw std::runtime_error("invalid boolean: " + value);
}

static size_t packed_b_storage_bytes(size_t rows, size_t cols) {
    const auto checked_add = [](size_t lhs, size_t rhs, const char * what) {
        if (rhs > std::numeric_limits<size_t>::max() - lhs) {
            throw std::overflow_error(what);
        }
        return lhs + rhs;
    };
    const auto checked_mul = [](size_t lhs, size_t rhs, const char * what) {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
            throw std::overflow_error(what);
        }
        return lhs * rhs;
    };
    const auto checked_ceil_div = [&checked_add](size_t numerator, size_t denominator, const char * what) {
        if (denominator == 0) {
            throw std::overflow_error(what);
        }
        return checked_add(numerator, denominator - 1, what) / denominator;
    };

    const size_t blocks_per_page = gemmini_page_packed_b_j_blocks_per_page();
    if (blocks_per_page == 0) {
        throw std::overflow_error("packed B geometry has zero blocks per page");
    }
    const size_t k_blocks = checked_ceil_div(rows, DIM, "packed B K block count overflow");
    const size_t j_blocks = checked_ceil_div(cols, DIM, "packed B J block count overflow");
    const size_t j_pages = checked_ceil_div(
        j_blocks, blocks_per_page, "packed B J page count overflow");
    const size_t pages = checked_mul(k_blocks, j_pages, "packed B page count overflow");
    if (pages > std::numeric_limits<size_t>::max() / GEMMINI_PAGE_PACKED_PAGE_BYTES) {
        throw std::overflow_error("packed B byte size overflow");
    }
    return pages * GEMMINI_PAGE_PACKED_PAGE_BYTES;
}

static size_t row_major_b_storage_bytes(size_t rows, size_t cols) {
    if (rows != 0 && cols > std::numeric_limits<size_t>::max() / rows) {
        throw std::overflow_error("row-major B element count overflow");
    }
    const size_t elements = rows * cols;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(elem_t)) {
        throw std::overflow_error("row-major B byte size overflow");
    }
    return elements * sizeof(elem_t);
}

static std::string tensor_name(const ggml_tensor * t) {
    if (t == nullptr) {
        return {};
    }

    const char * name = ggml_get_name(t);
    return name == nullptr ? std::string() : std::string(name);
}

static bool tensor_type_can_load_to_float(enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits != nullptr && traits->to_float != nullptr;
}

static bool can_pack_weight_tensor(
        const ggml_tensor * tensor,
        enum ggml_gemmini_weight_role role) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    if (!ggml_gemmini_weight_role_requires_pack(role)) {
        return false;
    }

    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
            tensor->ne[2] <= 0 || tensor->ne[3] <= 0) {
        return false;
    }

    if (!tensor_type_can_load_to_float(tensor->type)) {
        return false;
    }

    if (tensor->nb[0] != ggml_type_size(tensor->type)) {
        return false;
    }

    return true;
}

static bool should_emit_weight_pack_tensor(
        const ggml_tensor * tensor,
        bool file_page_packed_b,
        enum ggml_gemmini_weight_role role) {
    return can_pack_weight_tensor(tensor, role) &&
        ggml_gemmini_weight_pack_should_emit_entry(
            file_page_packed_b,
            static_cast<uint64_t>(tensor->ne[0]));
}

static void encode_and_store_b_output_row(
        const float * src,
        size_t cols_in,
        size_t cols_out,
        size_t out_col,
        bool page_packed_b,
        elem_t * dst) {
    if (!page_packed_b) {
        for (size_t k = 0; k < cols_in; ++k) {
            dst[k * cols_out + out_col] = static_cast<elem_t>(
                ggml_gemmini_float_to_bf16(src[k]));
        }
        return;
    }

    const size_t j_block = out_col / DIM;
    const size_t j_in_block = out_col % DIM;
    const size_t packed_row_stride = gemmini_page_packed_b_j_blocks_per_page() * DIM;
    const size_t k_blocks = gemmini_ceil_div_size(cols_in, DIM);
    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        elem_t * block = gemmini_page_packed_b_block_addr_mut(
            dst, k_block, j_block, cols_out);
        const size_t valid_rows = std::min(
            static_cast<size_t>(DIM), cols_in - k_block * DIM);
        for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
            const size_t k = k_block * DIM + k_in_block;
            block[k_in_block * packed_row_stride + j_in_block] =
                static_cast<elem_t>(ggml_gemmini_float_to_bf16(src[k]));
        }
    }
}

static void copy_and_store_b_output_row(
        const elem_t * src,
        size_t cols_in,
        size_t cols_out,
        size_t out_col,
        bool page_packed_b,
        elem_t * dst) {
    if (!page_packed_b) {
        for (size_t k = 0; k < cols_in; ++k) {
            dst[k * cols_out + out_col] = src[k];
        }
        return;
    }

    const size_t j_block = out_col / DIM;
    const size_t j_in_block = out_col % DIM;
    const size_t packed_row_stride = gemmini_page_packed_b_j_blocks_per_page() * DIM;
    const size_t k_blocks = gemmini_ceil_div_size(cols_in, DIM);
    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        elem_t * block = gemmini_page_packed_b_block_addr_mut(
            dst, k_block, j_block, cols_out);
        const size_t valid_rows = std::min(
            static_cast<size_t>(DIM), cols_in - k_block * DIM);
        for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
            block[k_in_block * packed_row_stride + j_in_block] =
                src[k_block * DIM + k_in_block];
        }
    }
}

static void tensor_row_to_float(
        const ggml_tensor * t,
        const uint8_t * row_ptr,
        int64_t cols,
        std::vector<float> & tmp) {
    tmp.resize(cols);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(tmp.data(), row_ptr, cols * sizeof(float));
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error("tensor type cannot be converted to float");
    }
    traits->to_float(row_ptr, tmp.data(), cols);
}

template <typename T>
static void write_pod(std::ostream & out, const T & value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("failed to write weight pack");
    }
}

static void write_bytes(std::ostream & out, const void * data, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error("failed to write weight pack");
    }
}

struct packed_slice {
    std::string name;
    int64_t slice_i2 = 0;
    int64_t slice_i3 = 0;
    int64_t cols_in = 0;
    int64_t cols_out = 0;
    enum ggml_gemmini_weight_role role = GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY;
    std::vector<elem_t> data;
};

static packed_slice pack_slice(
        const ggml_tensor * tensor,
        const std::string & name,
        const uint8_t * slice_ptr,
        int64_t slice_i2,
        int64_t slice_i3,
        bool page_packed_b,
        enum ggml_gemmini_weight_role role) {
    const int64_t cols_in = tensor->ne[0];
    const int64_t cols_out = tensor->ne[1];

    packed_slice slice;
    slice.name = name;
    slice.slice_i2 = slice_i2;
    slice.slice_i3 = slice_i3;
    slice.cols_in = cols_in;
    slice.cols_out = cols_out;
    slice.role = role;
    const size_t data_bytes = page_packed_b
        ? packed_b_storage_bytes(static_cast<size_t>(cols_in), static_cast<size_t>(cols_out))
        : row_major_b_storage_bytes(
            static_cast<size_t>(cols_in), static_cast<size_t>(cols_out));
    slice.data.assign(data_bytes / sizeof(elem_t), 0);

    std::vector<float> tmp;
    for (int64_t out_col = 0; out_col < cols_out; ++out_col) {
        const uint8_t * row_ptr = slice_ptr + out_col * tensor->nb[1];
        if (tensor->type == GGML_TYPE_BF16) {
            copy_and_store_b_output_row(
                reinterpret_cast<const elem_t *>(row_ptr),
                static_cast<size_t>(cols_in),
                static_cast<size_t>(cols_out),
                static_cast<size_t>(out_col),
                page_packed_b,
                slice.data.data());
        } else {
            tensor_row_to_float(tensor, row_ptr, cols_in, tmp);
            encode_and_store_b_output_row(
                tmp.data(),
                static_cast<size_t>(cols_in),
                static_cast<size_t>(cols_out),
                static_cast<size_t>(out_col),
                page_packed_b,
                slice.data.data());
        }
    }

    return slice;
}

static void write_slice(std::ostream & out, const packed_slice & slice) {
    if (slice.name.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("tensor name too long for weight pack");
    }

    ggml_gemmini_weight_pack_entry_header_v3 header = {};
    header.name_size = static_cast<uint32_t>(slice.name.size());
    header.reserved = slice.role == GGML_GEMMINI_WEIGHT_ROLE_SHARED
        ? GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_SHARED
        : GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY;
    header.slice_i2 = slice.slice_i2;
    header.slice_i3 = slice.slice_i3;
    header.cols_in = slice.cols_in;
    header.cols_out = slice.cols_out;
    header.data_count = slice.data.size();

    write_pod(out, header);
    write_bytes(out, slice.name.data(), slice.name.size());
    write_bytes(out, slice.data.data(), slice.data.size() * sizeof(elem_t));
}

static void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " --model MODEL.gguf --output MODEL.gguf.gemmini-pack"
              << " [--page-packed-b BOOL] [--hybrid-sparse-gguf BOOL]\n";
}

static options parse_args(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_path = argv[++i];
        } else if (arg == "--page-packed-b" && i + 1 < argc) {
            opts.page_packed_b = parse_bool_value(argv[++i]);
        } else if (arg == "--hybrid-sparse-gguf" && i + 1 < argc) {
            opts.hybrid_sparse_gguf = parse_bool_value(argv[++i]);
        } else if (arg == "--thin-model-output" && i + 1 < argc) {
            opts.thin_model_output_path = argv[++i];
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("invalid command line");
        }
    }

    if (opts.model_path.empty() || opts.output_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--model and --output are required");
    }
    if (opts.hybrid_sparse_gguf && opts.thin_model_output_path.empty()) {
        throw std::runtime_error("--hybrid-sparse-gguf requires --thin-model-output");
    }
    if (!opts.thin_model_output_path.empty()) {
        opts.hybrid_sparse_gguf = true;
    }
    if (paths_refer_to_same_file(opts.model_path, opts.output_path)) {
        throw std::runtime_error("weight-pack output must differ from the source GGUF");
    }
    if (!opts.thin_model_output_path.empty() &&
            paths_refer_to_same_file(opts.model_path, opts.thin_model_output_path)) {
        throw std::runtime_error("thin-model output must differ from the source GGUF");
    }
    if (!opts.thin_model_output_path.empty() &&
            paths_refer_to_same_file(opts.output_path, opts.thin_model_output_path)) {
        throw std::runtime_error("weight-pack and thin-model outputs must differ");
    }

    return opts;
}

static uint64_t fingerprint_file(const std::string & path) {
    static constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
    static constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open model for fingerprint: " + path);
    }
    std::array<char, 1u << 20> buffer = {};
    uint64_t hash = kFnvOffset;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<uint8_t>(buffer[static_cast<size_t>(i)]);
            hash *= kFnvPrime;
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while fingerprinting model: " + path);
    }
    return hash;
}

static void reject_split_gguf(const std::string & path) {
    gguf_init_params params = {};
    params.no_alloc = true;
    gguf_context * context = gguf_init_from_file(path.c_str(), params);
    if (context == nullptr) {
        throw std::runtime_error("failed to inspect GGUF split metadata: " + path);
    }
    const int64_t split_key = gguf_find_key(context, "split.count");
    const uint16_t split_count = split_key < 0 ? 0 : gguf_get_val_u16(context, split_key);
    gguf_free(context);
    if (split_count > 1) {
        throw std::runtime_error(
            "hybrid Gemmini artifacts do not yet support split GGUF models");
    }
}

struct file_range {
    uint64_t offset = 0;
    uint64_t size = 0;
};

static void copy_file_range(
        std::ifstream & source,
        std::fstream & destination,
        uint64_t offset,
        uint64_t size,
        std::vector<char> & buffer) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(offset));
    destination.seekp(static_cast<std::streamoff>(offset));
    if (!source || !destination) {
        throw std::runtime_error("failed to seek while creating sparse GGUF");
    }
    while (size != 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(size, buffer.size()));
        source.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (source.gcount() != static_cast<std::streamsize>(chunk)) {
            throw std::runtime_error("short read while creating sparse GGUF");
        }
        destination.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!destination) {
            throw std::runtime_error("write failed while creating sparse GGUF");
        }
        size -= chunk;
    }
}

static uint64_t write_sparse_hybrid_model(
        const options & opts,
        const llama_model * model,
        uint64_t model_fingerprint,
        const std::string & temporary_path) {
    if (opts.thin_model_output_path.empty()) {
        return 0;
    }

    gguf_init_params params = {};
    params.no_alloc = true;
    gguf_context * gguf = gguf_init_from_file(opts.model_path.c_str(), params);
    if (gguf == nullptr) {
        throw std::runtime_error("failed to read source GGUF metadata for sparse output");
    }

    std::vector<file_range> retained;
    retained.push_back({0, static_cast<uint64_t>(gguf_get_data_offset(gguf))});
    uint64_t removed_bytes = 0;
    const int64_t tensor_count = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < tensor_count; ++i) {
        const char * raw_name = gguf_get_tensor_name(gguf, i);
        const std::string name = raw_name == nullptr ? std::string() : std::string(raw_name);
        const unsigned usage = llama_internal_get_tensor_usage(model, name.c_str());
        const enum ggml_gemmini_weight_role role = usage == 0
            ? GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY
            : ggml_gemmini_classify_weight_usage(usage);
        const uint64_t size = static_cast<uint64_t>(gguf_get_tensor_size(gguf, i));
        const uint64_t offset = static_cast<uint64_t>(gguf_get_data_offset(gguf)) +
            static_cast<uint64_t>(gguf_get_tensor_offset(gguf, i));
        if (ggml_gemmini_weight_role_requires_original(role)) {
            retained.push_back({offset, size});
        } else {
            removed_bytes += size;
        }
    }
    gguf_free(gguf);

    std::sort(retained.begin(), retained.end(), [](const file_range & lhs, const file_range & rhs) {
        return lhs.offset < rhs.offset;
    });
    std::vector<file_range> merged;
    for (const file_range & range : retained) {
        if (range.size == 0) {
            continue;
        }
        if (!merged.empty() && range.offset <= merged.back().offset + merged.back().size) {
            const uint64_t end = std::max(
                merged.back().offset + merged.back().size, range.offset + range.size);
            merged.back().size = end - merged.back().offset;
        } else {
            merged.push_back(range);
        }
    }

    const uint64_t source_size = static_cast<uint64_t>(fs::file_size(opts.model_path));
    const fs::path output_parent = fs::path(opts.thin_model_output_path).parent_path();
    if (!output_parent.empty()) {
        fs::create_directories(output_parent);
    }
    {
        std::ofstream create(temporary_path, std::ios::binary | std::ios::trunc);
        if (!create) {
            throw std::runtime_error("failed to create sparse GGUF: " + temporary_path);
        }
        create.close();
        if (!create) {
            throw std::runtime_error("failed to close sparse GGUF: " + temporary_path);
        }
    }
    fs::resize_file(temporary_path, source_size + sizeof(ggml_gemmini_hybrid_footer));

    std::ifstream source(opts.model_path, std::ios::binary);
    std::fstream destination(temporary_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!source || !destination) {
        throw std::runtime_error("failed to open sparse GGUF streams");
    }
    std::vector<char> buffer(1u << 20);
    for (const file_range & range : merged) {
        copy_file_range(source, destination, range.offset, range.size, buffer);
    }

    ggml_gemmini_hybrid_footer footer = {};
    std::memcpy(footer.magic, GGML_GEMMINI_HYBRID_FOOTER_MAGIC,
        GGML_GEMMINI_HYBRID_FOOTER_MAGIC_SIZE);
    footer.model_fingerprint = model_fingerprint;
    destination.seekp(static_cast<std::streamoff>(source_size));
    write_pod(destination, footer);
    destination.flush();
    if (!destination) {
        throw std::runtime_error("failed to finalize sparse GGUF");
    }
    destination.close();
    if (!destination) {
        throw std::runtime_error("failed to close sparse GGUF");
    }
    source.close();
    return removed_bytes;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_args(argc, argv);

        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                std::fprintf(stderr, "%s", text);
            }
        }, nullptr);

        ggml_backend_load_all();

        reject_split_gguf(opts.model_path);

        llama_model_params model_params = llama_model_default_params();
        llama_model * model = llama_model_load_from_file(opts.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load model: " + opts.model_path);
        }

        int64_t tensors = 0;
        int64_t omitted_k_le_1_tensors = 0;
        int64_t gemmini_only_tensors = 0;
        int64_t shared_tensors = 0;
        int64_t original_only_tensors = 0;
        uint64_t slice_count = 0;
        const auto & tensor_map = llama_internal_get_tensor_map(model);
        for (const auto & item : tensor_map) {
            const ggml_tensor * tensor = item.second;
            const enum ggml_gemmini_weight_role role = tensor_role(model, tensor);
            if (!ggml_gemmini_weight_role_requires_pack(role)) {
                ++original_only_tensors;
                continue;
            }
            if (!can_pack_weight_tensor(tensor, role)) {
                throw std::runtime_error(
                    "Gemmini MUL_MAT tensor cannot be represented by the BF16 pack: " +
                    tensor_name(tensor));
            }
            if (!should_emit_weight_pack_tensor(tensor, opts.page_packed_b, role)) {
                ++omitted_k_le_1_tensors;
                if (opts.hybrid_sparse_gguf) {
                    throw std::runtime_error(
                        "hybrid sparse mode cannot omit a Gemmini tensor: " + tensor_name(tensor));
                }
                continue;
            }

            if (role == GGML_GEMMINI_WEIGHT_ROLE_SHARED) {
                ++shared_tensors;
            } else {
                ++gemmini_only_tensors;
            }

            const uint64_t slices_i2 = static_cast<uint64_t>(tensor->ne[2]);
            const uint64_t slices_i3 = static_cast<uint64_t>(tensor->ne[3]);
            if (slices_i2 != 0 && slices_i3 > std::numeric_limits<uint64_t>::max() / slices_i2) {
                throw std::overflow_error("weight-pack slice count multiplication overflow");
            }
            const uint64_t tensor_slices = slices_i2 * slices_i3;
            if (tensor_slices > std::numeric_limits<uint64_t>::max() - slice_count) {
                throw std::overflow_error("weight-pack slice count addition overflow");
            }
            ++tensors;
            slice_count += tensor_slices;
        }

        const uint64_t model_fingerprint = fingerprint_file(opts.model_path);
        const std::string pack_temporary_path = make_temporary_path(opts.output_path);
        std::string thin_temporary_path;
        try {
            if (opts.hybrid_sparse_gguf) {
                thin_temporary_path = make_temporary_path(opts.thin_model_output_path);
            }
        } catch (...) {
            fs::remove(pack_temporary_path);
            llama_model_free(model);
            throw;
        }
        uint64_t removed_original_bytes = 0;
        uint64_t thin_logical_bytes = 0;
        uint64_t thin_allocated_bytes = 0;
        try {
            std::ofstream out(pack_temporary_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("failed to open output: " + pack_temporary_path);
            }

            ggml_gemmini_weight_pack_file_header header = {};
            std::memcpy(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC, GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE);
            header.version = GGML_GEMMINI_WEIGHT_PACK_VERSION;
            header.endian = GGML_GEMMINI_WEIGHT_PACK_ENDIAN;
            header.elem_size = sizeof(elem_t);
            header.scale_size = 0;
            header.entry_count = slice_count;

            write_pod(out, header);
            ggml_gemmini_weight_pack_geometry_v4 geometry = {};
            geometry.numeric_format = GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE;
            geometry.profile_id = GGML_GEMMINI_PROFILE_ID;
            geometry.dim = DIM;
            geometry.page_size = GEMMINI_PAGE_PACKED_PAGE_BYTES;
            geometry.max_bytes = MAX_BYTES;
            geometry.b_layout = opts.page_packed_b
                ? GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B
                : GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR;
            geometry.flags = opts.hybrid_sparse_gguf
                ? GGML_GEMMINI_WEIGHT_PACK_FLAG_HYBRID_SPARSE_GGUF
                : 0;
            geometry.model_fingerprint = model_fingerprint;
            write_pod(out, geometry);
            for (const auto & item : tensor_map) {
                const ggml_tensor * tensor = item.second;
                const enum ggml_gemmini_weight_role role = tensor_role(model, tensor);
                if (!should_emit_weight_pack_tensor(tensor, opts.page_packed_b, role)) {
                    continue;
                }

                const std::string name = tensor_name(tensor);
                const auto * base = reinterpret_cast<const uint8_t *>(tensor->data);
                for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                        const uint8_t * slice_ptr = base + i2 * tensor->nb[2] + i3 * tensor->nb[3];
                        write_slice(out, pack_slice(
                            tensor, name, slice_ptr, i2, i3, opts.page_packed_b, role));
                    }
                }
            }
            out.close();
            if (!out) {
                throw std::runtime_error("failed to close output: " + pack_temporary_path);
            }

            removed_original_bytes = write_sparse_hybrid_model(
                opts, model, model_fingerprint, thin_temporary_path);

            // Publish the thin model first and the matching pack last.  If the
            // process dies between renames, the runtime fingerprint check
            // rejects the mismatched pair instead of consuming sparse zeros.
            if (opts.hybrid_sparse_gguf) {
                thin_logical_bytes = static_cast<uint64_t>(fs::file_size(thin_temporary_path));
                thin_allocated_bytes = allocated_file_bytes(thin_temporary_path);
                static constexpr uint64_t kSparseAllocationSlop = UINT64_C(16) << 20;
                const uint64_t expected_allocated_upper_bound =
                    thin_logical_bytes - removed_original_bytes + kSparseAllocationSlop;
                if (removed_original_bytes > kSparseAllocationSlop &&
                        thin_allocated_bytes > expected_allocated_upper_bound) {
                    throw std::runtime_error(
                        "hybrid GGUF lost sparse holes while being generated");
                }
                publish_artifact(thin_temporary_path, opts.thin_model_output_path);
            }
            publish_artifact(pack_temporary_path, opts.output_path);
        } catch (...) {
            fs::remove(pack_temporary_path);
            if (!thin_temporary_path.empty()) {
                fs::remove(thin_temporary_path);
            }
            llama_model_free(model);
            throw;
        }
        llama_model_free(model);

        std::cout << "GEMMINI-WEIGHT-PACK-WROTE,path=" << opts.output_path
                  << ",tensors=" << tensors
                  << ",slices=" << slice_count
                  << ",omitted_k_le_1_tensors=" << omitted_k_le_1_tensors
                  << ",gemmini_only_tensors=" << gemmini_only_tensors
                  << ",shared_tensors=" << shared_tensors
                  << ",original_only_tensors=" << original_only_tensors
                  << ",hybrid_sparse_gguf=" << (opts.hybrid_sparse_gguf ? 1 : 0)
                  << ",removed_original_bytes=" << removed_original_bytes
                  << ",thin_logical_bytes=" << thin_logical_bytes
                  << ",thin_allocated_bytes=" << thin_allocated_bytes
                  << ",model_fingerprint=0x" << std::hex << model_fingerprint << std::dec
                  << ",page_packed_b=" << (opts.page_packed_b ? 1 : 0)
                  << ",numeric_format=bf16_rne"
                  << ",profile=" << GGML_GEMMINI_PROFILE_NAME
                  << ",profile_id=" << GGML_GEMMINI_PROFILE_ID
                  << ",dim=" << DIM
                  << ",page_size=" << GEMMINI_PAGE_PACKED_PAGE_BYTES
                  << ",max_bytes=" << MAX_BYTES << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "llama-firesim-pack: " << e.what() << "\n";
        return 1;
    }
}
