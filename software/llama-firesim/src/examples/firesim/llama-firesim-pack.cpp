#include "ggml-gemmini-pack.h"
#include "ggml.h"
#include "llama.h"
#include "include/gemmini_params.h"
#include "include/gemmini_page_packed.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

namespace fs = std::filesystem;

namespace {

struct options {
    std::string model_path;
    std::string output_path;
    bool page_packed_b = false;
};

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
    const size_t pages = gemmini_page_packed_b_page_count(rows, cols);
    if (pages > std::numeric_limits<size_t>::max() / GEMMINI_PAGE_PACKED_PAGE_BYTES) {
        throw std::overflow_error("packed B byte size overflow");
    }
    return pages * GEMMINI_PAGE_PACKED_PAGE_BYTES;
}

static bool contains_any(const std::string & text, std::initializer_list<const char *> needles) {
    for (const char * needle : needles) {
        if (text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool is_gemmini_model_weight_name(const std::string & name) {
    return contains_any(name, {
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "token_embd.weight",
        ".wq",
        ".wk",
        ".wv",
        ".wo",
    });
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

static bool can_pack_weight_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    if (!is_gemmini_model_weight_name(tensor_name(tensor))) {
        return false;
    }

    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
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

static elem_t quantize_value_to_i8(float value, float scale) {
    float q = std::round(value / scale);
    q = std::max(-128.0f, std::min(127.0f, q));
    return static_cast<elem_t>(q);
}

static float quantize_and_store_b_output_row(
        const float * src,
        size_t cols_in,
        size_t cols_out,
        size_t out_col,
        bool page_packed_b,
        elem_t * dst) {
    float max_abs = 0.0f;
    for (size_t k = 0; k < cols_in; ++k) {
        max_abs = std::max(max_abs, std::fabs(src[k]));
    }
    if (max_abs == 0.0f) {
        return 1.0f;
    }
    const float scale = max_abs / 127.0f;

    if (!page_packed_b) {
        for (size_t k = 0; k < cols_in; ++k) {
            dst[k * cols_out + out_col] = quantize_value_to_i8(src[k], scale);
        }
        return scale;
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
                quantize_value_to_i8(src[k], scale);
        }
    }
    return scale;
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
    std::vector<float> output_scales;
    std::vector<elem_t> data;
};

static packed_slice pack_slice(
        const ggml_tensor * tensor,
        const std::string & name,
        const uint8_t * slice_ptr,
        int64_t slice_i2,
        int64_t slice_i3,
        bool page_packed_b) {
    const int64_t cols_in = tensor->ne[0];
    const int64_t cols_out = tensor->ne[1];

    packed_slice slice;
    slice.name = name;
    slice.slice_i2 = slice_i2;
    slice.slice_i3 = slice_i3;
    slice.cols_in = cols_in;
    slice.cols_out = cols_out;
    slice.output_scales.resize(cols_out, 1.0f);
    const size_t data_bytes = page_packed_b
        ? packed_b_storage_bytes(static_cast<size_t>(cols_in), static_cast<size_t>(cols_out))
        : static_cast<size_t>(cols_in) * static_cast<size_t>(cols_out) * sizeof(elem_t);
    slice.data.assign(data_bytes / sizeof(elem_t), 0);

    std::vector<float> tmp;
    for (int64_t out_col = 0; out_col < cols_out; ++out_col) {
        const uint8_t * row_ptr = slice_ptr + out_col * tensor->nb[1];
        tensor_row_to_float(tensor, row_ptr, cols_in, tmp);
        slice.output_scales[out_col] = quantize_and_store_b_output_row(
            tmp.data(),
            static_cast<size_t>(cols_in),
            static_cast<size_t>(cols_out),
            static_cast<size_t>(out_col),
            page_packed_b,
            slice.data.data());
    }

    return slice;
}

static void write_slice(std::ostream & out, const packed_slice & slice) {
    if (slice.name.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("tensor name too long for weight pack");
    }

    ggml_gemmini_weight_pack_entry_header header = {};
    header.name_size = static_cast<uint32_t>(slice.name.size());
    header.slice_i2 = slice.slice_i2;
    header.slice_i3 = slice.slice_i3;
    header.cols_in = slice.cols_in;
    header.cols_out = slice.cols_out;
    header.scale_count = slice.output_scales.size();
    header.data_count = slice.data.size();

    write_pod(out, header);
    write_bytes(out, slice.name.data(), slice.name.size());
    write_bytes(out, slice.output_scales.data(), slice.output_scales.size() * sizeof(float));
    write_bytes(out, slice.data.data(), slice.data.size() * sizeof(elem_t));
}

static void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " --model MODEL.gguf --output MODEL.gguf.gemmini-pack"
              << " [--page-packed-b BOOL]\n";
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
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("invalid command line");
        }
    }

    if (opts.model_path.empty() || opts.output_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--model and --output are required");
    }

    return opts;
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

        llama_model_params model_params = llama_model_default_params();
        llama_model * model = llama_model_load_from_file(opts.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load model: " + opts.model_path);
        }

        int64_t tensors = 0;
        uint64_t slice_count = 0;
        const auto & tensor_map = llama_internal_get_tensor_map(model);
        for (const auto & item : tensor_map) {
            const ggml_tensor * tensor = item.second;
            if (!can_pack_weight_tensor(tensor)) {
                continue;
            }

            ++tensors;
            slice_count += static_cast<uint64_t>(tensor->ne[2]) * static_cast<uint64_t>(tensor->ne[3]);
        }

        const fs::path output_parent = fs::path(opts.output_path).parent_path();
        if (!output_parent.empty()) {
            fs::create_directories(output_parent);
        }
        const std::string tmp_path = opts.output_path + ".tmp";
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open output: " + tmp_path);
        }

        ggml_gemmini_weight_pack_file_header header = {};
        std::memcpy(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC, GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE);
        header.version = GGML_GEMMINI_WEIGHT_PACK_VERSION;
        header.endian = GGML_GEMMINI_WEIGHT_PACK_ENDIAN;
        header.elem_size = sizeof(elem_t);
        header.scale_size = sizeof(float);
        header.entry_count = slice_count;

        write_pod(out, header);
        ggml_gemmini_weight_pack_geometry_v2 geometry = {};
        geometry.dim = DIM;
        geometry.page_size = GEMMINI_PAGE_PACKED_PAGE_BYTES;
        geometry.max_bytes = MAX_BYTES;
        geometry.b_layout = opts.page_packed_b
            ? GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B
            : GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR;
        write_pod(out, geometry);
        for (const auto & item : tensor_map) {
            const ggml_tensor * tensor = item.second;
            if (!can_pack_weight_tensor(tensor)) {
                continue;
            }

            const std::string name = tensor_name(tensor);
            const auto * base = reinterpret_cast<const uint8_t *>(tensor->data);
            for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                    const uint8_t * slice_ptr = base + i2 * tensor->nb[2] + i3 * tensor->nb[3];
                    write_slice(out, pack_slice(
                        tensor, name, slice_ptr, i2, i3, opts.page_packed_b));
                }
            }
        }
        out.close();
        if (!out) {
            throw std::runtime_error("failed to close output: " + tmp_path);
        }

        fs::rename(tmp_path, opts.output_path);
        llama_model_free(model);

        std::cout << "GEMMINI-WEIGHT-PACK-WROTE,path=" << opts.output_path
                  << ",tensors=" << tensors
                  << ",slices=" << slice_count
                  << ",page_packed_b=" << (opts.page_packed_b ? 1 : 0)
                  << ",dim=" << DIM
                  << ",page_size=" << GEMMINI_PAGE_PACKED_PAGE_BYTES
                  << ",max_bytes=" << MAX_BYTES << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "llama-firesim-pack: " << e.what() << "\n";
        return 1;
    }
}
