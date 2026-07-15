#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-gemmini.h"
#include "ggml-gemmini-pack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" {
#include "include/gemmini_all.h"
}

namespace {

template <typename T, std::size_t Alignment>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() noexcept = default;

    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment> &) noexcept {}

    T * allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        void * ptr = nullptr;
        if (n != 0 && posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }

        return static_cast<T *>(ptr);
    }

    void deallocate(T * ptr, std::size_t) noexcept {
        std::free(ptr);
    }

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
static bool operator==(const aligned_allocator<T, Alignment> &, const aligned_allocator<U, Alignment> &) {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
static bool operator!=(const aligned_allocator<T, Alignment> &, const aligned_allocator<U, Alignment> &) {
    return false;
}

static constexpr std::size_t k_gemmini_dma_alignment = GEMMINI_PAGE_PACKED_PAGE_BYTES;

template <typename T>
using gemmini_aligned_vector = std::vector<T, aligned_allocator<T, k_gemmini_dma_alignment>>;

static uint64_t read_cycles_local() {
#if defined(__riscv)
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
#else
    return 0;
#endif
}

static bool env_flag(const char * name, bool fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }

    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0) {
        return false;
    }

    return true;
}

static int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    return std::strtol(value, nullptr, 0);
}

struct gemmini_page_packing_config {
    bool a = false;
    bool b = false;
    bool c = false;
    bool d = false;

    uint8_t mask() const {
        return static_cast<uint8_t>((a ? 1u : 0u) |
                                    (b ? 2u : 0u) |
                                    (c ? 4u : 0u) |
                                    (d ? 8u : 0u));
    }
};

static const gemmini_page_packing_config & gemmini_page_packing() {
    static const gemmini_page_packing_config config{
        env_flag("GGML_GEMMINI_PAGE_PACKED_A", false),
        env_flag("GGML_GEMMINI_PAGE_PACKED_B", false),
        env_flag("GGML_GEMMINI_PAGE_PACKED_C", false),
        env_flag("GGML_GEMMINI_PAGE_PACKED_D", false),
    };
    return config;
}

static size_t checked_mul_size(size_t lhs, size_t rhs, const char * what) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(std::string("Gemmini ") + what + " size overflow");
    }
    return lhs * rhs;
}

static size_t checked_page_storage_bytes(size_t pages, const char * what) {
    return checked_mul_size(pages, GEMMINI_PAGE_PACKED_PAGE_BYTES, what);
}

static size_t page_packed_a_storage_bytes(size_t rows, size_t cols) {
    return checked_page_storage_bytes(
        gemmini_page_packed_a_page_count(rows, cols), "packed A");
}

static size_t page_packed_b_storage_bytes(size_t rows, size_t cols) {
    return checked_page_storage_bytes(
        gemmini_page_packed_b_page_count(rows, cols), "packed B");
}

static size_t page_packed_acc_storage_bytes(
        size_t rows,
        size_t cols,
        size_t elem_size,
        const char * what) {
    return checked_page_storage_bytes(
        gemmini_page_packed_acc_page_count(rows, cols, elem_size), what);
}

static size_t row_major_storage_bytes(
        size_t rows,
        size_t cols,
        size_t elem_size,
        const char * what) {
    return checked_mul_size(checked_mul_size(rows, cols, what), elem_size, what);
}

static size_t a_storage_bytes(size_t rows, size_t cols, bool packed) {
    return packed ? page_packed_a_storage_bytes(rows, cols)
                  : row_major_storage_bytes(rows, cols, sizeof(elem_t), "A");
}

static size_t b_storage_bytes(size_t rows, size_t cols, bool packed) {
    return packed ? page_packed_b_storage_bytes(rows, cols)
                  : row_major_storage_bytes(rows, cols, sizeof(elem_t), "B");
}

static size_t acc_storage_bytes(size_t rows, size_t cols, bool packed, const char * what) {
    return packed ? page_packed_acc_storage_bytes(rows, cols, sizeof(acc_t), what)
                  : row_major_storage_bytes(rows, cols, sizeof(acc_t), what);
}

template <typename T>
static size_t storage_elements(size_t bytes, const char * what) {
    if (bytes % sizeof(T) != 0) {
        throw std::runtime_error(std::string("Gemmini ") + what + " byte size is not element-aligned");
    }
    return bytes / sizeof(T);
}

struct request_runtime_config {
    int active_mask = -1;
    int gemmini_id = -1;
    bool split_mode = false;
    bool single_mode = false;
};

static request_runtime_config & request_runtime() {
    static thread_local request_runtime_config config;
    return config;
}

static int gemmini_active_mask() {
    if (request_runtime().active_mask >= 0) {
        return request_runtime().active_mask & 0xf;
    }
    return env_int("GGML_GEMMINI_ACTIVE_MASK", 0xf) & 0xf;
}

static int popcount4(int value) {
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        count += (value >> i) & 1;
    }
    return count;
}

static int gemmini_max_offloads() {
    return env_int("GGML_GEMMINI_MAX_OFFLOADS", -1);
}

static int gemmini_tile_i() {
    return std::max(1, env_int("GGML_GEMMINI_TILE_I", 8));
}

static int gemmini_tile_j() {
    return std::max(1, env_int("GGML_GEMMINI_TILE_J", 4));
}

static int gemmini_tile_k() {
    return std::max(1, env_int("GGML_GEMMINI_TILE_K", 16));
}

static bool gemmini_use_counters() {
    return env_flag("GGML_GEMMINI_COUNTERS", false);
}

static void prefault_readonly_range(const void * ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0 || !env_flag("GGML_GEMMINI_PREFAULT", true)) {
        return;
    }

    constexpr size_t page_size = 4096;
    const volatile uint8_t * p = static_cast<const volatile uint8_t *>(ptr);
    uint8_t sink = 0;
    for (size_t offset = 0; offset < bytes; offset += page_size) {
        sink ^= p[offset];
    }
    sink ^= p[bytes - 1];
    (void) sink;
}

static void prefault_writable_range(void * ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0 || !env_flag("GGML_GEMMINI_PREFAULT", true)) {
        return;
    }

    constexpr size_t page_size = 4096;
    volatile uint8_t * p = static_cast<volatile uint8_t *>(ptr);
    for (size_t offset = 0; offset < bytes; offset += page_size) {
        p[offset] = p[offset];
    }
    p[bytes - 1] = p[bytes - 1];
}

static std::string tensor_name(const ggml_tensor * t) {
    if (t == nullptr) {
        return {};
    }

    const char * name = ggml_get_name(t);
    if (name == nullptr) {
        return {};
    }

    return name;
}

static std::string source_name(const ggml_tensor * t, int index) {
    if (t == nullptr || t->src[index] == nullptr) {
        return {};
    }
    return tensor_name(t->src[index]);
}

static int parse_layer_index_from(const std::string & text) {
    if (text.empty()) {
        return -1;
    }

    const auto dash = text.rfind('-');
    if (dash != std::string::npos && dash + 1 < text.size()) {
        size_t end = dash + 1;
        while (end < text.size() && text[end] >= '0' && text[end] <= '9') {
            ++end;
        }
        if (end > dash + 1) {
            return std::stoi(text.substr(dash + 1, end - dash - 1));
        }
    }

    const std::string marker = "blk.";
    const auto blk = text.find(marker);
    if (blk != std::string::npos) {
        size_t start = blk + marker.size();
        size_t end = start;
        while (end < text.size() && text[end] >= '0' && text[end] <= '9') {
            ++end;
        }
        if (end > start) {
            return std::stoi(text.substr(start, end - start));
        }
    }

    for (const char * cache_marker : {"cache_k_l", "cache_v_l"}) {
        const auto cache = text.find(cache_marker);
        if (cache == std::string::npos) {
            continue;
        }
        size_t start = cache + std::strlen(cache_marker);
        size_t end = start;
        while (end < text.size() && text[end] >= '0' && text[end] <= '9') {
            ++end;
        }
        if (end > start) {
            return std::stoi(text.substr(start, end - start));
        }
    }

    return -1;
}

static int parse_layer_index(const std::string & node_name, const std::string & src0_name) {
    int layer = parse_layer_index_from(node_name);
    if (layer >= 0) {
        return layer;
    }
    return parse_layer_index_from(src0_name);
}

static bool contains_any(const std::string & text, std::initializer_list<const char *> needles) {
    for (const char * needle : needles) {
        if (text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool is_qcur_name(const std::string & text) {
    return text.find("Qcur") != std::string::npos;
}

static bool is_kcur_name(const std::string & text) {
    return text.find("Kcur") != std::string::npos;
}

static bool is_vcur_name(const std::string & text) {
    return text.find("Vcur") != std::string::npos;
}

static bool is_cache_name(const std::string & text) {
    return contains_any(text, {"cache_k_l", "cache_v_l"});
}

static std::string classify_stage(const std::string & node_name, const std::string & src0_name, const std::string & op_name = {}) {
    const std::string joined = node_name + "|" + src0_name + "|" + op_name;

    if (op_name == "GET_ROWS" && src0_name.find("token_embd.weight") != std::string::npos) {
        return "input_embed";
    }
    if (op_name == "GET_ROWS" && contains_any(src0_name, {"attn_out", "l_out", "ffn_out"})) {
        return "residual_slice";
    }

    const bool token_embd_as_lm_head =
        src0_name.find("token_embd.weight") != std::string::npos &&
        (node_name.find("result_output") != std::string::npos ||
         node_name == "prepack" ||
         op_name == "MUL_MAT" ||
         op_name == "weight_repack");
    const bool output_weight_as_lm_head = src0_name == "output.weight";
    if (token_embd_as_lm_head || output_weight_as_lm_head || contains_any(joined, {"result_output", "lm_head"})) {
        return "lm_head";
    }

    if (node_name.find("kqv_out") != std::string::npos && op_name == "RESHAPE") {
        return "attn_output_reshape";
    }
    if (op_name == "FLASH_ATTN_EXT" || node_name.find("__fattn__") != std::string::npos) {
        return "attn_fused";
    }
    if (contains_any(joined, {"attn_inp_kq_mask", "attn_mask"})) {
        return "attn_mask";
    }

    if (op_name == "ROPE") {
        if (is_qcur_name(joined)) {
            return "q_rope";
        }
        if (is_kcur_name(joined)) {
            return "k_rope";
        }
        return "rope";
    }
    if (op_name == "SET_ROWS" && is_cache_name(node_name)) {
        return "kv_cache_write";
    }
    if ((op_name == "VIEW" || op_name == "PERMUTE") && is_cache_name(joined)) {
        return "kv_cache_read";
    }
    if (op_name == "RESHAPE" || op_name == "VIEW" || op_name == "PERMUTE") {
        if (is_qcur_name(joined)) {
            return op_name == "PERMUTE" ? "q_permute" : "q_reshape";
        }
        if (is_kcur_name(joined)) {
            return "k_reshape";
        }
        if (is_vcur_name(joined)) {
            return "v_reshape";
        }
    }

    if (contains_any(src0_name, {"attn_output.weight", ".wo"}) || contains_any(node_name, {"attn_out", "o_proj"})) {
        return "o_proj";
    }

    if (contains_any(src0_name, {"attn_q.weight", ".wq"}) || contains_any(node_name, {"q_proj"})) {
        return "q_proj";
    }
    if (contains_any(src0_name, {"attn_k.weight", ".wk"}) || contains_any(node_name, {"k_proj"})) {
        return "k_proj";
    }
    if (contains_any(src0_name, {"attn_v.weight", ".wv"}) || contains_any(node_name, {"v_proj"})) {
        return "v_proj";
    }
    if (contains_any(joined, {"wqkv", "qkv"})) {
        return "qkv_proj";
    }
    if (contains_any(joined, {"kq_soft_max", "softmax"})) {
        return "attn_softmax";
    }
    if (contains_any(joined, {"kqv", "attn_value"})) {
        return "attn_value";
    }
    if (contains_any(joined, {"kq", "attn_score"})) {
        return "attn_score";
    }

    if (op_name == "ADD" && node_name.find("ffn_inp") != std::string::npos) {
        return "attn_residual";
    }
    if (op_name == "ADD" && node_name.find("l_out") != std::string::npos) {
        return "ffn_residual";
    }

    if (node_name == "norm" || node_name.find("result_norm") != std::string::npos) {
        return "final_norm";
    }
    if (contains_any(node_name, {"attn_norm"}) ||
        (op_name == "RMS_NORM" && contains_any(src0_name, {"embd", "l_out"}))) {
        return "attn_norm";
    }
    if (contains_any(node_name, {"ffn_norm"}) ||
        (op_name == "RMS_NORM" && src0_name.find("ffn_inp") != std::string::npos)) {
        return "ffn_norm";
    }

    if (op_name == "GEGLU" ||
        op_name == "SWIGLU" ||
        contains_any(node_name, {"ffn_geglu", "ffn_swiglu"})) {
        return "ffn_activation";
    }
    if (contains_any(joined, {"ffn_gate"})) {
        return "ffn_gate";
    }
    if (contains_any(joined, {"ffn_up"})) {
        return "ffn_up";
    }
    if (contains_any(joined, {"ffn_down"})) {
        return "ffn_down";
    }
    if (contains_any(joined, {"norm", "rope", "rot", "pos_", "attn_scale", "embd"})) {
        return "other_cpu";
    }

    return "other_cpu";
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

static std::string effective_node_name(const ggml_tensor * t) {
    std::string name = tensor_name(t);
    if (!name.empty()) {
        return name;
    }

    std::string src0 = source_name(t, 0);
    if (!src0.empty()) {
        return src0;
    }

    return ggml_op_desc(t);
}

static std::string csv_escape(const std::string & value) {
    bool needs_quotes = value.find_first_of(",\"\n") != std::string::npos;
    if (!needs_quotes) {
        return value;
    }

    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
}

static std::string format_hex_mask(int32_t mask) {
    std::ostringstream out;
    out << "0x" << std::hex << (mask & 0xf);
    return out.str();
}

static std::string markdown_table_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '|':
                out += "\\|";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '\n':
                out += "<br>";
                break;
            case '\r':
                break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

struct ggml_backend_gemmini_context {
    int n_threads = 1;
};

struct weight_cache_entry {
    gemmini_aligned_vector<elem_t> data;
    std::vector<float> output_scales;
    int64_t rows = 0;
    int64_t cols = 0;
    bool page_packed_b = false;
};

struct prepacked_weight_entry {
    gemmini_aligned_vector<elem_t> data;
    std::vector<float> output_scales;
    int64_t rows = 0;
    int64_t cols = 0;
    bool page_packed_b = false;
};

struct gemmini_op_event {
    std::string run_label;
    std::string phase;
    std::string backend;
    std::string stage;
    std::string node_name;
    std::string src0_name;
    std::string op_name;
    int layer_index = -1;
    int token_index = -1;
    int64_t wall_us = 0;
    uint64_t wall_cycles = 0;
    int64_t weight_prepare_us = 0;
    uint64_t weight_prepare_cycles = 0;
    int64_t activation_quant_us = 0;
    uint64_t activation_quant_cycles = 0;
    int64_t prefault_us = 0;
    uint64_t prefault_cycles = 0;
    int64_t flush_us = 0;
    uint64_t flush_cycles = 0;
    int64_t job_configuration_us = 0;
    uint64_t job_configuration_cycles = 0;
    int64_t calculate_tiling_factors_us = 0;
    uint64_t calculate_tiling_factors_cycles = 0;
    int64_t gemmini_configuration_us = 0;
    uint64_t gemmini_configuration_cycles = 0;
    int64_t gemmini_run_us = 0;
    uint64_t gemmini_run_cycles = 0;
    int64_t gemmini_call_us = 0;
    uint64_t gemmini_call_cycles = 0;
    int64_t output_dequant_store_us = 0;
    uint64_t output_dequant_store_cycles = 0;
    uint64_t gemmini_total_cycles = 0;
    uint64_t gemmini_load_cycles = 0;
    uint64_t gemmini_preload_cycles = 0;
    uint64_t gemmini_compute_cycles = 0;
    uint64_t gemmini_store_cycles = 0;
    uint64_t gemmini_wait_cycles = 0;
    int64_t dim_i = 0;
    int64_t dim_j = 0;
    int64_t dim_k = 0;
    int64_t tile_i = 0;
    int64_t tile_j = 0;
    int64_t tile_k = 0;
    uint8_t page_packed_mask = 0;
};

struct token_event {
    std::string run_label;
    int token_index = -1;
    int token_id = -1;
    std::string piece;
};

struct run_totals {
    std::string model_name;
    std::string input_prompt;
    std::string final_output;
    int32_t gemmini_mask = 0;
    int32_t gemmini_count = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int64_t wall_us = 0;
    uint64_t wall_cycles = 0;
    int64_t ttft_us = 0;
    uint64_t ttft_cycles = 0;
    double tpot_us = 0.0;
    double tpot_cycles = 0.0;
};

struct pending_eval_state {
    const ggml_tensor * tensor = nullptr;
    int64_t start_us = 0;
    uint64_t start_cycles = 0;
};

struct last_gemmini_state {
    const ggml_tensor * tensor = nullptr;
    std::string node_name;
    std::string src0_name;
    std::string stage;
    int layer_index = -1;
    int token_index = -1;
    int64_t dim_i = 0;
    int64_t dim_j = 0;
    int64_t dim_k = 0;
    int64_t tile_i = 0;
    int64_t tile_j = 0;
    int64_t tile_k = 0;
    uint8_t page_packed_mask = 0;
    int64_t repack_us = 0;
    uint64_t repack_cycles = 0;
    int64_t weight_prepare_us = 0;
    uint64_t weight_prepare_cycles = 0;
    int64_t activation_quant_us = 0;
    uint64_t activation_quant_cycles = 0;
    int64_t prefault_us = 0;
    uint64_t prefault_cycles = 0;
    int64_t flush_us = 0;
    uint64_t flush_cycles = 0;
    int64_t job_configuration_us = 0;
    uint64_t job_configuration_cycles = 0;
    int64_t calculate_tiling_factors_us = 0;
    uint64_t calculate_tiling_factors_cycles = 0;
    int64_t gemmini_configuration_us = 0;
    uint64_t gemmini_configuration_cycles = 0;
    int64_t gemmini_run_us = 0;
    uint64_t gemmini_run_cycles = 0;
    int64_t gemmini_call_us = 0;
    uint64_t gemmini_call_cycles = 0;
    int64_t output_dequant_store_us = 0;
    uint64_t output_dequant_store_cycles = 0;
    uint64_t total_cycles = 0;
    uint64_t load_cycles = 0;
    uint64_t preload_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t store_cycles = 0;
    uint64_t wait_cycles = 0;
    bool valid = false;
};

struct profiler_thread_state {
    bool enabled = false;
    std::string current_run = "hybrid";
    std::string current_phase = "prefill";
    int token_index = -1;
    pending_eval_state pending_eval;
    last_gemmini_state last_gemmini;
    int64_t graph_call_profiled_us = 0;
    uint64_t graph_call_profiled_cycles = 0;
};

struct profiler_shared_state {
    std::mutex mutex;
    int64_t model_load_us = 0;
    uint64_t model_load_cycles = 0;
    std::string summary_model_name;
    std::string summary_input_prompt;
    std::string summary_final_output;
    std::map<std::string, run_totals> runs;
    std::vector<gemmini_op_event> op_events;
    std::vector<token_event> token_events;
};

struct offload_admission_state {
    std::unordered_set<const ggml_tensor *> admitted;
};

static profiler_thread_state & profiler_tls() {
    static thread_local profiler_thread_state state;
    return state;
}

static profiler_shared_state & profiler_shared() {
    static profiler_shared_state state;
    return state;
}

static offload_admission_state & offload_admission() {
    static thread_local offload_admission_state state;
    return state;
}

static void reset_offload_admission() {
    offload_admission() = offload_admission_state{};
}

static std::unordered_map<const void *, weight_cache_entry> & weight_cache() {
    static std::unordered_map<const void *, weight_cache_entry> cache;
    return cache;
}

static std::unordered_map<std::string, prepacked_weight_entry> & prepacked_weight_store() {
    static std::unordered_map<std::string, prepacked_weight_entry> store;
    return store;
}

static std::mutex & weight_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

static void profiler_add_event(const gemmini_op_event & event) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.op_events.push_back(event);
}

static std::string current_run_label() {
    return profiler_tls().current_run.empty() ? std::string("hybrid") : profiler_tls().current_run;
}

static std::string prepacked_weight_key(
        const std::string & name,
        int64_t slice_i2,
        int64_t slice_i3,
        int64_t cols_in,
        int64_t cols_out) {
    std::ostringstream out;
    out << name << '\x1f'
        << slice_i2 << '\x1f'
        << slice_i3 << '\x1f'
        << cols_in << '\x1f'
        << cols_out;
    return out.str();
}

template <typename T>
static bool read_pod(std::istream & in, T & value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    return static_cast<bool>(in);
}

static bool read_bytes(std::istream & in, void * data, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    in.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
}

static bool checked_b_storage_size(
        int64_t rows,
        int64_t cols,
        bool page_packed_b,
        uint64_t expected_count,
        size_t * size_out) {
    if (rows <= 0 || cols <= 0) {
        return false;
    }

    try {
        const size_t bytes = b_storage_bytes(
            static_cast<size_t>(rows), static_cast<size_t>(cols), page_packed_b);
        const size_t count = storage_elements<elem_t>(bytes, "B");
        if (expected_count != static_cast<uint64_t>(count)) {
            return false;
        }
        *size_out = count;
        return true;
    } catch (...) {
        return false;
    }
}

static gemmini_aligned_vector<elem_t> convert_b_layout(
        const elem_t * src,
        size_t rows,
        size_t cols,
        bool src_page_packed,
        bool dst_page_packed) {
    const size_t dst_bytes = b_storage_bytes(rows, cols, dst_page_packed);
    gemmini_aligned_vector<elem_t> dst(storage_elements<elem_t>(dst_bytes, "B"), 0);

    if (src_page_packed == dst_page_packed) {
        std::copy(src, src + dst.size(), dst.begin());
        return dst;
    }

    const size_t k_blocks = gemmini_ceil_div_size(rows, DIM);
    const size_t j_blocks = gemmini_ceil_div_size(cols, DIM);
    const size_t packed_row_stride = gemmini_page_packed_b_j_blocks_per_page() * DIM;

    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        const size_t valid_rows = std::min(static_cast<size_t>(DIM), rows - k_block * DIM);
        for (size_t j_block = 0; j_block < j_blocks; ++j_block) {
            const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - j_block * DIM);
            if (dst_page_packed) {
                elem_t * packed_block = gemmini_page_packed_b_block_addr_mut(
                    dst.data(), k_block, j_block, cols);
                for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
                    const elem_t * row_src = src + (k_block * DIM + k_in_block) * cols + j_block * DIM;
                    std::copy(row_src, row_src + valid_cols,
                              packed_block + k_in_block * packed_row_stride);
                }
            } else {
                const elem_t * packed_block = gemmini_page_packed_b_block_addr(
                    src, k_block, j_block, cols);
                for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
                    elem_t * row_dst = dst.data() + (k_block * DIM + k_in_block) * cols + j_block * DIM;
                    std::copy(packed_block + k_in_block * packed_row_stride,
                              packed_block + k_in_block * packed_row_stride + valid_cols,
                              row_dst);
                }
            }
        }
    }

    return dst;
}

static int load_prepacked_weight_file(const char * path) {
    if (path == nullptr || *path == '\0') {
        return -1;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return -1;
    }

    ggml_gemmini_weight_pack_file_header header = {};
    if (!read_pod(in, header)) {
        return -1;
    }

    if (std::memcmp(header.magic, GGML_GEMMINI_WEIGHT_PACK_MAGIC, GGML_GEMMINI_WEIGHT_PACK_MAGIC_SIZE) != 0 ||
            (header.version != GGML_GEMMINI_WEIGHT_PACK_VERSION_V1 &&
             header.version != GGML_GEMMINI_WEIGHT_PACK_VERSION) ||
            header.endian != GGML_GEMMINI_WEIGHT_PACK_ENDIAN ||
            header.elem_size != sizeof(elem_t) ||
            header.scale_size != sizeof(float)) {
        return -1;
    }

    bool file_page_packed_b = false;
    if (header.version == GGML_GEMMINI_WEIGHT_PACK_VERSION) {
        ggml_gemmini_weight_pack_geometry_v2 geometry = {};
        if (!read_pod(in, geometry) ||
                geometry.dim != DIM ||
                geometry.page_size != GEMMINI_PAGE_PACKED_PAGE_BYTES ||
                geometry.max_bytes != MAX_BYTES ||
                (geometry.b_layout != GGML_GEMMINI_WEIGHT_LAYOUT_ROW_MAJOR &&
                 geometry.b_layout != GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B)) {
            return -1;
        }
        file_page_packed_b = geometry.b_layout == GGML_GEMMINI_WEIGHT_LAYOUT_PAGE_PACKED_B;
    }

    std::unordered_map<std::string, prepacked_weight_entry> loaded;
    loaded.reserve(static_cast<size_t>(std::min<uint64_t>(header.entry_count, 4096)));

    for (uint64_t index = 0; index < header.entry_count; ++index) {
        ggml_gemmini_weight_pack_entry_header entry_header = {};
        if (!read_pod(in, entry_header)) {
            return -1;
        }

        if (entry_header.name_size == 0 ||
                entry_header.cols_in <= 0 ||
                entry_header.cols_out <= 0 ||
                entry_header.scale_count != static_cast<uint64_t>(entry_header.cols_out)) {
            return -1;
        }

        size_t data_count = 0;
        if (!checked_b_storage_size(
                    entry_header.cols_in,
                    entry_header.cols_out,
                    file_page_packed_b,
                    entry_header.data_count,
                    &data_count)) {
            return -1;
        }

        std::string name(entry_header.name_size, '\0');
        if (!read_bytes(in, name.data(), name.size())) {
            return -1;
        }

        prepacked_weight_entry entry;
        entry.rows = entry_header.cols_in;
        entry.cols = entry_header.cols_out;
        entry.page_packed_b = file_page_packed_b;
        entry.output_scales.resize(static_cast<size_t>(entry_header.scale_count));
        entry.data.resize(data_count);

        if (!read_bytes(in, entry.output_scales.data(), entry.output_scales.size() * sizeof(float)) ||
                !read_bytes(in, entry.data.data(), entry.data.size() * sizeof(elem_t))) {
            return -1;
        }

        const std::string key = prepacked_weight_key(
            name,
            entry_header.slice_i2,
            entry_header.slice_i3,
            entry.rows,
            entry.cols);
        loaded.emplace(key, std::move(entry));
    }

    char extra = 0;
    if (in.read(&extra, 1)) {
        return -1;
    }

    const size_t loaded_count = loaded.size();
    {
        std::lock_guard<std::mutex> lock(weight_cache_mutex());
        prepacked_weight_store().swap(loaded);
    }

    return loaded_count > static_cast<size_t>(std::numeric_limits<int>::max()) ?
        std::numeric_limits<int>::max() :
        static_cast<int>(loaded_count);
}

static bool cache_prepacked_weight_slice(
        const void * slice_key,
        const std::string & tensor_name_value,
        int64_t slice_i2,
        int64_t slice_i3,
        int64_t cols_in,
        int64_t cols_out) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());

    auto & cache = weight_cache();
    if (cache.find(slice_key) != cache.end()) {
        return true;
    }

    auto & store = prepacked_weight_store();
    const std::string key = prepacked_weight_key(tensor_name_value, slice_i2, slice_i3, cols_in, cols_out);
    auto it = store.find(key);
    if (it == store.end()) {
        return false;
    }

    weight_cache_entry entry;
    entry.rows = it->second.rows;
    entry.cols = it->second.cols;
    entry.output_scales = std::move(it->second.output_scales);
    entry.page_packed_b = gemmini_page_packing().b;
    if (it->second.page_packed_b == entry.page_packed_b) {
        entry.data = std::move(it->second.data);
    } else {
        entry.data = convert_b_layout(
            it->second.data.data(),
            static_cast<size_t>(entry.rows),
            static_cast<size_t>(entry.cols),
            it->second.page_packed_b,
            entry.page_packed_b);
    }

    cache.emplace(slice_key, std::move(entry));
    store.erase(it);
    return true;
}

static elem_t quantize_value_to_i8(float value, float scale) {
    float q = std::round(value / scale);
    q = std::max(-128.0f, std::min(127.0f, q));
    return static_cast<elem_t>(q);
}

static float quantize_row_to_i8(
        const float * src,
        int64_t cols,
        elem_t * dst) {
    float max_abs = 0.0f;
    for (int64_t i = 0; i < cols; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    if (max_abs == 0.0f) {
        std::fill(dst, dst + cols, 0);
        return 1.0f;
    }

    const float scale = max_abs / 127.0f;
    for (int64_t i = 0; i < cols; ++i) {
        dst[i] = quantize_value_to_i8(src[i], scale);
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
    GGML_ASSERT(traits != nullptr && traits->to_float != nullptr);
    traits->to_float(row_ptr, tmp.data(), cols);
}

static bool tensor_type_can_store_from_float(enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits != nullptr && traits->from_float_ref != nullptr;
}

static void store_float_row_to_tensor(
        const ggml_tensor * t,
        uint8_t * row_ptr,
        const float * values,
        int64_t cols) {
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(row_ptr, values, cols * sizeof(float));
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    GGML_ASSERT(traits != nullptr && traits->from_float_ref != nullptr);
    traits->from_float_ref(values, row_ptr, cols);
}

static float quantize_and_store_b_output_row(
        elem_t * dst,
        const float * src,
        size_t out_col,
        size_t cols_in,
        size_t cols_out,
        bool page_packed_b) {
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
        const size_t valid_rows = std::min(static_cast<size_t>(DIM), cols_in - k_block * DIM);
        for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
            block[k_in_block * packed_row_stride + j_in_block] =
                quantize_value_to_i8(src[k_block * DIM + k_in_block], scale);
        }
    }
    return scale;
}

static void store_quantized_a_row(
        elem_t * dst,
        const elem_t * quant_row,
        size_t row,
        size_t cols) {
    const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
    const size_t packed_row_stride = page_k_blocks * DIM;
    const size_t i_block = row / DIM;
    const size_t i_in_block = row % DIM;
    const size_t k_blocks = gemmini_ceil_div_size(cols, DIM);

    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        elem_t * block = gemmini_page_packed_a_block_addr_mut(dst, i_block, k_block, cols);
        const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - k_block * DIM);
        std::copy(
            quant_row + k_block * DIM,
            quant_row + k_block * DIM + valid_cols,
            block + i_in_block * packed_row_stride);
    }
}

static float quantize_row_to_packed_a(
        const float * src,
        size_t cols,
        elem_t * dst,
        size_t row) {
    float max_abs = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        max_abs = std::max(max_abs, std::fabs(src[col]));
    }
    if (max_abs == 0.0f) {
        return 1.0f;
    }
    const float scale = max_abs / 127.0f;

    const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
    const size_t packed_row_stride = page_k_blocks * DIM;
    const size_t i_block = row / DIM;
    const size_t i_in_block = row % DIM;
    const size_t k_blocks = gemmini_ceil_div_size(cols, DIM);
    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        elem_t * block = gemmini_page_packed_a_block_addr_mut(
            dst, i_block, k_block, cols);
        elem_t * packed_row = block + i_in_block * packed_row_stride;
        const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - k_block * DIM);
        for (size_t k_in_block = 0; k_in_block < valid_cols; ++k_in_block) {
            const size_t col = k_block * DIM + k_in_block;
            packed_row[k_in_block] = quantize_value_to_i8(src[col], scale);
        }
    }
    return scale;
}

static void quantize_activation_matrix(
        const ggml_tensor * src,
        const uint8_t * slice,
        size_t row_stride_bytes,
        size_t rows,
        size_t cols,
        bool page_packed_a,
        gemmini_aligned_vector<elem_t> & quantized,
        std::vector<float> & row_scales,
        std::vector<float> & tmp) {
    const size_t bytes = a_storage_bytes(rows, cols, page_packed_a);
    quantized.assign(storage_elements<elem_t>(bytes, "A"), 0);
    row_scales.assign(rows, 1.0f);

    for (size_t slab_row = 0; slab_row < rows; slab_row += DIM) {
        const size_t slab_end = std::min(rows, slab_row + static_cast<size_t>(DIM));
        for (size_t row = slab_row; row < slab_end; ++row) {
            const uint8_t * row_ptr = slice + row * row_stride_bytes;
            const float * values = nullptr;
            if (src->type == GGML_TYPE_F32) {
                values = reinterpret_cast<const float *>(row_ptr);
            } else {
                tensor_row_to_float(src, row_ptr, static_cast<int64_t>(cols), tmp);
                values = tmp.data();
            }

            if (page_packed_a) {
                row_scales[row] = quantize_row_to_packed_a(
                    values, cols, quantized.data(), row);
            } else {
                row_scales[row] = quantize_row_to_i8(
                    values, static_cast<int64_t>(cols), quantized.data() + row * cols);
            }
        }
    }
}

static void dequantize_and_store_output(
        const ggml_tensor * dst,
        uint8_t * dst_slice,
        size_t dst_row_stride_bytes,
        size_t rows,
        size_t cols,
        const acc_t * accum,
        bool page_packed_c,
        const std::vector<float> & a_row_scales,
        const std::vector<float> & b_output_scales,
        std::vector<float> & output_row) {
    const size_t page_j_blocks =
        gemmini_page_packed_acc_j_blocks_per_page(sizeof(acc_t));
    const size_t packed_row_stride = page_j_blocks * DIM;
    const size_t j_blocks = gemmini_ceil_div_size(cols, DIM);

    for (size_t row = 0; row < rows; ++row) {
        float * direct_row = dst->type == GGML_TYPE_F32
            ? reinterpret_cast<float *>(dst_slice + row * dst_row_stride_bytes)
            : output_row.data();
        const float a_scale = a_row_scales[row];

        if (!page_packed_c) {
            const acc_t * acc_row = accum + row * cols;
            for (size_t col = 0; col < cols; ++col) {
                direct_row[col] = static_cast<float>(acc_row[col]) * a_scale * b_output_scales[col];
            }
        } else {
            const size_t i_block = row / DIM;
            const size_t i_in_block = row % DIM;
            for (size_t j_block = 0; j_block < j_blocks; ++j_block) {
                const acc_t * block = static_cast<const acc_t *>(
                    gemmini_page_packed_acc_block_addr(
                        accum, i_block, j_block, cols, sizeof(acc_t)));
                const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - j_block * DIM);
                const acc_t * acc_row = block + i_in_block * packed_row_stride;
                for (size_t j_in_block = 0; j_in_block < valid_cols; ++j_in_block) {
                    const size_t col = j_block * DIM + j_in_block;
                    direct_row[col] = static_cast<float>(acc_row[j_in_block]) *
                        a_scale * b_output_scales[col];
                }
            }
        }

        if (dst->type != GGML_TYPE_F32) {
            store_float_row_to_tensor(
                dst,
                dst_slice + row * dst_row_stride_bytes,
                output_row.data(),
                static_cast<int64_t>(cols));
        }
    }
}

static gemmini_aligned_vector<elem_t> convert_a_from_row_major(
        const elem_t * src,
        size_t rows,
        size_t cols,
        bool page_packed_a) {
    const size_t bytes = a_storage_bytes(rows, cols, page_packed_a);
    gemmini_aligned_vector<elem_t> dst(storage_elements<elem_t>(bytes, "A"), 0);
    if (!page_packed_a) {
        std::copy(src, src + rows * cols, dst.begin());
        return dst;
    }

    for (size_t row = 0; row < rows; ++row) {
        store_quantized_a_row(dst.data(), src + row * cols, row, cols);
    }
    return dst;
}

static gemmini_aligned_vector<acc_t> convert_acc_layout(
        const acc_t * src,
        size_t rows,
        size_t cols,
        bool src_page_packed,
        bool dst_page_packed,
        const char * what) {
    const size_t bytes = acc_storage_bytes(rows, cols, dst_page_packed, what);
    gemmini_aligned_vector<acc_t> dst(storage_elements<acc_t>(bytes, what), 0);
    if (src_page_packed == dst_page_packed) {
        std::copy(src, src + dst.size(), dst.begin());
        return dst;
    }

    const size_t page_j_blocks =
        gemmini_page_packed_acc_j_blocks_per_page(sizeof(acc_t));
    const size_t packed_row_stride = page_j_blocks * DIM;
    const size_t i_blocks = gemmini_ceil_div_size(rows, DIM);
    const size_t j_blocks = gemmini_ceil_div_size(cols, DIM);

    for (size_t i_block = 0; i_block < i_blocks; ++i_block) {
        const size_t valid_rows = std::min(static_cast<size_t>(DIM), rows - i_block * DIM);
        for (size_t j_block = 0; j_block < j_blocks; ++j_block) {
            const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - j_block * DIM);
            if (dst_page_packed) {
                acc_t * block = static_cast<acc_t *>(gemmini_page_packed_acc_block_addr_mut(
                    dst.data(), i_block, j_block, cols, sizeof(acc_t)));
                for (size_t i_in_block = 0; i_in_block < valid_rows; ++i_in_block) {
                    const acc_t * row_src = src + (i_block * DIM + i_in_block) * cols + j_block * DIM;
                    std::copy(row_src, row_src + valid_cols,
                              block + i_in_block * packed_row_stride);
                }
            } else {
                const acc_t * block = static_cast<const acc_t *>(gemmini_page_packed_acc_block_addr(
                    src, i_block, j_block, cols, sizeof(acc_t)));
                for (size_t i_in_block = 0; i_in_block < valid_rows; ++i_in_block) {
                    acc_t * row_dst = dst.data() + (i_block * DIM + i_in_block) * cols + j_block * DIM;
                    std::copy(block + i_in_block * packed_row_stride,
                              block + i_in_block * packed_row_stride + valid_cols,
                              row_dst);
                }
            }
        }
    }
    return dst;
}

static void populate_weight_cache_entry(
        weight_cache_entry & entry,
        const ggml_tensor * src0,
        const uint8_t * slice_ptr,
        int64_t cols_out,
        int64_t cols_in,
        size_t row_stride_bytes,
        int layer_index,
        const std::string & node_name,
        const std::string & src0_name) {
    uint64_t start_cycles = read_cycles_local();
    int64_t start_us = ggml_time_us();

    entry.rows = cols_in;
    entry.cols = cols_out;
    entry.page_packed_b = gemmini_page_packing().b;
    entry.data.assign(
        storage_elements<elem_t>(
            b_storage_bytes(
                static_cast<size_t>(cols_in),
                static_cast<size_t>(cols_out),
                entry.page_packed_b),
            "B"),
        0);
    entry.output_scales.resize(cols_out, 1.0f);

    std::vector<float> tmp;
    for (int64_t out_col = 0; out_col < cols_out; ++out_col) {
        const uint8_t * row_ptr = slice_ptr + out_col * row_stride_bytes;
        tensor_row_to_float(src0, row_ptr, cols_in, tmp);
        entry.output_scales[out_col] = quantize_and_store_b_output_row(
            entry.data.data(),
            tmp.data(),
            static_cast<size_t>(out_col),
            static_cast<size_t>(cols_in),
            static_cast<size_t>(cols_out),
            entry.page_packed_b);
    }

    int64_t end_us = ggml_time_us();
    uint64_t end_cycles = read_cycles_local();

    gemmini_op_event repack_event;
    repack_event.run_label = current_run_label();
    repack_event.phase = "load/repack";
    repack_event.backend = "gemmini";
    repack_event.stage = classify_stage(node_name, src0_name);
    repack_event.node_name = node_name;
    repack_event.src0_name = src0_name;
    repack_event.op_name = "weight_repack";
    repack_event.layer_index = layer_index;
    repack_event.token_index = profiler_tls().token_index;
    repack_event.wall_us = end_us - start_us;
    repack_event.wall_cycles = end_cycles - start_cycles;
    repack_event.dim_i = 0;
    repack_event.dim_j = cols_out;
    repack_event.dim_k = cols_in;
    repack_event.page_packed_mask = gemmini_page_packing().mask();
    profiler_add_event(repack_event);
}

static weight_cache_entry & get_or_create_weight_cache(
        const ggml_tensor * src0,
        const void * slice_key,
        const uint8_t * slice_ptr,
        int64_t cols_out,
        int64_t cols_in,
        size_t row_stride_bytes,
        int layer_index,
        const std::string & node_name,
        const std::string & src0_name) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    auto & cache = weight_cache();
    auto it = cache.find(slice_key);
    if (it != cache.end()) {
        if (it->second.page_packed_b != gemmini_page_packing().b) {
            it->second.data = convert_b_layout(
                it->second.data.data(),
                static_cast<size_t>(it->second.rows),
                static_cast<size_t>(it->second.cols),
                it->second.page_packed_b,
                gemmini_page_packing().b);
            it->second.page_packed_b = gemmini_page_packing().b;
        }
        return it->second;
    }

    weight_cache_entry entry;
    populate_weight_cache_entry(
        entry,
        src0,
        slice_ptr,
        cols_out,
        cols_in,
        row_stride_bytes,
        layer_index,
        node_name,
        src0_name);

    auto inserted = cache.emplace(slice_key, std::move(entry));
    return inserted.first->second;
}

static bool tensor_type_can_load_to_float(enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits != nullptr && traits->to_float != nullptr;
}

static bool can_prepack_weight_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    const std::string name = tensor_name(tensor);
    if (!is_gemmini_model_weight_name(name)) {
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

struct gemmini_matmul_metrics {
    int64_t prefault_us = 0;
    uint64_t prefault_cycles = 0;
    int64_t flush_us = 0;
    uint64_t flush_cycles = 0;
    int64_t job_configuration_us = 0;
    uint64_t job_configuration_cycles = 0;
    int64_t calculate_tiling_factors_us = 0;
    uint64_t calculate_tiling_factors_cycles = 0;
    int64_t gemmini_configuration_us = 0;
    uint64_t gemmini_configuration_cycles = 0;
    int64_t gemmini_run_us = 0;
    uint64_t gemmini_run_cycles = 0;
    uint64_t total_cycles = 0;
    uint64_t load_cycles = 0;
    uint64_t preload_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t store_cycles = 0;
    uint64_t wait_cycles = 0;
    int64_t tile_i = 0;
    int64_t tile_j = 0;
    int64_t tile_k = 0;
};

static void gemmini_flush_mask(int gemmini_mask) {
    for (int i = 0; i < total_gemmini_num; ++i) {
        if (((gemmini_mask >> i) & 1) == 0) {
            continue;
        }

        switch (i) {
            case 0:
                gemmini_flush(custom0, 0);
                break;
            case 1:
                gemmini_flush(custom1, 0);
                break;
            case 2:
                gemmini_flush(custom2, 0);
                break;
            case 3:
                gemmini_flush(custom3, 0);
                break;
        }
    }
}

static void gemmini_flush_once(int gemmini_mask) {
    static int flushed_mask = 0;
    static std::mutex flush_mutex;
    std::lock_guard<std::mutex> lock(flush_mutex);
    const int missing_mask = gemmini_mask & ~flushed_mask;
    if (missing_mask == 0) {
        return;
    }

    gemmini_flush_mask(missing_mask);
    flushed_mask |= missing_mask;
}

static bool split_tiling_fits(size_t tI, size_t tJ, size_t tK) {
    constexpr size_t max_A_rows = TOTAL_SPAD_ROWS / 16;
    constexpr size_t max_B_rows = TOTAL_SPAD_ROWS / 16;
    constexpr size_t max_acc_rows = TOTAL_ACC_ROWS / 8;

    return tiled_matmul_A_spad_rows(tI, tJ, tK) <= max_A_rows &&
           tiled_matmul_B_spad_rows(tI, tJ, tK) <= max_B_rows &&
           tiled_matmul_total_acc_rows(tI, tJ) <= max_acc_rows;
}

static void choose_split_tiling_factors(shared_multi_matmul_job_t * job) {
    job->gemmini_num = 0;
    for (int i = 0; i < total_gemmini_num; ++i) {
        if ((job->gemmini_list >> i) & 1) {
            ++job->gemmini_num;
        }
    }
    if (job->gemmini_num == 0) {
        job->gemmini_num = 1;
    }

    job->dim_I_padded = (job->dim_I / DIM + (job->dim_I % DIM != 0)) * DIM;
    job->dim_J_padded = (job->dim_J / DIM + (job->dim_J % DIM != 0)) * DIM;
    job->dim_K_padded = (job->dim_K / DIM + (job->dim_K % DIM != 0)) * DIM;

    const size_t max_i = std::max<size_t>(1, job->dim_I_padded / DIM);
    const size_t max_j = std::max<size_t>(1, job->dim_J_padded / DIM);
    const size_t max_k = std::max<size_t>(1, job->dim_K_padded / DIM);

    size_t tI = std::min(max_i, static_cast<size_t>(gemmini_tile_i()));
    size_t tJ = std::min(max_j, static_cast<size_t>(gemmini_tile_j()));
    size_t tK = std::min(max_k, static_cast<size_t>(gemmini_tile_k()));

    while (!split_tiling_fits(tI, tJ, tK)) {
        if (tK > 1) {
            --tK;
        } else if (tI > 1) {
            --tI;
        } else if (tJ > 1) {
            --tJ;
        } else {
            break;
        }
    }

    while (true) {
        bool increased = false;

        if (tJ < max_j && split_tiling_fits(tI, tJ + 1, tK)) {
            ++tJ;
            increased = true;
        }
        if (tI < max_i && split_tiling_fits(tI + 1, tJ, tK)) {
            ++tI;
            increased = true;
        }
        if (tK < max_k && split_tiling_fits(tI, tJ, tK + 1)) {
            ++tK;
            increased = true;
        }

        if (!increased) {
            break;
        }
    }

    job->tile_I = tI;
    job->tile_J = tJ;
    job->tile_K = tK;
    job->sp_addr_range = TOTAL_SPAD_ROWS / 4;
    job->acc_addr_range = TOTAL_ACC_ROWS / 4;
    job->sp_addr_A_stacked = tiled_matmul_A_spad_rows(tI, tJ, tK);
    job->sp_addr_B_stacked = tiled_matmul_B_spad_rows(tI, tJ, tK);
    job->acc_addr_stacked = tiled_matmul_total_acc_rows(tI, tJ);
}

static gemmini_matmul_metrics run_gemmini_matmul(
        int gemmini_mask,
        size_t rows,
        size_t cols_out,
        size_t cols_in,
        const elem_t * a,
        const elem_t * b,
        const acc_t * d,
        acc_t * out,
        const gemmini_page_packing_config & packing) {
    gemmini_matmul_metrics metrics;
    const bool split_mode = request_runtime().split_mode;
    const bool single_mode = request_runtime().single_mode && !split_mode;
    const bool use_counters = !split_mode && gemmini_use_counters();
    const int gemmini_id = request_runtime().gemmini_id;
    const int requested_mask = gemmini_mask & 0xf;
    const int single_gemmini_id = 3;
    const int run_gemmini_mask = single_mode ? (1 << single_gemmini_id) : requested_mask;
    const char * run_mode = single_mode ? "single_reduced_bank_conflict" : "shared_multi";
    const bool page_packed_d = d != nullptr && packing.d;
    const uint8_t effective_packing_mask = static_cast<uint8_t>(
        (packing.a ? 1u : 0u) |
        (packing.b ? 2u : 0u) |
        (packing.c ? 4u : 0u) |
        (page_packed_d ? 8u : 0u));
    const size_t a_bytes = a_storage_bytes(rows, cols_in, packing.a);
    const size_t b_bytes = b_storage_bytes(cols_in, cols_out, packing.b);
    const size_t c_bytes = acc_storage_bytes(rows, cols_out, packing.c, "C");
    const size_t d_bytes = d == nullptr ? 0 : acc_storage_bytes(rows, cols_out, page_packed_d, "D");

    if (env_flag("GGML_GEMMINI_TRACE", false)) {
        std::fprintf(stderr,
            "GEMMINI-RUN-BEGIN,mode=%s,mask=0x%x,requested_mask=0x%x,counters=%d,split=%d,gemmini_id=%d,rows=%zu,cols_out=%zu,cols_in=%zu,packing=0x%x,page_packed_a=%d,page_packed_b=%d,page_packed_c=%d,page_packed_d=%d\n",
            run_mode,
            run_gemmini_mask,
            requested_mask,
            use_counters ? 1 : 0,
            split_mode ? 1 : 0,
            single_mode ? single_gemmini_id : gemmini_id,
            rows,
            cols_out,
            cols_in,
            effective_packing_mask,
            packing.a ? 1 : 0,
            packing.b ? 1 : 0,
            packing.c ? 1 : 0,
            page_packed_d ? 1 : 0);
        std::fflush(stderr);
    }

    constexpr size_t kCounterLoad = 0;
    constexpr size_t kCounterStore = 1;
    constexpr size_t kCounterExecute = 2;
    constexpr size_t kCounterPreload = 3;
    constexpr size_t kCounterLoadWait = 4;
    constexpr size_t kCounterStoreWait = 5;
    constexpr size_t kCounterScratchpadWait = 6;
    constexpr size_t kCounterAccWait = 7;

    if (use_counters) {
        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr, "GEMMINI-RUN-COUNTERS,mask=0x%x\n", run_gemmini_mask);
            std::fflush(stderr);
        }
        for (int i = 0; i < total_gemmini_num; ++i) {
            if (((run_gemmini_mask >> i) & 1) == 0) {
                continue;
            }

            counter_reset(i);
            counter_configure(i, kCounterLoad, LOAD_ACTIVE_CYCLE);
            counter_configure(i, kCounterStore, STORE_ACTIVE_CYCLE);
            counter_configure(i, kCounterExecute, EXE_ACTIVE_CYCLE);
            counter_configure(i, kCounterPreload, EXE_PRELOAD_HAZ_CYCLE);
            counter_configure(i, kCounterLoadWait, LOAD_DMA_WAIT_CYCLE);
            counter_configure(i, kCounterStoreWait, STORE_DMA_WAIT_CYCLE);
            counter_configure(i, kCounterScratchpadWait, SCRATCHPAD_A_WAIT_CYCLE);
            counter_configure(i, kCounterAccWait, ACC_A_WAIT_CYCLE);
        }
    }

    uint64_t component_cycles_start = read_cycles_local();
    int64_t component_us_start = ggml_time_us();
    prefault_readonly_range(a, a_bytes);
    prefault_readonly_range(b, b_bytes);
    prefault_readonly_range(d, d_bytes);
    prefault_writable_range(out, c_bytes);
    metrics.prefault_us += ggml_time_us() - component_us_start;
    metrics.prefault_cycles += read_cycles_local() - component_cycles_start;

    if (env_flag("GGML_GEMMINI_TRACE", false)) {
        std::fprintf(stderr, "GEMMINI-RUN-FLUSH,mask=0x%x\n", run_gemmini_mask);
        std::fflush(stderr);
    }
    component_cycles_start = read_cycles_local();
    component_us_start = ggml_time_us();
    gemmini_flush_once(run_gemmini_mask);
    metrics.flush_us += ggml_time_us() - component_us_start;
    metrics.flush_cycles += read_cycles_local() - component_cycles_start;

    const uint64_t start_cycles = read_cycles_local();
    component_cycles_start = start_cycles;
    component_us_start = ggml_time_us();
    if (single_mode) {
        const int custom_num = custom3;
        tiled_matmul_single_job_t job;
        std::memset(&job, 0, sizeof(job));
        job.custom_num = custom_num;
        job.dim_I = rows;
        job.dim_J = cols_out;
        job.dim_K = cols_in;
        job.A = a;
        job.B = b;
        job.D = d;
        job.C = out;
        job.stride_A = packing.a ? GEMMINI_PAGE_PACKED_STRIDE(cols_in) : cols_in;
        job.stride_B = packing.b ? GEMMINI_PAGE_PACKED_STRIDE(cols_out) : cols_out;
        job.stride_D = d == nullptr ? 0 : (page_packed_d ? GEMMINI_PAGE_PACKED_STRIDE(cols_out) : cols_out);
        job.stride_C = packing.c ? GEMMINI_PAGE_PACKED_STRIDE(cols_out) : cols_out;
        job.A_scale_factor = MVIN_SCALE_IDENTITY;
        job.B_scale_factor = MVIN_SCALE_IDENTITY;
        job.D_scale_factor = MVIN_SCALE_IDENTITY;
        job.act = NO_ACTIVATION;
        job.scale = ACC_SCALE_IDENTITY;
        job.bert_scale = 0;
        job.repeating_bias = false;
        job.a_transpose = false;
        job.b_transpose = false;
        job.full_C = true;
        job.low_D = false;
        job.weightA = 1;
        job.dataflow = WEIGHT_STATIONARY;
        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr,
                "GEMMINI-RUN-CALL,mode=single_reduced_bank_conflict,mask=0x%x,gemmini_id=%d,custom_num=%d\n",
                run_gemmini_mask,
                single_gemmini_id,
                custom_num);
            std::fflush(stderr);
        }
        metrics.job_configuration_us += ggml_time_us() - component_us_start;
        metrics.job_configuration_cycles += read_cycles_local() - component_cycles_start;

        component_cycles_start = read_cycles_local();
        component_us_start = ggml_time_us();
        const tiled_matmul_auto_factors_t tiling = tiled_matmul_auto_reduced_bank_conflict_factors(
            rows,
            cols_out,
            cols_in,
            NO_ACTIVATION,
            WS);
        metrics.tile_i = static_cast<int64_t>(tiling.tile_I);
        metrics.tile_j = static_cast<int64_t>(tiling.tile_J);
        metrics.tile_k = static_cast<int64_t>(tiling.tile_K);
        job.tile_I = tiling.tile_I;
        job.tile_J = tiling.tile_J;
        job.tile_K = tiling.tile_K;
        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr,
                "GEMMINI-TILING,mode=single_reduced_bank_conflict,split=0,gemmini_id=%d,tile_i=%zu,tile_j=%zu,tile_k=%zu,sp_A=%zu,sp_B=%zu,acc=%zu\n",
                single_gemmini_id,
                tiling.tile_I,
                tiling.tile_J,
                tiling.tile_K,
                tiled_matmul_A_spad_rows(tiling.tile_I, tiling.tile_J, tiling.tile_K),
                tiled_matmul_B_spad_rows(tiling.tile_I, tiling.tile_J, tiling.tile_K),
                tiled_matmul_total_acc_rows(tiling.tile_I, tiling.tile_J));
            std::fflush(stderr);
        }
        metrics.calculate_tiling_factors_us += ggml_time_us() - component_us_start;
        metrics.calculate_tiling_factors_cycles += read_cycles_local() - component_cycles_start;

        component_cycles_start = read_cycles_local();
        component_us_start = ggml_time_us();
        tiled_matmul_single_job_init(&job);
        metrics.gemmini_configuration_us += ggml_time_us() - component_us_start;
        metrics.gemmini_configuration_cycles += read_cycles_local() - component_cycles_start;

        component_cycles_start = read_cycles_local();
        component_us_start = ggml_time_us();
        while (!job.done) {
            tiled_matmul_single_job_step(&job);
        }
        gemmini_fence();
        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr, "GEMMINI-RUN-END,mode=single_reduced_bank_conflict,total_cycles=%llu\n",
                static_cast<unsigned long long>(read_cycles_local() - start_cycles));
            std::fflush(stderr);
        }
        metrics.gemmini_run_us += ggml_time_us() - component_us_start;
        metrics.gemmini_run_cycles += read_cycles_local() - component_cycles_start;
    } else {
        shared_multi_matmul_job_t job;
        std::memset(&job, 0, sizeof(job));
        job.tile_id = 0;
        job.gemmini_list = requested_mask;
        if (split_mode) {
            const int partition = std::max(0, std::min(total_gemmini_num - 1, gemmini_id));
            job.sp_addr_start_stack = TOTAL_SPAD_ROWS * partition / 16;
            job.sp_addr_end_stack = TOTAL_SPAD_ROWS * partition / 16;
            job.acc_addr_start_stack = TOTAL_ACC_ROWS * partition / 8;
            job.sp_addr_range = TOTAL_SPAD_ROWS / 4;
            job.acc_addr_range = TOTAL_ACC_ROWS / 4;
        } else {
            job.sp_addr_start_stack = 0;
            job.sp_addr_end_stack = 0;
            job.acc_addr_start_stack = 0;
            job.sp_addr_range = TOTAL_SPAD_ROWS;
            job.acc_addr_range = TOTAL_ACC_ROWS;
        }
        job.dim_I = rows;
        job.dim_J = cols_out;
        job.dim_K = cols_in;
        job.A = a;
        job.B = b;
        job.D = d;
        job.C = out;
        job.stride_A = packing.a ? GEMMINI_PAGE_PACKED_STRIDE(cols_in) : cols_in;
        job.stride_B = packing.b ? GEMMINI_PAGE_PACKED_STRIDE(cols_out) : cols_out;
        job.stride_D = d == nullptr ? 0 : (page_packed_d ? GEMMINI_PAGE_PACKED_STRIDE(cols_out) : cols_out);
        job.stride_C = packing.c ? GEMMINI_PAGE_PACKED_STRIDE(cols_out) : cols_out;
        job.A_scale_factor = MVIN_SCALE_IDENTITY;
        job.B_scale_factor = MVIN_SCALE_IDENTITY;
        job.D_scale_factor = MVIN_SCALE_IDENTITY;
        job.act = NO_ACTIVATION;
        job.scale = ACC_SCALE_IDENTITY;
        job.bert_scale = 0;
        job.repeating_bias = false;
        job.a_transpose = false;
        job.b_transpose = false;
        job.full_C = true;
        job.low_D = false;
        job.weightA = 1;
        job.dataflow = WEIGHT_STATIONARY;

        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr,
                "GEMMINI-RUN-CALL,mode=shared_multi,mask=0x%x,split=%d,gemmini_id=%d,sp_start=%zu,sp_end=%zu,acc_start=%zu,sp_range=%zu,acc_range=%zu\n",
                requested_mask,
                split_mode ? 1 : 0,
                gemmini_id,
                job.sp_addr_start_stack,
                job.sp_addr_end_stack,
                job.acc_addr_start_stack,
                job.sp_addr_range,
                job.acc_addr_range);
            std::fflush(stderr);
        }
        metrics.job_configuration_us += ggml_time_us() - component_us_start;
        metrics.job_configuration_cycles += read_cycles_local() - component_cycles_start;

        component_cycles_start = read_cycles_local();
        component_us_start = ggml_time_us();
        if (split_mode) {
            choose_split_tiling_factors(&job);
        } else {
            shared_multi_choose_tiling_factors(&job);
        }
        metrics.tile_i = static_cast<int64_t>(job.tile_I);
        metrics.tile_j = static_cast<int64_t>(job.tile_J);
        metrics.tile_k = static_cast<int64_t>(job.tile_K);
        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr,
                "GEMMINI-TILING,mode=shared_multi,split=%d,gemmini_id=%d,tile_i=%zu,tile_j=%zu,tile_k=%zu,sp_A=%zu,sp_B=%zu,acc=%zu\n",
                split_mode ? 1 : 0,
                gemmini_id,
                job.tile_I,
                job.tile_J,
                job.tile_K,
                job.sp_addr_A_stacked,
                job.sp_addr_B_stacked,
                job.acc_addr_stacked);
            std::fflush(stderr);
        }
        metrics.calculate_tiling_factors_us += ggml_time_us() - component_us_start;
        metrics.calculate_tiling_factors_cycles += read_cycles_local() - component_cycles_start;

        component_cycles_start = read_cycles_local();
        component_us_start = ggml_time_us();
        shared_multi_tiled_matmul_job_init(&job);
        metrics.gemmini_configuration_us += ggml_time_us() - component_us_start;
        metrics.gemmini_configuration_cycles += read_cycles_local() - component_cycles_start;

        component_cycles_start = read_cycles_local();
        component_us_start = ggml_time_us();
        while (!job.done) {
            shared_multi_tiled_matmul_job_step(&job);
        }

        gemmini_fence();
        if (env_flag("GGML_GEMMINI_TRACE", false)) {
            std::fprintf(stderr, "GEMMINI-RUN-END,mode=shared_multi,total_cycles=%llu\n",
                static_cast<unsigned long long>(read_cycles_local() - start_cycles));
            std::fflush(stderr);
        }
        metrics.gemmini_run_us += ggml_time_us() - component_us_start;
        metrics.gemmini_run_cycles += read_cycles_local() - component_cycles_start;
    }

    const uint64_t end_cycles = read_cycles_local();
    metrics.total_cycles = end_cycles - start_cycles;

    if (use_counters) {
        for (int i = 0; i < total_gemmini_num; ++i) {
            if (((run_gemmini_mask >> i) & 1) == 0) {
                continue;
            }

            metrics.load_cycles += counter_read(i, kCounterLoad);
            metrics.store_cycles += counter_read(i, kCounterStore);
            metrics.compute_cycles += counter_read(i, kCounterExecute);
            metrics.preload_cycles += counter_read(i, kCounterPreload);
            metrics.wait_cycles += counter_read(i, kCounterLoadWait);
            metrics.wait_cycles += counter_read(i, kCounterStoreWait);
            metrics.wait_cycles += counter_read(i, kCounterScratchpadWait);
            metrics.wait_cycles += counter_read(i, kCounterAccWait);
        }
    }

    return metrics;
}

static bool gemmini_can_offload(const ggml_tensor * op) {
    if (env_flag("GGML_GEMMINI_DISABLE", false)) {
        return false;
    }

    if (gemmini_active_mask() == 0) {
        return false;
    }

    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    if (!tensor_type_can_store_from_float(op->type)) {
        return false;
    }

    const ggml_tensor * dst = op;
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ne0 != ne01 || ne1 != ne11 || ne2 != ne12 || ne3 != ne13) {
        return false;
    }

    if (ggml_blck_size(op->type) <= 0 || ne0 % ggml_blck_size(op->type) != 0) {
        return false;
    }

    if (nb0 != ggml_type_size(op->type) ||
            nb00 != ggml_type_size(src0->type) ||
            nb10 != ggml_type_size(src1->type)) {
        return false;
    }

    if (ne1 < 1) {
        return false;
    }

    return tensor_type_can_load_to_float(src0->type) && tensor_type_can_load_to_float(src1->type);
}

static bool gemmini_admit_offload(const ggml_tensor * op) {
    const int max_offloads = gemmini_max_offloads();
    if (max_offloads < 0) {
        return true;
    }
    if (max_offloads == 0) {
        return false;
    }

    auto & admitted = offload_admission().admitted;
    if (admitted.find(op) != admitted.end()) {
        return true;
    }
    if (admitted.size() >= static_cast<size_t>(max_offloads)) {
        return false;
    }

    admitted.insert(op);
    if (env_flag("GGML_GEMMINI_TRACE", false)) {
        const std::string node_name = effective_node_name(op);
        const std::string src0_name = source_name(op, 0);
        std::fprintf(stderr,
            "GEMMINI-ADMIT,index=%zu,max=%d,node=%s,src0=%s\n",
            admitted.size() - 1,
            max_offloads,
            node_name.c_str(),
            src0_name.c_str());
        std::fflush(stderr);
    }
    return true;
}

static void ggml_backend_gemmini_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t rows = ne1;
    const int64_t cols_out = ne01;
    const int64_t cols_in = ne10;

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const std::string node_name = effective_node_name(dst);
    const std::string src0_name = tensor_name(src0);
    const int layer_index = parse_layer_index(node_name, src0_name);
    const std::string stage = classify_stage(node_name, src0_name);
    const int active_mask = gemmini_active_mask();
    const gemmini_page_packing_config & packing = gemmini_page_packing();

    std::vector<float> tmp;
    std::vector<float> output_row(cols_out);
    gemmini_aligned_vector<elem_t> quant_a;
    std::vector<float> a_row_scales(rows, 1.0f);
    gemmini_aligned_vector<acc_t> accum(
        storage_elements<acc_t>(
            acc_storage_bytes(
                static_cast<size_t>(rows),
                static_cast<size_t>(cols_out),
                packing.c,
                "C"),
            "C"),
        0);

    uint64_t total_cycles = 0;
    uint64_t load_cycles = 0;
    uint64_t preload_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t store_cycles = 0;
    uint64_t wait_cycles = 0;
    int64_t tile_i = 0;
    int64_t tile_j = 0;
    int64_t tile_k = 0;
    int64_t weight_prepare_us = 0;
    uint64_t weight_prepare_cycles = 0;
    int64_t activation_quant_us = 0;
    uint64_t activation_quant_cycles = 0;
    int64_t prefault_us = 0;
    uint64_t prefault_cycles = 0;
    int64_t flush_us = 0;
    uint64_t flush_cycles = 0;
    int64_t job_configuration_us = 0;
    uint64_t job_configuration_cycles = 0;
    int64_t calculate_tiling_factors_us = 0;
    uint64_t calculate_tiling_factors_cycles = 0;
    int64_t gemmini_configuration_us = 0;
    uint64_t gemmini_configuration_cycles = 0;
    int64_t gemmini_run_us = 0;
    uint64_t gemmini_run_cycles = 0;
    int64_t gemmini_call_us = 0;
    uint64_t gemmini_call_cycles = 0;
    int64_t output_dequant_store_us = 0;
    uint64_t output_dequant_store_cycles = 0;

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const uint8_t * src0_slice = reinterpret_cast<const uint8_t *>(src0->data) + i02 * nb02 + i03 * nb03;
            const uint8_t * src1_slice = reinterpret_cast<const uint8_t *>(src1->data) + i12 * nb12 + i13 * nb13;
            uint8_t * dst_slice = reinterpret_cast<uint8_t *>(dst->data) + i12 * nb2 + i13 * nb3;

            weight_cache_entry transient_wcache;
            const bool cache_src0 = is_gemmini_model_weight_name(src0_name);
            weight_cache_entry * wcache = nullptr;
            const uint64_t weight_prepare_cycles_start = read_cycles_local();
            const int64_t weight_prepare_us_start = ggml_time_us();
            if (cache_src0) {
                wcache = &get_or_create_weight_cache(
                    src0,
                    src0_slice,
                    src0_slice,
                    cols_out,
                    cols_in,
                    nb01,
                    layer_index,
                    node_name,
                    src0_name);
            } else {
                populate_weight_cache_entry(
                    transient_wcache,
                    src0,
                    src0_slice,
                    cols_out,
                    cols_in,
                    nb01,
                    layer_index,
                    node_name,
                    src0_name);
                wcache = &transient_wcache;
            }
            weight_prepare_us += ggml_time_us() - weight_prepare_us_start;
            weight_prepare_cycles += read_cycles_local() - weight_prepare_cycles_start;
            GGML_ASSERT(wcache->page_packed_b == packing.b);

            const uint64_t activation_quant_cycles_start = read_cycles_local();
            const int64_t activation_quant_us_start = ggml_time_us();
            quantize_activation_matrix(
                src1,
                src1_slice,
                nb11,
                static_cast<size_t>(rows),
                static_cast<size_t>(cols_in),
                packing.a,
                quant_a,
                a_row_scales,
                tmp);
            activation_quant_us += ggml_time_us() - activation_quant_us_start;
            activation_quant_cycles += read_cycles_local() - activation_quant_cycles_start;

            if (env_flag("GGML_GEMMINI_TRACE", false)) {
                std::fprintf(stderr,
                    "GEMMINI-DISPATCH,stage=%s,layer=%d,rows=%lld,cols_out=%lld,cols_in=%lld,mask=0x%x,packing=0x%x,node=%s,src0=%s\n",
                    stage.c_str(),
                    layer_index,
                    static_cast<long long>(rows),
                    static_cast<long long>(cols_out),
                    static_cast<long long>(cols_in),
                    active_mask,
                    packing.mask(),
                    node_name.c_str(),
                    src0_name.c_str());
                std::fflush(stderr);
            }

            const uint64_t gemmini_call_cycles_start = read_cycles_local();
            const int64_t gemmini_call_us_start = ggml_time_us();
            const gemmini_matmul_metrics metrics = run_gemmini_matmul(
                active_mask,
                rows,
                cols_out,
                cols_in,
                quant_a.data(),
                wcache->data.data(),
                nullptr,
                accum.data(),
                packing);
            gemmini_call_us += ggml_time_us() - gemmini_call_us_start;
            gemmini_call_cycles += read_cycles_local() - gemmini_call_cycles_start;

            total_cycles += metrics.total_cycles;
            prefault_us += metrics.prefault_us;
            prefault_cycles += metrics.prefault_cycles;
            flush_us += metrics.flush_us;
            flush_cycles += metrics.flush_cycles;
            job_configuration_us += metrics.job_configuration_us;
            job_configuration_cycles += metrics.job_configuration_cycles;
            calculate_tiling_factors_us += metrics.calculate_tiling_factors_us;
            calculate_tiling_factors_cycles += metrics.calculate_tiling_factors_cycles;
            gemmini_configuration_us += metrics.gemmini_configuration_us;
            gemmini_configuration_cycles += metrics.gemmini_configuration_cycles;
            gemmini_run_us += metrics.gemmini_run_us;
            gemmini_run_cycles += metrics.gemmini_run_cycles;
            load_cycles += metrics.load_cycles;
            preload_cycles += metrics.preload_cycles;
            compute_cycles += metrics.compute_cycles;
            store_cycles += metrics.store_cycles;
            wait_cycles += metrics.wait_cycles;
            tile_i = metrics.tile_i;
            tile_j = metrics.tile_j;
            tile_k = metrics.tile_k;

            const uint64_t output_dequant_store_cycles_start = read_cycles_local();
            const int64_t output_dequant_store_us_start = ggml_time_us();
            dequantize_and_store_output(
                dst,
                dst_slice,
                nb1,
                static_cast<size_t>(rows),
                static_cast<size_t>(cols_out),
                accum.data(),
                packing.c,
                a_row_scales,
                wcache->output_scales,
                output_row);
            output_dequant_store_us += ggml_time_us() - output_dequant_store_us_start;
            output_dequant_store_cycles += read_cycles_local() - output_dequant_store_cycles_start;
        }
    }

    profiler_tls().last_gemmini.tensor = dst;
    profiler_tls().last_gemmini.node_name = node_name;
    profiler_tls().last_gemmini.src0_name = src0_name;
    profiler_tls().last_gemmini.stage = stage;
    profiler_tls().last_gemmini.layer_index = layer_index;
    profiler_tls().last_gemmini.token_index = profiler_tls().token_index;
    profiler_tls().last_gemmini.dim_i = rows;
    profiler_tls().last_gemmini.dim_j = cols_out;
    profiler_tls().last_gemmini.dim_k = cols_in;
    profiler_tls().last_gemmini.tile_i = tile_i;
    profiler_tls().last_gemmini.tile_j = tile_j;
    profiler_tls().last_gemmini.tile_k = tile_k;
    profiler_tls().last_gemmini.page_packed_mask = packing.mask();
    profiler_tls().last_gemmini.repack_us = 0;
    profiler_tls().last_gemmini.repack_cycles = 0;
    profiler_tls().last_gemmini.weight_prepare_us = weight_prepare_us;
    profiler_tls().last_gemmini.weight_prepare_cycles = weight_prepare_cycles;
    profiler_tls().last_gemmini.activation_quant_us = activation_quant_us;
    profiler_tls().last_gemmini.activation_quant_cycles = activation_quant_cycles;
    profiler_tls().last_gemmini.prefault_us = prefault_us;
    profiler_tls().last_gemmini.prefault_cycles = prefault_cycles;
    profiler_tls().last_gemmini.flush_us = flush_us;
    profiler_tls().last_gemmini.flush_cycles = flush_cycles;
    profiler_tls().last_gemmini.job_configuration_us = job_configuration_us;
    profiler_tls().last_gemmini.job_configuration_cycles = job_configuration_cycles;
    profiler_tls().last_gemmini.calculate_tiling_factors_us = calculate_tiling_factors_us;
    profiler_tls().last_gemmini.calculate_tiling_factors_cycles = calculate_tiling_factors_cycles;
    profiler_tls().last_gemmini.gemmini_configuration_us = gemmini_configuration_us;
    profiler_tls().last_gemmini.gemmini_configuration_cycles = gemmini_configuration_cycles;
    profiler_tls().last_gemmini.gemmini_run_us = gemmini_run_us;
    profiler_tls().last_gemmini.gemmini_run_cycles = gemmini_run_cycles;
    profiler_tls().last_gemmini.gemmini_call_us = gemmini_call_us;
    profiler_tls().last_gemmini.gemmini_call_cycles = gemmini_call_cycles;
    profiler_tls().last_gemmini.output_dequant_store_us = output_dequant_store_us;
    profiler_tls().last_gemmini.output_dequant_store_cycles = output_dequant_store_cycles;
    profiler_tls().last_gemmini.total_cycles = total_cycles;
    profiler_tls().last_gemmini.load_cycles = load_cycles;
    profiler_tls().last_gemmini.preload_cycles = preload_cycles;
    profiler_tls().last_gemmini.compute_cycles = compute_cycles;
    profiler_tls().last_gemmini.store_cycles = store_cycles;
    profiler_tls().last_gemmini.wait_cycles = wait_cycles;
    profiler_tls().last_gemmini.valid = true;
}

static const char * ggml_backend_gemmini_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "GEMMINI";
}

static void ggml_backend_gemmini_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_gemmini_context *>(backend->context);
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_gemmini_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_gemmini_mul_mat(node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_gemmini_i = {
    /* .get_name            = */ ggml_backend_gemmini_get_name,
    /* .free                = */ ggml_backend_gemmini_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_gemmini_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

static ggml_guid_t ggml_backend_gemmini_guid() {
    static ggml_guid guid = { 0x6a, 0x9a, 0x4f, 0x18, 0x1f, 0x4b, 0x47, 0xcd, 0x8e, 0x6e, 0xb4, 0x9b, 0x4a, 0xb0, 0x3c, 0x21 };
    return &guid;
}

} // namespace

void ggml_gemmini_weight_cache_clear(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    weight_cache().clear();
}

void ggml_gemmini_weight_pack_clear(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    prepacked_weight_store().clear();
}

int ggml_gemmini_weight_pack_load(const char * path) {
    return load_prepacked_weight_file(path);
}

int ggml_gemmini_prepack_weight(const ggml_tensor * tensor) {
    if (!can_prepack_weight_tensor(tensor)) {
        return 0;
    }

    const std::string src0_name = tensor_name(tensor);
    const int layer_index = parse_layer_index("prepack", src0_name);
    int slices = 0;

    const auto * base = reinterpret_cast<const uint8_t *>(tensor->data);
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            const uint8_t * slice_ptr = base + i2 * tensor->nb[2] + i3 * tensor->nb[3];
            if (!cache_prepacked_weight_slice(
                        slice_ptr,
                        src0_name,
                        i2,
                        i3,
                        tensor->ne[0],
                        tensor->ne[1])) {
                get_or_create_weight_cache(
                    tensor,
                    slice_ptr,
                    slice_ptr,
                    tensor->ne[1],
                    tensor->ne[0],
                    tensor->nb[1],
                    layer_index,
                    "prepack",
                    src0_name);
            }
            ++slices;
        }
    }

    return slices;
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    auto * ctx = new ggml_backend_gemmini_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_gemmini_guid(),
        /* .iface   = */ ggml_backend_gemmini_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

static void ggml_backend_gemmini_set_n_threads(ggml_backend_t backend, int n_threads) {
    auto * ctx = static_cast<ggml_backend_gemmini_context *>(backend->context);
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_gemmini_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "GEMMINI";
}

static const char * ggml_backend_gemmini_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Gemmini tiled int8 accelerator backend";
}

static void ggml_backend_gemmini_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_gemmini_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_gemmini_device_get_name(dev);
    props->description = ggml_backend_gemmini_device_get_description(dev);
    props->type = ggml_backend_gemmini_device_get_type(dev);
    ggml_backend_gemmini_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_gemmini_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_gemmini_init();
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_gemmini_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (env_flag("GGML_GEMMINI_DISABLE", false) || gemmini_active_mask() == 0) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            return gemmini_can_offload(op) && gemmini_admit_offload(op);
        default:
            return false;
    }
}

static bool ggml_backend_gemmini_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_gemmini_device_i = {
    /* .get_name             = */ ggml_backend_gemmini_device_get_name,
    /* .get_description      = */ ggml_backend_gemmini_device_get_description,
    /* .get_memory           = */ ggml_backend_gemmini_device_get_memory,
    /* .get_type             = */ ggml_backend_gemmini_device_get_type,
    /* .get_props            = */ ggml_backend_gemmini_device_get_props,
    /* .init_backend         = */ ggml_backend_gemmini_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_gemmini_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_gemmini_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_gemmini_device_supports_op,
    /* .supports_buft        = */ ggml_backend_gemmini_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// registry interface

static const char * ggml_backend_gemmini_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "GEMMINI";
}

static size_t ggml_backend_gemmini_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_gemmini_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &device;
}

static void * ggml_backend_gemmini_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return reinterpret_cast<void *>(ggml_backend_gemmini_set_n_threads);
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_gemmini_reg_i = {
    /* .get_name         = */ ggml_backend_gemmini_reg_get_name,
    /* .get_device_count = */ ggml_backend_gemmini_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gemmini_reg_get_device,
    /* .get_proc_address = */ ggml_backend_gemmini_get_proc_address,
};

ggml_backend_reg_t ggml_backend_gemmini_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_gemmini_reg_i,
        /* .context     = */ nullptr,
    };

    return &reg;
}

void ggml_gemmini_profiler_reset(void) {
    {
        auto & shared = profiler_shared();
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.model_load_us = 0;
        shared.model_load_cycles = 0;
        shared.summary_model_name.clear();
        shared.summary_input_prompt.clear();
        shared.summary_final_output.clear();
        shared.runs.clear();
        shared.op_events.clear();
        shared.token_events.clear();
    }
    profiler_tls() = profiler_thread_state{};
    reset_offload_admission();
}

void ggml_gemmini_profiler_reset_events(void) {
    {
        auto & shared = profiler_shared();
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.model_load_us = 0;
        shared.model_load_cycles = 0;
        shared.summary_model_name.clear();
        shared.summary_input_prompt.clear();
        shared.summary_final_output.clear();
        shared.runs.clear();
        shared.op_events.clear();
        shared.token_events.clear();
    }
    profiler_tls() = profiler_thread_state{};
    reset_offload_admission();
}

void ggml_gemmini_profiler_set_enabled(bool enabled) {
    profiler_tls().enabled = enabled;
}

bool ggml_gemmini_profiler_is_enabled(void) {
    return profiler_tls().enabled;
}

void ggml_gemmini_runtime_configure_request(int32_t active_mask, int32_t gemmini_id, bool split_mode, bool single_mode) {
    request_runtime().active_mask = active_mask;
    request_runtime().gemmini_id = gemmini_id;
    request_runtime().split_mode = split_mode;
    request_runtime().single_mode = single_mode;
}

void ggml_gemmini_runtime_clear_request(void) {
    request_runtime() = request_runtime_config{};
}

int ggml_gemmini_smoke_matmul(int32_t rows_i, int32_t cols_out_i, int32_t cols_in_i) {
    if (rows_i <= 0 || cols_out_i <= 0 || cols_in_i <= 0) {
        std::fprintf(stderr, "GEMMINI-SMOKE-ERROR,reason=bad_shape,rows=%d,cols_out=%d,cols_in=%d\n",
            rows_i, cols_out_i, cols_in_i);
        std::fflush(stderr);
        return -1;
    }

    const int mask = gemmini_active_mask();
    if (mask == 0) {
        std::fprintf(stderr, "GEMMINI-SMOKE-ERROR,reason=active_mask_zero\n");
        std::fflush(stderr);
        return -2;
    }

    try {
        const size_t rows = static_cast<size_t>(rows_i);
        const size_t cols_out = static_cast<size_t>(cols_out_i);
        const size_t cols_in = static_cast<size_t>(cols_in_i);
        const gemmini_page_packing_config & packing = gemmini_page_packing();

        gemmini_aligned_vector<elem_t> a_row_major(rows * cols_in);
        gemmini_aligned_vector<elem_t> b_row_major(cols_in * cols_out);
        gemmini_aligned_vector<acc_t> d_row_major(rows * cols_out);

        for (size_t r = 0; r < rows; ++r) {
            for (size_t k = 0; k < cols_in; ++k) {
                const int value = static_cast<int>((r * 17 + k * 7) % 15) - 7;
                a_row_major[r * cols_in + k] = static_cast<elem_t>(value);
            }
        }

        for (size_t k = 0; k < cols_in; ++k) {
            for (size_t c = 0; c < cols_out; ++c) {
                const int value = static_cast<int>((k * 13 + c * 5) % 11) - 5;
                b_row_major[k * cols_out + c] = static_cast<elem_t>(value);
            }
        }

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols_out; ++c) {
                d_row_major[r * cols_out + c] = static_cast<acc_t>(
                    static_cast<int>((r * 3 + c * 7) % 13) - 6);
            }
        }

        gemmini_aligned_vector<elem_t> a = convert_a_from_row_major(
            a_row_major.data(), rows, cols_in, packing.a);
        gemmini_aligned_vector<elem_t> b = convert_b_layout(
            b_row_major.data(), cols_in, cols_out, false, packing.b);
        gemmini_aligned_vector<acc_t> d = convert_acc_layout(
            d_row_major.data(), rows, cols_out, false, packing.d, "D");
        gemmini_aligned_vector<acc_t> out(
            storage_elements<acc_t>(acc_storage_bytes(rows, cols_out, packing.c, "C"), "C"),
            0);

        std::printf("GEMMINI-SMOKE-BEGIN,rows=%zu,cols_out=%zu,cols_in=%zu,mask=0x%x,page_packed_a=%d,page_packed_b=%d,page_packed_c=%d,page_packed_d=%d\n",
            rows,
            cols_out,
            cols_in,
            mask,
            packing.a ? 1 : 0,
            packing.b ? 1 : 0,
            packing.c ? 1 : 0,
            packing.d ? 1 : 0);
        std::fflush(stdout);

        const gemmini_matmul_metrics metrics =
            run_gemmini_matmul(
                mask,
                rows,
                cols_out,
                cols_in,
                a.data(),
                b.data(),
                d.data(),
                out.data(),
                packing);

        gemmini_aligned_vector<acc_t> out_row_major = convert_acc_layout(
            out.data(), rows, cols_out, packing.c, false, "C");

        std::printf("GEMMINI-SMOKE-RUN-DONE,total_cycles=%llu\n",
            static_cast<unsigned long long>(metrics.total_cycles));
        std::fflush(stdout);

        size_t mismatch_count = 0;
        size_t first_r = 0;
        size_t first_c = 0;
        int64_t first_expected = 0;
        int64_t first_actual = 0;

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols_out; ++c) {
                int64_t expected = d_row_major[r * cols_out + c];
                for (size_t k = 0; k < cols_in; ++k) {
                    expected += static_cast<int64_t>(a_row_major[r * cols_in + k]) *
                                static_cast<int64_t>(b_row_major[k * cols_out + c]);
                }

                const int64_t actual = static_cast<int64_t>(out_row_major[r * cols_out + c]);
                if (actual != expected) {
                    if (mismatch_count == 0) {
                        first_r = r;
                        first_c = c;
                        first_expected = expected;
                        first_actual = actual;
                    }
                    ++mismatch_count;
                }
            }
        }

        if (mismatch_count != 0) {
            std::printf("GEMMINI-SMOKE-FAIL,mismatches=%zu,first_r=%zu,first_c=%zu,expected=%lld,actual=%lld\n",
                mismatch_count,
                first_r,
                first_c,
                static_cast<long long>(first_expected),
                static_cast<long long>(first_actual));
            std::fflush(stdout);
            return 1;
        }

        std::printf("GEMMINI-SMOKE-PASS,total_cycles=%llu\n",
            static_cast<unsigned long long>(metrics.total_cycles));
        std::fflush(stdout);
        return 0;
    } catch (...) {
        std::fprintf(stderr, "GEMMINI-SMOKE-ERROR,reason=exception\n");
        std::fflush(stderr);
        return -3;
    }
}

void ggml_gemmini_profiler_start_run(const char * run_label) {
    reset_offload_admission();
    profiler_tls().current_run = run_label ? run_label : "hybrid";
    profiler_tls().current_phase = "prefill";
    profiler_tls().token_index = -1;
    profiler_tls().pending_eval = pending_eval_state{};
    profiler_tls().last_gemmini = last_gemmini_state{};
    profiler_tls().graph_call_profiled_us = 0;
    profiler_tls().graph_call_profiled_cycles = 0;
}

void ggml_gemmini_profiler_set_phase(const char * phase, int32_t token_index) {
    profiler_tls().current_phase = phase ? phase : "prefill";
    profiler_tls().token_index = token_index;
}

void ggml_gemmini_profiler_graph_overhead_begin(void) {
    profiler_tls().graph_call_profiled_us = 0;
    profiler_tls().graph_call_profiled_cycles = 0;
}

void ggml_gemmini_profiler_graph_overhead_end(const char * phase, int32_t token_index, int64_t wall_us, uint64_t cycles) {
    if (!profiler_tls().enabled) {
        return;
    }

    const int64_t profiled_us = profiler_tls().graph_call_profiled_us;
    const uint64_t profiled_cycles = profiler_tls().graph_call_profiled_cycles;
    const int64_t overhead_us = wall_us > profiled_us ? wall_us - profiled_us : 0;
    const uint64_t overhead_cycles = cycles > profiled_cycles ? cycles - profiled_cycles : 0;
    if (overhead_us == 0 && overhead_cycles == 0) {
        return;
    }

    gemmini_op_event event;
    event.run_label = current_run_label();
    event.phase = "graph_build/scheduler_overhead";
    event.backend = "llama_runtime";
    event.stage = "graph_build/scheduler_overhead";
    event.node_name = phase ? phase : profiler_tls().current_phase;
    event.op_name = "llama_decode";
    event.token_index = token_index;
    event.wall_us = overhead_us;
    event.wall_cycles = overhead_cycles;
    profiler_add_event(event);
}

void ggml_gemmini_profiler_note_model_load(int64_t wall_us, uint64_t cycles) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.model_load_us = wall_us;
    shared.model_load_cycles = cycles;
}

void ggml_gemmini_profiler_note_prompt(int32_t prompt_tokens) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.runs[current_run_label()].prompt_tokens = prompt_tokens;
}

void ggml_gemmini_profiler_note_summary_metadata(const char * model_name, const char * input_prompt, const char * final_output) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.summary_model_name = model_name ? model_name : "";
    shared.summary_input_prompt = input_prompt ? input_prompt : "";
    shared.summary_final_output = final_output ? final_output : "";
}

void ggml_gemmini_profiler_note_gemmini_config(int32_t active_mask) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    run_totals & run = shared.runs[current_run_label()];
    run.gemmini_mask = active_mask & 0xf;
    run.gemmini_count = popcount4(run.gemmini_mask);
}

void ggml_gemmini_profiler_note_run_metadata(const char * model_name, const char * input_prompt, const char * final_output) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    run_totals & run = shared.runs[current_run_label()];
    run.model_name = model_name ? model_name : "";
    run.input_prompt = input_prompt ? input_prompt : "";
    run.final_output = final_output ? final_output : "";
}

void ggml_gemmini_profiler_note_sampling(int32_t token_index, int64_t wall_us, uint64_t cycles) {
    gemmini_op_event event;
    event.run_label = current_run_label();
    event.phase = "sampling";
    event.backend = profiler_tls().current_run == "cpu_baseline" ? "cpu_baseline" : "cpu_fallback";
    event.stage = "other_cpu";
    event.node_name = "sampling";
    event.op_name = "sampling";
    event.token_index = token_index;
    event.wall_us = wall_us;
    event.wall_cycles = cycles;
    profiler_add_event(event);
}

void ggml_gemmini_profiler_note_run_total(
        int64_t wall_us,
        uint64_t cycles,
        int32_t generated_tokens,
        int64_t ttft_us,
        uint64_t ttft_cycles,
        double tpot_us,
        double tpot_cycles) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    run_totals & run = shared.runs[current_run_label()];
    run.wall_us = wall_us;
    run.wall_cycles = cycles;
    run.generated_tokens = generated_tokens;
    run.ttft_us = ttft_us;
    run.ttft_cycles = ttft_cycles;
    run.tpot_us = tpot_us;
    run.tpot_cycles = tpot_cycles;
}

void ggml_gemmini_profiler_note_token(int32_t token_index, int32_t token_id, const char * piece) {
    auto & shared = profiler_shared();
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.token_events.push_back(token_event{
        current_run_label(),
        token_index,
        token_id,
        piece ? piece : "",
    });
}

void ggml_gemmini_profiler_eval_begin(const ggml_tensor * t, uint64_t cycles, int64_t time_us) {
    if (!profiler_tls().enabled) {
        return;
    }

    profiler_tls().pending_eval.tensor = t;
    profiler_tls().pending_eval.start_cycles = cycles;
    profiler_tls().pending_eval.start_us = time_us;
}

void ggml_gemmini_profiler_eval_end(const ggml_tensor * t, uint64_t cycles, int64_t time_us, bool cpu_baseline) {
    if (!profiler_tls().enabled) {
        return;
    }

    const std::string node_name = effective_node_name(t);
    const std::string src0_name = source_name(t, 0);

    gemmini_op_event event;
    event.run_label = current_run_label();
    event.phase = profiler_tls().current_phase;
    event.backend = cpu_baseline ? "cpu_baseline" : "cpu_fallback";
    event.node_name = node_name;
    event.src0_name = src0_name;
    event.op_name = ggml_op_desc(t);
    event.stage = classify_stage(node_name, src0_name, event.op_name);
    event.layer_index = parse_layer_index(node_name, src0_name);
    event.token_index = profiler_tls().token_index;
    event.wall_us = time_us - profiler_tls().pending_eval.start_us;
    event.wall_cycles = cycles - profiler_tls().pending_eval.start_cycles;
    profiler_tls().graph_call_profiled_us += event.wall_us;
    profiler_tls().graph_call_profiled_cycles += event.wall_cycles;

    if (!cpu_baseline && profiler_tls().last_gemmini.valid && profiler_tls().last_gemmini.tensor == t) {
        event.backend = "gemmini";
        event.stage = profiler_tls().last_gemmini.stage;
        event.layer_index = profiler_tls().last_gemmini.layer_index;
        event.token_index = profiler_tls().last_gemmini.token_index;
        event.weight_prepare_us = profiler_tls().last_gemmini.weight_prepare_us;
        event.weight_prepare_cycles = profiler_tls().last_gemmini.weight_prepare_cycles;
        event.activation_quant_us = profiler_tls().last_gemmini.activation_quant_us;
        event.activation_quant_cycles = profiler_tls().last_gemmini.activation_quant_cycles;
        event.prefault_us = profiler_tls().last_gemmini.prefault_us;
        event.prefault_cycles = profiler_tls().last_gemmini.prefault_cycles;
        event.flush_us = profiler_tls().last_gemmini.flush_us;
        event.flush_cycles = profiler_tls().last_gemmini.flush_cycles;
        event.job_configuration_us = profiler_tls().last_gemmini.job_configuration_us;
        event.job_configuration_cycles = profiler_tls().last_gemmini.job_configuration_cycles;
        event.calculate_tiling_factors_us = profiler_tls().last_gemmini.calculate_tiling_factors_us;
        event.calculate_tiling_factors_cycles = profiler_tls().last_gemmini.calculate_tiling_factors_cycles;
        event.gemmini_configuration_us = profiler_tls().last_gemmini.gemmini_configuration_us;
        event.gemmini_configuration_cycles = profiler_tls().last_gemmini.gemmini_configuration_cycles;
        event.gemmini_run_us = profiler_tls().last_gemmini.gemmini_run_us;
        event.gemmini_run_cycles = profiler_tls().last_gemmini.gemmini_run_cycles;
        event.gemmini_call_us = profiler_tls().last_gemmini.gemmini_call_us;
        event.gemmini_call_cycles = profiler_tls().last_gemmini.gemmini_call_cycles;
        event.output_dequant_store_us = profiler_tls().last_gemmini.output_dequant_store_us;
        event.output_dequant_store_cycles = profiler_tls().last_gemmini.output_dequant_store_cycles;
        event.gemmini_total_cycles = profiler_tls().last_gemmini.total_cycles;
        event.gemmini_load_cycles = profiler_tls().last_gemmini.load_cycles;
        event.gemmini_preload_cycles = profiler_tls().last_gemmini.preload_cycles;
        event.gemmini_compute_cycles = profiler_tls().last_gemmini.compute_cycles;
        event.gemmini_store_cycles = profiler_tls().last_gemmini.store_cycles;
        event.gemmini_wait_cycles = profiler_tls().last_gemmini.wait_cycles;
        event.dim_i = profiler_tls().last_gemmini.dim_i;
        event.dim_j = profiler_tls().last_gemmini.dim_j;
        event.dim_k = profiler_tls().last_gemmini.dim_k;
        event.tile_i = profiler_tls().last_gemmini.tile_i;
        event.tile_j = profiler_tls().last_gemmini.tile_j;
        event.tile_k = profiler_tls().last_gemmini.tile_k;
        event.page_packed_mask = profiler_tls().last_gemmini.page_packed_mask;
        profiler_tls().last_gemmini.valid = false;
    }

    profiler_add_event(event);
}

struct aggregate_entry {
    int64_t wall_us = 0;
    uint64_t wall_cycles = 0;
};

struct stage_aggregate_entry {
    aggregate_entry total;
    aggregate_entry gemmini;
    aggregate_entry cpu;
    std::map<std::string, aggregate_entry> gemmini_components;
};

struct matmul_shape_key {
    std::string run_label;
    std::string phase;
    int64_t dim_i = 0;
    int64_t dim_j = 0;
    int64_t dim_k = 0;
    int64_t tile_i = 0;
    int64_t tile_j = 0;
    int64_t tile_k = 0;
    uint8_t page_packed_mask = 0;

    bool operator<(const matmul_shape_key & other) const {
        if (run_label != other.run_label) {
            return run_label < other.run_label;
        }
        if (phase != other.phase) {
            return phase < other.phase;
        }
        if (dim_i != other.dim_i) {
            return dim_i < other.dim_i;
        }
        if (dim_j != other.dim_j) {
            return dim_j < other.dim_j;
        }
        if (dim_k != other.dim_k) {
            return dim_k < other.dim_k;
        }
        if (tile_i != other.tile_i) {
            return tile_i < other.tile_i;
        }
        if (tile_j != other.tile_j) {
            return tile_j < other.tile_j;
        }
        if (tile_k != other.tile_k) {
            return tile_k < other.tile_k;
        }
        return page_packed_mask < other.page_packed_mask;
    }
};

struct matmul_shape_entry {
    int64_t count = 0;
    aggregate_entry total;
    std::map<std::string, aggregate_entry> components;
};

static double pct_u64(uint64_t value, uint64_t total) {
    if (total == 0) {
        return 0.0;
    }
    return (100.0 * static_cast<double>(value)) / static_cast<double>(total);
}

static double pct_i64(int64_t value, int64_t total) {
    if (total == 0) {
        return 0.0;
    }
    return (100.0 * static_cast<double>(value)) / static_cast<double>(total);
}

static std::map<std::string, aggregate_entry> aggregate_by_run(
        const std::map<std::pair<std::string, std::string>, aggregate_entry> & values) {
    std::map<std::string, aggregate_entry> by_run;
    for (const auto & kv : values) {
        by_run[kv.first.first].wall_us += kv.second.wall_us;
        by_run[kv.first.first].wall_cycles += kv.second.wall_cycles;
    }
    return by_run;
}

static bool is_decode_graph_phase(const std::string & phase) {
    return phase == "prefill" || phase == "decode";
}

static void add_stage_component(
        std::map<std::string, aggregate_entry> & values,
        const std::string & name,
        int64_t wall_us,
        uint64_t wall_cycles) {
    if (wall_us == 0 && wall_cycles == 0) {
        return;
    }
    values[name].wall_us += wall_us;
    values[name].wall_cycles += wall_cycles;
}

static void add_stage_event(stage_aggregate_entry & entry, const gemmini_op_event & event) {
    entry.total.wall_us += event.wall_us;
    entry.total.wall_cycles += event.wall_cycles;

    aggregate_entry gemmini_time;
    if (event.backend == "gemmini") {
        gemmini_time.wall_us = event.wall_us;
        gemmini_time.wall_cycles = event.wall_cycles;
        add_stage_component(entry.gemmini_components, "weight_prepare",
            event.weight_prepare_us, event.weight_prepare_cycles);
        add_stage_component(entry.gemmini_components, "activation_quant",
            event.activation_quant_us, event.activation_quant_cycles);
        add_stage_component(entry.gemmini_components, "prefault",
            event.prefault_us, event.prefault_cycles);
        add_stage_component(entry.gemmini_components, "flush",
            event.flush_us, event.flush_cycles);
        add_stage_component(entry.gemmini_components, "job_configuration",
            event.job_configuration_us, event.job_configuration_cycles);
        add_stage_component(entry.gemmini_components, "calculate_tiling_factors",
            event.calculate_tiling_factors_us, event.calculate_tiling_factors_cycles);
        add_stage_component(entry.gemmini_components, "gemmini_configuration",
            event.gemmini_configuration_us, event.gemmini_configuration_cycles);
        add_stage_component(entry.gemmini_components, "gemmini_run",
            event.gemmini_run_us, event.gemmini_run_cycles);
        add_stage_component(entry.gemmini_components, "output_dequant_store",
            event.output_dequant_store_us, event.output_dequant_store_cycles);
    }

    entry.gemmini.wall_us += gemmini_time.wall_us;
    entry.gemmini.wall_cycles += gemmini_time.wall_cycles;
    entry.cpu.wall_us += event.wall_us - gemmini_time.wall_us;
    entry.cpu.wall_cycles += event.wall_cycles - gemmini_time.wall_cycles;
}

static std::string stage_summary_phase_label(const gemmini_op_event & event) {
    if (event.phase == "decode" && event.token_index >= 0) {
        return "decode" + std::to_string(event.token_index);
    }
    return event.phase;
}

static int stage_summary_phase_sort_rank(const std::string & phase) {
    if (phase == "prefill") {
        return 0;
    }
    const std::string prefix = "decode";
    if (phase.rfind(prefix, 0) == 0 && phase.size() > prefix.size()) {
        const std::string suffix = phase.substr(prefix.size());
        bool numeric = true;
        for (char ch : suffix) {
            numeric = numeric && ch >= '0' && ch <= '9';
        }
        if (numeric) {
            return std::stoi(suffix) + 1;
        }
    }
    return std::numeric_limits<int>::max();
}

static std::vector<std::string> sorted_stage_summary_phases(
        const std::map<std::string, std::map<std::string, stage_aggregate_entry>> & values) {
    std::vector<std::string> phases;
    phases.reserve(values.size());
    for (const auto & kv : values) {
        phases.push_back(kv.first);
    }
    std::sort(phases.begin(), phases.end(),
        [](const std::string & lhs, const std::string & rhs) {
            const int lhs_rank = stage_summary_phase_sort_rank(lhs);
            const int rhs_rank = stage_summary_phase_sort_rank(rhs);
            if (lhs_rank != rhs_rank) {
                return lhs_rank < rhs_rank;
            }
            return lhs < rhs;
        });
    return phases;
}

static aggregate_entry aggregate_stage_entries(const std::map<std::string, stage_aggregate_entry> & values) {
    aggregate_entry total;
    for (const auto & kv : values) {
        total.wall_us += kv.second.total.wall_us;
        total.wall_cycles += kv.second.total.wall_cycles;
    }
    return total;
}

static int stage_sort_rank(const std::string & stage) {
    static const std::vector<std::string> order = {
        "input_embed",
        "residual_slice",
        "attn_norm",
        "q_proj",
        "q_reshape",
        "q_rope",
        "q_permute",
        "k_proj",
        "k_reshape",
        "k_rope",
        "v_proj",
        "v_reshape",
        "qkv_proj",
        "kv_cache_write",
        "kv_cache_read",
        "attn_mask",
        "attn_score",
        "attn_softmax",
        "attn_fused",
        "attn_output_reshape",
        "attn_value",
        "o_proj",
        "attn_residual",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_activation",
        "ffn_swiglu",
        "ffn_down",
        "ffn_residual",
        "final_norm",
        "lm_head",
        "other_cpu",
        "graph_build/scheduler_overhead",
    };

    const auto it = std::find(order.begin(), order.end(), stage);
    if (it == order.end()) {
        return static_cast<int>(order.size());
    }
    return static_cast<int>(it - order.begin());
}

static std::vector<std::pair<std::string, stage_aggregate_entry>> sorted_stage_entries(
        const std::map<std::string, stage_aggregate_entry> & values) {
    std::vector<std::pair<std::string, stage_aggregate_entry>> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto & lhs, const auto & rhs) {
            const int lhs_rank = stage_sort_rank(lhs.first);
            const int rhs_rank = stage_sort_rank(rhs.first);
            if (lhs_rank != rhs_rank) {
                return lhs_rank < rhs_rank;
            }
            return lhs.first < rhs.first;
        });
    return sorted;
}

static aggregate_entry runtime_denominator(
        const std::string & run_label,
        const std::map<std::string, run_totals> & runs,
        const std::map<std::string, aggregate_entry> & profiled_by_run) {
    aggregate_entry denom;

    const auto profiled_it = profiled_by_run.find(run_label);
    if (profiled_it != profiled_by_run.end()) {
        denom = profiled_it->second;
    }

    const auto run_it = runs.find(run_label);
    if (run_it != runs.end()) {
        denom.wall_us = std::max<int64_t>(denom.wall_us, run_it->second.wall_us);
        denom.wall_cycles = std::max<uint64_t>(denom.wall_cycles, run_it->second.wall_cycles);
    }

    return denom;
}

static aggregate_entry runtime_overhead(
        const std::string & run_label,
        const std::map<std::string, run_totals> & runs,
        const std::map<std::string, aggregate_entry> & profiled_by_run) {
    aggregate_entry overhead;

    const auto run_it = runs.find(run_label);
    if (run_it == runs.end()) {
        return overhead;
    }

    const auto profiled_it = profiled_by_run.find(run_label);
    const aggregate_entry profiled = profiled_it == profiled_by_run.end() ? aggregate_entry{} : profiled_it->second;

    if (run_it->second.wall_us > profiled.wall_us) {
        overhead.wall_us = run_it->second.wall_us - profiled.wall_us;
    }
    if (run_it->second.wall_cycles > profiled.wall_cycles) {
        overhead.wall_cycles = run_it->second.wall_cycles - profiled.wall_cycles;
    }

    return overhead;
}

static bool has_aggregate_value(const aggregate_entry & value) {
    return value.wall_us != 0 || value.wall_cycles != 0;
}

static std::string metric_i64_or_na(bool valid, int64_t value) {
    return valid ? std::to_string(value) : "n/a";
}

static std::string metric_u64_or_na(bool valid, uint64_t value) {
    return valid ? std::to_string(value) : "n/a";
}

static std::string metric_double_or_na(bool valid, double value) {
    if (!valid) {
        return "n/a";
    }
    std::ostringstream out;
    out << value;
    return out.str();
}

static std::string metric_double_fixed_or_na(bool valid, double value) {
    if (!valid) {
        return "n/a";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

static void add_aggregate(
        std::map<std::pair<std::string, std::string>, aggregate_entry> & values,
        const std::string & run_label,
        const std::string & name,
        int64_t wall_us,
        uint64_t wall_cycles) {
    if (wall_us == 0 && wall_cycles == 0) {
        return;
    }
    values[{run_label, name}].wall_us += wall_us;
    values[{run_label, name}].wall_cycles += wall_cycles;
}

static const std::vector<std::string> & backend_gemmini_breakdown_order() {
    static const std::vector<std::string> order = {
        "weight_prepare",
        "activation_quant",
        "prefault",
        "flush",
        "job_configuration",
        "calculate_tiling_factors",
        "gemmini_configuration",
        "gemmini_run",
        "output_dequant_store",
        "other_host",
    };
    return order;
}

static std::string matmul_shape_string(const matmul_shape_key & key) {
    std::ostringstream out;
    out << key.dim_i << "x" << key.dim_j << "x" << key.dim_k;
    return out.str();
}

static std::string matmul_tiling_string(const matmul_shape_key & key) {
    std::ostringstream out;
    out << key.tile_i << "x" << key.tile_j << "x" << key.tile_k;
    return out.str();
}

static bool is_gemmini_mul_mat_event(const gemmini_op_event & event) {
    return event.backend == "gemmini" &&
        event.op_name == "MUL_MAT" &&
        is_decode_graph_phase(event.phase) &&
        event.dim_i > 0 &&
        event.dim_j > 0 &&
        event.dim_k > 0;
}

static aggregate_entry gemmini_event_component(const gemmini_op_event & event, const std::string & name) {
    if (name == "weight_prepare") {
        return {event.weight_prepare_us, event.weight_prepare_cycles};
    }
    if (name == "activation_quant") {
        return {event.activation_quant_us, event.activation_quant_cycles};
    }
    if (name == "prefault") {
        return {event.prefault_us, event.prefault_cycles};
    }
    if (name == "flush") {
        return {event.flush_us, event.flush_cycles};
    }
    if (name == "job_configuration") {
        return {event.job_configuration_us, event.job_configuration_cycles};
    }
    if (name == "calculate_tiling_factors") {
        return {event.calculate_tiling_factors_us, event.calculate_tiling_factors_cycles};
    }
    if (name == "gemmini_configuration") {
        return {event.gemmini_configuration_us, event.gemmini_configuration_cycles};
    }
    if (name == "gemmini_run") {
        return {event.gemmini_run_us, event.gemmini_run_cycles};
    }
    if (name == "output_dequant_store") {
        return {event.output_dequant_store_us, event.output_dequant_store_cycles};
    }
    if (name == "other_host") {
        const int64_t accounted_us =
            event.weight_prepare_us +
            event.activation_quant_us +
            event.prefault_us +
            event.flush_us +
            event.job_configuration_us +
            event.calculate_tiling_factors_us +
            event.gemmini_configuration_us +
            event.gemmini_run_us +
            event.output_dequant_store_us;
        const uint64_t accounted_cycles =
            event.weight_prepare_cycles +
            event.activation_quant_cycles +
            event.prefault_cycles +
            event.flush_cycles +
            event.job_configuration_cycles +
            event.calculate_tiling_factors_cycles +
            event.gemmini_configuration_cycles +
            event.gemmini_run_cycles +
            event.output_dequant_store_cycles;

        aggregate_entry other;
        if (event.wall_us > accounted_us) {
            other.wall_us = event.wall_us - accounted_us;
        }
        if (event.wall_cycles > accounted_cycles) {
            other.wall_cycles = event.wall_cycles - accounted_cycles;
        }
        return other;
    }

    return {};
}

static void add_matmul_shape_event(matmul_shape_entry & entry, const gemmini_op_event & event) {
    ++entry.count;
    entry.total.wall_us += event.wall_us;
    entry.total.wall_cycles += event.wall_cycles;
    for (const std::string & name : backend_gemmini_breakdown_order()) {
        const aggregate_entry component = gemmini_event_component(event, name);
        if (has_aggregate_value(component)) {
            entry.components[name].wall_us += component.wall_us;
            entry.components[name].wall_cycles += component.wall_cycles;
        }
    }
}

static aggregate_entry stage_gemmini_component(const stage_aggregate_entry & entry, const std::string & name) {
    if (name == "other_host") {
        aggregate_entry accounted;
        for (const auto & kv : entry.gemmini_components) {
            accounted.wall_us += kv.second.wall_us;
            accounted.wall_cycles += kv.second.wall_cycles;
        }

        aggregate_entry other;
        if (entry.gemmini.wall_us > accounted.wall_us) {
            other.wall_us = entry.gemmini.wall_us - accounted.wall_us;
        }
        if (entry.gemmini.wall_cycles > accounted.wall_cycles) {
            other.wall_cycles = entry.gemmini.wall_cycles - accounted.wall_cycles;
        }
        return other;
    }

    const auto it = entry.gemmini_components.find(name);
    return it == entry.gemmini_components.end() ? aggregate_entry{} : it->second;
}

int ggml_gemmini_profiler_write_results(const char * results_dir) {
    const std::string dir = results_dir ? results_dir : ".";
    int64_t model_load_us = 0;
    uint64_t model_load_cycles = 0;
    std::string summary_model_name;
    std::string summary_input_prompt;
    std::string summary_final_output;
    std::map<std::string, run_totals> runs;
    std::vector<gemmini_op_event> op_events;
    std::vector<token_event> token_events;

    {
        auto & shared = profiler_shared();
        std::lock_guard<std::mutex> lock(shared.mutex);
        model_load_us = shared.model_load_us;
        model_load_cycles = shared.model_load_cycles;
        summary_model_name = shared.summary_model_name;
        summary_input_prompt = shared.summary_input_prompt;
        summary_final_output = shared.summary_final_output;
        runs = shared.runs;
        op_events = shared.op_events;
        token_events = shared.token_events;
    }

    {
        std::ofstream out(dir + "/run_summary.csv");
        if (!out) {
            return -1;
        }

        out << "run_label,gemmini_mask,gemmini_count,page_packed_a,page_packed_b,page_packed_c,page_packed_d,prompt_tokens,generated_tokens,total_us,total_cycles,ttft_us,ttft_cycles,tpot_us_per_token,tpot_cycles_per_token,input_prompt,final_output\n";
        const gemmini_page_packing_config & packing = gemmini_page_packing();
        for (const auto & kv : runs) {
            const run_totals & totals = kv.second;
            out << csv_escape(kv.first) << ","
                << format_hex_mask(totals.gemmini_mask) << ","
                << totals.gemmini_count << ","
                << (packing.a ? 1 : 0) << ","
                << (packing.b ? 1 : 0) << ","
                << (packing.c ? 1 : 0) << ","
                << (packing.d ? 1 : 0) << ","
                << totals.prompt_tokens << ","
                << totals.generated_tokens << ","
                << totals.wall_us << ","
                << totals.wall_cycles << ","
                << metric_i64_or_na(totals.generated_tokens > 0, totals.ttft_us) << ","
                << metric_u64_or_na(totals.generated_tokens > 0, totals.ttft_cycles) << ","
                << metric_double_or_na(totals.generated_tokens > 1, totals.tpot_us) << ","
                << metric_double_or_na(totals.generated_tokens > 1, totals.tpot_cycles) << ","
                << csv_escape(totals.input_prompt) << ","
                << csv_escape(totals.final_output) << "\n";
        }
    }

    {
        std::ofstream out(dir + "/token_trace.csv");
        if (!out) {
            return -1;
        }

        out << "run_label,token_index,token_id,piece\n";
        for (const auto & token : token_events) {
            out << csv_escape(token.run_label) << ","
                << token.token_index << ","
                << token.token_id << ","
                << csv_escape(token.piece) << "\n";
        }
    }

    {
        std::ofstream out(dir + "/op_profile.csv");
        if (!out) {
            return -1;
        }

        out << "run_label,phase,token_index,backend,stage,layer,node_name,src0_name,op,wall_us,wall_cycles,weight_prepare_us,weight_prepare_cycles,activation_quant_us,activation_quant_cycles,prefault_us,prefault_cycles,flush_us,flush_cycles,job_configuration_us,job_configuration_cycles,calculate_tiling_factors_us,calculate_tiling_factors_cycles,gemmini_configuration_us,gemmini_configuration_cycles,gemmini_run_us,gemmini_run_cycles,gemmini_call_us,gemmini_call_cycles,output_dequant_store_us,output_dequant_store_cycles,gemmini_total_cycles,gemmini_load_cycles,gemmini_preload_cycles,gemmini_compute_cycles,gemmini_store_cycles,gemmini_wait_cycles,dim_i,dim_j,dim_k,tile_i,tile_j,tile_k,page_packed_a,page_packed_b,page_packed_c,page_packed_d\n";
        for (const auto & event : op_events) {
            out << csv_escape(event.run_label) << ","
                << csv_escape(event.phase) << ","
                << event.token_index << ","
                << csv_escape(event.backend) << ","
                << csv_escape(event.stage) << ","
                << event.layer_index << ","
                << csv_escape(event.node_name) << ","
                << csv_escape(event.src0_name) << ","
                << csv_escape(event.op_name) << ","
                << event.wall_us << ","
                << event.wall_cycles << ","
                << event.weight_prepare_us << ","
                << event.weight_prepare_cycles << ","
                << event.activation_quant_us << ","
                << event.activation_quant_cycles << ","
                << event.prefault_us << ","
                << event.prefault_cycles << ","
                << event.flush_us << ","
                << event.flush_cycles << ","
                << event.job_configuration_us << ","
                << event.job_configuration_cycles << ","
                << event.calculate_tiling_factors_us << ","
                << event.calculate_tiling_factors_cycles << ","
                << event.gemmini_configuration_us << ","
                << event.gemmini_configuration_cycles << ","
                << event.gemmini_run_us << ","
                << event.gemmini_run_cycles << ","
                << event.gemmini_call_us << ","
                << event.gemmini_call_cycles << ","
                << event.output_dequant_store_us << ","
                << event.output_dequant_store_cycles << ","
                << event.gemmini_total_cycles << ","
                << event.gemmini_load_cycles << ","
                << event.gemmini_preload_cycles << ","
                << event.gemmini_compute_cycles << ","
                << event.gemmini_store_cycles << ","
                << event.gemmini_wait_cycles << ","
                << event.dim_i << ","
                << event.dim_j << ","
                << event.dim_k << ","
                << event.tile_i << ","
                << event.tile_j << ","
                << event.tile_k << ","
                << ((event.page_packed_mask & 0x1) != 0) << ","
                << ((event.page_packed_mask & 0x2) != 0) << ","
                << ((event.page_packed_mask & 0x4) != 0) << ","
                << ((event.page_packed_mask & 0x8) != 0) << "\n";
        }
    }

    std::map<std::string, std::map<std::string, std::map<std::string, stage_aggregate_entry>>> phase_stage_totals;
    std::map<std::string, std::map<std::string, std::map<std::string, stage_aggregate_entry>>> stage_summary_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> backend_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> phase_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> repack_stage_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> gemmini_host_component_totals;
    std::map<std::pair<std::string, std::string>, uint64_t> gemmini_component_totals;
    std::map<std::string, uint64_t> gemmini_total_cycles_by_run;
    std::map<matmul_shape_key, matmul_shape_entry> gemmini_matmul_shape_totals;

    for (const auto & event : op_events) {
        if (is_gemmini_mul_mat_event(event)) {
            const matmul_shape_key key{
                event.run_label,
                event.phase,
                event.dim_i,
                event.dim_j,
                event.dim_k,
                event.tile_i,
                event.tile_j,
                event.tile_k,
                event.page_packed_mask,
            };
            add_matmul_shape_event(gemmini_matmul_shape_totals[key], event);
        }

        if (event.phase == "load/repack") {
            repack_stage_totals[{event.run_label, event.stage}].wall_us += event.wall_us;
            repack_stage_totals[{event.run_label, event.stage}].wall_cycles += event.wall_cycles;
        } else {
            if (is_decode_graph_phase(event.phase)) {
                add_stage_event(phase_stage_totals[event.run_label][event.phase][event.stage], event);

                const std::string summary_phase = stage_summary_phase_label(event);
                add_stage_event(stage_summary_totals[event.run_label][summary_phase][event.stage], event);
            }

            if (event.backend == "gemmini") {
                add_aggregate(
                    backend_totals,
                    event.run_label,
                    "Gemmini",
                    event.wall_us,
                    event.wall_cycles);
            }

            phase_totals[{event.run_label, event.phase}].wall_us += event.wall_us;
            phase_totals[{event.run_label, event.phase}].wall_cycles += event.wall_cycles;
        }

        if (event.backend == "gemmini") {
            add_aggregate(gemmini_host_component_totals, event.run_label, "weight_prepare",
                event.weight_prepare_us, event.weight_prepare_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "activation_quant",
                event.activation_quant_us, event.activation_quant_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "prefault",
                event.prefault_us, event.prefault_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "flush",
                event.flush_us, event.flush_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "job_configuration",
                event.job_configuration_us, event.job_configuration_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "calculate_tiling_factors",
                event.calculate_tiling_factors_us, event.calculate_tiling_factors_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "gemmini_configuration",
                event.gemmini_configuration_us, event.gemmini_configuration_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "gemmini_run",
                event.gemmini_run_us, event.gemmini_run_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "output_dequant_store",
                event.output_dequant_store_us, event.output_dequant_store_cycles);

            gemmini_total_cycles_by_run[event.run_label] += event.gemmini_total_cycles;
            gemmini_component_totals[{event.run_label, "load"}] += event.gemmini_load_cycles;
            gemmini_component_totals[{event.run_label, "preload"}] += event.gemmini_preload_cycles;
            gemmini_component_totals[{event.run_label, "compute"}] += event.gemmini_compute_cycles;
            gemmini_component_totals[{event.run_label, "store"}] += event.gemmini_store_cycles;
            gemmini_component_totals[{event.run_label, "wait"}] += event.gemmini_wait_cycles;
        }
    }

    for (const auto & run_kv : runs) {
        const std::pair<std::string, std::string> gemmini_key{run_kv.first, "Gemmini"};
        const aggregate_entry gemmini = backend_totals[gemmini_key];
        aggregate_entry cpu;
        if (run_kv.second.wall_us > gemmini.wall_us) {
            cpu.wall_us = run_kv.second.wall_us - gemmini.wall_us;
        }
        if (run_kv.second.wall_cycles > gemmini.wall_cycles) {
            cpu.wall_cycles = run_kv.second.wall_cycles - gemmini.wall_cycles;
        }
        backend_totals[{run_kv.first, "CPU"}] = cpu;
    }

    std::map<std::string, aggregate_entry> gemmini_breakdown_by_run = aggregate_by_run(gemmini_host_component_totals);
    for (const auto & run_kv : runs) {
        const aggregate_entry gemmini = backend_totals[{run_kv.first, "Gemmini"}];
        const aggregate_entry accounted = gemmini_breakdown_by_run[run_kv.first];
        const int64_t other_us = gemmini.wall_us > accounted.wall_us ? gemmini.wall_us - accounted.wall_us : 0;
        const uint64_t other_cycles = gemmini.wall_cycles > accounted.wall_cycles ? gemmini.wall_cycles - accounted.wall_cycles : 0;
        add_aggregate(gemmini_host_component_totals, run_kv.first, "other_host", other_us, other_cycles);
    }

    const std::map<std::string, aggregate_entry> backend_profiled_by_run = aggregate_by_run(backend_totals);
    const std::map<std::string, aggregate_entry> phase_profiled_by_run = aggregate_by_run(phase_totals);
    const std::map<std::string, aggregate_entry> repack_by_run = aggregate_by_run(repack_stage_totals);
    const std::map<std::string, aggregate_entry> gemmini_host_by_run = aggregate_by_run(gemmini_host_component_totals);

    {
        std::ofstream out(dir + "/stage_summary.csv");
        if (!out) {
            return -1;
        }

        out << "run_label,phase,stage,total_us,total_cycles,pct_of_phase,gemmini_us,gemmini_cycles,cpu_us,cpu_cycles,pct_gemmini_of_stage,pct_cpu_of_stage";
        for (const std::string & name : backend_gemmini_breakdown_order()) {
            out << ",pct_" << name << "_of_stage_gemmini";
        }
        out << "\n";
        for (const auto & run_kv : stage_summary_totals) {
            for (const std::string & phase : sorted_stage_summary_phases(run_kv.second)) {
                const auto phase_it = run_kv.second.find(phase);
                const aggregate_entry denom = aggregate_stage_entries(phase_it->second);
                for (const auto & stage_kv : sorted_stage_entries(phase_it->second)) {
                    out << csv_escape(run_kv.first) << ","
                        << csv_escape(phase_it->first) << ","
                        << csv_escape(stage_kv.first) << ","
                        << stage_kv.second.total.wall_us << ","
                        << stage_kv.second.total.wall_cycles << ","
                        << pct_i64(stage_kv.second.total.wall_us, denom.wall_us) << ","
                        << stage_kv.second.gemmini.wall_us << ","
                        << stage_kv.second.gemmini.wall_cycles << ","
                        << stage_kv.second.cpu.wall_us << ","
                        << stage_kv.second.cpu.wall_cycles << ","
                        << pct_i64(stage_kv.second.gemmini.wall_us, stage_kv.second.total.wall_us) << ","
                        << pct_i64(stage_kv.second.cpu.wall_us, stage_kv.second.total.wall_us);
                    for (const std::string & name : backend_gemmini_breakdown_order()) {
                        const aggregate_entry component = stage_gemmini_component(stage_kv.second, name);
                        out << "," << pct_i64(component.wall_us, stage_kv.second.gemmini.wall_us);
                    }
                    out << "\n";
                }
            }
        }
    }

    {
        std::ofstream out(dir + "/matmul_summary.csv");
        if (!out) {
            return -1;
        }

        out << "run_label,phase,shape,dim_i,dim_j,dim_k,tile_i,tile_j,tile_k,page_packed_a,page_packed_b,page_packed_c,page_packed_d,count,total_us,total_cycles,avg_us,avg_cycles";
        for (const std::string & name : backend_gemmini_breakdown_order()) {
            out << ",pct_" << name << "_of_matmul_wall";
        }
        out << "\n";

        for (const auto & kv : gemmini_matmul_shape_totals) {
            const matmul_shape_key & key = kv.first;
            const matmul_shape_entry & entry = kv.second;
            out << csv_escape(key.run_label) << ","
                << csv_escape(key.phase) << ","
                << csv_escape(matmul_shape_string(key)) << ","
                << key.dim_i << ","
                << key.dim_j << ","
                << key.dim_k << ","
                << key.tile_i << ","
                << key.tile_j << ","
                << key.tile_k << ","
                << ((key.page_packed_mask & 0x1) != 0) << ","
                << ((key.page_packed_mask & 0x2) != 0) << ","
                << ((key.page_packed_mask & 0x4) != 0) << ","
                << ((key.page_packed_mask & 0x8) != 0) << ","
                << entry.count << ","
                << entry.total.wall_us << ","
                << entry.total.wall_cycles << ","
                << metric_double_fixed_or_na(entry.count > 0, static_cast<double>(entry.total.wall_us) / static_cast<double>(entry.count)) << ","
                << metric_double_fixed_or_na(entry.count > 0, static_cast<double>(entry.total.wall_cycles) / static_cast<double>(entry.count));
            for (const std::string & name : backend_gemmini_breakdown_order()) {
                const auto component_it = entry.components.find(name);
                const int64_t component_us = component_it == entry.components.end() ? 0 : component_it->second.wall_us;
                out << "," << metric_double_fixed_or_na(true, pct_i64(component_us, entry.total.wall_us));
            }
            out << "\n";
        }
    }

    {
        std::ofstream out(dir + "/backend_summary.csv");
        if (!out) {
            return -1;
        }

        out << "run_label,category,name,total_us,total_cycles,pct_of_runtime_breakdown,pct_of_parent\n";

        for (const auto & kv : backend_totals) {
            const aggregate_entry denom = runtime_denominator(kv.first.first, runs, backend_profiled_by_run);
            out << csv_escape(kv.first.first) << ",backend,"
                << csv_escape(kv.first.second) << ","
                << kv.second.wall_us << ","
                << kv.second.wall_cycles << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << "\n";
        }

        for (const auto & kv : phase_totals) {
            const aggregate_entry denom = runtime_denominator(kv.first.first, runs, phase_profiled_by_run);
            out << csv_escape(kv.first.first) << ",phase,"
                << csv_escape(kv.first.second) << ","
                << kv.second.wall_us << ","
                << kv.second.wall_cycles << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << "\n";
        }

        for (const auto & run_kv : runs) {
            const aggregate_entry overhead = runtime_overhead(run_kv.first, runs, phase_profiled_by_run);
            if (!has_aggregate_value(overhead)) {
                continue;
            }
            const aggregate_entry denom = runtime_denominator(run_kv.first, runs, phase_profiled_by_run);
            out << csv_escape(run_kv.first) << ",phase,runtime_overhead,"
                << overhead.wall_us << ","
                << overhead.wall_cycles << ","
                << pct_i64(overhead.wall_us, denom.wall_us) << ","
                << pct_i64(overhead.wall_us, denom.wall_us) << "\n";
        }

        for (const auto & kv : repack_stage_totals) {
            const auto run_it = runs.find(kv.first.first);
            const int64_t run_total_us = run_it == runs.end() ? 0 : run_it->second.wall_us;
            const aggregate_entry repack_total = repack_by_run.count(kv.first.first) ? repack_by_run.at(kv.first.first) : aggregate_entry{};
            out << csv_escape(kv.first.first) << ",repack_stage,"
                << csv_escape(kv.first.second) << ","
                << kv.second.wall_us << ","
                << kv.second.wall_cycles << ","
                << pct_i64(kv.second.wall_us, run_total_us) << ","
                << pct_i64(kv.second.wall_us, repack_total.wall_us) << "\n";
        }

        for (const auto & run_kv : runs) {
            const int64_t run_total_us = run_kv.second.wall_us;
            const aggregate_entry host_total = gemmini_host_by_run.count(run_kv.first) ? gemmini_host_by_run.at(run_kv.first) : aggregate_entry{};
            for (const std::string & name : backend_gemmini_breakdown_order()) {
                const auto component_it = gemmini_host_component_totals.find({run_kv.first, name});
                if (component_it == gemmini_host_component_totals.end()) {
                    continue;
                }
                const aggregate_entry & component = component_it->second;
                out << csv_escape(run_kv.first) << ",backend_gemmini_breakdown,"
                    << csv_escape(name) << ","
                    << component.wall_us << ","
                    << component.wall_cycles << ","
                    << pct_i64(component.wall_us, run_total_us) << ","
                    << pct_i64(component.wall_us, host_total.wall_us) << "\n";
            }
        }

        for (const auto & kv : gemmini_component_totals) {
            const uint64_t parent = gemmini_total_cycles_by_run[kv.first.first];
            out << csv_escape(kv.first.first) << ",gemmini_subphase,"
                << csv_escape(kv.first.second) << ","
                << 0 << ","
                << kv.second << ","
                << pct_u64(kv.second, runs[kv.first.first].wall_cycles) << ","
                << pct_u64(kv.second, parent) << "\n";
        }
    }

    {
        std::ofstream out(dir + "/summary.md");
        if (!out) {
            return -1;
        }

        out << "# llama-firesim summary\n\n";
        out << "| metric | value |\n";
        out << "| --- | --- |\n";
        out << "| model | " << markdown_table_escape(summary_model_name) << " |\n";
        out << "| input prompts | " << markdown_table_escape(summary_input_prompt) << " |\n";
        out << "| final outputs | " << markdown_table_escape(summary_final_output) << " |\n\n";
        out << "Model load: " << model_load_us << " us / " << model_load_cycles << " cycles\n\n";

        for (const auto & run_kv : runs) {
            const std::string & run_label = run_kv.first;
            const run_totals & totals = run_kv.second;

            out << "## " << run_label << "\n\n";
            out << "| metric | value |\n";
            out << "| --- | --- |\n";
            out << "| input prompt | " << markdown_table_escape(totals.input_prompt) << " |\n";
            out << "| final output | " << markdown_table_escape(totals.final_output) << " |\n";
            out << "| gemmini mask | " << format_hex_mask(totals.gemmini_mask) << " |\n";
            out << "| gemmini count | " << totals.gemmini_count << " |\n";
            out << "| prompt tokens | " << totals.prompt_tokens << " |\n";
            out << "| generated tokens | " << totals.generated_tokens << " |\n";
            out << "| total time (us) | " << totals.wall_us << " |\n";
            out << "| total cycles | " << totals.wall_cycles << " |\n";
            out << "| TTFT (us) | " << metric_i64_or_na(totals.generated_tokens > 0, totals.ttft_us) << " |\n";
            out << "| TTFT (cycles) | " << metric_u64_or_na(totals.generated_tokens > 0, totals.ttft_cycles) << " |\n";
            out << "| TPOT (us/token) | " << metric_double_or_na(totals.generated_tokens > 1, totals.tpot_us) << " |\n";
            out << "| TPOT (cycles/token) | " << metric_double_or_na(totals.generated_tokens > 1, totals.tpot_cycles) << " |\n\n";

            out << "### Phase Split\n\n";
            out << "| phase | total us | total cycles | % of runtime breakdown |\n";
            out << "| --- | --- | --- | --- |\n";
            const aggregate_entry phase_denom = runtime_denominator(run_label, runs, phase_profiled_by_run);
            for (const auto & kv : phase_totals) {
                if (kv.first.first != run_label) {
                    continue;
                }
                out << "| " << kv.first.second << " | "
                    << kv.second.wall_us << " | "
                    << kv.second.wall_cycles << " | "
                    << pct_i64(kv.second.wall_us, phase_denom.wall_us) << " |\n";
            }
            const aggregate_entry phase_overhead = runtime_overhead(run_label, runs, phase_profiled_by_run);
            if (has_aggregate_value(phase_overhead)) {
                out << "| runtime_overhead | "
                    << phase_overhead.wall_us << " | "
                    << phase_overhead.wall_cycles << " | "
                    << pct_i64(phase_overhead.wall_us, phase_denom.wall_us) << " |\n";
            }
            out << "\n";

            const auto run_stage_it = phase_stage_totals.find(run_label);
            if (run_stage_it != phase_stage_totals.end()) {
                const std::vector<std::pair<std::string, std::string>> stage_sections = {
                    {"prefill", "Prefill Stage Split"},
                    {"decode", "Decode Stage Split"},
                };
                for (const auto & phase_title : stage_sections) {
                    const auto phase_stage_it = run_stage_it->second.find(phase_title.first);
                    if (phase_stage_it == run_stage_it->second.end()) {
                        continue;
                    }

                    out << "### " << phase_title.second << "\n\n";
                    out << "| stage | total us | total cycles | % of " << phase_title.first << " stage time | % gemmini | % cpu";
                    for (const std::string & name : backend_gemmini_breakdown_order()) {
                        out << " | % " << name << " of gemmini";
                    }
                    out << " |\n";
                    out << "| --- | --- | --- | --- | --- | ---";
                    for (size_t i = 0; i < backend_gemmini_breakdown_order().size(); ++i) {
                        out << " | ---";
                    }
                    out << " |\n";
                    const aggregate_entry stage_denom = aggregate_stage_entries(phase_stage_it->second);
                    for (const auto & kv : sorted_stage_entries(phase_stage_it->second)) {
                        out << "| " << kv.first << " | "
                            << kv.second.total.wall_us << " | "
                            << kv.second.total.wall_cycles << " | "
                            << pct_i64(kv.second.total.wall_us, stage_denom.wall_us) << " | "
                            << pct_i64(kv.second.gemmini.wall_us, kv.second.total.wall_us) << " | "
                            << pct_i64(kv.second.cpu.wall_us, kv.second.total.wall_us);
                        for (const std::string & name : backend_gemmini_breakdown_order()) {
                            const aggregate_entry component = stage_gemmini_component(kv.second, name);
                            out << " | " << pct_i64(component.wall_us, kv.second.gemmini.wall_us);
                        }
                        out << " |\n";
                    }
                    out << "\n";
                }
            }

            bool wrote_matmul_header = false;
            const std::vector<std::pair<std::string, std::string>> matmul_sections = {
                {"prefill", "Prefill Gemmini Matmuls"},
                {"decode", "Decode Gemmini Matmuls"},
            };
            for (const auto & phase_title : matmul_sections) {
                bool has_rows = false;
                for (const auto & kv : gemmini_matmul_shape_totals) {
                    if (kv.first.run_label == run_label && kv.first.phase == phase_title.first) {
                        has_rows = true;
                        break;
                    }
                }
                if (!has_rows) {
                    continue;
                }

                if (!wrote_matmul_header) {
                    out << "### Gemmini Matmul Shape Analysis\n\n";
                    out << "`tile IxJxK` is the `job.tile_I/J/K` selected by the Gemmini tiling path. Subphase percentages are based on accumulated wall time for each shape+tiling group.\n\n";
                    wrote_matmul_header = true;
                }

                out << "#### " << phase_title.second << "\n\n";
                out << "| shape MxNxK | tile IxJxK | count | avg wall us | avg wall cycles";
                for (const std::string & name : backend_gemmini_breakdown_order()) {
                    out << " | % " << name;
                }
                out << " |\n";
                out << "| --- | --- | --- | --- | ---";
                for (size_t i = 0; i < backend_gemmini_breakdown_order().size(); ++i) {
                    out << " | ---";
                }
                out << " |\n";

                for (const auto & kv : gemmini_matmul_shape_totals) {
                    const matmul_shape_key & key = kv.first;
                    if (key.run_label != run_label || key.phase != phase_title.first) {
                        continue;
                    }

                    const matmul_shape_entry & entry = kv.second;
                    out << "| " << matmul_shape_string(key) << " | "
                        << matmul_tiling_string(key) << " | "
                        << entry.count << " | "
                        << metric_double_fixed_or_na(entry.count > 0, static_cast<double>(entry.total.wall_us) / static_cast<double>(entry.count)) << " | "
                        << metric_double_fixed_or_na(entry.count > 0, static_cast<double>(entry.total.wall_cycles) / static_cast<double>(entry.count));
                    for (const std::string & name : backend_gemmini_breakdown_order()) {
                        const auto component_it = entry.components.find(name);
                        const int64_t component_us = component_it == entry.components.end() ? 0 : component_it->second.wall_us;
                        out << " | " << metric_double_fixed_or_na(true, pct_i64(component_us, entry.total.wall_us));
                    }
                    out << " |\n";
                }
                out << "\n";
            }

            out << "### Backend Split\n\n";
            out << "| backend | total us | total cycles | % of runtime breakdown |\n";
            out << "| --- | --- | --- | --- |\n";
            const aggregate_entry backend_denom = runtime_denominator(run_label, runs, backend_profiled_by_run);
            for (const auto & kv : backend_totals) {
                if (kv.first.first != run_label) {
                    continue;
                }
                out << "| " << kv.first.second << " | "
                    << kv.second.wall_us << " | "
                    << kv.second.wall_cycles << " | "
                    << pct_i64(kv.second.wall_us, backend_denom.wall_us) << " |\n";
            }
            out << "\n";

            const aggregate_entry repack_total = repack_by_run.count(run_label) ? repack_by_run.at(run_label) : aggregate_entry{};
            if (has_aggregate_value(repack_total)) {
                out << "### Repack Overhead\n\n";
                out << "These events overlap the runtime phase/backend/stage breakdown above and are not included in those percentages.\n\n";
                out << "| stage | total us | total cycles | % of run | % of repack |\n";
                out << "| --- | --- | --- | --- | --- |\n";
                for (const auto & kv : repack_stage_totals) {
                    if (kv.first.first != run_label) {
                        continue;
                    }
                    out << "| " << kv.first.second << " | "
                        << kv.second.wall_us << " | "
                        << kv.second.wall_cycles << " | "
                        << pct_i64(kv.second.wall_us, totals.wall_us) << " | "
                        << pct_i64(kv.second.wall_us, repack_total.wall_us) << " |\n";
                }
                out << "| total | "
                    << repack_total.wall_us << " | "
                    << repack_total.wall_cycles << " | "
                    << pct_i64(repack_total.wall_us, totals.wall_us) << " | 100 |\n";
                out << "\n";
            }

            const aggregate_entry backend_gemmini_total = gemmini_host_by_run.count(run_label) ? gemmini_host_by_run.at(run_label) : aggregate_entry{};
            if (has_aggregate_value(backend_gemmini_total)) {
                out << "### Backend Gemmini Breakdown\n\n";
                out << "These subphases split the full wall time of operations with `backend=gemmini`. `other_host` closes the gap so the total matches Backend Split's Gemmini row.\n\n";
                out << "| subphase | total us | total cycles | % of run | % of backend Gemmini |\n";
                out << "| --- | --- | --- | --- | --- |\n";
                for (const std::string & name : backend_gemmini_breakdown_order()) {
                    const auto component_it = gemmini_host_component_totals.find({run_label, name});
                    if (component_it == gemmini_host_component_totals.end()) {
                        continue;
                    }
                    const aggregate_entry & component = component_it->second;
                    out << "| " << name << " | "
                        << component.wall_us << " | "
                        << component.wall_cycles << " | "
                        << pct_i64(component.wall_us, totals.wall_us) << " | "
                        << pct_i64(component.wall_us, backend_gemmini_total.wall_us) << " |\n";
                }
                out << "| total | "
                    << backend_gemmini_total.wall_us << " | "
                    << backend_gemmini_total.wall_cycles << " | "
                    << pct_i64(backend_gemmini_total.wall_us, totals.wall_us) << " | 100 |\n";
                out << "\n";
            }

            out << "### Gemmini Breakdown\n\n";
            out << "| subphase | cycles | % of gemmini cycles |\n";
            out << "| --- | --- | --- |\n";
            for (const auto & kv : gemmini_component_totals) {
                if (kv.first.first != run_label) {
                    continue;
                }
                out << "| " << kv.first.second << " | "
                    << kv.second << " | "
                    << pct_u64(kv.second, gemmini_total_cycles_by_run[run_label]) << " |\n";
            }
            out << "\n";
        }
    }

    return 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
