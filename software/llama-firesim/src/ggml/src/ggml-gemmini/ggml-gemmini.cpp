#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-gemmini-bf16.h"
#include "ggml-gemmini.h"
#include "ggml-gemmini-pack.h"
#include "ggml-gemmini-pack-reader.h"
#include "ggml-gemmini-page-packing.h"
#include "ggml-gemmini-profile.h"
#include "ggml-gemmini-vpu.h"
#include "ggml-gemmini-weight-select.h"
#include "ggml-gemmini-runtime-opt.h"

#include <algorithm>
#include <array>
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
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

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

static bool flash_env_flag(const char * requested_name, const char * uppercase_alias, bool fallback) {
    if (std::getenv(requested_name) != nullptr) {
        return env_flag(requested_name, fallback);
    }
    return env_flag(uppercase_alias, fallback);
}

struct activation_stage_key {
    const ggml_tensor * tensor = nullptr;
    const uint8_t * slice = nullptr;
    size_t row_stride_bytes = 0;
    size_t rows = 0;
    size_t cols = 0;
    enum ggml_type type = GGML_TYPE_COUNT;
    bool page_packed_a = false;
};

static bool operator==(const activation_stage_key & lhs, const activation_stage_key & rhs) {
    return lhs.tensor == rhs.tensor &&
        lhs.slice == rhs.slice &&
        lhs.row_stride_bytes == rhs.row_stride_bytes &&
        lhs.rows == rhs.rows &&
        lhs.cols == rhs.cols &&
        lhs.type == rhs.type &&
        lhs.page_packed_a == rhs.page_packed_a;
}

struct gemmini_thread_workspace {
    std::vector<float> conversion_row;
    std::vector<float> output_row;
    gemmini_aligned_vector<elem_t> encoded_a;
    gemmini_aligned_vector<acc_t> accum;
    activation_stage_key activation_key;
    ggml_gemmini_activation_cache_state activation_cache;

    void begin_activation_epoch() {
        ggml_gemmini_activation_cache_begin_epoch(activation_cache);
    }

    void end_activation_epoch() {
        ggml_gemmini_activation_cache_end_epoch(activation_cache);
    }
};

static gemmini_thread_workspace & gemmini_workspace() {
    static thread_local gemmini_thread_workspace workspace;
    return workspace;
}

template <typename T, typename Allocator>
static void ensure_workspace_size(std::vector<T, Allocator> & buffer, size_t elements) {
    // Never shrink a retained workspace. resize() value-initializes only when
    // the high-water mark grows; stable-shape calls therefore avoid the old
    // per-operation zero-fill and allocation/free cycle.
    if (buffer.size() < elements) {
        buffer.resize(elements);
    }
}

static int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    return std::strtol(value, nullptr, 0);
}

using gemmini_page_packing_config = ggml_gemmini_page_packing_config;
using flash_page_packing_config = ggml_gemmini_flash_page_packing_config;

static const gemmini_page_packing_config & gemmini_page_packing() {
    static const gemmini_page_packing_config config{
        env_flag("GGML_GEMMINI_PAGE_PACKED_A", false),
        env_flag("GGML_GEMMINI_PAGE_PACKED_B", false),
        env_flag("GGML_GEMMINI_PAGE_PACKED_C", false),
        env_flag("GGML_GEMMINI_PAGE_PACKED_D", false),
    };
    return config;
}

static const flash_page_packing_config & flash_page_packing() {
    // These exact, case-sensitive names intentionally do not share state
    // with GGML_GEMMINI_PAGE_PACKED_{A,B,C,D}. Defaults preserve the former
    // A=0/B=1 FlashAttention behavior while allowing K and V to diverge.
    static const flash_page_packing_config config{
        flash_env_flag("Flash_Q_PAGE_PACKED", "FLASH_Q_PAGE_PACKED", false),
        flash_env_flag("Flash_K_PAGE_PACKED", "FLASH_K_PAGE_PACKED", true),
        flash_env_flag("Flash_V_PAGE_PACKED", "FLASH_V_PAGE_PACKED", true),
    };
    return config;
}

static gemmini_page_packing_config gemmini_page_packing_for_shape(
        int64_t rows,
        int64_t cols_in,
        bool has_d) {
    GGML_ASSERT(rows > 0);
    GGML_ASSERT(cols_in > 0);
    return ggml_gemmini_effective_page_packing(
        gemmini_page_packing(),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(cols_in),
        has_d);
}

static bool gemmini_page_packed_b_for_k(int64_t cols_in) {
    GGML_ASSERT(cols_in > 0);
    return ggml_gemmini_page_pack_b_operand(
        gemmini_page_packing().b,
        static_cast<uint64_t>(cols_in));
}

static size_t checked_mul_size(size_t lhs, size_t rhs, const char * what) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(std::string("Gemmini ") + what + " size overflow");
    }
    return lhs * rhs;
}

static size_t checked_add_size(size_t lhs, size_t rhs, const char * what) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        throw std::overflow_error(std::string("Gemmini ") + what + " size overflow");
    }
    return lhs + rhs;
}

static size_t checked_ceil_div_size(size_t numerator, size_t denominator, const char * what) {
    if (denominator == 0) {
        throw std::overflow_error(std::string("Gemmini ") + what + " has zero divisor");
    }
    return checked_add_size(numerator, denominator - 1, what) / denominator;
}

static size_t checked_page_storage_bytes(size_t pages, const char * what) {
    return checked_mul_size(pages, GEMMINI_PAGE_PACKED_PAGE_BYTES, what);
}

static size_t page_packed_a_storage_bytes(size_t rows, size_t cols) {
    return checked_page_storage_bytes(
        gemmini_page_packed_a_page_count(rows, cols), "packed A");
}

static size_t page_packed_b_storage_bytes(size_t rows, size_t cols) {
    const size_t blocks_per_page = gemmini_page_packed_b_j_blocks_per_page();
    if (blocks_per_page == 0) {
        throw std::overflow_error("Gemmini packed B geometry has zero blocks per page");
    }
    const size_t k_blocks = checked_ceil_div_size(rows, DIM, "packed B K blocks");
    const size_t j_blocks = checked_ceil_div_size(cols, DIM, "packed B J blocks");
    const size_t j_pages = checked_ceil_div_size(
        j_blocks, blocks_per_page, "packed B J pages");
    return checked_page_storage_bytes(
        checked_mul_size(k_blocks, j_pages, "packed B pages"), "packed B");
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
    bool vpu_disabled = false;
    std::string vpu_failure_reason;
};

static request_runtime_config & request_runtime() {
    static thread_local request_runtime_config config;
    return config;
}

static int gemmini_active_mask() {
    if (request_runtime().active_mask >= 0) {
        return request_runtime().active_mask & GGML_GEMMINI_PROFILE_LOGICAL_MASK;
    }
    return env_int("GGML_GEMMINI_ACTIVE_MASK", GGML_GEMMINI_PROFILE_LOGICAL_MASK) &
        GGML_GEMMINI_PROFILE_LOGICAL_MASK;
}

static int gemmini_physical_mask(int logical_mask) {
    return static_cast<int>(ggml_gemmini_profile_physical_mask(
        static_cast<unsigned>(logical_mask)));
}

static std::array<std::mutex, 4> & gemmini_resource_mutexes() {
    static std::array<std::mutex, 4> mutexes;
    return mutexes;
}

static std::vector<std::unique_lock<std::mutex>> lock_gemmini_resources(int physical_mask) {
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(4);
    auto & mutexes = gemmini_resource_mutexes();
    for (int id = 0; id < 4; ++id) {
        if ((physical_mask & (1 << id)) != 0) {
            // Always lock in opcode order so overlapping multi-port requests
            // cannot deadlock. Disjoint split requests remain independent.
            locks.emplace_back(mutexes[id]);
        }
    }
    return locks;
}

struct weighted_rms_defer_registry {
    std::mutex mutex;
    std::unordered_map<const ggml_tensor *, std::unordered_set<ggml_backend_t>> owners;
};

static weighted_rms_defer_registry & deferred_weighted_rms_registry() {
    static weighted_rms_defer_registry registry;
    return registry;
}

static bool deferred_weighted_rms_contains(const ggml_tensor * tensor) {
    auto & registry = deferred_weighted_rms_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(tensor);
    return found != registry.owners.end() && !found->second.empty();
}

static bool deferred_weighted_rms_is_marked(
        ggml_backend_t      backend,
        const ggml_tensor * tensor) {
    auto & registry = deferred_weighted_rms_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(tensor);
    return found != registry.owners.end() && found->second.count(backend) != 0;
}

static void deferred_weighted_rms_mark(
        ggml_backend_t     backend,
        const ggml_tensor * tensor) {
    auto & registry = deferred_weighted_rms_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.owners[tensor].insert(backend);
}

static void deferred_weighted_rms_erase(
        ggml_backend_t     backend,
        const ggml_tensor * tensor) {
    auto & registry = deferred_weighted_rms_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(tensor);
    if (found == registry.owners.end()) {
        return;
    }
    found->second.erase(backend);
    if (found->second.empty()) {
        registry.owners.erase(found);
    }
}

static void deferred_weighted_rms_erase_backend(ggml_backend_t backend) {
    auto & registry = deferred_weighted_rms_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (auto it = registry.owners.begin(); it != registry.owners.end();) {
        it->second.erase(backend);
        if (it->second.empty()) {
            it = registry.owners.erase(it);
        } else {
            ++it;
        }
    }
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

static bool vpu_is_disabled_by_environment() {
    return env_flag("GGML_GEMMINI_DISABLE_VPU", false) ||
           env_flag("GGML_GEMMINI_VPU_DISABLE", false);
}

static bool flash_is_disabled_by_environment() {
    return vpu_is_disabled_by_environment() ||
           env_flag("GGML_GEMMINI_DISABLE_FLASH_ATTENTION", false) ||
           env_flag("GGML_GEMMINI_FLASH_ATTN_DISABLE", false);
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
    bool needs_quotes = value.find_first_of(",\"\r\n") != std::string::npos;
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
    out << "0x" << std::hex << (mask & GGML_GEMMINI_PROFILE_LOGICAL_MASK);
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
    ggml_backend_t cpu_fallback = nullptr;
};

struct weight_cache_entry {
    gemmini_aligned_vector<elem_t> data;
    int64_t rows = 0;
    int64_t cols = 0;
    bool page_packed_b = false;
    uint32_t role = 0;
};

struct prepacked_weight_entry {
    gemmini_aligned_vector<elem_t> data;
    int64_t rows = 0;
    int64_t cols = 0;
    bool page_packed_b = false;
    uint32_t role = 0;
};

struct weight_pack_state {
    bool loaded = false;
    bool hybrid_sparse_gguf = false;
    uint64_t model_fingerprint = 0;
    std::unordered_map<std::string, uint32_t> roles_by_name;
    std::unordered_set<const void *> packed_only_tensor_data;
    std::unordered_set<const void *> packed_only_slice_data;
};

struct backend_component_timing {
    int64_t wall_us = 0;
    uint64_t wall_cycles = 0;
};

struct gemmini_op_event {
    std::string run_label;
    std::string phase;
    std::string backend;
    std::string stage;
    std::string node_name;
    std::string src0_name;
    std::string op_name;
    std::string fallback_reason;
    int layer_index = -1;
    int token_index = -1;
    int64_t wall_us = 0;
    uint64_t wall_cycles = 0;
    int64_t accelerator_call_us = 0;
    uint64_t accelerator_call_cycles = 0;
    std::map<std::string, backend_component_timing> flash_attention_components;
    int64_t weight_cache_lookup_pack_us = 0;
    uint64_t weight_cache_lookup_pack_cycles = 0;
    int64_t activation_stage_pack_us = 0;
    uint64_t activation_stage_pack_cycles = 0;
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
    int64_t output_unpack_store_us = 0;
    uint64_t output_unpack_store_cycles = 0;
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
    uint8_t flash_page_packed_mask = 0;
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
    bool completed = false;
};

struct pending_eval_state {
    const ggml_tensor * tensor = nullptr;
    int64_t start_us = 0;
    uint64_t start_cycles = 0;
};

struct last_gemmini_state {
    const ggml_tensor * tensor = nullptr;
    std::string backend;
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
    uint8_t flash_page_packed_mask = 0;
    int64_t accelerator_call_us = 0;
    uint64_t accelerator_call_cycles = 0;
    std::map<std::string, backend_component_timing> flash_attention_components;
    int64_t repack_us = 0;
    uint64_t repack_cycles = 0;
    int64_t weight_cache_lookup_pack_us = 0;
    uint64_t weight_cache_lookup_pack_cycles = 0;
    int64_t activation_stage_pack_us = 0;
    uint64_t activation_stage_pack_cycles = 0;
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
    int64_t output_unpack_store_us = 0;
    uint64_t output_unpack_store_cycles = 0;
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
    std::unordered_map<const ggml_tensor *, std::string> fallback_reasons;
    int64_t graph_call_profiled_us = 0;
    uint64_t graph_call_profiled_cycles = 0;
};

struct profiler_shared_state {
    std::mutex mutex;
    uint64_t output_generation = 0;
    int64_t model_load_us = 0;
    uint64_t model_load_cycles = 0;
    std::string summary_model_name;
    std::string summary_input_prompt;
    std::string summary_final_output;
    std::unordered_set<std::string> started_run_labels;
    std::string duplicate_run_label;
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

struct profiler_output_state {
    uint64_t generation = std::numeric_limits<uint64_t>::max();
    size_t op_event_cursor = 0;
    size_t token_event_cursor = 0;
    std::unordered_set<std::string> exported_run_labels;
};

struct profiler_output_transaction {
    profiler_output_state & state;
    bool committed = false;

    ~profiler_output_transaction() {
        if (!committed) {
            // A retry must rebuild a coherent set of files instead of
            // appending after a partially failed multi-file export.
            state = profiler_output_state{};
        }
    }
};

static std::mutex & profiler_output_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<std::string, profiler_output_state> & profiler_outputs() {
    static std::unordered_map<std::string, profiler_output_state> outputs;
    return outputs;
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

static weight_pack_state & current_weight_pack() {
    static weight_pack_state state;
    return state;
}

static bool packed_only_weight_name(const std::string & name) {
    const auto & roles = current_weight_pack().roles_by_name;
    const auto it = roles.find(name);
    return current_weight_pack().hybrid_sparse_gguf && it != roles.end() &&
        it->second == GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY;
}

static bool packed_only_weight_tensor(const ggml_tensor * tensor) {
    // Views of a packed-only model weight must retain the same fail-closed
    // policy as the owning tensor.  Walk the bounded view chain instead of
    // relying only on the immediate view's name/data pointer.
    for (int depth = 0; tensor != nullptr && depth < 64;
            ++depth, tensor = tensor->view_src) {
        if (packed_only_weight_name(tensor_name(tensor)) ||
                current_weight_pack().packed_only_tensor_data.find(tensor->data) !=
                    current_weight_pack().packed_only_tensor_data.end()) {
            return true;
        }
    }
    return false;
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

static void note_cpu_fallback(const ggml_tensor * tensor, const std::string & reason) {
    if (tensor != nullptr && profiler_tls().enabled) {
        profiler_tls().fallback_reasons[tensor] = reason;
    }
    if (env_flag("GGML_GEMMINI_TRACE", false)) {
        std::fprintf(stderr,
            "GEMMINI-CPU-FALLBACK,profile=%s,node=%s,op=%s,reason=%s\n",
            GGML_GEMMINI_PROFILE_NAME,
            tensor == nullptr ? "" : effective_node_name(tensor).c_str(),
            tensor == nullptr ? "" : ggml_op_desc(tensor),
            reason.c_str());
        std::fflush(stderr);
    }
}

static void add_backend_component(
        std::map<std::string, backend_component_timing> & components,
        const char * name,
        int64_t wall_us,
        uint64_t wall_cycles) {
    if (wall_us == 0 && wall_cycles == 0) {
        return;
    }
    backend_component_timing & component = components[name];
    component.wall_us += wall_us;
    component.wall_cycles += wall_cycles;
}

static void mark_accelerator_result(
        ggml_tensor * tensor,
        const char * backend_name,
        int64_t start_us,
        uint64_t start_cycles,
        uint8_t page_packed_mask = 0,
        uint8_t flash_page_packed_mask = 0,
        const ggml_gemmini_vpu_result * accelerator_result = nullptr) {
    const int64_t elapsed_us = ggml_time_us() - start_us;
    const uint64_t elapsed_cycles = read_cycles_local() - start_cycles;
    last_gemmini_state state;
    state.tensor = tensor;
    state.backend = backend_name;
    state.node_name = effective_node_name(tensor);
    state.src0_name = source_name(tensor, 0);
    state.stage = classify_stage(state.node_name, state.src0_name, ggml_op_desc(tensor));
    state.layer_index = parse_layer_index(state.node_name, state.src0_name);
    state.token_index = profiler_tls().token_index;
    state.accelerator_call_us = elapsed_us;
    state.accelerator_call_cycles = elapsed_cycles;
    state.page_packed_mask = page_packed_mask;
    state.flash_page_packed_mask = flash_page_packed_mask;
    if (accelerator_result != nullptr && std::strcmp(backend_name, "flash_attention") == 0) {
        const ggml_gemmini_flash_breakdown & flash = accelerator_result->flash;
        add_backend_component(state.flash_attention_components, "input_validation",
            flash.input_validation_us, flash.input_validation_cycles);
        add_backend_component(state.flash_attention_components, "mask_prepare",
            flash.mask_prepare_us, flash.mask_prepare_cycles);
        add_backend_component(state.flash_attention_components, "workspace_prepare",
            flash.workspace_prepare_us, flash.workspace_prepare_cycles);
        add_backend_component(state.flash_attention_components, "plan_preflight",
            flash.plan_preflight_us, flash.plan_preflight_cycles);
        add_backend_component(state.flash_attention_components, "input_pack",
            flash.input_pack_us, flash.input_pack_cycles);
        add_backend_component(state.flash_attention_components, "fused_attention_run",
            flash.fused_attention_run_us, flash.fused_attention_run_cycles);
        add_backend_component(state.flash_attention_components, "output_unpack_store",
            flash.output_unpack_store_us, flash.output_unpack_store_cycles);
    }
    state.valid = true;
    profiler_tls().last_gemmini = std::move(state);
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

struct production_pack_load_context {
    std::unordered_map<std::string, prepacked_weight_entry> * loaded = nullptr;
    std::unordered_map<std::string, uint32_t> * roles_by_name = nullptr;
};

static uint16_t * allocate_production_pack_entry(
        void * context_ptr,
        const ggml_gemmini_weight_pack_v3_entry_descriptor & descriptor,
        std::string & reason) {
    auto * context = static_cast<production_pack_load_context *>(context_ptr);
    if (context == nullptr || context->loaded == nullptr) {
        reason = "production pack-load context is null";
        return nullptr;
    }

    prepacked_weight_entry entry;
    entry.rows = descriptor.cols_in;
    entry.cols = descriptor.cols_out;
    entry.page_packed_b = descriptor.page_packed_b;
    entry.role = descriptor.role;
    entry.data.resize(descriptor.data_count);

    const std::string key = prepacked_weight_key(
        descriptor.name,
        descriptor.slice_i2,
        descriptor.slice_i3,
        descriptor.cols_in,
        descriptor.cols_out);
    auto inserted = context->loaded->emplace(key, std::move(entry));
    if (!inserted.second) {
        reason = "duplicate v3 tensor slice";
        return nullptr;
    }
    if (context->roles_by_name != nullptr && descriptor.role != 0) {
        auto role_inserted = context->roles_by_name->emplace(descriptor.name, descriptor.role);
        if (!role_inserted.second && role_inserted.first->second != descriptor.role) {
            context->loaded->erase(inserted.first);
            reason = "inconsistent tensor roles across v4 slices";
            return nullptr;
        }
    }
    return reinterpret_cast<uint16_t *>(inserted.first->second.data.data());
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

    std::unordered_map<std::string, prepacked_weight_entry> loaded;
    std::unordered_map<std::string, uint32_t> roles_by_name;
    production_pack_load_context context{&loaded, &roles_by_name};
    const ggml_gemmini_weight_pack_expected_geometry expected{
        sizeof(elem_t),
        GGML_GEMMINI_WEIGHT_NUMERIC_FORMAT_BF16_RNE,
        GGML_GEMMINI_PROFILE_ID,
        DIM,
        GEMMINI_PAGE_PACKED_PAGE_BYTES,
        MAX_BYTES,
    };
    const ggml_gemmini_weight_pack_read_result result =
        ggml_gemmini_weight_pack_read_v3(
            in, expected, allocate_production_pack_entry, &context);
    if (result.status != GGML_GEMMINI_WEIGHT_PACK_READ_OK) {
        const char * reason = result.status == GGML_GEMMINI_WEIGHT_PACK_READ_LEGACY_VERSION
            ? "legacy_int8_pack"
            : ggml_gemmini_weight_pack_read_status_name(result.status);
        std::fprintf(stderr,
            "GEMMINI-WEIGHT-PACK-IGNORED,path=%s,version=%u,reason=%s,detail=%s\n",
            path,
            result.version,
            reason,
            result.reason.c_str());
        return -1;
    }

    const size_t loaded_count = loaded.size();
    {
        std::lock_guard<std::mutex> lock(weight_cache_mutex());
        prepacked_weight_store().swap(loaded);
        weight_pack_state state;
        state.loaded = true;
        state.hybrid_sparse_gguf = result.hybrid_sparse_gguf;
        state.model_fingerprint = result.model_fingerprint;
        state.roles_by_name = std::move(roles_by_name);
        current_weight_pack() = std::move(state);
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
        int64_t cols_out,
        uint32_t expected_role) {
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
    if (expected_role != 0 && it->second.role != expected_role) {
        return false;
    }

    weight_cache_entry entry;
    entry.rows = it->second.rows;
    entry.cols = it->second.cols;
    entry.page_packed_b = gemmini_page_packed_b_for_k(entry.rows);
    entry.role = it->second.role;
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

static elem_t encode_bf16(float value) {
    return static_cast<elem_t>(ggml_gemmini_float_to_bf16(value));
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

static void encode_and_store_b_output_row(
        elem_t * dst,
        const float * src,
        size_t out_col,
        size_t cols_in,
        size_t cols_out,
        bool page_packed_b) {
    if (!page_packed_b) {
        for (size_t k = 0; k < cols_in; ++k) {
            dst[k * cols_out + out_col] = encode_bf16(src[k]);
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
        const size_t valid_rows = std::min(static_cast<size_t>(DIM), cols_in - k_block * DIM);
        for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
            block[k_in_block * packed_row_stride + j_in_block] =
                encode_bf16(src[k_block * DIM + k_in_block]);
        }
    }
}

static void copy_and_store_b_output_row(
        elem_t * dst,
        const elem_t * src,
        size_t out_col,
        size_t cols_in,
        size_t cols_out,
        bool page_packed_b) {
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
        const size_t valid_rows = std::min(static_cast<size_t>(DIM), cols_in - k_block * DIM);
        for (size_t k_in_block = 0; k_in_block < valid_rows; ++k_in_block) {
            block[k_in_block * packed_row_stride + j_in_block] =
                src[k_block * DIM + k_in_block];
        }
    }
}

static void store_encoded_a_row(
        elem_t * dst,
        const elem_t * encoded_row,
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
            encoded_row + k_block * DIM,
            encoded_row + k_block * DIM + valid_cols,
            block + i_in_block * packed_row_stride);
    }
}

static void encode_f32_a_row(
        elem_t * dst,
        const float * values,
        size_t row,
        size_t cols,
        bool page_packed_a) {
    static_assert(sizeof(elem_t) == sizeof(uint16_t),
        "BF16 activation staging requires 16-bit Gemmini elements");

    if (!page_packed_a) {
        ggml_gemmini_encode_f32_row_to_bf16(
            values,
            reinterpret_cast<uint16_t *>(dst + row * cols),
            cols);
        return;
    }

    const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
    const size_t packed_row_stride = page_k_blocks * DIM;
    const size_t i_block = row / DIM;
    const size_t i_in_block = row % DIM;
    const size_t k_blocks = gemmini_ceil_div_size(cols, DIM);

    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        elem_t * block = gemmini_page_packed_a_block_addr_mut(
            dst, i_block, k_block, cols);
        const size_t col = k_block * DIM;
        const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - col);
        ggml_gemmini_encode_f32_row_to_bf16(
            values + col,
            reinterpret_cast<uint16_t *>(block + i_in_block * packed_row_stride),
            valid_cols);
    }
}

static void copy_bf16_a_row(
        elem_t * dst,
        const uint8_t * source,
        size_t row,
        size_t cols,
        bool page_packed_a) {
    static_assert(sizeof(elem_t) == sizeof(ggml_bf16_t),
        "BF16 activation staging requires matching GGML and Gemmini elements");

    if (!page_packed_a) {
        std::memcpy(dst + row * cols, source, cols * sizeof(elem_t));
        return;
    }

    const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
    const size_t packed_row_stride = page_k_blocks * DIM;
    const size_t i_block = row / DIM;
    const size_t i_in_block = row % DIM;
    const size_t k_blocks = gemmini_ceil_div_size(cols, DIM);

    for (size_t k_block = 0; k_block < k_blocks; ++k_block) {
        elem_t * block = gemmini_page_packed_a_block_addr_mut(
            dst, i_block, k_block, cols);
        const size_t col = k_block * DIM;
        const size_t valid_cols = std::min(static_cast<size_t>(DIM), cols - col);
        std::memcpy(
            block + i_in_block * packed_row_stride,
            source + col * sizeof(elem_t),
            valid_cols * sizeof(elem_t));
    }
}

static void encode_activation_matrix_bf16(
        const ggml_tensor * src,
        const uint8_t * slice,
        size_t row_stride_bytes,
        size_t rows,
        size_t cols,
        bool page_packed_a,
        gemmini_aligned_vector<elem_t> & encoded,
        std::vector<float> & tmp) {
    const size_t bytes = a_storage_bytes(rows, cols, page_packed_a);
    const size_t elements = storage_elements<elem_t>(bytes, "A");
    ensure_workspace_size(encoded, elements);

    if (src->type == GGML_TYPE_F32) {
        for (size_t row = 0; row < rows; ++row) {
            const float * values = reinterpret_cast<const float *>(
                slice + row * row_stride_bytes);
            encode_f32_a_row(
                encoded.data(), values, row, cols, page_packed_a);
        }
        return;
    }

    if (src->type == GGML_TYPE_BF16) {
        for (size_t row = 0; row < rows; ++row) {
            copy_bf16_a_row(
                encoded.data(),
                slice + row * row_stride_bytes,
                row,
                cols,
                page_packed_a);
        }
        return;
    }

    for (size_t row = 0; row < rows; ++row) {
        const uint8_t * row_ptr = slice + row * row_stride_bytes;
        tensor_row_to_float(src, row_ptr, static_cast<int64_t>(cols), tmp);
        encode_f32_a_row(
            encoded.data(), tmp.data(), row, cols, page_packed_a);
    }
}

static bool activation_cache_family(const std::string & stage, int & family, unsigned & member) {
    family = 0;
    member = 0;
    if (stage == "q_proj") {
        family = 1;
        member = 1u << 0;
        return true;
    }
    if (stage == "k_proj") {
        family = 1;
        member = 1u << 1;
        return true;
    }
    if (stage == "v_proj") {
        family = 1;
        member = 1u << 2;
        return true;
    }
    if (stage == "ffn_gate") {
        family = 2;
        member = 1u << 0;
        return true;
    }
    if (stage == "ffn_up") {
        family = 2;
        member = 1u << 1;
        return true;
    }
    return false;
}

static bool byte_ranges_overlap(
        const void * lhs_ptr,
        size_t lhs_bytes,
        const void * rhs_ptr,
        size_t rhs_bytes) {
    if (lhs_ptr == nullptr || rhs_ptr == nullptr || lhs_bytes == 0 || rhs_bytes == 0) {
        return false;
    }
    const uintptr_t lhs = reinterpret_cast<uintptr_t>(lhs_ptr);
    const uintptr_t rhs = reinterpret_cast<uintptr_t>(rhs_ptr);
    if (lhs > std::numeric_limits<uintptr_t>::max() - lhs_bytes ||
            rhs > std::numeric_limits<uintptr_t>::max() - rhs_bytes) {
        return true;
    }
    return lhs < rhs + rhs_bytes && rhs < lhs + lhs_bytes;
}

static bool dense_tensor_storage_bytes(const ggml_tensor * tensor, size_t & bytes) {
    if (tensor == nullptr || tensor->data == nullptr || !ggml_is_contiguous(tensor)) {
        return false;
    }
    bytes = ggml_nbytes(tensor);
    return true;
}

static bool can_write_gemmini_output_directly(
        const ggml_tensor * dst,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const uint8_t * dst_slice,
        size_t rows,
        size_t cols_out,
        size_t dst_row_stride_bytes,
        bool page_packed_c) {
    static_assert(std::is_same<acc_t, float>::value,
        "direct Gemmini output requires FP32 accumulators");

    const size_t dense_row_bytes = checked_mul_size(
        cols_out, sizeof(acc_t), "direct C row bytes");

    if (dst == nullptr || !ggml_gemmini_direct_c_layout_eligible(
            env_flag("GGML_GEMMINI_DIRECT_C", true),
            page_packed_c,
            dst->type == GGML_TYPE_F32,
            reinterpret_cast<uintptr_t>(dst_slice),
            alignof(acc_t),
            dst_row_stride_bytes,
            cols_out,
            sizeof(acc_t)) || dst_row_stride_bytes != dense_row_bytes) {
        return false;
    }

    const size_t output_bytes = checked_mul_size(
        checked_mul_size(rows, cols_out, "direct C elements"),
        sizeof(acc_t), "direct C bytes");
    size_t src0_bytes = 0;
    size_t src1_bytes = 0;
    if (!dense_tensor_storage_bytes(src0, src0_bytes) ||
            !dense_tensor_storage_bytes(src1, src1_bytes)) {
        return false;
    }
    return !byte_ranges_overlap(dst_slice, output_bytes, src0->data, src0_bytes) &&
        !byte_ranges_overlap(dst_slice, output_bytes, src1->data, src1_bytes);
}

static void store_accumulated_output(
        const ggml_tensor * dst,
        uint8_t * dst_slice,
        size_t dst_row_stride_bytes,
        size_t rows,
        size_t cols,
        const acc_t * accum,
        bool page_packed_c,
        std::vector<float> & output_row) {
    const size_t page_j_blocks =
        gemmini_page_packed_acc_j_blocks_per_page(sizeof(acc_t));
    const size_t packed_row_stride = page_j_blocks * DIM;
    const size_t j_blocks = gemmini_ceil_div_size(cols, DIM);

    for (size_t row = 0; row < rows; ++row) {
        float * direct_row = dst->type == GGML_TYPE_F32
            ? reinterpret_cast<float *>(dst_slice + row * dst_row_stride_bytes)
            : output_row.data();
        if (!page_packed_c) {
            const acc_t * acc_row = accum + row * cols;
            for (size_t col = 0; col < cols; ++col) {
                direct_row[col] = static_cast<float>(acc_row[col]);
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
                    direct_row[col] = static_cast<float>(acc_row[j_in_block]);
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
        store_encoded_a_row(dst.data(), src + row * cols, row, cols);
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
    entry.page_packed_b = gemmini_page_packed_b_for_k(cols_in);
    entry.data.assign(
        storage_elements<elem_t>(
            b_storage_bytes(
                static_cast<size_t>(cols_in),
                static_cast<size_t>(cols_out),
                entry.page_packed_b),
            "B"),
        0);
    std::vector<float> tmp;
    for (int64_t out_col = 0; out_col < cols_out; ++out_col) {
        const uint8_t * row_ptr = slice_ptr + out_col * row_stride_bytes;
        if (src0->type == GGML_TYPE_BF16) {
            copy_and_store_b_output_row(
                entry.data.data(),
                reinterpret_cast<const elem_t *>(row_ptr),
                static_cast<size_t>(out_col),
                static_cast<size_t>(cols_in),
                static_cast<size_t>(cols_out),
                entry.page_packed_b);
        } else {
            tensor_row_to_float(src0, row_ptr, cols_in, tmp);
            encode_and_store_b_output_row(
                entry.data.data(),
                tmp.data(),
                static_cast<size_t>(out_col),
                static_cast<size_t>(cols_in),
                static_cast<size_t>(cols_out),
                entry.page_packed_b);
        }
    }

    int64_t end_us = ggml_time_us();
    uint64_t end_cycles = read_cycles_local();

    if (profiler_tls().enabled) {
        gemmini_op_event repack_event;
        repack_event.run_label = current_run_label();
        repack_event.phase = "load/repack";
        repack_event.backend = "gemmini_bf16";
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
        repack_event.page_packed_mask = entry.page_packed_b ? 0x2u : 0u;
        profiler_add_event(repack_event);
    }
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
    const bool desired_page_packed_b = gemmini_page_packed_b_for_k(cols_in);
    if (it != cache.end()) {
        if (it->second.page_packed_b != desired_page_packed_b) {
            it->second.data = convert_b_layout(
                it->second.data.data(),
                static_cast<size_t>(it->second.rows),
                static_cast<size_t>(it->second.cols),
                it->second.page_packed_b,
                desired_page_packed_b);
            it->second.page_packed_b = desired_page_packed_b;
        }
        return it->second;
    }

    if (current_weight_pack().hybrid_sparse_gguf &&
            (packed_only_weight_name(src0_name) ||
             current_weight_pack().packed_only_slice_data.find(slice_key) !=
                current_weight_pack().packed_only_slice_data.end())) {
        throw std::runtime_error(
            "packed-only Gemmini weight is missing from the verified cache: " + src0_name);
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

static bool can_prepack_weight_tensor(
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

struct gemmini_matmul_metrics {
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
        const gemmini_page_packing_config & packing,
        size_t out_row_stride = 0) {
    gemmini_matmul_metrics metrics;
    const gemmini_page_packing_config checked_packing =
        ggml_gemmini_effective_page_packing(packing, rows, cols_in, d != nullptr);
    if (checked_packing.mask() != packing.mask()) {
        throw std::logic_error("Gemmini matmul received non-effective page-packing flags");
    }
    const bool split_mode = request_runtime().split_mode;
#if GGML_GEMMINI_PROFILE_ID == GGML_GEMMINI_PROFILE_SINGLE_ID
    constexpr bool single_mode = true;
#else
    constexpr bool single_mode = false;
#endif
    if (single_mode && split_mode) {
        throw std::runtime_error("Gemmini single profile does not support split mode");
    }
    const bool use_counters = !split_mode && gemmini_use_counters();
    const int gemmini_id = request_runtime().gemmini_id;
    const int requested_mask = gemmini_mask & GGML_GEMMINI_PROFILE_LOGICAL_MASK;
    const int run_gemmini_mask = gemmini_physical_mask(requested_mask);
    if (run_gemmini_mask == 0) {
        throw std::runtime_error("Gemmini logical active mask selects no compiled matrix port");
    }
    auto resource_locks = lock_gemmini_resources(run_gemmini_mask);
    const int physical_gemmini_id = single_mode ? XCUSTOM_ACC : gemmini_id;
    const char * run_mode = single_mode ? "single_1x16" : "shared_multi_4x8";
    const bool page_packed_d = packing.d;
    const uint8_t effective_packing_mask = packing.mask();
    const size_t stride_c = packing.c
        ? GEMMINI_PAGE_PACKED_STRIDE(cols_out)
        : (out_row_stride == 0 ? cols_out : out_row_stride);
    if (!packing.c && (stride_c < cols_out ||
            stride_c > GEMMINI_PAGE_PACKED_STRIDE_MASK ||
            stride_c > std::numeric_limits<uint32_t>::max() / sizeof(acc_t))) {
        throw std::logic_error("Gemmini matmul received invalid plain C stride");
    }

    if (env_flag("GGML_GEMMINI_TRACE", false)) {
        std::fprintf(stderr,
            "GEMMINI-RUN-BEGIN,mode=%s,mask=0x%x,requested_mask=0x%x,counters=%d,split=%d,gemmini_id=%d,rows=%zu,cols_out=%zu,cols_in=%zu,packing=0x%x,page_packed_a=%d,page_packed_b=%d,page_packed_c=%d,page_packed_d=%d\n",
            run_mode,
            run_gemmini_mask,
            requested_mask,
            use_counters ? 1 : 0,
            split_mode ? 1 : 0,
            physical_gemmini_id,
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

    if (env_flag("GGML_GEMMINI_TRACE", false)) {
        std::fprintf(stderr, "GEMMINI-RUN-FLUSH,mask=0x%x\n", run_gemmini_mask);
        std::fflush(stderr);
    }
    uint64_t component_cycles_start = read_cycles_local();
    int64_t component_us_start = ggml_time_us();
    gemmini_flush_once(run_gemmini_mask);
    metrics.flush_us += ggml_time_us() - component_us_start;
    metrics.flush_cycles += read_cycles_local() - component_cycles_start;

    const uint64_t start_cycles = read_cycles_local();
    component_cycles_start = start_cycles;
    component_us_start = ggml_time_us();
    if (single_mode) {
        const int custom_num = XCUSTOM_ACC;
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
        job.stride_C = stride_c;
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
                physical_gemmini_id,
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
                physical_gemmini_id,
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
        job.stride_C = stride_c;
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

static bool is_weighted_rms_norm_mul(
        const ggml_tensor * op,
        const ggml_tensor ** rms_norm,
        const ggml_tensor ** weight,
        char * reason,
        size_t reason_capacity) {
    if (op == nullptr || op->op != GGML_OP_MUL ||
            op->src[0] == nullptr || op->src[1] == nullptr) {
        if (reason != nullptr && reason_capacity != 0) {
            std::snprintf(reason, reason_capacity,
                "MUL is not an RMS_NORM + learned-weight pair");
        }
        return false;
    }

    const ggml_tensor * candidate_rms = nullptr;
    const ggml_tensor * candidate_weight = nullptr;
    if (op->src[0]->op == GGML_OP_RMS_NORM) {
        candidate_rms = op->src[0];
        candidate_weight = op->src[1];
    } else if (op->src[1]->op == GGML_OP_RMS_NORM) {
        candidate_rms = op->src[1];
        candidate_weight = op->src[0];
    } else {
        if (reason != nullptr && reason_capacity != 0) {
            std::snprintf(reason, reason_capacity,
                "MUL is not an RMS_NORM + learned-weight pair");
        }
        return false;
    }

    if (!ggml_gemmini_vpu_can_compute_weighted_rms_norm(
            candidate_rms, candidate_weight, op, reason, reason_capacity)) {
        return false;
    }
    if (rms_norm != nullptr) {
        *rms_norm = candidate_rms;
    }
    if (weight != nullptr) {
        *weight = candidate_weight;
    }
    return true;
}

static uint8_t vpu_effective_flash_page_packed_mask(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_FLASH_ATTN_EXT ||
        op->src[0] == nullptr || op->src[1] == nullptr) {
        return 0;
    }
    const auto packing = ggml_gemmini_effective_flash_page_packing(
        flash_page_packing(),
        static_cast<uint64_t>(op->src[0]->ne[1]),
        static_cast<uint64_t>(op->src[0]->ne[0]),
        static_cast<uint64_t>(op->src[1]->ne[1]));
    return packing.mask();
}

static bool vpu_can_offload(const ggml_tensor * op, char * reason, size_t reason_capacity) {
    if (op == nullptr) {
        std::snprintf(reason, reason_capacity, "operation is null");
        return false;
    }
    if (request_runtime().vpu_disabled) {
        std::snprintf(reason, reason_capacity, "VPU disabled for this request after: %s",
            request_runtime().vpu_failure_reason.c_str());
        return false;
    }
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        if (flash_is_disabled_by_environment()) {
            std::snprintf(reason, reason_capacity,
                "FlashAttention disabled by diagnostic environment variable");
            return false;
        }
    } else if (vpu_is_disabled_by_environment()) {
        std::snprintf(reason, reason_capacity,
            "VPU disabled by diagnostic environment variable");
        return false;
    }
    if (op->op == GGML_OP_MUL) {
        char weighted_reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
        if (is_weighted_rms_norm_mul(
                op, nullptr, nullptr, weighted_reason, sizeof(weighted_reason))) {
            return true;
        }
    }
    return ggml_gemmini_vpu_can_compute(
        op, static_cast<unsigned>(gemmini_active_mask()),
        op->op == GGML_OP_FLASH_ATTN_EXT ? flash_page_packing().mask() : 0,
        reason, reason_capacity);
}

static void ggml_backend_gemmini_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    static_assert(sizeof(elem_t) == sizeof(ggml_bf16_t),
        "direct BF16 activation input requires matching Gemmini elements");

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
    const gemmini_page_packing_config packing =
        gemmini_page_packing_for_shape(rows, cols_in, false);
    const bool direct_a_enabled = env_flag("GGML_GEMMINI_DIRECT_A", true);
    const size_t src1_storage_bytes = ggml_nbytes(src1);
    const size_t dst_storage_bytes = ggml_nbytes(dst);

    auto & workspace = gemmini_workspace();

    uint64_t total_cycles = 0;
    uint64_t load_cycles = 0;
    uint64_t preload_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t store_cycles = 0;
    uint64_t wait_cycles = 0;
    int64_t tile_i = 0;
    int64_t tile_j = 0;
    int64_t tile_k = 0;
    int64_t weight_cache_lookup_pack_us = 0;
    uint64_t weight_cache_lookup_pack_cycles = 0;
    int64_t activation_stage_pack_us = 0;
    uint64_t activation_stage_pack_cycles = 0;
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
    int64_t output_unpack_store_us = 0;
    uint64_t output_unpack_store_cycles = 0;

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const uint8_t * src0_slice = reinterpret_cast<const uint8_t *>(src0->data) + i02 * nb02 + i03 * nb03;
            const uint8_t * src1_slice = reinterpret_cast<const uint8_t *>(src1->data) + i12 * nb12 + i13 * nb13;
            uint8_t * dst_slice = reinterpret_cast<uint8_t *>(dst->data) + i12 * nb2 + i13 * nb3;

            weight_cache_entry transient_wcache;
            const bool cache_src0 = ggml_gemmini_is_model_weight_name(src0_name);
            weight_cache_entry * wcache = nullptr;
            const uint64_t weight_cache_lookup_pack_cycles_start = read_cycles_local();
            const int64_t weight_cache_lookup_pack_us_start = ggml_time_us();
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
            weight_cache_lookup_pack_us += ggml_time_us() - weight_cache_lookup_pack_us_start;
            weight_cache_lookup_pack_cycles += read_cycles_local() - weight_cache_lookup_pack_cycles_start;
            GGML_ASSERT(wcache->page_packed_b == packing.b);

            const uint64_t activation_stage_pack_cycles_start = read_cycles_local();
            const int64_t activation_stage_pack_us_start = ggml_time_us();
            const ggml_gemmini_direct_a_plan direct_a_plan =
                ggml_gemmini_make_direct_bf16_a_plan(
                    direct_a_enabled,
                    packing.a,
                    src1->type == GGML_TYPE_BF16,
                    ggml_is_contiguous(src1),
                    ggml_is_contiguous(dst),
                    reinterpret_cast<uintptr_t>(src1->data),
                    src1_storage_bytes,
                    reinterpret_cast<uintptr_t>(src1_slice),
                    nb11,
                    static_cast<size_t>(rows),
                    static_cast<size_t>(cols_in),
                    reinterpret_cast<uintptr_t>(dst->data),
                    dst_storage_bytes);
            const bool direct_a = direct_a_plan.direct;
            const elem_t * gemmini_a = nullptr;
            bool activation_cache_hit = false;
            if (direct_a) {
                gemmini_a = reinterpret_cast<const elem_t *>(direct_a_plan.address);
                ggml_gemmini_activation_cache_record_direct_input(
                    workspace.activation_cache);
                workspace.activation_key = {};
            } else {
                const activation_stage_key activation_key{
                    src1,
                    src1_slice,
                    nb11,
                    static_cast<size_t>(rows),
                    static_cast<size_t>(cols_in),
                    src1->type,
                    packing.a,
                };
                int activation_family = 0;
                unsigned activation_member = 0;
                const bool cacheable_activation = activation_cache_family(
                    stage, activation_family, activation_member);
                const bool activation_key_matches =
                    workspace.activation_key == activation_key;
                activation_cache_hit = cacheable_activation &&
                    ggml_gemmini_activation_cache_can_hit(
                        workspace.activation_cache,
                        activation_family,
                        activation_member,
                        activation_key_matches);
                if (activation_cache_hit) {
                    ggml_gemmini_activation_cache_record_hit(
                        workspace.activation_cache, activation_member);
                } else {
                    encode_activation_matrix_bf16(
                        src1,
                        src1_slice,
                        nb11,
                        static_cast<size_t>(rows),
                        static_cast<size_t>(cols_in),
                        packing.a,
                        workspace.encoded_a,
                        workspace.conversion_row);
                    workspace.activation_key = activation_key;
                    ggml_gemmini_activation_cache_record_miss(
                        workspace.activation_cache,
                        cacheable_activation ? activation_family : 0,
                        cacheable_activation ? activation_member : 0);
                }
                gemmini_a = workspace.encoded_a.data();
            }
            activation_stage_pack_us += ggml_time_us() - activation_stage_pack_us_start;
            activation_stage_pack_cycles += read_cycles_local() - activation_stage_pack_cycles_start;

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
            const bool direct_c = can_write_gemmini_output_directly(
                dst,
                src0,
                src1,
                dst_slice,
                static_cast<size_t>(rows),
                static_cast<size_t>(cols_out),
                nb1,
                packing.c);
            if (env_flag("GGML_GEMMINI_TRACE", false)) {
                std::fprintf(stderr,
                    "GEMMINI-MATMUL-OPT,stage=%s,activation_cache_hit=%d,direct_a=%d,direct_c=%d\n",
                    stage.c_str(),
                    activation_cache_hit ? 1 : 0,
                    direct_a ? 1 : 0,
                    direct_c ? 1 : 0);
                std::fflush(stderr);
            }
            acc_t * gemmini_output = nullptr;
            size_t output_stride = static_cast<size_t>(cols_out);
            if (direct_c) {
                gemmini_output = reinterpret_cast<acc_t *>(dst_slice);
                output_stride = nb1 / sizeof(acc_t);
            } else {
                const size_t accum_elements = storage_elements<acc_t>(
                    acc_storage_bytes(
                        static_cast<size_t>(rows),
                        static_cast<size_t>(cols_out),
                        packing.c,
                        "C"),
                    "C");
                ensure_workspace_size(workspace.accum, accum_elements);
                gemmini_output = workspace.accum.data();
            }
#if defined(__riscv)
            asm volatile ("" ::: "memory");
#endif
            const gemmini_matmul_metrics metrics = run_gemmini_matmul(
                active_mask,
                rows,
                cols_out,
                cols_in,
                gemmini_a,
                wcache->data.data(),
                nullptr,
                gemmini_output,
                packing,
                output_stride);
#if defined(__riscv)
            asm volatile ("" ::: "memory");
#endif
            gemmini_call_us += ggml_time_us() - gemmini_call_us_start;
            gemmini_call_cycles += read_cycles_local() - gemmini_call_cycles_start;

            total_cycles += metrics.total_cycles;
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

            if (!direct_c) {
                const uint64_t output_unpack_store_cycles_start = read_cycles_local();
                const int64_t output_unpack_store_us_start = ggml_time_us();
                if (dst->type != GGML_TYPE_F32) {
                    ensure_workspace_size(
                        workspace.output_row, static_cast<size_t>(cols_out));
                }
                store_accumulated_output(
                    dst,
                    dst_slice,
                    nb1,
                    static_cast<size_t>(rows),
                    static_cast<size_t>(cols_out),
                    workspace.accum.data(),
                    packing.c,
                    workspace.output_row);
                output_unpack_store_us += ggml_time_us() - output_unpack_store_us_start;
                output_unpack_store_cycles += read_cycles_local() - output_unpack_store_cycles_start;
            }
        }
    }

    profiler_tls().last_gemmini.tensor = dst;
    profiler_tls().last_gemmini.backend = "gemmini_bf16";
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
    profiler_tls().last_gemmini.flash_page_packed_mask = 0;
    profiler_tls().last_gemmini.flash_attention_components.clear();
    profiler_tls().last_gemmini.accelerator_call_us = gemmini_call_us;
    profiler_tls().last_gemmini.accelerator_call_cycles = gemmini_call_cycles;
    profiler_tls().last_gemmini.repack_us = 0;
    profiler_tls().last_gemmini.repack_cycles = 0;
    profiler_tls().last_gemmini.weight_cache_lookup_pack_us = weight_cache_lookup_pack_us;
    profiler_tls().last_gemmini.weight_cache_lookup_pack_cycles = weight_cache_lookup_pack_cycles;
    profiler_tls().last_gemmini.activation_stage_pack_us = activation_stage_pack_us;
    profiler_tls().last_gemmini.activation_stage_pack_cycles = activation_stage_pack_cycles;
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
    profiler_tls().last_gemmini.output_unpack_store_us = output_unpack_store_us;
    profiler_tls().last_gemmini.output_unpack_store_cycles = output_unpack_store_cycles;
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
    deferred_weighted_rms_erase_backend(backend);
    if (ctx->cpu_fallback != nullptr) {
        ggml_backend_free(ctx->cpu_fallback);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status compute_cpu_fallback(
        ggml_backend_gemmini_context * ctx,
        ggml_cgraph * cgraph,
        int first_node,
        int end_node) {
    if (ctx == nullptr || ctx->cpu_fallback == nullptr || first_node >= end_node) {
        return GGML_STATUS_FAILED;
    }
    ggml_cgraph view = ggml_graph_view(cgraph, first_node, end_node);
    return ggml_backend_graph_compute(ctx->cpu_fallback, &view);
}

static bool is_vpu_graph_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }
}

static void disable_vpu_after_hardware_error(const ggml_gemmini_vpu_result & result) {
    if (result.status != GGML_GEMMINI_VPU_STATUS_HARDWARE_ERROR) {
        return;
    }
    request_runtime().vpu_disabled = true;
    request_runtime().vpu_failure_reason = result.reason;
    std::fprintf(stderr,
        "GEMMINI-VPU-DISABLED,profile=%s,status=0x%llx,reason=%s\n",
        GGML_GEMMINI_PROFILE_NAME,
        static_cast<unsigned long long>(result.vpu_status),
        result.reason);
    std::fflush(stderr);
}

static void ggml_backend_gemmini_graph_optimize(
        ggml_backend_t backend,
        ggml_cgraph *  cgraph) {
    if (cgraph == nullptr) {
        return;
    }

    // Allocation can optimize the same tensor graph again after a reset. Drop
    // any previous hint for RMS nodes in this split before deriving the new
    // graph-proven set.
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node != nullptr && node->op == GGML_OP_RMS_NORM) {
            deferred_weighted_rms_erase(backend, node);
        }
    }

    if (vpu_is_disabled_by_environment() || request_runtime().vpu_disabled) {
        return;
    }

    for (int i = 0; i + 1 < cgraph->n_nodes; ++i) {
        ggml_tensor * rms = cgraph->nodes[i];
        if (rms == nullptr || rms->op != GGML_OP_RMS_NORM ||
                !ggml_can_fuse(cgraph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL})) {
            continue;
        }

        const ggml_tensor * admitted_rms = nullptr;
        const ggml_tensor * weight = nullptr;
        char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
        if (is_weighted_rms_norm_mul(
                cgraph->nodes[i + 1], &admitted_rms, &weight,
                reason, sizeof(reason)) && admitted_rms == rms) {
            GGML_UNUSED(weight);
            deferred_weighted_rms_mark(backend, rms);
        }
    }
}

static enum ggml_status ggml_backend_gemmini_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_gemmini_context *>(backend->context);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT) {
            if (packed_only_weight_tensor(node->src[0]) &&
                    (env_flag("GGML_GEMMINI_DISABLE", false) ||
                     gemmini_active_mask() == 0 ||
                     gemmini_max_offloads() >= 0 ||
                     !gemmini_can_offload(node))) {
                std::fprintf(stderr,
                    "GEMMINI-HYBRID-REQUIRED-OFFLOAD-REJECTED,node=%s,src0=%s\n",
                    effective_node_name(node).c_str(), source_name(node, 0).c_str());
                std::fflush(stderr);
                return GGML_STATUS_FAILED;
            }
            ggml_backend_gemmini_mul_mat(node);
            continue;
        }

        if (node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
                node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE ||
                node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        // The optimizer marked this RMS node while it still had the complete
        // backend split graph. The eval callback may since have reduced the
        // compute view, so never infer fusion legality from this local view
        // alone.
        const bool weighted_rms_marked =
            node->op == GGML_OP_RMS_NORM &&
            deferred_weighted_rms_is_marked(backend, node);
        if (weighted_rms_marked && i + 1 < cgraph->n_nodes &&
                ggml_can_fuse(cgraph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL})) {
            ggml_tensor * weighted = cgraph->nodes[i + 1];
            const ggml_tensor * admitted_rms = nullptr;
            const ggml_tensor * weight = nullptr;
            char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
            if (is_weighted_rms_norm_mul(
                    weighted, &admitted_rms, &weight, reason, sizeof(reason)) &&
                    admitted_rms == node &&
                    !vpu_is_disabled_by_environment() && !request_runtime().vpu_disabled) {
                const int64_t start_us = ggml_time_us();
                const uint64_t start_cycles = read_cycles_local();
                const ggml_gemmini_vpu_result result =
                    ggml_gemmini_vpu_compute_weighted_rms_norm(
                        node, weight, weighted);
                if (result.status == GGML_GEMMINI_VPU_STATUS_OK) {
                    mark_accelerator_result(weighted, "vpu", start_us, start_cycles);
                    ++i;
                    continue;
                }
                disable_vpu_after_hardware_error(result);
                if (!result.fallback_safe) {
                    return GGML_STATUS_FAILED;
                }
                std::snprintf(reason, sizeof(reason), "%s: %s",
                    ggml_gemmini_vpu_status_name(result.status), result.reason);
            } else if (reason[0] == '\0') {
                std::snprintf(reason, sizeof(reason), "%s",
                    request_runtime().vpu_disabled
                        ? request_runtime().vpu_failure_reason.c_str()
                        : "weighted RMSNorm VPU mapping disabled");
            }

            note_cpu_fallback(weighted, reason);
            const enum ggml_status status = compute_cpu_fallback(ctx, cgraph, i, i + 2);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            ++i;
            continue;
        }

        if (is_vpu_graph_op(node->op)) {
            char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
            if (vpu_can_offload(node, reason, sizeof(reason))) {
                std::vector<std::unique_lock<std::mutex>> flash_resource_locks;
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    flash_resource_locks = lock_gemmini_resources(
                        gemmini_physical_mask(gemmini_active_mask()));
                }
                const int64_t start_us = ggml_time_us();
                const uint64_t start_cycles = read_cycles_local();
                const ggml_gemmini_vpu_result result = ggml_gemmini_vpu_compute(
                    node, static_cast<unsigned>(gemmini_active_mask()),
                    node->op == GGML_OP_FLASH_ATTN_EXT ? flash_page_packing().mask() : 0);
                if (result.status == GGML_GEMMINI_VPU_STATUS_OK) {
                    mark_accelerator_result(
                        node,
                        node->op == GGML_OP_FLASH_ATTN_EXT ? "flash_attention" : "vpu",
                        start_us,
                        start_cycles,
                        0,
                        vpu_effective_flash_page_packed_mask(node),
                        &result);
                    continue;
                }
                disable_vpu_after_hardware_error(result);
                if (!result.fallback_safe) {
                    return GGML_STATUS_FAILED;
                }
                std::snprintf(reason, sizeof(reason), "%s: %s",
                    ggml_gemmini_vpu_status_name(result.status), result.reason);
            }

            note_cpu_fallback(node, reason[0] == '\0' ? "VPU admission rejected" : reason);
            const enum ggml_status status = compute_cpu_fallback(ctx, cgraph, i, i + 1);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            continue;
        }

        note_cpu_fallback(node, "operation has no Gemmini/VPU mapping");
        const enum ggml_status status = compute_cpu_fallback(ctx, cgraph, i, i + 1);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
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
    /* .graph_optimize      = */ ggml_backend_gemmini_graph_optimize,
};

static ggml_guid_t ggml_backend_gemmini_guid() {
    static ggml_guid guid = { 0x6a, 0x9a, 0x4f, 0x18, 0x1f, 0x4b, 0x47, 0xcd, 0x8e, 0x6e, 0xb4, 0x9b, 0x4a, 0xb0, 0x3c, 0x21 };
    return &guid;
}

} // namespace

void ggml_gemmini_weight_cache_clear(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    weight_cache().clear();
    current_weight_pack().packed_only_tensor_data.clear();
    current_weight_pack().packed_only_slice_data.clear();
}

void ggml_gemmini_weight_pack_clear(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    prepacked_weight_store().clear();
    current_weight_pack() = weight_pack_state{};
}

int ggml_gemmini_weight_pack_load(const char * path) {
    return load_prepacked_weight_file(path);
}

int ggml_gemmini_prepack_weight(const ggml_tensor * tensor, int32_t weight_role) {
    if (weight_role < GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY ||
            weight_role > GGML_GEMMINI_WEIGHT_ROLE_SHARED) {
        return -1;
    }
    const auto role = static_cast<enum ggml_gemmini_weight_role>(weight_role);
    if (!ggml_gemmini_weight_role_requires_pack(role)) {
        return 0;
    }
    if (!can_prepack_weight_tensor(tensor, role)) {
        return -1;
    }

    const std::string src0_name = tensor_name(tensor);
    uint32_t serialized_role = 0;
    {
        std::lock_guard<std::mutex> lock(weight_cache_mutex());
        const auto it = current_weight_pack().roles_by_name.find(src0_name);
        if (it != current_weight_pack().roles_by_name.end()) {
            serialized_role = it->second;
        }
    }
    const uint32_t expected_role = role == GGML_GEMMINI_WEIGHT_ROLE_SHARED
        ? GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_SHARED
        : GGML_GEMMINI_WEIGHT_PACK_ENTRY_ROLE_GEMMINI_ONLY;
    if (serialized_role != 0 && serialized_role != expected_role) {
        std::fprintf(stderr,
            "GEMMINI-WEIGHT-PACK-ROLE-MISMATCH,tensor=%s,expected=%u,actual=%u\n",
            src0_name.c_str(), expected_role, serialized_role);
        return -1;
    }
    if (current_weight_pack().hybrid_sparse_gguf && serialized_role == 0) {
        std::fprintf(stderr,
            "GEMMINI-WEIGHT-PACK-MISSING-TENSOR,tensor=%s\n", src0_name.c_str());
        return -1;
    }
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
                        tensor->ne[1],
                        serialized_role)) {
                if (current_weight_pack().hybrid_sparse_gguf) {
                    std::fprintf(stderr,
                        "GEMMINI-WEIGHT-PACK-MISSING-SLICE,tensor=%s,i2=%lld,i3=%lld\n",
                        src0_name.c_str(), static_cast<long long>(i2),
                        static_cast<long long>(i3));
                    return -1;
                }
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
            if (current_weight_pack().hybrid_sparse_gguf &&
                    role == GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY) {
                std::lock_guard<std::mutex> lock(weight_cache_mutex());
                current_weight_pack().packed_only_slice_data.insert(slice_ptr);
            }
            ++slices;
        }
    }

    if (current_weight_pack().hybrid_sparse_gguf &&
            role == GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY) {
        std::lock_guard<std::mutex> lock(weight_cache_mutex());
        current_weight_pack().packed_only_tensor_data.insert(tensor->data);
    }

    return slices;
}

bool ggml_gemmini_hybrid_sparse_mode(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    return current_weight_pack().hybrid_sparse_gguf;
}

uint64_t ggml_gemmini_weight_pack_model_fingerprint(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    return current_weight_pack().model_fingerprint;
}

int ggml_gemmini_weight_pack_finalize(void) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    if (!current_weight_pack().hybrid_sparse_gguf) {
        return 0;
    }
    if (!prepacked_weight_store().empty()) {
        std::fprintf(stderr,
            "GEMMINI-WEIGHT-PACK-UNBOUND-ENTRIES,count=%zu\n",
            prepacked_weight_store().size());
        return -1;
    }
    return 0;
}

int ggml_gemmini_weight_cache_mlock(void) {
#if defined(__linux__)
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    for (const auto & item : weight_cache()) {
        const auto & data = item.second.data;
        if (!data.empty() && mlock(data.data(), data.size() * sizeof(elem_t)) != 0) {
            return -1;
        }
    }
    return 0;
#else
    return -1;
#endif
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    auto * ctx = new ggml_backend_gemmini_context;
    ctx->cpu_fallback = ggml_backend_cpu_init();
    if (ctx->cpu_fallback == nullptr) {
        delete ctx;
        return nullptr;
    }

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
    ggml_backend_cpu_set_n_threads(ctx->cpu_fallback, n_threads);
}

// device interface

static const char * ggml_backend_gemmini_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "GEMMINI";
}

static const char * ggml_backend_gemmini_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Profile-locked BF16 Gemmini, VPU, and FlashAttention backend";
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

struct gemmini_packed_weight_buffer_context {
    void * base = nullptr;
    size_t mapped_size = 0;
};

static const char * gemmini_packed_weight_buffer_type_get_name(
        ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "GEMMINI_PACKED_WEIGHT";
}

static void gemmini_packed_weight_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<gemmini_packed_weight_buffer_context *>(buffer->context);
    if (context == nullptr) {
        return;
    }
#if defined(__linux__)
    if (context->base != nullptr && context->mapped_size != 0) {
        munmap(context->base, context->mapped_size);
    }
#else
    std::free(context->base);
#endif
    delete context;
}

static void * gemmini_packed_weight_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<gemmini_packed_weight_buffer_context *>(buffer->context);
    return context == nullptr ? nullptr : context->base;
}

static enum ggml_status gemmini_packed_weight_buffer_init_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    return packed_only_weight_name(tensor_name(tensor))
        ? GGML_STATUS_SUCCESS
        : GGML_STATUS_FAILED;
}

static void gemmini_packed_weight_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    // There is intentionally no canonical payload in this virtual buffer.
    // The verified page-packed cache is the only storage used by Gemmini.
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ASSERT(packed_only_weight_name(tensor_name(tensor)));
}

static void gemmini_packed_weight_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    // llama.cpp still walks the GGUF loading plan for this tensor.  Do not
    // dereference `data`: in a hybrid model it points at a sparse hole.
    GGML_UNUSED(buffer);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ASSERT(packed_only_weight_name(tensor_name(tensor)));
}

static void gemmini_packed_weight_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ABORT("attempted to read a Gemmini-only packed weight through a canonical buffer");
}

static void gemmini_packed_weight_buffer_clear(
        ggml_backend_buffer_t buffer,
        uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i gemmini_packed_weight_buffer_i = {
    /* .free_buffer     = */ gemmini_packed_weight_buffer_free,
    /* .get_base        = */ gemmini_packed_weight_buffer_get_base,
    /* .init_tensor     = */ gemmini_packed_weight_buffer_init_tensor,
    /* .memset_tensor   = */ gemmini_packed_weight_buffer_memset_tensor,
    /* .set_tensor      = */ gemmini_packed_weight_buffer_set_tensor,
    /* .get_tensor      = */ gemmini_packed_weight_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ gemmini_packed_weight_buffer_clear,
    /* .reset           = */ nullptr,
};

static ggml_backend_buffer_t gemmini_packed_weight_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    auto * context = new (std::nothrow) gemmini_packed_weight_buffer_context;
    if (context == nullptr) {
        return nullptr;
    }
    if (size != 0) {
#if defined(__linux__)
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_NORESERVE)
        flags |= MAP_NORESERVE;
#endif
        void * base = mmap(nullptr, size, PROT_NONE, flags, -1, 0);
        if (base == MAP_FAILED) {
            delete context;
            return nullptr;
        }
        context->base = base;
        context->mapped_size = size;
#else
        context->base = std::malloc(size);
        if (context->base == nullptr) {
            delete context;
            return nullptr;
        }
        context->mapped_size = size;
#endif
    }
    return ggml_backend_buffer_init(
        buft, gemmini_packed_weight_buffer_i, context, size);
}

static size_t gemmini_packed_weight_buffer_type_get_alignment(
        ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return k_gemmini_dma_alignment;
}

static bool gemmini_packed_weight_buffer_type_is_host(
        ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_t gemmini_packed_weight_buffer_type(void) {
    static ggml_backend_buffer_type type = {
        /* .iface   = */ {
            /* .get_name         = */ gemmini_packed_weight_buffer_type_get_name,
            /* .alloc_buffer     = */ gemmini_packed_weight_buffer_type_alloc_buffer,
            /* .get_alignment    = */ gemmini_packed_weight_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ gemmini_packed_weight_buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &type;
}

static bool tensor_uses_gemmini_packed_weight_buffer(const ggml_tensor * tensor) {
    for (int depth = 0; tensor != nullptr && depth < 64;
            ++depth, tensor = tensor->view_src) {
        if (tensor->buffer != nullptr &&
                ggml_backend_buffer_get_type(tensor->buffer) ==
                    gemmini_packed_weight_buffer_type()) {
            return true;
        }
    }
    return false;
}

static ggml_backend_t ggml_backend_gemmini_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_gemmini_init();
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // Activations, graph outputs, and mmap-backed canonical tensors all live
    // in ordinary target DRAM.  The PROT_NONE packed-weight type is exposed as
    // an extra model-weight-only candidate below, never as the compute default.
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_host_buffer_type(
        ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static ggml_backend_buffer_type_t * ggml_backend_gemmini_device_get_extra_buffers_type(
        ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t packed = gemmini_packed_weight_buffer_type();
    packed->device = dev;
    static ggml_backend_buffer_type_t extra_bufts[2] = {};
    extra_bufts[0] = packed;
    extra_bufts[1] = nullptr;
    return extra_bufts;
}

static bool ggml_backend_gemmini_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (op != nullptr) {
        unsigned packed_source_mask = 0;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const bool packed_source =
                tensor_uses_gemmini_packed_weight_buffer(op->src[i]) ||
                packed_only_weight_tensor(op->src[i]);
            if (packed_source) {
                packed_source_mask |= 1u << i;
            }
        }
        if (packed_source_mask != 0) {
            // A hybrid sparse GGUF intentionally has no canonical payload for
            // this weight. Keep a valid MUL_MAT on Gemmini even when a
            // diagnostic setting would normally send it to CPU; graph_compute
            // then reports an explicit admission failure instead of allowing
            // the CPU to multiply by sparse-hole zeros.
            return ggml_gemmini_packed_source_mask_allowed(
                       op->op == GGML_OP_MUL_MAT, packed_source_mask) &&
                packed_only_weight_tensor(op->src[0]);
        }
    }

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
        case GGML_OP_ADD:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL: {
            char reason[GGML_GEMMINI_VPU_REASON_CAPACITY] = {};
            const bool supported = vpu_can_offload(op, reason, sizeof(reason));
            if (!supported) {
                note_cpu_fallback(op, reason[0] == '\0' ? "VPU structural admission rejected" : reason);
            }
            return supported;
        }
        default:
            return false;
    }
}

static bool ggml_backend_gemmini_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft == gemmini_packed_weight_buffer_type() || ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_gemmini_device_i = {
    /* .get_name             = */ ggml_backend_gemmini_device_get_name,
    /* .get_description      = */ ggml_backend_gemmini_device_get_description,
    /* .get_memory           = */ ggml_backend_gemmini_device_get_memory,
    /* .get_type             = */ ggml_backend_gemmini_device_get_type,
    /* .get_props            = */ ggml_backend_gemmini_device_get_props,
    /* .init_backend         = */ ggml_backend_gemmini_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_gemmini_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_gemmini_device_get_host_buffer_type,
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
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return reinterpret_cast<void *>(ggml_backend_gemmini_device_get_extra_buffers_type);
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
        ++shared.output_generation;
        shared.model_load_us = 0;
        shared.model_load_cycles = 0;
        shared.summary_model_name.clear();
        shared.summary_input_prompt.clear();
        shared.summary_final_output.clear();
        shared.started_run_labels.clear();
        shared.duplicate_run_label.clear();
        shared.runs.clear();
        shared.op_events.clear();
        shared.token_events.clear();
    }
    profiler_tls() = profiler_thread_state{};
    reset_offload_admission();
    gemmini_workspace().end_activation_epoch();
}

void ggml_gemmini_profiler_reset_events(void) {
    {
        auto & shared = profiler_shared();
        std::lock_guard<std::mutex> lock(shared.mutex);
        ++shared.output_generation;
        shared.model_load_us = 0;
        shared.model_load_cycles = 0;
        shared.summary_model_name.clear();
        shared.summary_input_prompt.clear();
        shared.summary_final_output.clear();
        shared.started_run_labels.clear();
        shared.duplicate_run_label.clear();
        shared.runs.clear();
        shared.op_events.clear();
        shared.token_events.clear();
    }
    profiler_tls() = profiler_thread_state{};
    reset_offload_admission();
    gemmini_workspace().end_activation_epoch();
}

void ggml_gemmini_profiler_set_enabled(bool enabled) {
    profiler_tls().enabled = enabled;
}

bool ggml_gemmini_profiler_is_enabled(void) {
    return profiler_tls().enabled;
}

enum ggml_gemmini_hw_profile ggml_gemmini_compiled_profile(void) {
#if GGML_GEMMINI_PROFILE_ID == GGML_GEMMINI_PROFILE_SINGLE_ID
    return GGML_GEMMINI_HW_PROFILE_SINGLE_1X16;
#else
    return GGML_GEMMINI_HW_PROFILE_MULTI_4X8;
#endif
}

const char * ggml_gemmini_compiled_profile_name(void) {
    return GGML_GEMMINI_PROFILE_NAME;
}

int32_t ggml_gemmini_default_active_mask(void) {
    return static_cast<int32_t>(GGML_GEMMINI_PROFILE_LOGICAL_MASK);
}

bool ggml_gemmini_should_defer_eval(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->op == GGML_OP_RMS_NORM &&
        deferred_weighted_rms_contains(tensor);
}

bool ggml_gemmini_runtime_configure_request(int32_t active_mask, int32_t gemmini_id, bool split_mode) {
    if (!ggml_gemmini_profile_request_is_valid(active_mask, gemmini_id, split_mode)) {
        return false;
    }
    request_runtime().vpu_disabled = false;
    request_runtime().vpu_failure_reason.clear();
    request_runtime().active_mask = active_mask;
    request_runtime().gemmini_id = gemmini_id;
    request_runtime().split_mode = split_mode;
    return true;
}

void ggml_gemmini_runtime_clear_request(void) {
    request_runtime() = request_runtime_config{};
    gemmini_workspace().end_activation_epoch();
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
        const gemmini_page_packing_config packing =
            gemmini_page_packing_for_shape(rows_i, cols_in_i, true);

        gemmini_aligned_vector<elem_t> a_row_major(rows * cols_in);
        gemmini_aligned_vector<elem_t> b_row_major(cols_in * cols_out);
        gemmini_aligned_vector<acc_t> d_row_major(rows * cols_out);

        for (size_t r = 0; r < rows; ++r) {
            for (size_t k = 0; k < cols_in; ++k) {
                const int value = static_cast<int>((r * 17 + k * 7) % 15) - 7;
                a_row_major[r * cols_in + k] = encode_bf16(static_cast<float>(value) / 8.0f);
            }
        }

        for (size_t k = 0; k < cols_in; ++k) {
            for (size_t c = 0; c < cols_out; ++c) {
                const int value = static_cast<int>((k * 13 + c * 5) % 11) - 5;
                b_row_major[k * cols_out + c] = encode_bf16(static_cast<float>(value) / 8.0f);
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
        float first_expected = 0.0f;
        float first_actual = 0.0f;

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols_out; ++c) {
                float expected = d_row_major[r * cols_out + c];
                for (size_t k = 0; k < cols_in; ++k) {
                    expected += ggml_gemmini_bf16_to_float(
                                    static_cast<uint16_t>(a_row_major[r * cols_in + k])) *
                                ggml_gemmini_bf16_to_float(
                                    static_cast<uint16_t>(b_row_major[k * cols_out + c]));
                }

                const float actual = static_cast<float>(out_row_major[r * cols_out + c]);
                if (!std::isfinite(actual) ||
                        std::fabs(actual - expected) > 1.0e-3f + 1.0e-2f * std::fabs(expected)) {
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
            std::printf("GEMMINI-SMOKE-FAIL,mismatches=%zu,first_r=%zu,first_c=%zu,expected=%g,actual=%g\n",
                mismatch_count,
                first_r,
                first_c,
                static_cast<double>(first_expected),
                static_cast<double>(first_actual));
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
    gemmini_workspace().end_activation_epoch();
    profiler_tls().current_run = run_label != nullptr && *run_label != '\0'
        ? run_label
        : "hybrid";
    {
        auto & shared = profiler_shared();
        std::lock_guard<std::mutex> lock(shared.mutex);
        if (!shared.started_run_labels.insert(profiler_tls().current_run).second) {
            shared.duplicate_run_label = profiler_tls().current_run;
        }
    }
    profiler_tls().current_phase = "prefill";
    profiler_tls().token_index = -1;
    profiler_tls().pending_eval = pending_eval_state{};
    profiler_tls().last_gemmini = last_gemmini_state{};
    profiler_tls().fallback_reasons.clear();
    profiler_tls().graph_call_profiled_us = 0;
    profiler_tls().graph_call_profiled_cycles = 0;
}

void ggml_gemmini_profiler_set_phase(const char * phase, int32_t token_index) {
    profiler_tls().current_phase = phase ? phase : "prefill";
    profiler_tls().token_index = token_index;
}

void ggml_gemmini_profiler_graph_overhead_begin(void) {
    gemmini_workspace().begin_activation_epoch();
    profiler_tls().graph_call_profiled_us = 0;
    profiler_tls().graph_call_profiled_cycles = 0;
}

void ggml_gemmini_profiler_graph_overhead_end(const char * phase, int32_t token_index, int64_t wall_us, uint64_t cycles) {
    gemmini_workspace().end_activation_epoch();
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
    run.gemmini_mask = active_mask & GGML_GEMMINI_PROFILE_LOGICAL_MASK;
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
    if (!profiler_tls().enabled) {
        return;
    }

    gemmini_op_event event;
    event.run_label = current_run_label();
    event.phase = "sampling";
    event.backend = profiler_tls().current_run.rfind("cpu_baseline", 0) == 0
        ? "cpu_baseline"
        : "cpu_fallback";
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
    run.completed = true;
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
        event.backend = profiler_tls().last_gemmini.backend.empty()
            ? "gemmini_bf16"
            : profiler_tls().last_gemmini.backend;
        event.stage = profiler_tls().last_gemmini.stage;
        event.layer_index = profiler_tls().last_gemmini.layer_index;
        event.token_index = profiler_tls().last_gemmini.token_index;
        event.accelerator_call_us = profiler_tls().last_gemmini.accelerator_call_us;
        event.accelerator_call_cycles = profiler_tls().last_gemmini.accelerator_call_cycles;
        event.flash_attention_components = profiler_tls().last_gemmini.flash_attention_components;
        event.weight_cache_lookup_pack_us = profiler_tls().last_gemmini.weight_cache_lookup_pack_us;
        event.weight_cache_lookup_pack_cycles = profiler_tls().last_gemmini.weight_cache_lookup_pack_cycles;
        event.activation_stage_pack_us = profiler_tls().last_gemmini.activation_stage_pack_us;
        event.activation_stage_pack_cycles = profiler_tls().last_gemmini.activation_stage_pack_cycles;
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
        event.output_unpack_store_us = profiler_tls().last_gemmini.output_unpack_store_us;
        event.output_unpack_store_cycles = profiler_tls().last_gemmini.output_unpack_store_cycles;
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
        event.flash_page_packed_mask = profiler_tls().last_gemmini.flash_page_packed_mask;
        profiler_tls().last_gemmini.valid = false;
    }

    if (!cpu_baseline && event.backend == "cpu_fallback") {
        auto fallback = profiler_tls().fallback_reasons.find(t);
        if (fallback != profiler_tls().fallback_reasons.end()) {
            event.fallback_reason = std::move(fallback->second);
            profiler_tls().fallback_reasons.erase(fallback);
        }
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
    aggregate_entry vpu;
    aggregate_entry flash_attention;
    aggregate_entry cpu;
    std::map<std::string, aggregate_entry> gemmini_components;
    std::map<std::string, aggregate_entry> flash_attention_components;
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
    aggregate_entry vpu_time;
    aggregate_entry flash_attention_time;
    if (event.backend == "gemmini_bf16") {
        gemmini_time.wall_us = event.wall_us;
        gemmini_time.wall_cycles = event.wall_cycles;
    } else if (event.backend == "vpu") {
        vpu_time.wall_us = event.wall_us;
        vpu_time.wall_cycles = event.wall_cycles;
    } else if (event.backend == "flash_attention") {
        flash_attention_time.wall_us = event.wall_us;
        flash_attention_time.wall_cycles = event.wall_cycles;
    }
    if (event.backend == "gemmini_bf16") {
        add_stage_component(entry.gemmini_components, "weight_cache_lookup_pack",
            event.weight_cache_lookup_pack_us, event.weight_cache_lookup_pack_cycles);
        add_stage_component(entry.gemmini_components, "activation_stage_pack",
            event.activation_stage_pack_us, event.activation_stage_pack_cycles);
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
        add_stage_component(entry.gemmini_components, "output_unpack_store",
            event.output_unpack_store_us, event.output_unpack_store_cycles);
    } else if (event.backend == "flash_attention") {
        for (const auto & component : event.flash_attention_components) {
            add_stage_component(entry.flash_attention_components, component.first,
                component.second.wall_us, component.second.wall_cycles);
        }
    }

    entry.gemmini.wall_us += gemmini_time.wall_us;
    entry.gemmini.wall_cycles += gemmini_time.wall_cycles;
    entry.vpu.wall_us += vpu_time.wall_us;
    entry.vpu.wall_cycles += vpu_time.wall_cycles;
    entry.flash_attention.wall_us += flash_attention_time.wall_us;
    entry.flash_attention.wall_cycles += flash_attention_time.wall_cycles;
    entry.cpu.wall_us += event.wall_us - gemmini_time.wall_us - vpu_time.wall_us - flash_attention_time.wall_us;
    entry.cpu.wall_cycles += event.wall_cycles - gemmini_time.wall_cycles - vpu_time.wall_cycles - flash_attention_time.wall_cycles;
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
        "weight_cache_lookup_pack",
        "activation_stage_pack",
        "flush",
        "job_configuration",
        "calculate_tiling_factors",
        "gemmini_configuration",
        "gemmini_run",
        "output_unpack_store",
        "other_host",
    };
    return order;
}

static const std::vector<std::string> & backend_flash_attention_breakdown_order() {
    static const std::vector<std::string> order = {
        "input_validation",
        "mask_prepare",
        "workspace_prepare",
        "plan_preflight",
        "input_pack",
        "fused_attention_run",
        "output_unpack_store",
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
    return event.backend == "gemmini_bf16" &&
        event.op_name == "MUL_MAT" &&
        is_decode_graph_phase(event.phase) &&
        event.dim_i > 0 &&
        event.dim_j > 0 &&
        event.dim_k > 0;
}

static aggregate_entry gemmini_event_component(const gemmini_op_event & event, const std::string & name) {
    if (name == "weight_cache_lookup_pack") {
        return {event.weight_cache_lookup_pack_us, event.weight_cache_lookup_pack_cycles};
    }
    if (name == "activation_stage_pack") {
        return {event.activation_stage_pack_us, event.activation_stage_pack_cycles};
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
    if (name == "output_unpack_store") {
        return {event.output_unpack_store_us, event.output_unpack_store_cycles};
    }
    if (name == "other_host") {
        const int64_t accounted_us =
            event.weight_cache_lookup_pack_us +
            event.activation_stage_pack_us +
            event.flush_us +
            event.job_configuration_us +
            event.calculate_tiling_factors_us +
            event.gemmini_configuration_us +
            event.gemmini_run_us +
            event.output_unpack_store_us;
        const uint64_t accounted_cycles =
            event.weight_cache_lookup_pack_cycles +
            event.activation_stage_pack_cycles +
            event.flush_cycles +
            event.job_configuration_cycles +
            event.calculate_tiling_factors_cycles +
            event.gemmini_configuration_cycles +
            event.gemmini_run_cycles +
            event.output_unpack_store_cycles;

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

static aggregate_entry flash_attention_event_component(
        const gemmini_op_event & event,
        const std::string & name) {
    if (name == "other_host") {
        aggregate_entry accounted;
        for (const auto & component : event.flash_attention_components) {
            accounted.wall_us += component.second.wall_us;
            accounted.wall_cycles += component.second.wall_cycles;
        }

        aggregate_entry other;
        if (event.wall_us > accounted.wall_us) {
            other.wall_us = event.wall_us - accounted.wall_us;
        }
        if (event.wall_cycles > accounted.wall_cycles) {
            other.wall_cycles = event.wall_cycles - accounted.wall_cycles;
        }
        return other;
    }

    const auto it = event.flash_attention_components.find(name);
    if (it == event.flash_attention_components.end()) {
        return {};
    }
    return {it->second.wall_us, it->second.wall_cycles};
}

static aggregate_entry stage_flash_attention_component(
        const stage_aggregate_entry & entry,
        const std::string & name) {
    if (name == "other_host") {
        aggregate_entry accounted;
        for (const auto & component : entry.flash_attention_components) {
            accounted.wall_us += component.second.wall_us;
            accounted.wall_cycles += component.second.wall_cycles;
        }

        aggregate_entry other;
        if (entry.flash_attention.wall_us > accounted.wall_us) {
            other.wall_us = entry.flash_attention.wall_us - accounted.wall_us;
        }
        if (entry.flash_attention.wall_cycles > accounted.wall_cycles) {
            other.wall_cycles = entry.flash_attention.wall_cycles - accounted.wall_cycles;
        }
        return other;
    }

    const auto it = entry.flash_attention_components.find(name);
    return it == entry.flash_attention_components.end() ? aggregate_entry{} : it->second;
}

static const std::array<const char *, 7> & profiler_output_file_names() {
    static const std::array<const char *, 7> names = {
        "run_summary.csv",
        "token_trace.csv",
        "op_profile.csv",
        "stage_summary.csv",
        "matmul_summary.csv",
        "backend_summary.csv",
        "summary.md",
    };
    return names;
}

static bool truncate_profiler_output_files(const std::string & dir) {
    for (const char * name : profiler_output_file_names()) {
        std::ofstream out(dir + "/" + name, std::ios::out | std::ios::trunc);
        if (!out) {
            return false;
        }
    }
    return true;
}

static bool profiler_output_file_empty(const std::string & path) {
    std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
    return !in || in.tellg() == std::streampos(0);
}

int ggml_gemmini_profiler_write_results(const char * results_dir) {
    const std::string dir = results_dir ? results_dir : ".";
    std::unique_lock<std::mutex> output_lock(profiler_output_mutex());
    profiler_output_state & output_state = profiler_outputs()[dir];
    profiler_output_transaction output_transaction{output_state};
    int64_t model_load_us = 0;
    uint64_t model_load_cycles = 0;
    std::string summary_model_name;
    std::map<std::string, run_totals> runs;
    std::vector<gemmini_op_event> op_events;
    std::vector<token_event> token_events;
    size_t captured_op_event_end = 0;
    size_t captured_token_event_end = 0;
    size_t next_op_event_cursor = 0;
    size_t next_token_event_cursor = 0;
    bool fresh_output = false;

    {
        auto & shared = profiler_shared();
        std::lock_guard<std::mutex> lock(shared.mutex);
        if (!shared.duplicate_run_label.empty()) {
            std::fprintf(stderr,
                "GEMMINI-PROFILER-ERROR,reason=duplicate_run_label,run_label=%s\n",
                shared.duplicate_run_label.c_str());
            std::fflush(stderr);
            output_transaction.committed = true;
            return -2;
        }
        if (output_state.generation != shared.output_generation) {
            // A profiler reset starts a new result session. Preserve the old
            // reset semantics by replacing stale files exactly once, then use
            // append-only writes for the rest of this generation.
            if (!truncate_profiler_output_files(dir)) {
                return -1;
            }
            output_state = profiler_output_state{};
            output_state.generation = shared.output_generation;
            fresh_output = true;
        }

        model_load_us = shared.model_load_us;
        model_load_cycles = shared.model_load_cycles;
        summary_model_name = shared.summary_model_name;
        captured_op_event_end = shared.op_events.size();
        captured_token_event_end = shared.token_events.size();
        next_op_event_cursor = captured_op_event_end;
        next_token_event_cursor = captured_token_event_end;

        std::unordered_set<std::string> pending_run_labels;
        for (const auto & run : shared.runs) {
            if (run.second.completed &&
                    output_state.exported_run_labels.find(run.first) ==
                    output_state.exported_run_labels.end()) {
                pending_run_labels.insert(run.first);
                runs.insert(run);
            }
        }

        for (size_t i = output_state.op_event_cursor; i < captured_op_event_end; ++i) {
            if (pending_run_labels.find(shared.op_events[i].run_label) !=
                    pending_run_labels.end()) {
                op_events.push_back(shared.op_events[i]);
            } else {
                const auto run = shared.runs.find(shared.op_events[i].run_label);
                if (run == shared.runs.end() || !run->second.completed) {
                    next_op_event_cursor = std::min(next_op_event_cursor, i);
                }
            }
        }
        for (size_t i = output_state.token_event_cursor; i < captured_token_event_end; ++i) {
            if (pending_run_labels.find(shared.token_events[i].run_label) !=
                    pending_run_labels.end()) {
                token_events.push_back(shared.token_events[i]);
            } else {
                const auto run = shared.runs.find(shared.token_events[i].run_label);
                if (run == shared.runs.end() || !run->second.completed) {
                    next_token_event_cursor = std::min(next_token_event_cursor, i);
                }
            }
        }
    }

    if (runs.empty() && !fresh_output) {
        output_transaction.committed = true;
        return 0;
    }

    {
        const std::string path = dir + "/run_summary.csv";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "run_label,gemmini_mask,gemmini_count,page_packed_a,page_packed_b,page_packed_c,page_packed_d,prompt_tokens,generated_tokens,total_us,total_cycles,ttft_us,ttft_cycles,tpot_us_per_token,tpot_cycles_per_token,input_prompt,final_output,hardware_profile,flash_page_packed_q,flash_page_packed_k,flash_page_packed_v\n";
        }
        const gemmini_page_packing_config & packing = gemmini_page_packing();
        const flash_page_packing_config & flash_packing = flash_page_packing();
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
                << csv_escape(totals.final_output) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << ","
                << (flash_packing.queries ? 1 : 0) << ","
                << (flash_packing.keys ? 1 : 0) << ","
                << (flash_packing.values ? 1 : 0) << "\n";
        }
        out.flush();
        if (!out) {
            return -1;
        }
    }

    {
        const std::string path = dir + "/token_trace.csv";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "run_label,token_index,token_id,piece\n";
        }
        for (const auto & token : token_events) {
            out << csv_escape(token.run_label) << ","
                << token.token_index << ","
                << token.token_id << ","
                << csv_escape(token.piece) << "\n";
        }
        out.flush();
        if (!out) {
            return -1;
        }
    }

    {
        const std::string path = dir + "/op_profile.csv";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "run_label,phase,token_index,backend,hardware_profile,fallback_reason,stage,layer,node_name,src0_name,op,wall_us,wall_cycles,weight_cache_lookup_pack_us,weight_cache_lookup_pack_cycles,activation_stage_pack_us,activation_stage_pack_cycles,flush_us,flush_cycles,job_configuration_us,job_configuration_cycles,calculate_tiling_factors_us,calculate_tiling_factors_cycles,gemmini_configuration_us,gemmini_configuration_cycles,gemmini_run_us,gemmini_run_cycles,gemmini_call_us,gemmini_call_cycles,output_unpack_store_us,output_unpack_store_cycles,gemmini_total_cycles,gemmini_load_cycles,gemmini_preload_cycles,gemmini_compute_cycles,gemmini_store_cycles,gemmini_wait_cycles,dim_i,dim_j,dim_k,tile_i,tile_j,tile_k,page_packed_a,page_packed_b,page_packed_c,page_packed_d,accelerator_call_us,accelerator_call_cycles";
            for (const std::string & name : backend_flash_attention_breakdown_order()) {
                if (name != "other_host") {
                    out << ",flash_" << name << "_us,flash_" << name << "_cycles";
                }
            }
            out << ",flash_page_packed_q,flash_page_packed_k,flash_page_packed_v";
            out << "\n";
        }
        for (const auto & event : op_events) {
            out << csv_escape(event.run_label) << ","
                << csv_escape(event.phase) << ","
                << event.token_index << ","
                << csv_escape(event.backend) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << ","
                << csv_escape(event.fallback_reason) << ","
                << csv_escape(event.stage) << ","
                << event.layer_index << ","
                << csv_escape(event.node_name) << ","
                << csv_escape(event.src0_name) << ","
                << csv_escape(event.op_name) << ","
                << event.wall_us << ","
                << event.wall_cycles << ","
                << event.weight_cache_lookup_pack_us << ","
                << event.weight_cache_lookup_pack_cycles << ","
                << event.activation_stage_pack_us << ","
                << event.activation_stage_pack_cycles << ","
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
                << event.output_unpack_store_us << ","
                << event.output_unpack_store_cycles << ","
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
                << ((event.page_packed_mask & 0x8) != 0) << ","
                << event.accelerator_call_us << ","
                << event.accelerator_call_cycles;
            for (const std::string & name : backend_flash_attention_breakdown_order()) {
                if (name == "other_host") {
                    continue;
                }
                const aggregate_entry component = flash_attention_event_component(event, name);
                out << "," << component.wall_us << "," << component.wall_cycles;
            }
            out << "," << ((event.flash_page_packed_mask & 0x1) != 0)
                << "," << ((event.flash_page_packed_mask & 0x2) != 0)
                << "," << ((event.flash_page_packed_mask & 0x4) != 0);
            out << "\n";
        }
        out.flush();
        if (!out) {
            return -1;
        }
    }

    std::map<std::string, std::map<std::string, std::map<std::string, stage_aggregate_entry>>> phase_stage_totals;
    std::map<std::string, std::map<std::string, std::map<std::string, stage_aggregate_entry>>> stage_summary_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> backend_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> phase_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> repack_stage_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> gemmini_host_component_totals;
    std::map<std::pair<std::string, std::string>, aggregate_entry> flash_attention_component_totals;
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

            if (!event.backend.empty()) {
                add_aggregate(
                    backend_totals,
                    event.run_label,
                    event.backend,
                    event.wall_us,
                    event.wall_cycles);
            }

            phase_totals[{event.run_label, event.phase}].wall_us += event.wall_us;
            phase_totals[{event.run_label, event.phase}].wall_cycles += event.wall_cycles;
        }

        if (event.backend == "gemmini_bf16") {
            add_aggregate(gemmini_host_component_totals, event.run_label, "weight_cache_lookup_pack",
                event.weight_cache_lookup_pack_us, event.weight_cache_lookup_pack_cycles);
            add_aggregate(gemmini_host_component_totals, event.run_label, "activation_stage_pack",
                event.activation_stage_pack_us, event.activation_stage_pack_cycles);
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
            add_aggregate(gemmini_host_component_totals, event.run_label, "output_unpack_store",
                event.output_unpack_store_us, event.output_unpack_store_cycles);

            gemmini_total_cycles_by_run[event.run_label] += event.gemmini_total_cycles;
            gemmini_component_totals[{event.run_label, "load"}] += event.gemmini_load_cycles;
            gemmini_component_totals[{event.run_label, "preload"}] += event.gemmini_preload_cycles;
            gemmini_component_totals[{event.run_label, "compute"}] += event.gemmini_compute_cycles;
            gemmini_component_totals[{event.run_label, "store"}] += event.gemmini_store_cycles;
            gemmini_component_totals[{event.run_label, "wait"}] += event.gemmini_wait_cycles;
        } else if (event.backend == "flash_attention") {
            for (const std::string & name : backend_flash_attention_breakdown_order()) {
                const aggregate_entry component = flash_attention_event_component(event, name);
                add_aggregate(flash_attention_component_totals, event.run_label, name,
                    component.wall_us, component.wall_cycles);
            }
        }
    }

    const std::map<std::string, aggregate_entry> backend_events_by_run = aggregate_by_run(backend_totals);
    for (const auto & run_kv : runs) {
        const aggregate_entry profiled = backend_events_by_run.count(run_kv.first)
            ? backend_events_by_run.at(run_kv.first)
            : aggregate_entry{};
        const int64_t other_us = run_kv.second.wall_us > profiled.wall_us
            ? run_kv.second.wall_us - profiled.wall_us
            : 0;
        const uint64_t other_cycles = run_kv.second.wall_cycles > profiled.wall_cycles
            ? run_kv.second.wall_cycles - profiled.wall_cycles
            : 0;
        add_aggregate(
            backend_totals,
            run_kv.first,
            "runtime_unprofiled",
            other_us,
            other_cycles);
    }

    std::map<std::string, aggregate_entry> gemmini_breakdown_by_run = aggregate_by_run(gemmini_host_component_totals);
    for (const auto & run_kv : runs) {
        const auto gemmini_it = backend_totals.find({run_kv.first, "gemmini_bf16"});
        const aggregate_entry gemmini = gemmini_it == backend_totals.end()
            ? aggregate_entry{}
            : gemmini_it->second;
        const aggregate_entry accounted = gemmini_breakdown_by_run[run_kv.first];
        const int64_t other_us = gemmini.wall_us > accounted.wall_us ? gemmini.wall_us - accounted.wall_us : 0;
        const uint64_t other_cycles = gemmini.wall_cycles > accounted.wall_cycles ? gemmini.wall_cycles - accounted.wall_cycles : 0;
        add_aggregate(gemmini_host_component_totals, run_kv.first, "other_host", other_us, other_cycles);
    }

    const std::map<std::string, aggregate_entry> backend_profiled_by_run = aggregate_by_run(backend_totals);
    const std::map<std::string, aggregate_entry> phase_profiled_by_run = aggregate_by_run(phase_totals);
    const std::map<std::string, aggregate_entry> repack_by_run = aggregate_by_run(repack_stage_totals);
    const std::map<std::string, aggregate_entry> gemmini_host_by_run = aggregate_by_run(gemmini_host_component_totals);
    const std::map<std::string, aggregate_entry> flash_attention_by_run =
        aggregate_by_run(flash_attention_component_totals);

    {
        const std::string path = dir + "/stage_summary.csv";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "run_label,phase,stage,total_us,total_cycles,pct_of_phase,gemmini_us,gemmini_cycles,cpu_us,cpu_cycles,pct_gemmini_of_stage,pct_cpu_of_stage";
            for (const std::string & name : backend_gemmini_breakdown_order()) {
                out << ",pct_" << name << "_of_stage_gemmini";
            }
            out << ",vpu_us,vpu_cycles,flash_attention_us,flash_attention_cycles,pct_vpu_of_stage,pct_flash_attention_of_stage";
            for (const std::string & name : backend_flash_attention_breakdown_order()) {
                out << ",pct_" << name << "_of_stage_flash_attention";
            }
            out << ",hardware_profile\n";
        }
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
                    out << "," << stage_kv.second.vpu.wall_us
                        << "," << stage_kv.second.vpu.wall_cycles
                        << "," << stage_kv.second.flash_attention.wall_us
                        << "," << stage_kv.second.flash_attention.wall_cycles
                        << "," << pct_i64(stage_kv.second.vpu.wall_us, stage_kv.second.total.wall_us)
                        << "," << pct_i64(stage_kv.second.flash_attention.wall_us, stage_kv.second.total.wall_us);
                    for (const std::string & name : backend_flash_attention_breakdown_order()) {
                        const aggregate_entry component =
                            stage_flash_attention_component(stage_kv.second, name);
                        out << "," << pct_i64(
                            component.wall_us, stage_kv.second.flash_attention.wall_us);
                    }
                    out << "," << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
                }
            }
        }
        out.flush();
        if (!out) {
            return -1;
        }
    }

    {
        const std::string path = dir + "/matmul_summary.csv";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "run_label,phase,shape,dim_i,dim_j,dim_k,tile_i,tile_j,tile_k,page_packed_a,page_packed_b,page_packed_c,page_packed_d,count,total_us,total_cycles,avg_us,avg_cycles";
            for (const std::string & name : backend_gemmini_breakdown_order()) {
                out << ",pct_" << name << "_of_matmul_wall";
            }
            out << ",hardware_profile\n";
        }

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
            out << "," << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
        }
        out.flush();
        if (!out) {
            return -1;
        }
    }

    {
        const std::string path = dir + "/backend_summary.csv";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "run_label,category,name,total_us,total_cycles,pct_of_runtime_breakdown,pct_of_parent,hardware_profile\n";
        }

        for (const auto & kv : backend_totals) {
            const aggregate_entry denom = runtime_denominator(kv.first.first, runs, backend_profiled_by_run);
            out << csv_escape(kv.first.first) << ",backend,"
                << csv_escape(kv.first.second) << ","
                << kv.second.wall_us << ","
                << kv.second.wall_cycles << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
        }

        for (const auto & kv : phase_totals) {
            const aggregate_entry denom = runtime_denominator(kv.first.first, runs, phase_profiled_by_run);
            out << csv_escape(kv.first.first) << ",phase,"
                << csv_escape(kv.first.second) << ","
                << kv.second.wall_us << ","
                << kv.second.wall_cycles << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << ","
                << pct_i64(kv.second.wall_us, denom.wall_us) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
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
                << pct_i64(overhead.wall_us, denom.wall_us) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
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
                << pct_i64(kv.second.wall_us, repack_total.wall_us) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
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
                    << pct_i64(component.wall_us, host_total.wall_us) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
            }
        }

        for (const auto & run_kv : runs) {
            const int64_t run_total_us = run_kv.second.wall_us;
            const aggregate_entry flash_total = flash_attention_by_run.count(run_kv.first)
                ? flash_attention_by_run.at(run_kv.first)
                : aggregate_entry{};
            for (const std::string & name : backend_flash_attention_breakdown_order()) {
                const auto component_it =
                    flash_attention_component_totals.find({run_kv.first, name});
                if (component_it == flash_attention_component_totals.end()) {
                    continue;
                }
                const aggregate_entry & component = component_it->second;
                out << csv_escape(run_kv.first) << ",backend_flash_attention_breakdown,"
                    << csv_escape(name) << ","
                    << component.wall_us << ","
                    << component.wall_cycles << ","
                    << pct_i64(component.wall_us, run_total_us) << ","
                    << pct_i64(component.wall_us, flash_total.wall_us) << ","
                    << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
            }
        }

        for (const auto & kv : gemmini_component_totals) {
            const uint64_t parent = gemmini_total_cycles_by_run[kv.first.first];
            out << csv_escape(kv.first.first) << ",gemmini_subphase,"
                << csv_escape(kv.first.second) << ","
                << 0 << ","
                << kv.second << ","
                << pct_u64(kv.second, runs[kv.first.first].wall_cycles) << ","
                << pct_u64(kv.second, parent) << ","
                << csv_escape(GGML_GEMMINI_PROFILE_NAME) << "\n";
        }
        out.flush();
        if (!out) {
            return -1;
        }
    }

    {
        const std::string path = dir + "/summary.md";
        const bool write_header = profiler_output_file_empty(path);
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) {
            return -1;
        }

        if (write_header) {
            out << "# llama-firesim summary\n\n";
            out << "| metric | value |\n";
            out << "| --- | --- |\n";
            out << "| model | " << markdown_table_escape(summary_model_name) << " |\n";
            out << "| hardware profile | " << markdown_table_escape(GGML_GEMMINI_PROFILE_NAME) << " |\n\n";
            out << "Model load: " << model_load_us << " us / " << model_load_cycles << " cycles\n\n";
            out << "Prompt and final-output text is recorded in each run section below.\n\n";
        }

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
            out << "| Gemmini page packing requested (A/B/C/D) | "
                << (gemmini_page_packing().a ? 1 : 0) << "/"
                << (gemmini_page_packing().b ? 1 : 0) << "/"
                << (gemmini_page_packing().c ? 1 : 0) << "/"
                << (gemmini_page_packing().d ? 1 : 0) << " |\n";
            out << "| FlashAttention page packing requested (Q/K/V) | "
                << (flash_page_packing().queries ? 1 : 0) << "/"
                << (flash_page_packing().keys ? 1 : 0) << "/"
                << (flash_page_packing().values ? 1 : 0) << " |\n";
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
                    out << "| stage | total us | total cycles | % of " << phase_title.first
                        << " stage time | % gemmini | % vpu | % flash attention | % cpu";
                    for (const std::string & name : backend_gemmini_breakdown_order()) {
                        out << " | % " << name << " of gemmini";
                    }
                    for (const std::string & name : backend_flash_attention_breakdown_order()) {
                        out << " | % " << name << " of flash attention";
                    }
                    out << " |\n";
                    out << "| --- | --- | --- | --- | --- | --- | --- | ---";
                    for (size_t i = 0; i < backend_gemmini_breakdown_order().size(); ++i) {
                        out << " | ---";
                    }
                    for (size_t i = 0; i < backend_flash_attention_breakdown_order().size(); ++i) {
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
                            << pct_i64(kv.second.vpu.wall_us, kv.second.total.wall_us) << " | "
                            << pct_i64(kv.second.flash_attention.wall_us, kv.second.total.wall_us) << " | "
                            << pct_i64(kv.second.cpu.wall_us, kv.second.total.wall_us);
                        for (const std::string & name : backend_gemmini_breakdown_order()) {
                            const aggregate_entry component = stage_gemmini_component(kv.second, name);
                            out << " | " << pct_i64(component.wall_us, kv.second.gemmini.wall_us);
                        }
                        for (const std::string & name : backend_flash_attention_breakdown_order()) {
                            const aggregate_entry component =
                                stage_flash_attention_component(kv.second, name);
                            out << " | " << pct_i64(
                                component.wall_us, kv.second.flash_attention.wall_us);
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
                out << "These subphases split the full wall time of operations with `backend=gemmini_bf16`. `other_host` closes the gap so the total matches Backend Split's `gemmini_bf16` row.\n\n";
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

            const aggregate_entry backend_flash_attention_total =
                flash_attention_by_run.count(run_label)
                    ? flash_attention_by_run.at(run_label)
                    : aggregate_entry{};
            if (has_aggregate_value(backend_flash_attention_total)) {
                out << "### Backend FlashAttention Breakdown\n\n";
                out << "These measured host-visible subphases split operations with `backend=flash_attention`. `fused_attention_run` is the complete fused QK/online-softmax/PV accelerator interval; no unsupported internal ratio is inferred. `other_host` closes the gap to the Backend Split `flash_attention` row.\n\n";
                out << "| subphase | total us | total cycles | % of run | % of backend FlashAttention |\n";
                out << "| --- | --- | --- | --- | --- |\n";
                for (const std::string & name : backend_flash_attention_breakdown_order()) {
                    const auto component_it =
                        flash_attention_component_totals.find({run_label, name});
                    if (component_it == flash_attention_component_totals.end()) {
                        continue;
                    }
                    const aggregate_entry & component = component_it->second;
                    out << "| " << name << " | "
                        << component.wall_us << " | "
                        << component.wall_cycles << " | "
                        << pct_i64(component.wall_us, totals.wall_us) << " | "
                        << pct_i64(component.wall_us, backend_flash_attention_total.wall_us)
                        << " |\n";
                }
                out << "| total | "
                    << backend_flash_attention_total.wall_us << " | "
                    << backend_flash_attention_total.wall_cycles << " | "
                    << pct_i64(backend_flash_attention_total.wall_us, totals.wall_us)
                    << " | 100 |\n\n";
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
        out.flush();
        if (!out) {
            return -1;
        }
    }

    output_state.op_event_cursor = next_op_event_cursor;
    output_state.token_event_cursor = next_token_event_cursor;
    for (const auto & run : runs) {
        output_state.exported_run_labels.insert(run.first);
    }
    output_transaction.committed = true;

    return 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
