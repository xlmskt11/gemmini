#pragma once

#include <cstddef>
#include <string>

enum ggml_gemmini_weight_role {
    GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY = 0,
    GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY = 1,
    GGML_GEMMINI_WEIGHT_ROLE_SHARED = 2,
};

#define GGML_GEMMINI_WEIGHT_USAGE_ORIGINAL (1u << 0)
#define GGML_GEMMINI_WEIGHT_USAGE_MUL_MAT  (1u << 1)

static inline bool ggml_gemmini_weight_role_requires_pack(
        enum ggml_gemmini_weight_role role) {
    return role == GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY ||
           role == GGML_GEMMINI_WEIGHT_ROLE_SHARED;
}

static inline bool ggml_gemmini_weight_role_requires_original(
        enum ggml_gemmini_weight_role role) {
    return role != GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY;
}

// Packed-only storage is valid only for MUL_MAT's weight operand (src0).
// Keeping this rule independent from ggml internals makes the model-loader
// buffer selection regression-testable on the host.
static inline bool ggml_gemmini_packed_source_mask_allowed(
        bool is_mul_mat,
        unsigned packed_source_mask) {
    return is_mul_mat && packed_source_mask == 1u;
}

// Model tensors use the canonical *.weight suffix.  Keep the historical
// short aliases as well because a few converted architectures expose names
// such as layers.N.attention.wq without the canonical suffix.
static inline bool ggml_gemmini_is_model_weight_name(const std::string & name) {
    constexpr const char * weight_suffix = ".weight";
    constexpr std::size_t weight_suffix_size = 7;

    if (name.size() >= weight_suffix_size &&
            name.compare(name.size() - weight_suffix_size,
                         weight_suffix_size,
                         weight_suffix) == 0) {
        return true;
    }

    return name.find(".wq") != std::string::npos ||
           name.find(".wk") != std::string::npos ||
           name.find(".wv") != std::string::npos ||
           name.find(".wo") != std::string::npos;
}

static inline bool ggml_gemmini_is_dense_matmul_weight_name(const std::string & name) {
    if (name == "output.weight") {
        return true;
    }

    static constexpr const char * matrix_suffixes[] = {
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_qkv.weight",
        ".attn_output.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    };
    for (const char * suffix : matrix_suffixes) {
        const std::size_t suffix_size = std::char_traits<char>::length(suffix);
        if (name.size() >= suffix_size &&
                name.compare(name.size() - suffix_size, suffix_size, suffix) == 0) {
            return true;
        }
    }

    // Historical converters used these names without the canonical suffix.
    static constexpr const char * short_suffixes[] = { ".wq", ".wk", ".wv", ".wo" };
    for (const char * suffix : short_suffixes) {
        const std::size_t suffix_size = std::char_traits<char>::length(suffix);
        if (name.size() >= suffix_size &&
                name.compare(name.size() - suffix_size, suffix_size, suffix) == 0) {
            return true;
        }
    }

    return false;
}

// This is deliberately fail-closed. Unknown weights stay in canonical GGUF
// form instead of being removed based only on their rank or name suffix.
static inline enum ggml_gemmini_weight_role ggml_gemmini_classify_weight(
        const std::string & name,
        bool token_embedding_is_tied_output) {
    if (name == "token_embd.weight") {
        return token_embedding_is_tied_output
            ? GGML_GEMMINI_WEIGHT_ROLE_SHARED
            : GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY;
    }
    return ggml_gemmini_is_dense_matmul_weight_name(name)
        ? GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY
        : GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY;
}

static inline enum ggml_gemmini_weight_role ggml_gemmini_classify_weight_usage(
        unsigned usage) {
    const bool original = (usage & GGML_GEMMINI_WEIGHT_USAGE_ORIGINAL) != 0;
    const bool matmul = (usage & GGML_GEMMINI_WEIGHT_USAGE_MUL_MAT) != 0;
    if (original && matmul) {
        return GGML_GEMMINI_WEIGHT_ROLE_SHARED;
    }
    if (matmul) {
        return GGML_GEMMINI_WEIGHT_ROLE_GEMMINI_ONLY;
    }
    return GGML_GEMMINI_WEIGHT_ROLE_CPU_ONLY;
}
