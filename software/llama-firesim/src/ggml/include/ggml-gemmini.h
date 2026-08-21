#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_gemmini_reg(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_gemmini_init(void);

enum ggml_gemmini_hw_profile {
    GGML_GEMMINI_HW_PROFILE_SINGLE_1X16 = 1,
    GGML_GEMMINI_HW_PROFILE_MULTI_4X8 = 2,
};

GGML_BACKEND_API enum ggml_gemmini_hw_profile ggml_gemmini_compiled_profile(void);
GGML_BACKEND_API const char * ggml_gemmini_compiled_profile_name(void);
GGML_BACKEND_API int32_t ggml_gemmini_default_active_mask(void);

GGML_BACKEND_API void ggml_gemmini_profiler_reset(void);
GGML_BACKEND_API void ggml_gemmini_profiler_reset_events(void);
GGML_BACKEND_API void ggml_gemmini_profiler_set_enabled(bool enabled);
GGML_BACKEND_API bool ggml_gemmini_profiler_is_enabled(void);

// active_mask and gemmini_id are logical member identifiers. In the single
// profile logical member 0 is translated to the generated custom3 opcode.
// Returns false without changing the request when the tuple is invalid.
GGML_BACKEND_API bool ggml_gemmini_runtime_configure_request(int32_t active_mask, int32_t gemmini_id, bool split_mode);
GGML_BACKEND_API void ggml_gemmini_runtime_clear_request(void);
// The eval callback uses this hint to keep a graph-proven RMS_NORM + MUL pair
// in one backend graph view. The hint is read-only and thread-safe.
GGML_BACKEND_API bool ggml_gemmini_should_defer_eval(const struct ggml_tensor * tensor);
GGML_BACKEND_API void ggml_gemmini_weight_cache_clear(void);
GGML_BACKEND_API void ggml_gemmini_weight_pack_clear(void);
GGML_BACKEND_API int ggml_gemmini_weight_pack_load(const char * path);
GGML_BACKEND_API int ggml_gemmini_prepack_weight(
    const struct ggml_tensor * tensor, int32_t weight_role);
GGML_BACKEND_API bool ggml_gemmini_hybrid_sparse_mode(void);
GGML_BACKEND_API uint64_t ggml_gemmini_weight_pack_model_fingerprint(void);
GGML_BACKEND_API int ggml_gemmini_weight_pack_finalize(void);
GGML_BACKEND_API int ggml_gemmini_weight_cache_mlock(void);

GGML_BACKEND_API void ggml_gemmini_profiler_start_run(const char * run_label);
GGML_BACKEND_API void ggml_gemmini_profiler_set_phase(const char * phase, int32_t token_index);
GGML_BACKEND_API void ggml_gemmini_profiler_graph_overhead_begin(void);
GGML_BACKEND_API void ggml_gemmini_profiler_graph_overhead_end(const char * phase, int32_t token_index, int64_t wall_us, uint64_t cycles);
GGML_BACKEND_API void ggml_gemmini_profiler_note_model_load(int64_t wall_us, uint64_t cycles);
GGML_BACKEND_API void ggml_gemmini_profiler_note_prompt(int32_t prompt_tokens);
GGML_BACKEND_API void ggml_gemmini_profiler_note_summary_metadata(const char * model_name, const char * input_prompt, const char * final_output);
GGML_BACKEND_API void ggml_gemmini_profiler_note_gemmini_config(int32_t active_mask);
GGML_BACKEND_API void ggml_gemmini_profiler_note_run_metadata(const char * model_name, const char * input_prompt, const char * final_output);
GGML_BACKEND_API void ggml_gemmini_profiler_note_sampling(int32_t token_index, int64_t wall_us, uint64_t cycles);
GGML_BACKEND_API void ggml_gemmini_profiler_note_run_total(
        int64_t wall_us,
        uint64_t cycles,
        int32_t generated_tokens,
        int64_t ttft_us,
        uint64_t ttft_cycles,
        double tpot_us,
        double tpot_cycles);
GGML_BACKEND_API void ggml_gemmini_profiler_note_token(int32_t token_index, int32_t token_id, const char * piece);

GGML_BACKEND_API void ggml_gemmini_profiler_eval_begin(const struct ggml_tensor * t, uint64_t cycles, int64_t time_us);
GGML_BACKEND_API void ggml_gemmini_profiler_eval_end(const struct ggml_tensor * t, uint64_t cycles, int64_t time_us, bool cpu_baseline);

// The first write after a profiler reset replaces the prior result session.
// Later writes append only completed, not-yet-exported runs for this session.
// Run labels must be unique within one reset generation.
GGML_BACKEND_API int ggml_gemmini_profiler_write_results(const char * results_dir);
GGML_BACKEND_API int ggml_gemmini_smoke_matmul(int32_t rows, int32_t cols_out, int32_t cols_in);

#ifdef __cplusplus
}
#endif
