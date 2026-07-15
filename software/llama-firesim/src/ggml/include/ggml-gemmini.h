#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_gemmini_reg(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_gemmini_init(void);

GGML_BACKEND_API void ggml_gemmini_profiler_reset(void);
GGML_BACKEND_API void ggml_gemmini_profiler_reset_events(void);
GGML_BACKEND_API void ggml_gemmini_profiler_set_enabled(bool enabled);
GGML_BACKEND_API bool ggml_gemmini_profiler_is_enabled(void);

GGML_BACKEND_API void ggml_gemmini_runtime_configure_request(int32_t active_mask, int32_t gemmini_id, bool split_mode, bool single_mode);
GGML_BACKEND_API void ggml_gemmini_runtime_clear_request(void);
GGML_BACKEND_API void ggml_gemmini_weight_cache_clear(void);
GGML_BACKEND_API void ggml_gemmini_weight_pack_clear(void);
GGML_BACKEND_API int ggml_gemmini_weight_pack_load(const char * path);
GGML_BACKEND_API int ggml_gemmini_prepack_weight(const struct ggml_tensor * tensor);

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

GGML_BACKEND_API int ggml_gemmini_profiler_write_results(const char * results_dir);
GGML_BACKEND_API int ggml_gemmini_smoke_matmul(int32_t rows, int32_t cols_out, int32_t cols_in);

#ifdef __cplusplus
}
#endif
