#include "ggml-gemmini.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

struct output_file {
    const char * name;
    const char * header;
};

constexpr std::array<output_file, 7> outputs = {{
    {"run_summary.csv", "run_label,gemmini_mask,gemmini_count,"},
    {"token_trace.csv", "run_label,token_index,token_id,piece\n"},
    {"op_profile.csv", "run_label,phase,token_index,backend,"},
    {"stage_summary.csv", "run_label,phase,stage,total_us,"},
    {"matmul_summary.csv", "run_label,phase,shape,dim_i,"},
    {"backend_summary.csv", "run_label,category,name,total_us,"},
    {"summary.md", "# llama-firesim summary\n"},
}};

using snapshots = std::unordered_map<std::string, std::string>;

static bool fail(const std::string & message) {
    std::cerr << "GEMMINI-PROFILER-APPEND-TEST-FAIL," << message << "\n";
    return false;
}

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::in | std::ios::binary);
    std::ostringstream bytes;
    bytes << input.rdbuf();
    return bytes.str();
}

static snapshots capture(const std::filesystem::path & directory) {
    snapshots result;
    for (const output_file & output : outputs) {
        result.emplace(output.name, read_file(directory / output.name));
    }
    return result;
}

static size_t count_substring(
        const std::string & value,
        const std::string & needle) {
    size_t count = 0;
    size_t offset = 0;
    while ((offset = value.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

static bool headers_once(const snapshots & files) {
    for (const output_file & output : outputs) {
        const auto found = files.find(output.name);
        if (found == files.end() || count_substring(found->second, output.header) != 1) {
            return fail(std::string("header-count,file=") + output.name);
        }
    }
    return true;
}

static bool byte_equal(const snapshots & lhs, const snapshots & rhs) {
    for (const output_file & output : outputs) {
        if (lhs.at(output.name) != rhs.at(output.name)) {
            return fail(std::string("unexpected-byte-change,file=") + output.name);
        }
    }
    return true;
}

static bool prefix_preserved(const snapshots & prefix, const snapshots & value) {
    for (const output_file & output : outputs) {
        const std::string & old_bytes = prefix.at(output.name);
        const std::string & new_bytes = value.at(output.name);
        if (new_bytes.size() < old_bytes.size() ||
                new_bytes.compare(0, old_bytes.size(), old_bytes) != 0) {
            return fail(std::string("prefix-not-preserved,file=") + output.name);
        }
    }
    return true;
}

static bool labels_appended_once(
        const snapshots & one_run,
        const snapshots & two_runs,
        const std::string & first,
        const std::string & second) {
    for (const output_file & output : outputs) {
        const size_t first_before = count_substring(one_run.at(output.name), first);
        const size_t first_after = count_substring(two_runs.at(output.name), first);
        const size_t second_after = count_substring(two_runs.at(output.name), second);
        if (first_after != first_before || second_after != first_before) {
            return fail(std::string("run-label-count,file=") + output.name);
        }
    }
    return true;
}

static ggml_tensor make_profiled_tensor() {
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.op = GGML_OP_ADD;
    tensor.ne[0] = 1;
    tensor.ne[1] = 1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = sizeof(float);
    tensor.nb[2] = sizeof(float);
    tensor.nb[3] = sizeof(float);
    std::strncpy(tensor.name, "profiler_append_test", sizeof(tensor.name) - 1);
    return tensor;
}

static void begin_run(const std::string & label, int seed) {
    ggml_gemmini_profiler_start_run(label.c_str());
    ggml_gemmini_profiler_set_enabled(true);
    ggml_gemmini_profiler_note_summary_metadata(
        "profiler-append-model", "session prompt", "session output");
    ggml_gemmini_profiler_note_gemmini_config(0xf);
    ggml_gemmini_profiler_note_prompt(8 + seed);
    ggml_gemmini_profiler_note_run_metadata(
        "profiler-append-model", "run prompt", "run output");
    ggml_gemmini_profiler_set_phase("prefill", -1);

    static ggml_tensor tensor = make_profiled_tensor();
    const uint64_t start_cycles = static_cast<uint64_t>(1000 + seed * 100);
    const int64_t start_us = static_cast<int64_t>(2000 + seed * 100);
    ggml_gemmini_profiler_eval_begin(&tensor, start_cycles, start_us);
    ggml_gemmini_profiler_eval_end(
        &tensor, start_cycles + 17, start_us + 7, true);
    ggml_gemmini_profiler_note_token(0, 100 + seed, "piece");
}

static void complete_current_run(int seed) {
    ggml_gemmini_profiler_note_run_total(
        100 + seed,
        static_cast<uint64_t>(1000 + seed),
        1,
        60 + seed,
        static_cast<uint64_t>(600 + seed),
        0.0,
        0.0);
}

static bool write_results(const std::filesystem::path & directory) {
    return ggml_gemmini_profiler_write_results(directory.c_str()) == 0 ||
        fail("write-results");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " RESULTS_DIR\n";
        return 2;
    }

    const std::filesystem::path directory(argv[1]);
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        fail("create-directory");
        return 1;
    }

    const std::string run_a = "profiler_append_run_A";
    const std::string run_b = "profiler_append_run_B";
    const std::string run_c = "profiler_append_run_C";
    const std::string run_d = "profiler_append_run_D";

    ggml_gemmini_profiler_reset();
    begin_run(run_a, 1);
    complete_current_run(1);
    if (!write_results(directory)) {
        return 1;
    }
    const snapshots after_a = capture(directory);
    if (!headers_once(after_a)) {
        return 1;
    }

    begin_run(run_b, 2);
    complete_current_run(2);
    if (!write_results(directory)) {
        return 1;
    }
    const snapshots after_b = capture(directory);
    if (!prefix_preserved(after_a, after_b) ||
            !headers_once(after_b) ||
            !labels_appended_once(after_a, after_b, run_a, run_b)) {
        return 1;
    }

    if (!write_results(directory) ||
            !byte_equal(after_b, capture(directory))) {
        return 1;
    }

    ggml_gemmini_profiler_reset();
    begin_run(run_c, 3);
    complete_current_run(3);
    if (!write_results(directory)) {
        return 1;
    }
    const snapshots after_c = capture(directory);
    if (!headers_once(after_c)) {
        return 1;
    }
    for (const output_file & output : outputs) {
        const std::string & bytes = after_c.at(output.name);
        if (bytes.find(run_a) != std::string::npos ||
                bytes.find(run_b) != std::string::npos) {
            fail(std::string("reset-left-stale-run,file=") + output.name);
            return 1;
        }
    }

    begin_run(run_d, 4);
    if (!write_results(directory) ||
            !byte_equal(after_c, capture(directory))) {
        return 1;
    }

    complete_current_run(4);
    if (!write_results(directory)) {
        return 1;
    }
    const snapshots after_d = capture(directory);
    if (!prefix_preserved(after_c, after_d) ||
            !headers_once(after_d) ||
            !labels_appended_once(after_c, after_d, run_c, run_d)) {
        return 1;
    }

    // Reusing a label within one reset generation is ambiguous: the map key
    // and the append cursor can no longer identify two independent runs.
    // The public writer must reject it without touching any output file.
    ggml_gemmini_profiler_start_run(run_d.c_str());
    if (ggml_gemmini_profiler_write_results(directory.c_str()) != -2 ||
            !byte_equal(after_d, capture(directory))) {
        fail("duplicate-label-gate");
        return 1;
    }

    std::cout << "GEMMINI-PROFILER-APPEND-TEST-PASS\n";
    return 0;
}
