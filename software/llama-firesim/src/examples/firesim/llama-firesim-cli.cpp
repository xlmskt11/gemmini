#include "ggml-gemmini.h"
#include "ggml.h"
#include "llama.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <clocale>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#endif

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

namespace fs = std::filesystem;

namespace {

static std::mutex & output_mutex() {
    static std::mutex mutex;
    return mutex;
}

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

static int32_t parse_int32_exact(const std::string & value) {
    size_t parsed = 0;
    const long long result = std::stoll(value, &parsed, 0);
    if (parsed != value.size() ||
            result < std::numeric_limits<int32_t>::min() ||
            result > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("invalid integer");
    }
    return static_cast<int32_t>(result);
}

static int32_t env_int32(const char * name, int32_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    try {
        return parse_int32_exact(value);
    } catch (...) {
        return fallback;
    }
}

static std::string trim_copy(const std::string & value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static bool parse_bool_value(const std::string & value) {
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON") {
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off" || value == "OFF") {
        return false;
    }
    throw std::runtime_error("invalid boolean");
}

static void configure_memory_locking() {
    if (!env_flag("LLAMA_FIRESIM_MLOCK", true)) {
        std::cout << "MLOCKALL-DISABLED\n";
        std::cout.flush();
        return;
    }

#if defined(__linux__)
    struct rlimit limit;
    if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0 && limit.rlim_cur != limit.rlim_max) {
        struct rlimit raised = limit;
        raised.rlim_cur = limit.rlim_max;
        if (setrlimit(RLIMIT_MEMLOCK, &raised) != 0) {
            std::perror("setrlimit(RLIMIT_MEMLOCK) failed");
        }
    }

    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::perror("mlockall failed");
        if (env_flag("LLAMA_FIRESIM_MLOCK_STRICT", false)) {
            throw std::runtime_error("mlockall failed");
        }
        std::cout << "MLOCKALL-WARN,continuing_without_locked_memory\n";
        std::cout.flush();
        return;
    }

    std::cout << "MLOCKALL-OK\n";
    std::cout.flush();
#else
    std::cout << "MLOCKALL-UNSUPPORTED\n";
    std::cout.flush();
#endif
}

static std::atomic<bool> abort_requested = false;

static void signal_handler(int signo) {
    if (signo == SIGINT || signo == SIGTERM) {
        abort_requested.store(true, std::memory_order_relaxed);
    }
}

static constexpr const char * DEFAULT_MODEL_PATH = "/root/models/model.gguf";

struct options {
    std::string model_path = DEFAULT_MODEL_PATH;
    std::string backend = "gemmini";
    std::string gemmini_mode = "multi";
    std::string results_dir = "/root/llama-results";
    std::string prompt;
    int32_t n_ctx = 2048;
    int32_t n_predict = 16;
    int32_t smoke_rows = 16;
    int32_t smoke_cols_out = 16;
    int32_t smoke_cols_in = 16;
    int32_t sweep_prompt_min = 64;
    int32_t sweep_prompt_max = 256;
    int32_t sweep_prompt_step = 16;
    std::vector<int32_t> sweep_masks = {0x1, 0x3, 0x7, 0xf};
    int page_packed_a = -1;
    int page_packed_b = -1;
    int page_packed_c = -1;
    int page_packed_d = -1;
    bool benchmark = false;
    bool arrival_benchmark = false;
    bool sweep = false;
    bool interactive = false;
    bool smoke_only = false;
};

struct gemmini_prepack_result {
    bool enabled = false;
    bool pack_loaded = false;
    int32_t tensors = 0;
    int32_t slices = 0;
    int32_t pack_entries = 0;
    int64_t wall_us = 0;
    uint64_t cycles = 0;
    std::string pack_path;
};

static void print_usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " [--backend gemmini|cpu] [--gemmini-mode multi|single] [--ctx-size N]\n"
        << "       [--n-predict N] [--results-dir DIR] [--prompt TEXT] [--benchmark] [--arrival-benchmark] [--sweep] [--interactive]\n"
        << "       [--gemmini-page-packed-a BOOL] [--gemmini-page-packed-b BOOL] [--gemmini-page-packed-c BOOL] [--gemmini-page-packed-d BOOL]\n"
        << "       [--sweep-prompt-min N] [--sweep-prompt-max N] [--sweep-prompt-step N] [--sweep-masks MASK[,MASK...]]\n"
        << "       " << argv0 << " --gemmini-smoke [M N K]\n";
}

static std::vector<int32_t> parse_mask_list(const std::string & value) {
    std::vector<int32_t> masks;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (item.empty()) {
            throw std::runtime_error("empty mask in --sweep-masks");
        }
        const int32_t mask = parse_int32_exact(item);
        if (mask < 0 || mask > 0xf) {
            throw std::runtime_error("--sweep-masks entries must be in [0, 0xf]");
        }
        masks.push_back(mask);
    }
    if (masks.empty()) {
        throw std::runtime_error("--sweep-masks must not be empty");
    }
    return masks;
}

static options parse_args(int argc, char ** argv) {
    options opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (arg == "--gemmini-mode" && i + 1 < argc) {
            opts.gemmini_mode = argv[++i];
        } else if (arg == "--gemmini-page-packed-a" && i + 1 < argc) {
            opts.page_packed_a = parse_bool_value(argv[++i]) ? 1 : 0;
        } else if (arg == "--gemmini-page-packed-b" && i + 1 < argc) {
            opts.page_packed_b = parse_bool_value(argv[++i]) ? 1 : 0;
        } else if (arg == "--gemmini-page-packed-c" && i + 1 < argc) {
            opts.page_packed_c = parse_bool_value(argv[++i]) ? 1 : 0;
        } else if (arg == "--gemmini-page-packed-d" && i + 1 < argc) {
            opts.page_packed_d = parse_bool_value(argv[++i]) ? 1 : 0;
        } else if (arg == "--ctx-size" && i + 1 < argc) {
            opts.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--n-predict" && i + 1 < argc) {
            opts.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--results-dir" && i + 1 < argc) {
            opts.results_dir = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            opts.prompt = argv[++i];
        } else if (arg == "--benchmark") {
            opts.benchmark = true;
        } else if (arg == "--arrival-benchmark") {
            opts.arrival_benchmark = true;
        } else if (arg == "--sweep") {
            opts.sweep = true;
        } else if (arg == "--sweep-prompt-min" && i + 1 < argc) {
            opts.sweep_prompt_min = std::stoi(argv[++i]);
        } else if (arg == "--sweep-prompt-max" && i + 1 < argc) {
            opts.sweep_prompt_max = std::stoi(argv[++i]);
        } else if (arg == "--sweep-prompt-step" && i + 1 < argc) {
            opts.sweep_prompt_step = std::stoi(argv[++i]);
        } else if (arg == "--sweep-masks" && i + 1 < argc) {
            opts.sweep_masks = parse_mask_list(argv[++i]);
        } else if (arg == "--interactive") {
            opts.interactive = true;
        } else if (arg == "--gemmini-smoke") {
            opts.smoke_only = true;
            if (i + 3 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                opts.smoke_rows = std::stoi(argv[++i]);
                opts.smoke_cols_out = std::stoi(argv[++i]);
                opts.smoke_cols_in = std::stoi(argv[++i]);
            }
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("invalid command line");
        }
    }

	    if (opts.backend != "gemmini" && opts.backend != "cpu") {
	        throw std::runtime_error("--backend must be gemmini or cpu");
	    }

        if (opts.gemmini_mode != "multi" && opts.gemmini_mode != "single") {
            throw std::runtime_error("--gemmini-mode must be multi or single");
        }

	    if (opts.n_ctx <= 0) {
	        throw std::runtime_error("--ctx-size must be > 0");
	    }

	    if (opts.n_predict < 0) {
	        throw std::runtime_error("--n-predict must be >= 0");
	    }

        if (opts.sweep_prompt_min <= 0 || opts.sweep_prompt_max < opts.sweep_prompt_min || opts.sweep_prompt_step <= 0) {
            throw std::runtime_error("invalid sweep prompt range");
        }

    return opts;
}

static void apply_page_packing_options(const options & opts) {
    const struct {
        const char * env_name;
        int value;
    } values[] = {
        {"GGML_GEMMINI_PAGE_PACKED_A", opts.page_packed_a},
        {"GGML_GEMMINI_PAGE_PACKED_B", opts.page_packed_b},
        {"GGML_GEMMINI_PAGE_PACKED_C", opts.page_packed_c},
        {"GGML_GEMMINI_PAGE_PACKED_D", opts.page_packed_d},
    };

    for (const auto & item : values) {
        if (item.value >= 0) {
            ::setenv(item.env_name, item.value != 0 ? "1" : "0", 1);
        }
    }
}

struct eval_state {
    bool cpu_baseline = false;
    const char * phase = "idle";
    int32_t token_index = -1;
    int64_t last_progress_us = 0;
    uint64_t op_count = 0;
    bool quiet = false;
};

static bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<eval_state *>(user_data);
    const bool profiler_enabled = ggml_gemmini_profiler_is_enabled();

    if (ask) {
        if (profiler_enabled) {
            ggml_gemmini_profiler_eval_begin(t, read_cycles_local(), ggml_time_us());
        }
        return true;
    }

    const int64_t now_us = ggml_time_us();
    if (profiler_enabled) {
        ggml_gemmini_profiler_eval_end(t, read_cycles_local(), now_us, state->cpu_baseline);
    }

    ++state->op_count;
    if (!state->quiet && now_us - state->last_progress_us >= 2000000) {
        std::cout << "PHASE-PROGRESS,phase=" << state->phase
                  << ",token_index=" << state->token_index
                  << ",ops=" << state->op_count << "\n";
        std::cout.flush();
        state->last_progress_us = now_us;
    }
    return true;
}

static bool decode_abort_callback(void *) {
    return abort_requested.load(std::memory_order_relaxed);
}

static std::string token_to_piece_string(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }
    return std::string(buf, buf + n);
}

class app {
public:
    explicit app(options opts) : opts_(std::move(opts)) {
        std::setlocale(LC_NUMERIC, "C");
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                std::fprintf(stderr, "%s", text);
            }
        }, nullptr);

        configure_memory_locking();

        std::cout << "GEMMINI-PAGE-PACKING,A="
                  << (env_flag("GGML_GEMMINI_PAGE_PACKED_A", false) ? 1 : 0)
                  << ",B=" << (env_flag("GGML_GEMMINI_PAGE_PACKED_B", false) ? 1 : 0)
                  << ",C=" << (env_flag("GGML_GEMMINI_PAGE_PACKED_C", false) ? 1 : 0)
                  << ",D=" << (env_flag("GGML_GEMMINI_PAGE_PACKED_D", false) ? 1 : 0)
                  << "\n";
        std::cout.flush();

        ggml_backend_load_all();

	        fs::create_directories(opts_.results_dir);

	        if (opts_.smoke_only) {
	            return;
	        }

	        load_model(opts_.model_path, false);
	    }

    ~app() {
        ggml_gemmini_weight_cache_clear();
        if (model_ != nullptr) {
            llama_model_free(model_);
        }
    }

    int run() {
        if (opts_.smoke_only) {
            ggml_gemmini_runtime_configure_request(
                env_int32("GGML_GEMMINI_ACTIVE_MASK", 0xf) & 0xf,
                -1,
                false,
                opts_.gemmini_mode == "single");
            const int status = ggml_gemmini_smoke_matmul(
                opts_.smoke_rows,
                opts_.smoke_cols_out,
                opts_.smoke_cols_in);
            ggml_gemmini_runtime_clear_request();
            return status;
        }

        if (!opts_.prompt.empty()) {
            run_command(opts_.prompt, opts_.n_predict, opts_.benchmark);
            return 0;
        }

        if (opts_.arrival_benchmark) {
            run_arrival_benchmark(1);
            return 0;
        }

        if (opts_.sweep) {
            run_prompt_token_sweep();
            return 0;
        }

        if (!opts_.interactive) {
            throw std::runtime_error("either --prompt or --interactive is required");
	        }

	        std::cout << "llama-firesim ready\n";
		        std::cout << "Commands: :help, :backend [gemmini|cpu], :gemmini-mode [multi|single], :ctx-size [N|max], :n-predict [N|max], :active-mask [MASK], :max-offloads [N], :trace [0|1], :decode N PROMPT, :prompt-tokens PROMPT_TOKENS [N] [TAG], :benchmark [N] PROMPT, :arrival-benchmark [N], :gemmini-smoke [M N K], :quit\n";
	        std::cout.flush();

        std::string line;
        while (true) {
            std::cout << "PROMPT> ";
            std::cout.flush();
            if (!std::getline(std::cin, line)) {
                return poweroff_and_exit();
            }

            if (line.empty()) {
                continue;
            }

            if (line == ":quit") {
                return poweroff_and_exit();
            }

	            if (line == ":help") {
	                std::cout << ":backend [gemmini|cpu]\n";
	                std::cout << ":gemmini-mode [multi|single]\n";
	                std::cout << ":ctx-size [N|max]\n";
	                std::cout << ":n-predict [N|max]\n";
	                std::cout << ":active-mask [MASK]\n";
	                std::cout << ":max-offloads [N]\n";
	                std::cout << ":trace [0|1|on|off]\n";
	                std::cout << ":decode N PROMPT\n";
	                std::cout << ":prompt-tokens PROMPT_TOKENS [N] [TAG]\n";
	                std::cout << ":benchmark [N] PROMPT\n";
	                std::cout << ":arrival-benchmark [N]\n";
                std::cout << ":gemmini-smoke [M N K]\n";
                std::cout << ":quit\n";
	                continue;
	            }

	            if (line == ":backend") {
	                std::cout << "BACKEND,current=" << opts_.backend << "\n";
	                std::cout.flush();
	                continue;
	            }

	            if (line.rfind(":backend ", 0) == 0) {
	                const std::string value = line.substr(std::strlen(":backend "));
	                if (value == "gemmini" || value == "cpu") {
                    opts_.backend = value;
                    std::cout << "backend set to " << opts_.backend << "\n";
                } else {
                    std::cout << "invalid backend: " << value << "\n";
                }
	                continue;
	            }

	            if (line == ":gemmini-mode" || line.rfind(":gemmini-mode ", 0) == 0) {
	                if (line == ":gemmini-mode") {
	                    print_gemmini_mode();
	                    continue;
	                }

	                const std::string value = trim_copy(line.substr(std::strlen(":gemmini-mode ")));
	                try {
	                    set_gemmini_mode(value);
	                } catch (const std::exception & e) {
	                    std::cout << "GEMMINI-MODE-ERROR,reason=" << e.what() << "\n";
	                    std::cout << "usage: :gemmini-mode [multi|single]\n";
	                    std::cout.flush();
	                }
	                continue;
	            }

	            if (line == ":ctx-size" || line.rfind(":ctx-size ", 0) == 0) {
	                if (line == ":ctx-size") {
	                    print_ctx_size();
	                    continue;
	                }

	                const std::string value = trim_copy(line.substr(std::strlen(":ctx-size ")));
	                try {
	                    set_ctx_size(value);
	                } catch (const std::exception & e) {
	                    std::cout << "CTX-SIZE-ERROR,reason=" << e.what() << "\n";
	                    std::cout << "usage: :ctx-size [N|max]\n";
	                    std::cout.flush();
	                }
	                continue;
	            }

	            if (line == ":n-predict" || line.rfind(":n-predict ", 0) == 0) {
	                if (line == ":n-predict") {
	                    print_n_predict();
	                    continue;
	                }

	                const std::string value = trim_copy(line.substr(std::strlen(":n-predict ")));
	                try {
	                    set_n_predict(value);
	                } catch (const std::exception & e) {
	                    std::cout << "N-PREDICT-ERROR,reason=" << e.what() << "\n";
	                    std::cout << "usage: :n-predict [N|max]\n";
	                }
	                std::cout.flush();
	                continue;
	            }

	            if (line == ":active-mask" || line.rfind(":active-mask ", 0) == 0) {
	                if (line == ":active-mask") {
	                    print_active_mask();
	                    continue;
	                }

	                const std::string value = trim_copy(line.substr(std::strlen(":active-mask ")));
	                try {
	                    set_active_mask(value);
	                } catch (const std::exception & e) {
	                    std::cout << "ACTIVE-MASK-ERROR,reason=" << e.what() << "\n";
	                    std::cout << "usage: :active-mask [MASK]\n";
	                    std::cout.flush();
	                }
	                continue;
	            }

	            if (line == ":max-offloads" || line.rfind(":max-offloads ", 0) == 0) {
	                if (line == ":max-offloads") {
	                    print_max_offloads();
	                    continue;
	                }

	                const std::string value = trim_copy(line.substr(std::strlen(":max-offloads ")));
	                try {
	                    set_max_offloads(value);
	                } catch (const std::exception & e) {
	                    std::cout << "MAX-OFFLOADS-ERROR,reason=" << e.what() << "\n";
	                    std::cout << "usage: :max-offloads [N]\n";
	                    std::cout.flush();
	                }
	                continue;
	            }

	            if (line == ":trace" || line.rfind(":trace ", 0) == 0) {
	                if (line == ":trace") {
	                    print_trace();
	                    continue;
	                }

	                const std::string value = trim_copy(line.substr(std::strlen(":trace ")));
	                try {
	                    set_trace(value);
	                } catch (const std::exception & e) {
	                    std::cout << "TRACE-ERROR,reason=" << e.what() << "\n";
	                    std::cout << "usage: :trace [0|1|on|off]\n";
	                    std::cout.flush();
	                }
	                continue;
	            }

	            if (line.rfind(":decode ", 0) == 0) {
                std::istringstream args(line.substr(std::strlen(":decode ")));
                int32_t decode_tokens = 0;
                std::string prompt;
                args >> decode_tokens;
                std::getline(args, prompt);
                const size_t prompt_start = prompt.find_first_not_of(' ');
                if (!args || decode_tokens < 0 || prompt_start == std::string::npos) {
                    std::cout << "usage: :decode N PROMPT\n";
                    std::cout.flush();
                    continue;
                }
                run_command(prompt.substr(prompt_start), decode_tokens, false);
                continue;
            }

            if (line == ":prompt-tokens" || line.rfind(":prompt-tokens ", 0) == 0) {
                if (line == ":prompt-tokens") {
                    std::cout << "usage: :prompt-tokens PROMPT_TOKENS [N] [TAG]\n";
                    std::cout.flush();
                    continue;
                }

                std::istringstream args(line.substr(std::strlen(":prompt-tokens ")));
                int32_t prompt_tokens = 0;
                int32_t decode_tokens = opts_.n_predict;
                std::string tag;
                args >> prompt_tokens;
                if (!args || prompt_tokens <= 0) {
                    std::cout << "usage: :prompt-tokens PROMPT_TOKENS [N] [TAG]\n";
                    std::cout.flush();
                    continue;
                }

                if (args >> decode_tokens) {
                    std::string extra;
                    if (decode_tokens < 0) {
                        std::cout << "usage: :prompt-tokens PROMPT_TOKENS [N] [TAG]\n";
                        std::cout.flush();
                        continue;
                    }
                    if (args >> tag && (args >> extra)) {
                        std::cout << "usage: :prompt-tokens PROMPT_TOKENS [N] [TAG]\n";
                        std::cout.flush();
                        continue;
                    }
                }

                try {
                    run_prompt_tokens_command(prompt_tokens, decode_tokens, tag);
                } catch (const std::exception & e) {
                    std::cout << "PROMPT-TOKENS-ERROR,reason=" << e.what() << "\n";
                    std::cout.flush();
                }
                continue;
            }

            if (line.rfind(":benchmark ", 0) == 0) {
                std::string rest = line.substr(std::strlen(":benchmark "));
                int32_t bench_predict = opts_.n_predict;
                size_t space = rest.find(' ');
                if (space != std::string::npos) {
                    bool numeric = !rest.substr(0, space).empty();
                    for (size_t i = 0; numeric && i < space; ++i) {
                        numeric = rest[i] >= '0' && rest[i] <= '9';
                    }
                    if (numeric) {
                        bench_predict = std::stoi(rest.substr(0, space));
                        rest = rest.substr(space + 1);
                    }
                }
                run_command(rest, bench_predict, true);
                continue;
            }

            if (line == ":arrival-benchmark" || line.rfind(":arrival-benchmark ", 0) == 0) {
                int32_t decode_tokens = 1;
                if (line.size() > std::strlen(":arrival-benchmark")) {
                    std::istringstream args(line.substr(std::strlen(":arrival-benchmark ")));
                    args >> decode_tokens;
                    if (!args || decode_tokens < 0) {
                        std::cout << "usage: :arrival-benchmark [decode_tokens]\n";
                        std::cout.flush();
                        continue;
                    }
                }
                run_arrival_benchmark(decode_tokens);
                continue;
            }

            if (line == ":gemmini-smoke" || line.rfind(":gemmini-smoke ", 0) == 0) {
                int32_t rows = 16;
                int32_t cols_out = 16;
                int32_t cols_in = 16;
                if (line.size() > std::strlen(":gemmini-smoke")) {
                    std::istringstream args(line.substr(std::strlen(":gemmini-smoke ")));
                    args >> rows >> cols_out >> cols_in;
                    if (!args) {
                        std::cout << "usage: :gemmini-smoke [M N K]\n";
                        std::cout.flush();
                        continue;
                    }
                }

                const int rc = ggml_gemmini_smoke_matmul(rows, cols_out, cols_in);
                std::cout << "GEMMINI-SMOKE-STATUS,rc=" << rc << "\n";
                std::cout.flush();
                continue;
            }

            run_command(line, opts_.n_predict, false);
        }
    }

private:
    struct request_gemmini_config {
        int32_t active_mask = -1;
        int32_t gemmini_id = -1;
        bool split_mode = false;
        bool single_mode = false;
    };

	    struct prompt_payload {
	        std::string text;
	        std::vector<llama_token> tokens;
	    };

    struct request_timing {
        std::string mode;
        int32_t request_id = 0;
        std::string run_label;
        int64_t arrival_us = 0;
        int64_t start_us = 0;
        int64_t finish_us = 0;
        uint64_t arrival_cycles = 0;
        uint64_t start_cycles = 0;
        uint64_t finish_cycles = 0;
        int32_t active_mask = 0;
        int32_t gemmini_id = -1;
        int32_t prompt_tokens = 0;
        int32_t generated_tokens = 0;
        bool aborted = false;
    };

	    struct run_result {
	        int32_t generated_tokens = 0;
	        bool aborted = false;
	        std::string final_output;
	    };

		    int32_t model_max_ctx_size() const {
		        if (model_ == nullptr) {
		            return 0;
		        }
		        return llama_model_n_ctx_train(model_);
		    }

    std::string current_model_name() const {
        const std::string filename = fs::path(opts_.model_path).filename().string();
        return filename.empty() ? opts_.model_path : filename;
    }

    std::string gemmini_weight_pack_path() const {
        const char * value = std::getenv("GGML_GEMMINI_WEIGHT_PACK");
        if (value != nullptr) {
            if (*value == '\0' || std::strcmp(value, "0") == 0) {
                return {};
            }
            return value;
        }

        return opts_.model_path + ".gemmini-pack";
    }

    static void append_transcript_entry(std::string & transcript, const std::string & entry) {
        if (!transcript.empty()) {
            transcript += "\n";
        }
        transcript += entry;
    }

    std::string prompt_run_label(const std::string & base, int32_t prompt_index) const {
        if (!opts_.interactive) {
            return base;
        }
        std::ostringstream out;
        out << base << "_prompt" << std::setw(4) << std::setfill('0') << prompt_index;
        return out.str();
    }

    static int32_t single_mask_from(int32_t active_mask, int32_t gemmini_id) {
        (void) active_mask;
        (void) gemmini_id;
        return 0x8;
    }

    int32_t current_run_gemmini_mask(bool cpu_baseline, const request_gemmini_config * request_config) const {
        if (cpu_baseline) {
            return 0;
        }
        const bool single_mode = request_config != nullptr ?
            request_config->single_mode : opts_.gemmini_mode == "single";
        if (request_config != nullptr) {
            return single_mode ?
                single_mask_from(request_config->active_mask, request_config->gemmini_id) :
                request_config->active_mask & 0xf;
        }
        const int32_t mask = env_int32("GGML_GEMMINI_ACTIVE_MASK", 0xf) & 0xf;
        return single_mode ? single_mask_from(mask, -1) : mask;
    }

    void publish_summary_metadata() const {
        const std::string model_name = current_model_name();
        ggml_gemmini_profiler_note_summary_metadata(
            model_name.c_str(),
            transcript_input_prompts_.c_str(),
            transcript_final_outputs_.c_str());
    }

		    int32_t max_n_predict_for_prompt_tokens(int32_t prompt_tokens) const {
	        if (prompt_tokens < 0 || opts_.n_ctx <= prompt_tokens) {
	            return 0;
	        }
	        return opts_.n_ctx - prompt_tokens;
	    }

	    int32_t max_n_predict_without_prompt() const {
	        return max_n_predict_for_prompt_tokens(1);
	    }

	    bool clamp_n_predict_to_ctx() {
	        const int32_t max_predict = max_n_predict_without_prompt();
	        if (opts_.n_predict > max_predict) {
	            opts_.n_predict = max_predict;
	            return true;
	        }
	        return false;
	    }

	    gemmini_prepack_result prepack_gemmini_weights() {
	        gemmini_prepack_result result;
	        result.enabled = env_flag("GGML_GEMMINI_PREPACK_WEIGHTS", true);
	        if (!result.enabled || model_ == nullptr) {
	            return result;
	        }

	        const uint64_t start_cycles = read_cycles_local();
	        const int64_t start_us = ggml_time_us();

            ggml_gemmini_weight_pack_clear();
            result.pack_path = gemmini_weight_pack_path();
            if (!result.pack_path.empty() && fs::exists(result.pack_path)) {
                const int loaded = ggml_gemmini_weight_pack_load(result.pack_path.c_str());
                if (loaded >= 0) {
                    result.pack_loaded = true;
                    result.pack_entries = loaded;
                } else {
                    std::cout << "GEMMINI-WEIGHT-PACK-ERROR,path=" << result.pack_path << "\n";
                    std::cout.flush();
                }
            }

	        const auto & tensors = llama_internal_get_tensor_map(model_);
	        for (const auto & item : tensors) {
	            const int slices = ggml_gemmini_prepack_weight(item.second);
	            if (slices > 0) {
	                ++result.tensors;
	                result.slices += slices;
	            }
	        }

	        result.wall_us = ggml_time_us() - start_us;
	        result.cycles = read_cycles_local() - start_cycles;
	        return result;
	    }

	    void load_model(const std::string & path, bool announce) {
	        uint64_t load_cycles_start = read_cycles_local();
	        const int64_t load_us_start = ggml_time_us();

	        llama_model_params model_params = llama_model_default_params();
	        llama_model * loaded = llama_model_load_from_file(path.c_str(), model_params);
	        if (loaded == nullptr) {
	            throw std::runtime_error("failed to load model: " + path);
	        }

	        ggml_gemmini_weight_cache_clear();
	        if (model_ != nullptr) {
	            llama_model_free(model_);
	        }
	        model_ = loaded;
	        opts_.model_path = path;

	        const gemmini_prepack_result prepack = prepack_gemmini_weights();
	        if (prepack.enabled) {
                if (prepack.pack_loaded) {
                    std::cout << "GEMMINI-WEIGHT-PACK-LOADED,path=" << prepack.pack_path
                              << ",entries=" << prepack.pack_entries << "\n";
                }
	            std::cout << "GEMMINI-PREPACK-DONE,tensors=" << prepack.tensors
	                      << ",slices=" << prepack.slices
	                      << ",wall_us=" << prepack.wall_us
	                      << ",cycles=" << prepack.cycles << "\n";
	            std::cout.flush();
	        }

	        const int64_t load_us_end = ggml_time_us();
	        const uint64_t load_cycles_end = read_cycles_local();

	        model_load_us_ = load_us_end - load_us_start;
	        model_load_cycles_ = load_cycles_end - load_cycles_start;
	        ggml_gemmini_profiler_reset();

	        bool ctx_clamped = false;
	        const int32_t max_ctx = model_max_ctx_size();
	        if (max_ctx > 0 && opts_.n_ctx > max_ctx) {
	            opts_.n_ctx = max_ctx;
	            ctx_clamped = true;
	        }
	        const bool predict_clamped = clamp_n_predict_to_ctx();

	        if (announce) {
	            std::cout << "MODEL-LOADED,path=" << opts_.model_path
	                      << ",load_us=" << model_load_us_
	                      << ",max_ctx_size=" << max_ctx
	                      << ",ctx_size=" << opts_.n_ctx
	                      << ",n_predict=" << opts_.n_predict << "\n";
	            if (ctx_clamped) {
	                std::cout << "CTX-SIZE-CLAMPED,max_ctx_size=" << max_ctx << "\n";
	            }
	            if (predict_clamped) {
	                std::cout << "N-PREDICT-CLAMPED,max=" << max_n_predict_without_prompt() << "\n";
	            }
	            std::cout.flush();
	        }
	    }

	    void print_ctx_size() const {
	        std::cout << "CTX-SIZE,current=" << opts_.n_ctx
	                  << ",max_ctx_size=" << model_max_ctx_size() << "\n";
	        std::cout.flush();
	    }

	    void print_n_predict() const {
	        std::cout << "N-PREDICT,current=" << opts_.n_predict
	                  << ",max_without_prompt=" << max_n_predict_without_prompt()
	                  << ",ctx_size=" << opts_.n_ctx << "\n";
	        std::cout.flush();
	    }

	    void print_active_mask() const {
	        const int32_t mask = env_int32("GGML_GEMMINI_ACTIVE_MASK", 0xf) & 0xf;
	        std::cout << "ACTIVE-MASK,current=0x" << std::hex << mask << std::dec << "\n";
	        std::cout.flush();
	    }

	    void print_gemmini_mode() const {
	        std::cout << "GEMMINI-MODE,current=" << opts_.gemmini_mode << "\n";
	        std::cout.flush();
	    }

	    void print_max_offloads() const {
	        std::cout << "MAX-OFFLOADS,current=" << env_int32("GGML_GEMMINI_MAX_OFFLOADS", -1) << "\n";
	        std::cout.flush();
	    }

	    void print_trace() const {
	        std::cout << "TRACE,current=" << (env_flag("GGML_GEMMINI_TRACE", false) ? 1 : 0) << "\n";
	        std::cout.flush();
	    }

	    void set_ctx_size(const std::string & value) {
	        const int32_t max_ctx = model_max_ctx_size();
	        int32_t n_ctx = 0;
	        if (value == "max") {
	            if (max_ctx <= 0) {
	                throw std::runtime_error("current model does not report max_ctx_size");
	            }
	            n_ctx = max_ctx;
	        } else {
	            n_ctx = parse_int32_exact(value);
	        }

	        if (n_ctx <= 0) {
	            throw std::runtime_error("ctx-size must be > 0");
	        }
	        if (max_ctx > 0 && n_ctx > max_ctx) {
	            throw std::runtime_error("ctx-size exceeds current model max_ctx_size");
	        }

	        opts_.n_ctx = n_ctx;
	        const bool predict_clamped = clamp_n_predict_to_ctx();
	        std::cout << "CTX-SIZE-SET,current=" << opts_.n_ctx
	                  << ",max_ctx_size=" << max_ctx << "\n";
	        if (predict_clamped) {
	            std::cout << "N-PREDICT-CLAMPED,max=" << max_n_predict_without_prompt() << "\n";
	        }
	        std::cout.flush();
	    }

	    void set_n_predict(const std::string & value) {
	        const int32_t max_predict = max_n_predict_without_prompt();
	        const int32_t n_predict = value == "max" ? max_predict : parse_int32_exact(value);
	        if (n_predict < 0) {
	            throw std::runtime_error("n-predict must be >= 0");
	        }
	        if (n_predict > max_predict) {
	            throw std::runtime_error("n-predict exceeds max_without_prompt");
	        }

	        opts_.n_predict = n_predict;
	        print_n_predict();
	    }

	    void set_active_mask(const std::string & value) {
	        const int32_t mask = parse_int32_exact(value);
	        if (mask < 0 || mask > 0xf) {
	            throw std::runtime_error("active-mask must be in [0, 0xf]");
	        }

	        std::ostringstream formatted;
	        formatted << "0x" << std::hex << mask;
	        const std::string env_value = formatted.str();
	        ::setenv("GGML_GEMMINI_ACTIVE_MASK", env_value.c_str(), 1);
	        print_active_mask();
	    }

	    void set_gemmini_mode(const std::string & value) {
	        if (value != "multi" && value != "single") {
	            throw std::runtime_error("gemmini-mode must be multi or single");
	        }
	        opts_.gemmini_mode = value;
	        print_gemmini_mode();
	    }

	    void set_max_offloads(const std::string & value) {
	        const int32_t max_offloads = parse_int32_exact(value);
	        if (max_offloads < -1) {
	            throw std::runtime_error("max-offloads must be >= -1");
	        }

	        const std::string env_value = std::to_string(max_offloads);
	        ::setenv("GGML_GEMMINI_MAX_OFFLOADS", env_value.c_str(), 1);
	        print_max_offloads();
	    }

	    void set_trace(const std::string & value) {
	        const bool enabled = parse_bool_value(value);
	        ::setenv("GGML_GEMMINI_TRACE", enabled ? "1" : "0", 1);
	        print_trace();
	    }

	    prompt_payload tokenize_prompt(const std::string & prompt) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_);
        const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }

        prompt_payload payload;
        payload.text = prompt;
        payload.tokens.resize(n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), payload.tokens.data(), payload.tokens.size(), true, true) < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }
        return payload;
    }

    prompt_payload make_synthetic_prompt(int32_t target_tokens) const {
        if (target_tokens <= 0) {
            throw std::runtime_error("target token count must be positive");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model_);
        const std::vector<std::string> pieces = {" a", " the", " hi", " x"};
        for (const std::string & piece : pieces) {
            std::string text;
            for (int i = 0; i < target_tokens * 4; ++i) {
                text += piece;
                const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
                if (n == target_tokens) {
                    return tokenize_prompt(text);
                }
                if (n > target_tokens) {
                    break;
                }
            }
        }

        prompt_payload payload;
        payload.text = "<synthetic-" + std::to_string(target_tokens) + "-token-prompt>";
        payload.tokens.reserve(target_tokens);
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos >= 0) {
            payload.tokens.push_back(bos);
        }

        const std::string seed = " a";
        const int n_seed = -llama_tokenize(vocab, seed.c_str(), seed.size(), nullptr, 0, false, true);
        if (n_seed <= 0) {
            throw std::runtime_error("failed to tokenize synthetic seed");
        }
        std::vector<llama_token> seed_tokens(n_seed);
        if (llama_tokenize(vocab, seed.c_str(), seed.size(), seed_tokens.data(), seed_tokens.size(), false, true) < 0) {
            throw std::runtime_error("failed to tokenize synthetic seed");
        }

        while (static_cast<int32_t>(payload.tokens.size()) < target_tokens) {
            for (llama_token token : seed_tokens) {
                if (static_cast<int32_t>(payload.tokens.size()) >= target_tokens) {
                    break;
                }
                payload.tokens.push_back(token);
            }
        }
        return payload;
    }

    run_result run_once(
            const prompt_payload & prompt,
            const std::string & run_label,
            bool cpu_baseline,
            int32_t n_predict,
            const request_gemmini_config * request_config = nullptr,
            bool quiet = false) {
        if (request_config == nullptr) {
            set_backend_mode(cpu_baseline);
            if (!cpu_baseline && opts_.gemmini_mode == "single") {
                ggml_gemmini_runtime_configure_request(
                    0x8,
                    3,
                    false,
                    true);
            } else {
                ggml_gemmini_runtime_clear_request();
            }
        } else {
            ::unsetenv("GGML_GEMMINI_DISABLE");
            ggml_gemmini_runtime_configure_request(
                request_config->active_mask,
                request_config->gemmini_id,
                request_config->split_mode,
                request_config->single_mode);
        }
        abort_requested.store(false, std::memory_order_relaxed);

        ggml_gemmini_profiler_start_run(run_label.c_str());
        ggml_gemmini_profiler_note_gemmini_config(current_run_gemmini_mask(cpu_baseline, request_config));
        ggml_gemmini_profiler_set_enabled(env_flag("GGML_GEMMINI_PROFILE", true));

	        const llama_vocab * vocab = llama_model_get_vocab(model_);
	        const int n_prompt = static_cast<int>(prompt.tokens.size());
	        if (n_prompt > opts_.n_ctx) {
	            throw std::runtime_error("prompt token count exceeds ctx-size");
	        }
	        const int32_t max_predict = max_n_predict_for_prompt_tokens(n_prompt);
	        if (n_predict > max_predict) {
	            std::ostringstream error;
	            error << "n-predict exceeds max for prompt: max=" << max_predict
	                  << ",prompt_tokens=" << n_prompt
	                  << ",ctx_size=" << opts_.n_ctx;
	            throw std::runtime_error(error.str());
	        }

	        ggml_gemmini_profiler_note_prompt(n_prompt);

        eval_state eval_state_{cpu_baseline};
        eval_state_.quiet = quiet;

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = opts_.n_ctx;
        ctx_params.n_batch = std::max<int32_t>(n_prompt, 1);
        ctx_params.n_ubatch = ctx_params.n_batch;
        ctx_params.n_threads = 1;
        ctx_params.n_threads_batch = 1;
        ctx_params.no_perf = false;
        ctx_params.cb_eval = eval_callback;
        ctx_params.cb_eval_user_data = &eval_state_;
        ctx_params.abort_callback = decode_abort_callback;
        ctx_params.abort_callback_data = nullptr;

        llama_context * ctx = llama_init_from_model(model_, ctx_params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create llama_context");
        }

        auto * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        uint64_t run_cycles_start = read_cycles_local();
        const int64_t run_us_start = ggml_time_us();

        if (!quiet) {
            std::cout << "RUN-START," << run_label << ",backend=" << (cpu_baseline ? "cpu" : "gemmini") << "\n";
            std::cout << "PROMPT[" << run_label << "]: " << prompt.text;
            std::cout.flush();
        }

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.tokens.data()), prompt.tokens.size());

        eval_state_.phase = "prefill";
        eval_state_.token_index = -1;
        eval_state_.last_progress_us = ggml_time_us();
        eval_state_.op_count = 0;
        if (!quiet) {
            std::cout << "\nPREFILL-START," << run_label << ",prompt_tokens=" << n_prompt << "\n";
            std::cout.flush();
        }
        ggml_gemmini_profiler_set_phase("prefill", -1);
        ggml_gemmini_profiler_graph_overhead_begin();
        const int64_t prefill_decode_us_start = ggml_time_us();
        const uint64_t prefill_decode_cycles_start = read_cycles_local();
        const int prefill_decode_rc = llama_decode(ctx, batch);
        const int64_t prefill_decode_us_end = ggml_time_us();
        const uint64_t prefill_decode_cycles_end = read_cycles_local();
        ggml_gemmini_profiler_graph_overhead_end(
            "prefill",
            -1,
            prefill_decode_us_end - prefill_decode_us_start,
            prefill_decode_cycles_end - prefill_decode_cycles_start);
        if (prefill_decode_rc != 0) {
            const bool aborted = abort_requested.load(std::memory_order_relaxed);
            llama_sampler_free(sampler);
            llama_free(ctx);
            if (aborted) {
                if (!quiet) {
                    std::cout << "RUN-ABORTED," << run_label << ",phase=prefill\n";
                    std::cout.flush();
                }
                ggml_gemmini_runtime_clear_request();
                return {0, true, ""};
            }
            throw std::runtime_error("prefill llama_decode failed");
        }
        if (!quiet) {
            std::cout << "PREFILL-END," << run_label << ",ops=" << eval_state_.op_count << "\n";
            std::cout.flush();
        }

        int32_t generated = 0;
        int64_t first_token_us = 0;
        int64_t last_token_us = 0;
        uint64_t first_token_cycles = 0;
        uint64_t last_token_cycles = 0;
        std::string generated_text;
        while (generated < n_predict) {
            const int64_t sample_us_start = ggml_time_us();
            const uint64_t sample_cycles_start = read_cycles_local();
            const llama_token token = llama_sampler_sample(sampler, ctx, -1);
            const int64_t sample_us_end = ggml_time_us();
            const uint64_t sample_cycles_end = read_cycles_local();

            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }

            const std::string piece = token_to_piece_string(vocab, token);
            generated_text += piece;
            if (generated == 0) {
                first_token_us = sample_us_end;
                first_token_cycles = sample_cycles_end;
            }
            last_token_us = sample_us_end;
            last_token_cycles = sample_cycles_end;
            if (!quiet) {
                std::cout << " -> " << piece;
                std::cout.flush();
            }

            ggml_gemmini_profiler_note_token(generated, token, piece.c_str());
            ggml_gemmini_profiler_note_sampling(generated, sample_us_end - sample_us_start, sample_cycles_end - sample_cycles_start);
            ++generated;

            if (generated >= n_predict) {
                break;
            }

            batch = llama_batch_get_one(const_cast<llama_token *>(&token), 1);
            eval_state_.phase = "decode";
            eval_state_.token_index = generated - 1;
            eval_state_.last_progress_us = ggml_time_us();
            eval_state_.op_count = 0;
            if (!quiet) {
                std::cout << "\nDECODE-START," << run_label << ",token_index=" << (generated - 1) << "\n";
                std::cout.flush();
            }
            ggml_gemmini_profiler_set_phase("decode", generated - 1);
            ggml_gemmini_profiler_graph_overhead_begin();
            const int64_t decode_us_start = ggml_time_us();
            const uint64_t decode_cycles_start = read_cycles_local();
            const int decode_rc = llama_decode(ctx, batch);
            const int64_t decode_us_end = ggml_time_us();
            const uint64_t decode_cycles_end = read_cycles_local();
            ggml_gemmini_profiler_graph_overhead_end(
                "decode",
                generated - 1,
                decode_us_end - decode_us_start,
                decode_cycles_end - decode_cycles_start);
            if (decode_rc != 0) {
                const bool aborted = abort_requested.load(std::memory_order_relaxed);
                llama_sampler_free(sampler);
                llama_free(ctx);
                if (aborted) {
                    if (!quiet) {
                        std::cout << "RUN-ABORTED," << run_label << ",phase=decode,token_index=" << (generated - 1) << "\n";
                        std::cout.flush();
                    }
                    ggml_gemmini_runtime_clear_request();
                    return {generated, true, ""};
                }
                throw std::runtime_error("decode llama_decode failed");
            }
            if (!quiet) {
                std::cout << "DECODE-END," << run_label << ",token_index=" << (generated - 1) << ",ops=" << eval_state_.op_count << "\n";
                std::cout.flush();
            }
        }

        if (!quiet) {
            std::cout << "\n";
            std::cout << "FINAL[" << run_label << "]: " << prompt.text << generated_text << "\n";
            std::cout << "RUN-END," << run_label << ",generated_tokens=" << generated << "\n";
            std::cout.flush();
        }

        const int64_t run_us_end = ggml_time_us();
        const uint64_t run_cycles_end = read_cycles_local();
        const int64_t ttft_us = generated > 0 ? first_token_us - run_us_start : 0;
        const uint64_t ttft_cycles = generated > 0 && first_token_cycles >= run_cycles_start ?
            first_token_cycles - run_cycles_start : 0;
        const double tpot_us = generated > 1 ?
            static_cast<double>(last_token_us - first_token_us) / static_cast<double>(generated - 1) : 0.0;
        const double tpot_cycles = generated > 1 && last_token_cycles >= first_token_cycles ?
            static_cast<double>(last_token_cycles - first_token_cycles) / static_cast<double>(generated - 1) : 0.0;
        const std::string final_output = prompt.text + generated_text;
        const std::string model_name = current_model_name();
        ggml_gemmini_profiler_note_run_metadata(model_name.c_str(), prompt.text.c_str(), final_output.c_str());
        ggml_gemmini_profiler_note_run_total(
            run_us_end - run_us_start,
            run_cycles_end - run_cycles_start,
            generated,
            ttft_us,
            ttft_cycles,
            tpot_us,
            tpot_cycles);

        llama_sampler_free(sampler);
        llama_free(ctx);
        ggml_gemmini_runtime_clear_request();

        return {generated, false, final_output};
    }

    run_result run_once(const std::string & prompt, const std::string & run_label, bool cpu_baseline, int32_t n_predict) {
        return run_once(tokenize_prompt(prompt), run_label, cpu_baseline, n_predict);
    }

    void run_command(const std::string & prompt, int32_t n_predict, bool benchmark) {
        const int32_t prompt_index = prompt_run_index_++;
        ggml_gemmini_profiler_note_model_load(model_load_us_, model_load_cycles_);

        if (benchmark) {
            const std::string hybrid_label = prompt_run_label("hybrid", prompt_index);
            const std::string cpu_label = prompt_run_label("cpu_baseline", prompt_index);
            const run_result hybrid = run_once(prompt, hybrid_label, false, n_predict);
            if (hybrid.aborted) {
                return;
            }
            const run_result cpu = run_once(prompt, cpu_label, true, n_predict);
            if (cpu.aborted) {
                return;
            }
            append_transcript_entry(transcript_input_prompts_, prompt);
            append_transcript_entry(
                transcript_final_outputs_,
                std::string("hybrid: ") + hybrid.final_output + "\n" +
                std::string("cpu_baseline: ") + cpu.final_output);
            publish_summary_metadata();
            if (ggml_gemmini_profiler_write_results(opts_.results_dir.c_str()) != 0) {
                throw std::runtime_error("failed to write benchmark results");
            }
            std::cout << "BENCHMARK-DONE\n";
            std::cout.flush();
            return;
        }

        const bool cpu_baseline = (opts_.backend == "cpu");
        const std::string run_label = prompt_run_label(cpu_baseline ? "cpu_baseline" : "hybrid", prompt_index);
        const run_result result = run_once(prompt, run_label, cpu_baseline, n_predict);
        if (result.aborted) {
            return;
        }
        append_transcript_entry(transcript_input_prompts_, prompt);
        append_transcript_entry(transcript_final_outputs_, result.final_output);
        publish_summary_metadata();
        if (ggml_gemmini_profiler_write_results(opts_.results_dir.c_str()) != 0) {
            throw std::runtime_error("failed to write results");
        }
        std::cout << "RESULTS-WRITTEN," << opts_.results_dir << "\n";
        std::cout.flush();
    }

    void run_prompt_tokens_command(int32_t prompt_tokens, int32_t n_predict, const std::string & tag) {
        if (opts_.backend != "gemmini" && opts_.backend != "cpu") {
            throw std::runtime_error("backend must be gemmini or cpu");
        }
        if (prompt_tokens <= 0) {
            throw std::runtime_error("prompt_tokens must be positive");
        }
        if (n_predict < 0) {
            throw std::runtime_error("n_predict must be >= 0");
        }
        if (prompt_tokens > opts_.n_ctx) {
            throw std::runtime_error("prompt token count exceeds ctx-size");
        }
        const int32_t max_predict = max_n_predict_for_prompt_tokens(prompt_tokens);
        if (n_predict > max_predict) {
            std::ostringstream error;
            error << "n-predict exceeds max for prompt: max=" << max_predict
                  << ",prompt_tokens=" << prompt_tokens
                  << ",ctx_size=" << opts_.n_ctx;
            throw std::runtime_error(error.str());
        }

        prompt_payload prompt = make_synthetic_prompt(prompt_tokens);
        if (static_cast<int32_t>(prompt.tokens.size()) != prompt_tokens) {
            throw std::runtime_error("failed to build synthetic prompt with requested token count");
        }

        const int32_t prompt_index = prompt_run_index_++;
        ggml_gemmini_profiler_note_model_load(model_load_us_, model_load_cycles_);

        const bool cpu_baseline = (opts_.backend == "cpu");
        const int32_t run_mask = current_run_gemmini_mask(cpu_baseline, nullptr);
        std::ostringstream label_base;
        label_base << (cpu_baseline ? "cpu" : opts_.gemmini_mode)
                   << "_mask0x" << std::hex << run_mask << std::dec
                   << "_ptok" << prompt_tokens
                   << "_dtok" << n_predict;
        const std::string run_label = prompt_run_label(label_base.str(), prompt_index);

        std::cout << "PROMPT-TOKENS-START,run_label=" << run_label
                  << ",tag=" << tag
                  << ",backend=" << opts_.backend
                  << ",gemmini_mode=" << opts_.gemmini_mode
                  << ",mask=0x" << std::hex << run_mask << std::dec
                  << ",prompt_tokens=" << prompt_tokens
                  << ",n_predict=" << n_predict << "\n";
        std::cout.flush();

        const run_result result = run_once(prompt, run_label, cpu_baseline, n_predict, nullptr, true);
        if (result.aborted) {
            std::cout << "PROMPT-TOKENS-ABORTED,run_label=" << run_label
                      << ",tag=" << tag
                      << ",prompt_tokens=" << prompt_tokens
                      << ",n_predict=" << n_predict
                      << ",generated_tokens=" << result.generated_tokens << "\n";
            std::cout.flush();
            return;
        }

        publish_summary_metadata();
        if (ggml_gemmini_profiler_write_results(opts_.results_dir.c_str()) != 0) {
            throw std::runtime_error("failed to write prompt-token results");
        }
        std::cout << "PROMPT-TOKENS-DONE,run_label=" << run_label
                  << ",tag=" << tag
                  << ",prompt_tokens=" << prompt_tokens
                  << ",n_predict=" << n_predict
                  << ",generated_tokens=" << result.generated_tokens
                  << ",results_dir=" << opts_.results_dir << "\n";
        std::cout.flush();
    }

    std::string sweep_run_label(int32_t active_mask, int32_t prompt_tokens) const {
        std::ostringstream out;
        out << "mask0x" << std::hex << active_mask << std::dec << "_ptok" << prompt_tokens;
        return out.str();
    }

    void run_prompt_token_sweep() {
        if (opts_.backend != "gemmini") {
            throw std::runtime_error("--sweep requires --backend gemmini");
        }

        ggml_gemmini_profiler_reset();
        ggml_gemmini_profiler_note_model_load(model_load_us_, model_load_cycles_);

        int32_t cases = 0;
        for (int32_t active_mask : opts_.sweep_masks) {
            request_gemmini_config config;
            config.active_mask = active_mask;
            config.gemmini_id = -1;
            config.split_mode = false;

            for (int32_t prompt_tokens = opts_.sweep_prompt_min;
                    prompt_tokens <= opts_.sweep_prompt_max;
                    prompt_tokens += opts_.sweep_prompt_step) {
                prompt_payload prompt = make_synthetic_prompt(prompt_tokens);
                if (static_cast<int32_t>(prompt.tokens.size()) != prompt_tokens) {
                    throw std::runtime_error("failed to build synthetic prompt with requested token count");
                }

                const std::string run_label = sweep_run_label(active_mask, prompt_tokens);
                std::cout << "SWEEP-CASE-START,run_label=" << run_label
                          << ",active_mask=0x" << std::hex << active_mask << std::dec
                          << ",prompt_tokens=" << prompt_tokens
                          << ",n_predict=" << opts_.n_predict << "\n";
                std::cout.flush();

                const run_result result = run_once(prompt, run_label, false, opts_.n_predict, &config, true);
                if (result.aborted) {
                    std::cout << "SWEEP-ABORTED,run_label=" << run_label
                              << ",generated_tokens=" << result.generated_tokens << "\n";
                    std::cout.flush();
                    return;
                }

                ++cases;
                std::cout << "SWEEP-CASE-DONE,run_label=" << run_label
                          << ",generated_tokens=" << result.generated_tokens << "\n";
                std::cout.flush();
            }
        }

        publish_summary_metadata();
        if (ggml_gemmini_profiler_write_results(opts_.results_dir.c_str()) != 0) {
            throw std::runtime_error("failed to write sweep results");
        }
        std::cout << "SWEEP-DONE,cases=" << cases
                  << ",results_dir=" << opts_.results_dir << "\n";
        std::cout.flush();
    }

    static int64_t estimated_cycles_for_us(int64_t delta_us) {
        const char * value = std::getenv("LLAMA_FIRESIM_CYCLES_PER_US");
        const int64_t cycles_per_us = value == nullptr ? 1000 : std::strtoll(value, nullptr, 0);
        return delta_us * cycles_per_us;
    }

    static void wait_until_us(int64_t target_us) {
        while (true) {
            const int64_t now = ggml_time_us();
            if (now >= target_us) {
                return;
            }
            const int64_t remaining = target_us - now;
            const int64_t sleep_us = std::min<int64_t>(remaining, 10000);
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }

    request_timing run_arrival_request(
            const prompt_payload & prompt,
            const std::string & mode,
            int32_t request_id,
            int64_t base_us,
            uint64_t base_cycles,
            int64_t interval_us,
            int32_t n_predict,
            const request_gemmini_config & config) {
        request_timing timing;
        timing.mode = mode;
        timing.request_id = request_id;
        timing.run_label = mode + "_req" + std::to_string(request_id);
        timing.arrival_us = base_us + request_id * interval_us;
        timing.arrival_cycles = base_cycles + static_cast<uint64_t>(estimated_cycles_for_us(timing.arrival_us - base_us));
        timing.active_mask = config.active_mask;
        timing.gemmini_id = config.gemmini_id;
        timing.prompt_tokens = static_cast<int32_t>(prompt.tokens.size());

        wait_until_us(timing.arrival_us);
        timing.start_us = ggml_time_us();
        timing.start_cycles = read_cycles_local();

        {
            std::lock_guard<std::mutex> lock(output_mutex());
            std::cout << "ARRIVAL-REQUEST-START,mode=" << mode
                      << ",request_id=" << request_id
                      << ",run_label=" << timing.run_label
                      << ",mask=0x" << std::hex << config.active_mask << std::dec
                      << ",gemmini_id=" << config.gemmini_id
                      << ",prompt_tokens=" << timing.prompt_tokens << "\n";
            std::cout.flush();
        }

        const run_result result = run_once(prompt, timing.run_label, false, n_predict, &config, true);

        timing.finish_us = ggml_time_us();
        timing.finish_cycles = read_cycles_local();
        timing.generated_tokens = result.generated_tokens;
        timing.aborted = result.aborted;

        {
            std::lock_guard<std::mutex> lock(output_mutex());
            std::cout << "ARRIVAL-REQUEST-DONE,mode=" << mode
                      << ",request_id=" << request_id
                      << ",run_label=" << timing.run_label
                      << ",e2e_us=" << (timing.finish_us - timing.arrival_us)
                      << ",queue_us=" << (timing.start_us - timing.arrival_us)
                      << ",service_us=" << (timing.finish_us - timing.start_us)
                      << ",generated_tokens=" << timing.generated_tokens
                      << ",aborted=" << (timing.aborted ? 1 : 0) << "\n";
            std::cout.flush();
        }

        return timing;
    }

    void write_arrival_outputs(const std::vector<request_timing> & timings, int64_t interval_us) {
        fs::create_directories(opts_.results_dir);

        {
            std::ofstream out(opts_.results_dir + "/arrival_request_summary.csv");
            if (!out) {
                throw std::runtime_error("failed to write arrival_request_summary.csv");
            }
            out << "mode,request_id,run_label,arrival_us,start_us,finish_us,e2e_us,queue_us,service_us,arrival_cycles,start_cycles,finish_cycles,e2e_cycles,active_mask,gemmini_id,prompt_tokens,generated_tokens\n";
            for (const request_timing & t : timings) {
                out << t.mode << ","
                    << t.request_id << ","
                    << t.run_label << ","
                    << t.arrival_us << ","
                    << t.start_us << ","
                    << t.finish_us << ","
                    << (t.finish_us - t.arrival_us) << ","
                    << (t.start_us - t.arrival_us) << ","
                    << (t.finish_us - t.start_us) << ","
                    << t.arrival_cycles << ","
                    << t.start_cycles << ","
                    << t.finish_cycles << ","
                    << (t.finish_cycles > t.arrival_cycles ? t.finish_cycles - t.arrival_cycles : 0) << ","
                    << "0x" << std::hex << t.active_mask << std::dec << ","
                    << t.gemmini_id << ","
                    << t.prompt_tokens << ","
                    << t.generated_tokens << "\n";
            }
        }

        struct mode_totals {
            int count = 0;
            int64_t first_arrival = 0;
            int64_t last_finish = 0;
            int64_t sum_e2e = 0;
            int64_t max_e2e = 0;
            int64_t sum_queue = 0;
            int64_t sum_service = 0;
        };

        std::map<std::string, mode_totals> totals;
        for (const request_timing & t : timings) {
            mode_totals & m = totals[t.mode];
            if (m.count == 0) {
                m.first_arrival = t.arrival_us;
                m.last_finish = t.finish_us;
            }
            ++m.count;
            m.first_arrival = std::min(m.first_arrival, t.arrival_us);
            m.last_finish = std::max(m.last_finish, t.finish_us);
            const int64_t e2e = t.finish_us - t.arrival_us;
            m.sum_e2e += e2e;
            m.max_e2e = std::max(m.max_e2e, e2e);
            m.sum_queue += t.start_us - t.arrival_us;
            m.sum_service += t.finish_us - t.start_us;
        }

        {
            std::ofstream out(opts_.results_dir + "/arrival_mode_summary.csv");
            if (!out) {
                throw std::runtime_error("failed to write arrival_mode_summary.csv");
            }
            out << "mode,request_count,interval_us,makespan_us,throughput_req_per_s,avg_e2e_us,max_e2e_us,avg_queue_us,avg_service_us\n";
            out << std::fixed << std::setprecision(6);
            for (const auto & kv : totals) {
                const mode_totals & m = kv.second;
                const int64_t makespan = std::max<int64_t>(1, m.last_finish - m.first_arrival);
                out << kv.first << ","
                    << m.count << ","
                    << interval_us << ","
                    << makespan << ","
                    << (static_cast<double>(m.count) * 1000000.0 / static_cast<double>(makespan)) << ","
                    << (m.sum_e2e / std::max(1, m.count)) << ","
                    << m.max_e2e << ","
                    << (m.sum_queue / std::max(1, m.count)) << ","
                    << (m.sum_service / std::max(1, m.count)) << "\n";
            }
        }

        write_arrival_stage_summary(timings);
        append_arrival_summary(timings, totals);
    }

    void write_arrival_stage_summary(const std::vector<request_timing> & timings) {
        std::map<std::string, request_timing> by_label;
        for (const request_timing & t : timings) {
            by_label[t.run_label] = t;
        }

        std::ifstream in(opts_.results_dir + "/stage_summary.csv");
        std::ofstream out(opts_.results_dir + "/arrival_stage_summary.csv");
        if (!out) {
            throw std::runtime_error("failed to write arrival_stage_summary.csv");
        }
        out << "mode,request_id,run_label,stage,total_us,total_cycles,pct_of_runtime_breakdown\n";
        if (!in) {
            return;
        }

        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string field;
            while (std::getline(ss, field, ',')) {
                fields.push_back(field);
            }
            if (fields.size() < 5) {
                continue;
            }
            auto it = by_label.find(fields[0]);
            if (it == by_label.end()) {
                continue;
            }
            out << it->second.mode << ","
                << it->second.request_id << ","
                << fields[0] << ","
                << fields[1] << ","
                << fields[2] << ","
                << fields[3] << ","
                << fields[4] << "\n";
        }
    }

    template <typename Totals>
    void append_arrival_summary(const std::vector<request_timing> & timings, const Totals & totals) {
        std::ofstream out(opts_.results_dir + "/summary.md", std::ios::app);
        if (!out) {
            return;
        }

        out << "## Arrival Benchmark\n\n";
        out << "| mode | requests | throughput req/s | avg e2e us | max e2e us | avg queue us | avg service us |\n";
        out << "| --- | --- | --- | --- | --- | --- | --- |\n";
        out << std::fixed << std::setprecision(6);
        for (const auto & kv : totals) {
            const auto & m = kv.second;
            const int64_t makespan = std::max<int64_t>(1, m.last_finish - m.first_arrival);
            out << "| " << kv.first << " | "
                << m.count << " | "
                << (static_cast<double>(m.count) * 1000000.0 / static_cast<double>(makespan)) << " | "
                << (m.sum_e2e / std::max(1, m.count)) << " | "
                << m.max_e2e << " | "
                << (m.sum_queue / std::max(1, m.count)) << " | "
                << (m.sum_service / std::max(1, m.count)) << " |\n";
        }
        out << "\n";

        out << "| mode | request | mask | gemmini | prompt tokens | generated tokens | e2e us | queue us | service us |\n";
        out << "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (const request_timing & t : timings) {
            out << "| " << t.mode << " | "
                << t.request_id << " | 0x"
                << std::hex << t.active_mask << std::dec << " | "
                << t.gemmini_id << " | "
                << t.prompt_tokens << " | "
                << t.generated_tokens << " | "
                << (t.finish_us - t.arrival_us) << " | "
                << (t.start_us - t.arrival_us) << " | "
                << (t.finish_us - t.start_us) << " |\n";
        }
        out << "\n";
    }

    void run_arrival_benchmark(int32_t n_predict) {
        constexpr int32_t request_count = 4;
        constexpr int32_t prompt_tokens = 256;
        constexpr int64_t interval_us = 1000000;

        std::cout << "ARRIVAL-BENCHMARK-START,requests=" << request_count
                  << ",interval_us=" << interval_us
                  << ",prompt_tokens=" << prompt_tokens
                  << ",decode_tokens=" << n_predict << "\n";
        std::cout.flush();

        prompt_payload prompt = make_synthetic_prompt(prompt_tokens);
        if (static_cast<int32_t>(prompt.tokens.size()) != prompt_tokens) {
            throw std::runtime_error("failed to build 256-token prompt");
        }

        ggml_gemmini_profiler_reset();
        ggml_gemmini_profiler_note_model_load(model_load_us_, model_load_cycles_);
        request_gemmini_config warmup_cfg{0xf, -1, false, false};
        const run_result warmup = run_once(prompt, "warmup", false, n_predict, &warmup_cfg, true);
        if (warmup.aborted) {
            std::cout << "ARRIVAL-BENCHMARK-ABORTED,phase=warmup\n";
            std::cout.flush();
            return;
        }

        ggml_gemmini_profiler_reset_events();
        ggml_gemmini_profiler_note_model_load(model_load_us_, model_load_cycles_);

        std::vector<request_timing> timings;
        timings.reserve(request_count * 2);

        const int64_t group_base_us = ggml_time_us();
        const uint64_t group_base_cycles = read_cycles_local();
        for (int32_t i = 0; i < request_count; ++i) {
            request_gemmini_config cfg{0xf, -1, false, false};
            timings.push_back(run_arrival_request(prompt, "group_seq", i, group_base_us, group_base_cycles, interval_us, n_predict, cfg));
        }

        const int64_t split_base_us = ggml_time_us();
        const uint64_t split_base_cycles = read_cycles_local();
        std::vector<request_timing> split_timings(request_count);
        std::vector<std::thread> workers;
        std::vector<std::string> errors(request_count);
        workers.reserve(request_count);
        for (int32_t i = 0; i < request_count; ++i) {
            workers.emplace_back([&, i]() {
                try {
                    request_gemmini_config cfg{1 << i, i, true, false};
                    split_timings[i] = run_arrival_request(prompt, "split_1gem", i, split_base_us, split_base_cycles, interval_us, n_predict, cfg);
                } catch (const std::exception & e) {
                    errors[i] = e.what();
                }
            });
        }
        for (std::thread & worker : workers) {
            worker.join();
        }
        for (const std::string & error : errors) {
            if (!error.empty()) {
                throw std::runtime_error("split worker failed: " + error);
            }
        }
        timings.insert(timings.end(), split_timings.begin(), split_timings.end());

        publish_summary_metadata();
        if (ggml_gemmini_profiler_write_results(opts_.results_dir.c_str()) != 0) {
            throw std::runtime_error("failed to write profiler results");
        }
        write_arrival_outputs(timings, interval_us);

        std::cout << "ARRIVAL-BENCHMARK-DONE\n";
        std::cout.flush();
    }

    void set_backend_mode(bool cpu_baseline) {
        if (cpu_baseline) {
            ::setenv("GGML_GEMMINI_DISABLE", "1", 1);
        } else {
            ::unsetenv("GGML_GEMMINI_DISABLE");
            if (std::getenv("GGML_GEMMINI_ACTIVE_MASK") == nullptr) {
                ::setenv("GGML_GEMMINI_ACTIVE_MASK", "0xf", 0);
            }
            if (std::getenv("GGML_GEMMINI_PROFILE") == nullptr) {
                ::setenv("GGML_GEMMINI_PROFILE", "1", 0);
            }
        }
    }

    int poweroff_and_exit() {
        std::cout << "powering off\n";
        std::cout.flush();
        std::system("/sbin/poweroff -f");
        return 0;
    }

    options opts_;
    llama_model * model_ = nullptr;
    int64_t model_load_us_ = 0;
    uint64_t model_load_cycles_ = 0;
    int32_t prompt_run_index_ = 0;
    std::string transcript_input_prompts_;
    std::string transcript_final_outputs_;
};

} // namespace

int main(int argc, char ** argv) {
    try {
        options opts = parse_args(argc, argv);
        apply_page_packing_options(opts);
        app app_(std::move(opts));
        return app_.run();
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
