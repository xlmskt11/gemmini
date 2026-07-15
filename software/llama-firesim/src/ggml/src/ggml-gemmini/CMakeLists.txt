if (NOT GEMMINI_SW_DIR)
    message(FATAL_ERROR "GEMMINI_SW_DIR must point at the FireSim target-design gemmini-rocc-tests directory")
endif()

ggml_add_backend_library(ggml-gemmini
    ggml-gemmini.cpp
)

target_include_directories(ggml-gemmini PRIVATE
    "${GEMMINI_SW_DIR}"
)

target_compile_options(ggml-gemmini PRIVATE
    -fpermissive
)
