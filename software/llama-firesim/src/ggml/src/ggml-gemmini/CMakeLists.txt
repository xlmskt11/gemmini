if (NOT GEMMINI_SW_DIR)
    message(FATAL_ERROR "GEMMINI_SW_DIR must point at the FireSim target-design gemmini-rocc-tests directory")
endif()

include("${CMAKE_SOURCE_DIR}/cmake/llama-firesim-hw-profile.cmake")

ggml_add_backend_library(ggml-gemmini
    ggml-gemmini.cpp
    ggml-gemmini-pack-reader.cpp
    ggml-gemmini-vpu.cpp
)

llama_firesim_apply_hw_profile(ggml-gemmini)

target_include_directories(ggml-gemmini PRIVATE
    "${GEMMINI_SW_DIR}"
)

target_compile_options(ggml-gemmini PRIVATE
    -fpermissive
)
