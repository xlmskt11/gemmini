set(TARGET llama-firesim-cli)

if (NOT GEMMINI_SW_DIR)
    message(FATAL_ERROR "GEMMINI_SW_DIR is required for Gemmini layout and weight-pack geometry")
endif()

include("${CMAKE_SOURCE_DIR}/cmake/llama-firesim-hw-profile.cmake")

add_executable(${TARGET}
    llama-firesim-cli.cpp
)

target_link_libraries(${TARGET} PRIVATE
    llama
    ggml
    Threads::Threads
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
llama_firesim_apply_hw_profile(${TARGET})

set(PACK_TARGET llama-firesim-pack)

add_executable(${PACK_TARGET}
    llama-firesim-pack.cpp
)

target_link_libraries(${PACK_TARGET} PRIVATE
    llama
    ggml
    Threads::Threads
)

target_include_directories(${PACK_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${GEMMINI_SW_DIR}
)

target_compile_features(${PACK_TARGET} PRIVATE cxx_std_17)
llama_firesim_apply_hw_profile(${PACK_TARGET})

set(PAGE_PACKED_LAYOUT_TEST_TARGET llama-firesim-page-packed-layout-test)

add_executable(${PAGE_PACKED_LAYOUT_TEST_TARGET}
    llama-firesim-page-packed-layout-test.cpp
)

target_include_directories(${PAGE_PACKED_LAYOUT_TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${GEMMINI_SW_DIR}
)

target_compile_features(${PAGE_PACKED_LAYOUT_TEST_TARGET} PRIVATE cxx_std_17)
llama_firesim_apply_hw_profile(${PAGE_PACKED_LAYOUT_TEST_TARGET})

set(BF16_TEST_TARGET llama-firesim-bf16-test)

add_executable(${BF16_TEST_TARGET}
    llama-firesim-bf16-test.cpp
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-pack-reader.cpp
)

target_include_directories(${BF16_TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${GEMMINI_SW_DIR}
)

target_compile_features(${BF16_TEST_TARGET} PRIVATE cxx_std_17)
llama_firesim_apply_hw_profile(${BF16_TEST_TARGET})

set(MATMUL_OPT_TEST_TARGET llama-firesim-matmul-opt-test)

add_executable(${MATMUL_OPT_TEST_TARGET}
    llama-firesim-matmul-opt-test.cpp
)

target_include_directories(${MATMUL_OPT_TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini
)

target_compile_features(${MATMUL_OPT_TEST_TARGET} PRIVATE cxx_std_17)

set(FLASH_OPT_TEST_TARGET llama-firesim-flash-opt-test)

add_executable(${FLASH_OPT_TEST_TARGET}
    llama-firesim-flash-opt-test.cpp
)

target_include_directories(${FLASH_OPT_TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini
)

target_compile_features(${FLASH_OPT_TEST_TARGET} PRIVATE cxx_std_17)

set(VPU_ADMISSION_TEST_TARGET llama-firesim-vpu-admission-test)

add_executable(${VPU_ADMISSION_TEST_TARGET}
    llama-firesim-vpu-admission-test.cpp
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-vpu.cpp
)

target_include_directories(${VPU_ADMISSION_TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini
    ${GEMMINI_SW_DIR}
)

target_compile_definitions(${VPU_ADMISSION_TEST_TARGET} PRIVATE
    GGML_GEMMINI_VPU_ADMISSION_ONLY=1
)

target_compile_features(${VPU_ADMISSION_TEST_TARGET} PRIVATE cxx_std_17)
llama_firesim_apply_hw_profile(${VPU_ADMISSION_TEST_TARGET})

# This exercises the real profiler implementation, so it is meaningful only
# in the production configuration where the Gemmini backend target exists.
if (TARGET ggml-gemmini)
    set(PROFILER_APPEND_TEST_TARGET llama-firesim-profiler-append-test)

    add_executable(${PROFILER_APPEND_TEST_TARGET}
        llama-firesim-profiler-append-test.cpp
    )

    target_link_libraries(${PROFILER_APPEND_TEST_TARGET} PRIVATE
        ggml-gemmini
        ggml-cpu
        ggml-base
        Threads::Threads
    )

    target_include_directories(${PROFILER_APPEND_TEST_TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
    )

    target_compile_features(${PROFILER_APPEND_TEST_TARGET} PRIVATE cxx_std_17)
    # Keep the target-side regression binary independent of a dynamic loader.
    # The production CPU backend still requires Linux futex support.
    target_link_options(${PROFILER_APPEND_TEST_TARGET} PRIVATE -static)
    llama_firesim_apply_hw_profile(${PROFILER_APPEND_TEST_TARGET})
endif()
