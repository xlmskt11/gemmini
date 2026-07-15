set(TARGET llama-firesim-cli)

if (NOT GEMMINI_SW_DIR)
    message(FATAL_ERROR "GEMMINI_SW_DIR is required for Gemmini layout and weight-pack geometry")
endif()

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

set(PAGE_PACKED_LAYOUT_TEST_TARGET llama-firesim-page-packed-layout-test)

add_executable(${PAGE_PACKED_LAYOUT_TEST_TARGET}
    llama-firesim-page-packed-layout-test.cpp
)

target_include_directories(${PAGE_PACKED_LAYOUT_TEST_TARGET} PRIVATE
    ${GEMMINI_SW_DIR}
)

target_compile_features(${PAGE_PACKED_LAYOUT_TEST_TARGET} PRIVATE cxx_std_17)
