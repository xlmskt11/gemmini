include_guard(GLOBAL)

set(LLAMA_FIRESIM_HW_PROFILE "" CACHE STRING
    "llama-firesim hardware profile (single or multi)")
set_property(CACHE LLAMA_FIRESIM_HW_PROFILE PROPERTY STRINGS single multi)

if (LLAMA_FIRESIM_HW_PROFILE STREQUAL "single")
    set(LLAMA_FIRESIM_HW_PROFILE_COMPILE_DEFINITION
        LLAMA_FIRESIM_HW_PROFILE_SINGLE=1 CACHE INTERNAL
        "llama-firesim selected hardware profile compile definition" FORCE)
elseif (LLAMA_FIRESIM_HW_PROFILE STREQUAL "multi")
    set(LLAMA_FIRESIM_HW_PROFILE_COMPILE_DEFINITION
        LLAMA_FIRESIM_HW_PROFILE_MULTI=1 CACHE INTERNAL
        "llama-firesim selected hardware profile compile definition" FORCE)
else()
    message(FATAL_ERROR
        "LLAMA_FIRESIM_HW_PROFILE must be selected explicitly as single or multi")
endif()

if (NOT GEMMINI_SW_DIR)
    message(FATAL_ERROR
        "GEMMINI_SW_DIR must point at the elaborated target-design gemmini-rocc-tests directory")
endif()

set(LLAMA_FIRESIM_HW_PROFILE_HEADER
    "${CMAKE_SOURCE_DIR}/ggml/include/ggml-gemmini-profile.h"
    CACHE INTERNAL "Selected llama-firesim Gemmini/VPU ABI profile header")

if (NOT EXISTS "${LLAMA_FIRESIM_HW_PROFILE_HEADER}")
    message(FATAL_ERROR
        "Missing llama-firesim hardware profile header: ${LLAMA_FIRESIM_HW_PROFILE_HEADER}")
endif()

set(LLAMA_FIRESIM_REQUIRED_GENERATED_HEADERS
    include/gemmini_params.h
    include/vpu_params.h
    include/gemmini_all.h
    include/gemmini_counter.h
    include/gemmini_page_packed.h
    include/gemmini_tiling.h
    include/vpu.h
    include/vpu_kernels.h
    include/vpu_flashattention_kernel.h
    rocc-software/src/xcustom.h)

foreach(relative_header IN LISTS LLAMA_FIRESIM_REQUIRED_GENERATED_HEADERS)
    if (NOT EXISTS "${GEMMINI_SW_DIR}/${relative_header}")
        message(FATAL_ERROR
            "Missing target-design generated software header: ${GEMMINI_SW_DIR}/${relative_header}")
    endif()
endforeach()

function(llama_firesim_apply_hw_profile target_name)
    if (NOT TARGET "${target_name}")
        message(FATAL_ERROR
            "Cannot apply llama-firesim hardware profile to missing target ${target_name}")
    endif()

    target_sources("${target_name}" PRIVATE
        "${LLAMA_FIRESIM_HW_PROFILE_HEADER}")
    # ggml-gemmini-profile.h directly includes the elaborated target-design
    # parameter headers. Apply this to every consumer, including CLI and host
    # admission tests, before force-including the profile header.
    target_include_directories("${target_name}" BEFORE PRIVATE
        "${GEMMINI_SW_DIR}"
        "${CMAKE_SOURCE_DIR}/ggml/include")
    target_compile_definitions("${target_name}" PRIVATE
        "${LLAMA_FIRESIM_HW_PROFILE_COMPILE_DEFINITION}")
    target_compile_options("${target_name}" PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:-include${LLAMA_FIRESIM_HW_PROFILE_HEADER}>"
        "$<$<COMPILE_LANGUAGE:CXX>:-include${LLAMA_FIRESIM_HW_PROFILE_HEADER}>")
endfunction()

message(STATUS
    "llama-firesim profile: ${LLAMA_FIRESIM_HW_PROFILE}; hardware ABI source: ${GEMMINI_SW_DIR}/include/{gemmini_params.h,vpu_params.h}")
