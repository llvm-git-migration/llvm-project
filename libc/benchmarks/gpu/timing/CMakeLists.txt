message(STATUS "In GPU subdir")
# if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
#     message(STATUS "Target arch is NOT gpu")
#     return()
# endif()

foreach(target nvptx)
    add_subdirectory(${target})
    list(APPEND target_gpu_timing libc.benchmarks.gpu.timing.${target}.${target}_timing)
endforeach()

message(STATUS "GPU TIMING: ${target_gpu_timing}")

add_header_library(
    timing
    HDRS
        timing.h
    DEPENDS
        ${target_gpu_timing}
)
