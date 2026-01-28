cmake_minimum_required(VERSION 4.0)
project(dbscan CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------
# CPU DBSCAN
# ------------------------------
add_library(cpu_dbscan STATIC
        src/cpu_dbscan.cpp
        src/common.h
        src/helper.h
)

target_include_directories(cpu_dbscan PUBLIC src)

# ------------------------------
# GPU DBSCAN
# ------------------------------
add_library(gpu_dbscan STATIC
        src/gpu_dbscan.cu
        src/common.h
        src/helper.h
        src/cuda_helper.h
)

set_target_properties(gpu_dbscan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gpu_dbscan PUBLIC src)

# ------------------------------
# IO
# ------------------------------
add_library(io STATIC
        src/io.cpp
        src/common.h
        src/helper.h
)

target_include_directories(io PUBLIC src)

# ------------------------------
# MAIN
# ------------------------------
add_executable(dbscan src/main.cu)

target_link_libraries(dbscan
        PRIVATE cpu_dbscan
        PRIVATE gpu_dbscan
        PRIVATE io
)
