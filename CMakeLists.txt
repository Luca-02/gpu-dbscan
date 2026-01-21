cmake_minimum_required(VERSION 4.0)
project(gpu-scan CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(gpu-scan src/main.cu
        src/io.cpp
        src/cpu_dbscan.cpp
        src/gpu_dbscan.cu
        src/dbscan.h
        src/common.h
)

set_target_properties(gpu-scan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
