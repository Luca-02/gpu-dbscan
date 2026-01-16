cmake_minimum_required(VERSION 4.0)
project(gpu-scan CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(gpu-scan src/main.cu
        src/parser.cpp
        src/structure/point.cpp
        src/dbscan.cpp
)

set_target_properties(gpu-scan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
