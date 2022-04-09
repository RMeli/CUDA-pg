cmake_minimum_required(VERSION 3.19)

project(info LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CUDA_ARCHITECTURES "35;50;72")

find_package(CUDA 10.0 REQUIRED)

include_directories(../utils)

add_executable(axpy main.cu
    ../utils/err.cu
    ../utils/timing.cu
)
