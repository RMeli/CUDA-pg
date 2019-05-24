cmake_minimum_required(VERSION 3.10)

project(info LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall")

find_package(CUDA 9.0 REQUIRED)

add_executable(info main.cu)
