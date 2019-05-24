#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>

void cuda_check_status(cudaError_t status){
    if(status != cudaSuccess){
        std::cerr << "ERROR: "
            << cudaGetErrorString(status)
            << std::endl;

        exit(1);
    }
}

template <typename T>
T* malloc_host(std::size_t n){
    T* ptr = new T[n];
    return ptr;
}

template <typename T>
T* malloc_device(std::size_t n){
    void* ptr; // Declare void pointer
    auto status = cudaMalloc(&ptr, n * sizeof(T)); // Try memory allocation
    cuda_check_status(status); // Check allocation
    return (T*)ptr; // Return pointer of type T
}

#endif // UTILS_H