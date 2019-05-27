#ifndef MEM_H
#define MEM_H

#include "err.h"

template <typename T>
T* malloc_host(std::size_t n, T value=T()){
    T* ptr = new T[n]; // Allocate memory
    std::fill(ptr, ptr + n, value); // Fill memorys
    return ptr; // Return pointer of type T*
}

template <typename T>
void free_host(T* ptr){
    delete[] ptr;
    ptr = nullptr;
}

template <typename T>
T* malloc_device(std::size_t n){
    void* ptr{nullptr}; // Declare void pointer
    auto status = cudaMalloc(&ptr, n * sizeof(T)); // Try memory allocation
    cuda_check_status(status); // Check allocation
    return (T*)ptr; // Return pointer of type T*
}

#endif // MEM_H