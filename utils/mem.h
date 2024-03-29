#ifndef MEM_H
#define MEM_H

#include <algorithm>

#include "err.h"

/**
 * @brief Allocate memory on the host
 *
 * @tparam T
 * @param n Number of memory blocks to allocate
 * @param value Initization value for memory blocks
 * @return T* Pointer to allocated memory
 */
template <typename T> T* malloc_host(std::size_t n, T value = T()) {
    T* host_ptr = new T[n];                   // Allocate memory
    std::fill(host_ptr, host_ptr + n, value); // Fill memorys
    return host_ptr;                          // Return pointer of type T*
}

/**
 * @brief Free host memory
 *
 * @tparam T
 * @param host_ptr Host pointer
 */
template <typename T> void free_host(T* host_ptr) {
    delete[] host_ptr;
    host_ptr = nullptr;
}

/**
 * @brief Copy host memory to host memory
 *
 * @tparam T
 * @param host_ptr_from Host pointer to copy from
 * @param host_prt_to Host pointer to copy to
 * @param n Number of memory blocks
 */
template <typename T>
void copy_host_to_host(T* host_ptr_from, T* host_prt_to, std::size_t n) {
    std::copy(host_ptr_from, host_ptr_from + n, host_prt_to);
}

/**
 * @brief Allocate memory on the device
 *
 * @tparam T
 * @param n Number of memory blocks to allocate
 * @return T* Pointer to allocated memory
 */
template <typename T> T* malloc_device(std::size_t n) {
    void* device_ptr{nullptr}; // Declare void pointer
    auto status =
        cudaMalloc(&device_ptr, n * sizeof(T)); // Try memory allocation
    cuda_check_status(status);                  // Check allocation
    return (T*)device_ptr;                      // Return pointer of type T*
}

/**
 * @brief Allocate memory on the device and set memory with @param value
 *
 * @tparam T
 * @param n Number of memory blocks to allocate
 * @param value Initization value for memory blocks
 * @return T* Pointer to allocated memory
 */
template <typename T> T* malloc_memset_device(std::size_t n, T value = T()) {
    void* device_ptr{nullptr}; // Declare void pointer
    auto status =
        cudaMalloc(&device_ptr, n * sizeof(T)); // Try memory allocation
    cuda_check_status(status);                  // Check allocation
    status = cudaMemset(device_ptr, value,
                        n * sizeof(T)); // Try memory initialization
    cuda_check_status(status);          // Check initialization
    return (T*)device_ptr;              // Return pointer of type T*
}

/**
 * @brief Free device memory
 *
 * @tparam T
 * @param device_ptr Device pointer
 */
template <typename T> void free_device(T* device_ptr) {
    cudaFree(device_ptr); // Free device pointer
    device_ptr = nullptr;
}

/**
 * @brief Copy host memory to device memory
 *
 * @tparam T
 * @param host_ptr Host pointer
 * @param device_ptr Device pointer
 * @param n Number of memory blocks to copy
 */
template <typename T>
void copy_host_to_device(T* host_ptr, T* device_ptr, std::size_t n) {
    auto status =
        cudaMemcpy(device_ptr, host_ptr, n * sizeof(T), cudaMemcpyHostToDevice);
    cuda_check_status(status);
}

/**
 * @brief Copy host memory to device memory asynchronously
 *
 * @tparam T
 * @param host_ptr Host pointer
 * @param device_ptr Device pointer
 * @param n Number of memory blocks to copy
 * @param stream CUDA stream
 */
template <typename T>
void copy_host_to_device_async(T* host_ptr, T* device_ptr, std::size_t n,
                               cudaStream_t stream) {
    auto status = cudaMemcpyAsync(device_ptr, host_ptr, n * sizeof(T),
                                  cudaMemcpyHostToDevice, stream);
    cuda_check_status(status);
}

/**
 * @brief Copy device memory to host memory
 *
 * @tparam T
 * @param device_ptr Device pointer
 * @param host_ptr Host pointer
 * @param n Number of memory blocks to copy
 */
template <typename T>
void copy_device_to_host(T* device_ptr, T* host_ptr, std::size_t n) {
    auto status =
        cudaMemcpy(host_ptr, device_ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
    cuda_check_status(status);
}

/**
 * @brief Copy device memory to host memory
 *
 * @tparam T
 * @param device_ptr Device pointer
 * @param host_ptr Host pointer
 * @param n Number of memory blocks to copy
 */
template <typename T>
void copy_device_to_host_async(T* device_ptr, T* host_ptr, std::size_t n,
                               cudaStream_t stream) {
    auto status = cudaMemcpyAsync(host_ptr, device_ptr, n * sizeof(T),
                                  cudaMemcpyDeviceToHost, stream);
    cuda_check_status(status);
}

/**
 * @brief Allocate page-locked memory on the host
 *
 * @tparam T
 * @param n Number of memory blocks to allocate
 * @return T* Pointer to allocated memory
 */
template <typename T> T* cualloc_host(std::size_t n) {
    void* host_ptr{nullptr}; // Declare void pointer
    auto status = cudaHostAlloc(
        &host_ptr, n * sizeof(T),
        cudaHostAllocDefault); // Try page-locked memory allocation
    cuda_check_status(status); // Check allocation
    return (T*)host_ptr;       // Return pointer of type T*
}

/**
 * @brief Free page-locked memory on the host
 *
 * @tparam T
 * @param host_ptr Device pointer
 */
template <typename T> void free_cuhost(T* host_ptr) {
    auto status = cudaFreeHost(host_ptr); // Free page-locked memory
    cuda_check_status(status);
    host_ptr = nullptr;
}

/**
 * @brief Allocate page-locked mapped memory on the host
 *
 * @tparam T
 * @param n Number of memory blocks to allocate
 * @param value Initization value for memory blocks
 * @return T* Pointer to allocated memory
 */
template <typename T> T* malloc_mapped(std::size_t n, T value = T()) {
    void* host_ptr{nullptr}; // Declare void pointer
    auto status = cudaHostAlloc(
        &host_ptr, n * sizeof(T),
        cudaHostAllocMapped);  // Try page-locked mapped memory allocation
    cuda_check_status(status); // Check allocation
    std::fill((T*)host_ptr, (T*)host_ptr + n, value); // Fill memorys
    return (T*)host_ptr; // Return pointer of type T*
}

/**
 * @brief Allocate read-only page-locked mapped memory on the host
 *
 * @tparam T
 * @param n Number of memory blocks to allocate
 * @param value Initization value for memory blocks
 * @return T* Pointer to allocated memory
 */
template <typename T> T* malloc_mapped_readonly(std::size_t n, T value = T()) {
    void* host_ptr{nullptr}; // Declare void pointer
    auto status = cudaHostAlloc(
        &host_ptr, n * sizeof(T),
        cudaHostAllocWriteCombined |
            cudaHostAllocMapped); // Try page-locked mapped memory allocation
    cuda_check_status(status);    // Check allocation
    std::fill((T*)host_ptr, (T*)host_ptr + n, value); // Fill memorys
    return (T*)host_ptr; // Return pointer of type T*
}

#endif // MEM_H