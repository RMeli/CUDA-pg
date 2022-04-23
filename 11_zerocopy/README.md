# ZEROCOPY

## Description

Zero-copy dot (inner) product between two vectors.

## CUDA Concepts

### Mapped Memory

The `cudaHostAlloc()`, used to allocate pinned memory (with the argument `cudaHostAllocDefault`) can be used to allocate mapped memory using the `cudaHostAllocMapped` argument. 

```cpp
template <typename T> T* malloc_mapped(std::size_t n) {
    void* host_ptr{nullptr}; // Declare void pointer
    auto status = cudaHostAlloc(
        &host_ptr, n * sizeof(T),
        cudaHostAllocMapped);  // Try page-locked mapped memory allocation
    cuda_check_status(status); // Check allocation
    return (T*)host_ptr;       // Return pointer of type T*
}
```

Mapped memory is also pinned (it can't be paged out of or relocated to physical memory), but it can be accessed from the GPU. It is also called zero-copy memory.

For buffers that are read-only for the GPU, performance can be improved by using the `cudaHostAllocWriteCombined`:

```cpp
template <typename T> T* malloc_mapped_readonly(std::size_t n) {
    void* host_ptr{nullptr}; // Declare void pointer
    auto status = cudaHostAlloc(
        &host_ptr, n * sizeof(T),
        cudaHostAllocWriteCombined |
            cudaHostAllocMapped); // Try page-locked mapped memory allocation
    cuda_check_status(status);    // Check allocation
    return (T*)host_ptr;          // Return pointer of type T*
}
```

Write-combined memory can be extremely inefficient if the CPU also needs to read it.

### Access Mapped Memory

The CPU and the GPU have a different virtual memory space, therefore buffers have different addresses on the CPU and the GPU. `cudaHostGetDevicePointer()` allow to get a valid GPU pointer.

### Mapped Memory Support

The device support of mapped memory can be verified as follows:

```cpp
cudaDeviceProp prop;
int device;
auto status = cudaGetDevice(&device);
cuda_check_status(status);
status = cudaGetDeviceProperties(&prop, device);
cuda_check_status(status);
if (prop.canMapHostMemory != 1) {
    cout << "Device cannot map memory." << endl;
    return 0;
}
```