# HIST

Compute histogram of random 8-bit values.

## CUDA Concepts

### Atomic Operations

An _atomic operation_ is an operation that cannot be broken down into smaller parts by other threads.

### `cudaMemset`

`cudaMemset()` can be used to directly initialise the device memory, if there is no data to be copied from the beginning.

```cpp
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
```

### `atomicAdd`

`atomicAdd()` allows to perform atomic additions, ensuring that only one thread modifies the result at a time.
