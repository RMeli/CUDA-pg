namespace dot {

#include <cassert>

#include "mem.h"

void dot_cpu(double* x, double* y, double& r, std::size_t n) {
    r = 0.0;
    for (std::size_t i{0}; i < n; i++) {
        r += x[i] * y[i];
    }
}

// Static shared memory allocation
// Define numThreadsPerBlock as template parameter of the kernel function
template <std::size_t numThreadsPerBlock = 256>
__global__ void dot_kernel(double* x, double* y, double* r, std::size_t n) {
    // For the reduction we need numThreadsPerBlock to be a power of 2
    // http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    assert(numThreadsPerBlock &&
           !(numThreadsPerBlock & (numThreadsPerBlock - 1)));

    // Schared variable amongs thread of the same block
    // Each block has its own copy of this variable
    __shared__ double cache[numThreadsPerBlock];

    std::size_t i{threadIdx.x + blockIdx.x * blockDim.x};
    std::size_t cacheIndex{threadIdx.x};

    double tmp{0.0};
    while (i < n) {
        tmp += x[i] * y[i];

        // Move current thread to next available index
        // Skip all indices assigned to all other threads
        // blockDim.x threads per block times grid.Dim blocks
        i += blockDim.x * gridDim.x;
    }

    // Store temporary reduction for this thread in cache
    cache[cacheIndex] = tmp;

    // Synchronize threads
    // Wait for all threads to fill the shared variable
    __syncthreads();

    // Use every other thread to compute the reduction
    // numThreadsPerBlock needs to be a power of 2
    // The complexity of this reduction is log2(numThreadsPerBlock)
    // The naive reduction (looping over the cache) would have linear complexity
    std::size_t j{blockDim.x / 2};
    while (j != 0) {
        // The first blockDim.x / 2 threads perform the computation
        // The last blockDim.x / 2 threads (cacheIndex >= j) do nothing
        // Example | Cache before reduction: [a, b, c, d]
        if (cacheIndex < j) {
            // Thread cacheIndex adds the value of cacheIndex + j to its own
            // cache
            cache[cacheIndex] += cache[cacheIndex + j];
        }
        __syncthreads();

        // Example | Cache after reduction: [a + c, b + d, c, d]
        // Perform the same computation for the first j/2 threads
        // Example | Chache after next computation: [a + c + b + d, b + d, c, d]
        j /= 2;
    }

    // Use first thread to copy the result of the reduction in r
    // r collects the reductions for different blocks of threads
    if (cacheIndex == 0) {
        // The result of the reduction is stored in cache[0]
        r[blockIdx.x] = cache[0];
    }
}

template <std::size_t numBlocks = 1024, std::size_t numThreadsPerBlock = 256>
void dot_gpu(double* x_host, double* y_host, double& r, std::size_t n) {
    double* x_device{nullptr};
    double* y_device{nullptr};

    x_device = malloc_device<double>(n);
    y_device = malloc_device<double>(n);

    copy_host_to_device(x_host, x_device, n);
    copy_host_to_device(y_host, y_device, n);

    // Results of partial sums computed with numBlocks
    double* r_host{nullptr};
    double* r_device{nullptr};
    r_host = malloc_host<double>(numBlocks);
    r_device = malloc_device<double>(numBlocks);

    dot::dot_kernel<numThreadsPerBlock>
        <<<numBlocks, numThreadsPerBlock>>>(x_device, y_device, r_device, n);

    copy_device_to_host(r_device, r_host, numBlocks);

    // Compute final reduction on the host
    r = 0.0;
    for (std::size_t i{0}; i < numBlocks; i++) {
        r += r_host[i];
    }

    free_device(x_device);
    free_device(y_device);
    free_device(r_device);
    free_host(r_host);
}

} // namespace dot