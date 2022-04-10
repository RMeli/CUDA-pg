
using namespace std;

namespace hist {
void hist_cpu(unsigned char* buffer, const size_t n, unsigned int* hist) {
    for (size_t i{0}; i < n; i++) {
        // buffer is unsigned char (8-bit byte): 256 possible values
        hist[buffer[i]]++;
    }
}

__global__ void hist_kernel_global(unsigned char* buffer, const size_t n,
                                   unsigned int* hist) {
    auto i{threadIdx.x + blockIdx.x * blockDim.x};
    auto offset{blockDim.x * gridDim.x};

    while (i < n) {
        // Atomic operation
        // Only one thread at a time can access the memory location
        // Avoids race conditions
        atomicAdd(&(hist[buffer[i]]), 1);

        i += offset;
    }
}

__global__ void hist_kernel_shared(unsigned char* buffer, const size_t n,
                                   unsigned int* hist) {
    // Create temporary histogram shared between threads of the same block
    // Should improve performance by reducing simultaneous access to memory
    __shared__ unsigned int tmp[256];
    tmp[threadIdx.x] =
        0;           // Initialise memory (each thread initialises one element)
    __syncthreads(); // Wait for all threads to finish initialising

    auto i{threadIdx.x + blockIdx.x * blockDim.x};
    auto offset{blockDim.x * gridDim.x};

    while (i < n) {
        // Atomic operation
        // Only one thread at a time can access the memory location
        // Avoids race conditions
        atomicAdd(&(tmp[buffer[i]]), 1);

        i += offset;
    }

    // Wait for all threads to finish updating temporary histogram
    __syncthreads();

    // Add temporary results from threads into global memory
    // Each threads is in charge of one bin
    atomicAdd(&(hist[threadIdx.x]), tmp[threadIdx.x]);
}

} // namespace hist