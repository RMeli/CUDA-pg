#include <iomanip>
#include <iostream>

#include "mem.h"
#include "timing.h"

constexpr size_t n{1024 * 1024}; // Chunk size of arrays to copy
constexpr size_t N{500 * n};     // Total size of arrays to copy

__global__ void kernel(int* a, int* b, int* c);

using namespace std;
int main() {
    CUDATimer ct;
    double ctime{0.0};

    // Get device properties
    cudaDeviceProp prop;
    int device;
    auto status = cudaGetDevice(&device);
    cuda_check_status(status);
    status = cudaGetDeviceProperties(&prop, device);
    cuda_check_status(status);

    // Check if device supports overlaps
    // If not, there is no speedup to be had by using CUDA streams
    if (!prop.deviceOverlap) {
        cout << "Device does not support overlaps." << endl;
        return 0;
    }

    // Create CUDA stream
    cudaStream_t stream;
    status = cudaStreamCreate(&stream);
    cuda_check_status(status);

    // Allocate page-locked host memory
    // Page-locked memory is needed for asyncronous memory transfers
    auto a_host = cualloc_host<int>(N);
    auto b_host = cualloc_host<int>(N);
    auto c_host = cualloc_host<int>(N);

    for (size_t i{0}; i < N; ++i) {
        a_host[i] = i % 10;
        b_host[i] = i % 20;
    }

    // Allocate device memory
    auto a_dev = malloc_device<int>(N);
    auto b_dev = malloc_device<int>(N);
    auto c_dev = malloc_device<int>(N);

    cout << "no stream... ";
    ct.start();
    copy_host_to_device(a_host, a_dev, N);
    copy_host_to_device(a_host, a_dev, N);
    kernel<<<N / 256, 256, 0>>>(a_dev, a_dev, a_dev);
    copy_device_to_host(c_dev, c_host, N);
    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    cout << "stream... ";
    ct.start();

    // Work on data in chunks
    // If the device support overlap, this can speedup the applications
    // The device can deal with (async) copies and computations running in
    // parallel
    for (size_t i{0}; i < N; i += n) {
        // Copy data to from locked memory to device
        // Async copy means that host execution goes immediately to next time
        // Copy might not have finished (or even started) yet
        copy_host_to_device_async(a_host + i, a_dev + i, n, stream);
        copy_host_to_device_async(a_host + i, a_dev + i, n, stream);

        // Start computation as soon as stream is ready
        // (as soon as the previous operation on the same stream is complete)
        kernel<<<n / 256, 256, 0, stream>>>(a_dev, b_dev, c_dev);

        copy_device_to_host_async(c_dev + i, c_host + i, n, stream);
    }

    // Memory copies and kernel executions have been queued in the stream
    // When the loop finishes, copies and computations might still be running
    // We need to wait for all computations/copies to finish
    status = cudaStreamSynchronize(stream);
    cuda_check_status(status);

    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    free_cuhost(a_host);
    free_cuhost(b_host);
    free_cuhost(c_host);
    free_device(a_dev);
    free_device(b_dev);
    free_device(c_dev);

    status = cudaStreamDestroy(stream);
    cuda_check_status(status);

    return 0;
}

__global__ void kernel(int* a, int* b, int* c) {
    size_t i{threadIdx.x + blockIdx.x * blockDim.x};
    if (i < N) {
        size_t i1 = (i + 1) % 256;
        size_t i2 = (i + 2) % 256;
        float as = (a[i] + a[i1] + a[i2]) / 3.0;
        float bs = (b[i] + b[i1] + b[i2]) / 3.0;
        c[i] = (as + bs) / 2;
    }
}