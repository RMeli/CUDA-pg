#include <iomanip>
#include <iostream>

#include "mem.h"
#include "timing.h"

constexpr size_t n{1024 * 1024}; // Chunk size of arrays to copy
constexpr size_t N{100 * n};     // Total size of arrays to copy

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

    // Create CUDA streams
    cudaStream_t stream0, stream1;
    status = cudaStreamCreate(&stream0);
    cuda_check_status(status);
    status = cudaStreamCreate(&stream1);
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
    // Every stream uses a different buffer
    auto a_dev_0 = malloc_device<int>(N);
    auto b_dev_0 = malloc_device<int>(N);
    auto c_dev_0 = malloc_device<int>(N);
    auto a_dev_1 = malloc_device<int>(N);
    auto b_dev_1 = malloc_device<int>(N);
    auto c_dev_1 = malloc_device<int>(N);

    cout << "no stream... ";
    ct.start();
    copy_host_to_device(a_host, a_dev_0, N);
    copy_host_to_device(a_host, a_dev_0, N);
    kernel<<<N / 256, 256, 0>>>(a_dev_0, a_dev_0, a_dev_0);
    copy_device_to_host(c_dev_0, c_host, N);
    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    cout << "stream... ";
    ct.start();

    // Work on data in chunks
    // If the device support overlap, this can speedup the applications
    // The device can deal with (async) copies and computations running in
    // parallel
    // Loop for twice the size of the chunck since we use two streams
    for (size_t i{0}; i < N; i += n * 2) {
        // Copy data to from locked memory to device
        // Async copy means that host execution goes immediately to next time
        // Copy might not have finished (or even started) yet

        // Copy using stream0
        copy_host_to_device_async(a_host + i, a_dev_0 + i, n, stream0);
        copy_host_to_device_async(a_host + i, a_dev_0 + i, n, stream0);

        // Copy using stream1
        copy_host_to_device_async(a_host + i + n, a_dev_1 + i, n, stream1);
        copy_host_to_device_async(a_host + i + n, a_dev_1 + i, n, stream1);

        // Start computation as soon as stream is ready
        // (as soon as the previous operation on the same stream is complete)

        // Computation for stream0
        kernel<<<n / 256, 256, 0, stream0>>>(a_dev_0, b_dev_0, c_dev_0);
        kernel<<<n / 256, 256, 0, stream1>>>(a_dev_1, b_dev_1, c_dev_1);

        // Copy back to locked memory from stream0
        copy_device_to_host_async(c_dev_0 + i, c_host + i, n, stream0);
        copy_device_to_host_async(c_dev_1 + i, c_host + i + n, n, stream1);
    }

    // Memory copies and kernel executions have been queued in the stream
    // When the loop finishes, copies and computations might still be running
    // We need to wait for all computations/copies to finish
    // Sync both streams
    status = cudaStreamSynchronize(stream0);
    cuda_check_status(status);
    status = cudaStreamSynchronize(stream1);
    cuda_check_status(status);

    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    free_cuhost(a_host);
    free_cuhost(b_host);
    free_cuhost(c_host);
    free_device(a_dev_0);
    free_device(b_dev_0);
    free_device(c_dev_0);
    free_device(a_dev_1);
    free_device(b_dev_1);
    free_device(c_dev_1);

    status = cudaStreamDestroy(stream0);
    cuda_check_status(status);
    status = cudaStreamDestroy(stream1);
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