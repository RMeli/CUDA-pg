#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

#include "mem.h"
#include "timing.h"

#include "hist.h"

using namespace std;

constexpr size_t buffer_size{100 * 1024 * 1024}; // 100 MB
constexpr size_t num_bins{256}; // 256 bins (values of unsigned char)

void init_hist(unsigned int[num_bins]);
void check_hist(unsigned int[num_bins], size_t);

int main() {

    auto buffer = malloc_host<unsigned char>(buffer_size);
    unsigned int histogram[num_bins];

    Timer t;
    double time{0.0};
    CUDATimer ct;
    double ctime{0.0};

    // Initialize the buffer with random values
    default_random_engine generator;
    uniform_int_distribution<unsigned char> distribution(0, 255);
    for (size_t i{0}; i < buffer_size; ++i) {
        buffer[i] = distribution(generator);
    }

    // Compute the histogram on CPU
    init_hist(histogram);
    cout << "hist (cpu)... ";
    t.start();
    hist::hist_cpu(buffer, buffer_size, histogram);
    time = t.stop();
    cout << fixed << setprecision(0) << time << " ms" << endl;
    check_hist(histogram, buffer_size);

    // Get device properties
    // Optimal performance here is achieved when the number of blocks
    // is twice the number of GPU multiprocessors
    cudaDeviceProp props;
    auto status = cudaGetDeviceProperties(&props, 0);
    cuda_check_status(status);
    const int n_blocks{props.multiProcessorCount};

    // Compute the histogram on GPU (with global memory)
    init_hist(histogram);
    cout << "hist (gpu | global)... ";
    ct.start();

    // Allocate device memory and copy data to device or set device memory
    auto hist_device = malloc_memset_device<unsigned int>(num_bins, 0);
    auto buffer_device = malloc_device<unsigned char>(buffer_size);
    copy_host_to_device(buffer, buffer_device, buffer_size);

    hist::hist_kernel_global<<<n_blocks * 2, 256>>>(buffer_device, buffer_size,
                                                    hist_device);

    // Copy histogram back to host
    copy_device_to_host(hist_device, histogram, num_bins);

    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << " ms" << endl;
    check_hist(histogram, buffer_size);

    // Compute the histogram on GPU (with shared memory)
    init_hist(histogram);
    cout << "hist (gpu | shared)... ";
    ct.start();

    // Allocate device memory and copy data to device or set device memory
    hist_device = malloc_memset_device<unsigned int>(num_bins, 0);
    buffer_device = malloc_device<unsigned char>(buffer_size);
    copy_host_to_device(buffer, buffer_device, buffer_size);

    hist::hist_kernel_shared<<<n_blocks * 2, 256>>>(buffer_device, buffer_size,
                                                    hist_device);

    // Copy histogram back to host
    copy_device_to_host(hist_device, histogram, num_bins);

    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << " ms" << endl;
    check_hist(histogram, buffer_size);

    // Cleanup
    free_host(buffer);
    free_device(hist_device);
    free_device(buffer_device);

    return 0;
}

void init_hist(unsigned int histogram[num_bins]) {
    for (size_t i{0}; i < num_bins; i++) {
        histogram[i] = 0;
    }
}

void check_hist(unsigned int histogram[num_bins], const size_t bs) {
    unsigned int sum{0};
    for (size_t i{0}; i < num_bins; ++i) {
        sum += histogram[i];
    }
    assert(sum == bs);
}