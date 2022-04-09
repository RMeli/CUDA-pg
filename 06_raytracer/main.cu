#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "raytracer.h"
#include "sphere.h"

#include "err.h"
#include "mem.h"
#include "ppm.h"
#include "timing.h"

using namespace std;

// Allocate memory for array of Sphere on the GPU
constexpr size_t num_spheres{25};
__constant__ Sphere s_constant[num_spheres];

int main() {
    // Image size needs to be a multiple of 16
    // The code launch 16x16 threads for each grid point
    // There is no safeguard to avoid computation in additional threads
    // Not using multiples of 16 results in a distorted image (from GPU)
    constexpr size_t width{1024}, height{1024};
    constexpr size_t n{width * height * 3};

    std::default_random_engine e(42);
    std::uniform_real_distribution<double> uniform_rgb(0, 1);
    std::uniform_real_distribution<double> uniform_xyz(-500, 500);
    std::uniform_real_distribution<double> uniform_radius(10, 100);

    Timer t;
    double time{0.0};
    CUDATimer ct;
    double ctime{0.0};

    // Kernel launch values
    dim3 grid(width / 16, height / 16);
    dim3 threads(16, 16);

    // Output files
    std::ofstream outcpu("raytracer_cpu.ppm", std::ios::binary);
    std::ofstream outgpu("raytracer_gpu.ppm", std::ios::binary);
    std::ofstream outgpuconst("raytracer_gpu_const.ppm", std::ios::binary);

    // The spheres in this array compose to scene to be ray traced
    // Allocate memory on the host
    Sphere* s_host = malloc_host<Sphere>(num_spheres);

    // Inisitlise Sphere on the host
    for (size_t i{0}; i < num_spheres; i++) {
        s_host[i].r = uniform_rgb(e);
        s_host[i].g = uniform_rgb(e);
        s_host[i].b = uniform_rgb(e);

        s_host[i].x = uniform_xyz(e);
        s_host[i].y = uniform_xyz(e);
        s_host[i].z = uniform_xyz(e);

        s_host[i].radius = uniform_radius(e);
    }

    // Allocate image on the host
    char* image = malloc_host<char>(n);

    std::cout << "raytracer (cpu)... " << std::flush;
    t.start();
    raytracer_cpu(image, s_host, width, height, num_spheres);
    time = t.stop();
    std::cout << time << " ms" << std::endl << std::flush;

    if (image != nullptr) {
        utils::write_ppm(image, width, height, outcpu);
        free_host(image);
    }

    // Allocate global memory on device and copy spheres there
    Sphere* s_device = malloc_device<Sphere>(num_spheres);
    copy_host_to_device(s_host, s_device, num_spheres);

    image = malloc_host<char>(n);

    cout << "raytracer (gpu)... " << flush;
    ct.start();
    char* image_device = malloc_device<char>(n);
    raytracer_kernel<<<grid, threads>>>(image_device, s_device, width, height,
                                        num_spheres);
    copy_device_to_host(image_device, image, n);
    ctime = ct.stop();
    std::cout << std::setprecision(1) << ctime << " ms" << std::endl
              << std::flush;

    free_device(image_device);
    if (image != nullptr) {
        utils::write_ppm(image, width, height, outgpu);
        free_host(image);
    }

    // Copy Sphere on the host to __constant__ memory
    auto status =
        cudaMemcpyToSymbol(s_constant, s_host, num_spheres * sizeof(Sphere), 0,
                           cudaMemcpyHostToDevice);
    cuda_check_status(status);
    free_host(s_host);

    image = malloc_host<char>(n);

    // Free memory and nullify pointer
    // This pointer is re-used below to point to the constant memory
    free_device(s_device);

    // Get address of symbol (constant memory) on the device
    // Need to cast to void** in order to avoid 'error: no instance of
    // overloaded function "cudaGetSymbolAddress" matches the argument list'
    cuda_check_status(cudaGetSymbolAddress((void**)&s_device, s_constant));

    cout << "raytracer (gpu | const)... " << flush;
    ct.start();
    image_device = malloc_device<char>(n);
    raytracer_kernel<<<grid, threads>>>(image_device, s_device, width, height,
                                        num_spheres);
    copy_device_to_host(image_device, image, n);
    ctime = ct.stop();
    std::cout << std::setprecision(1) << ctime << " ms" << std::endl
              << std::flush;

    free_device(image_device);
    if (image != nullptr) {
        utils::write_ppm(image, width, height, outgpuconst);
        free_host(image);
    }

    return 0;
}