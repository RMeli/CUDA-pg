#include <fstream>
#include <iostream>
#include <random>

#include "raytracer.h"
#include "sphere.h"

#include "mem.h"
#include "ppm.h"
#include "timing.h"

using namespace std;

// Allocate memory for array of Sphere on the GPU
constexpr size_t num_spheres{10};
__constant__ Sphere s_device[num_spheres];

int main() {
    constexpr size_t width{16}, height{16};
    constexpr size_t n{width * height * 3};

    std::default_random_engine e(42);
    std::uniform_real_distribution<double> uniform_rgb(0, 1);
    std::uniform_real_distribution<double> uniform_xyz(-500, 500);
    std::uniform_real_distribution<double> uniform_radius(10, 100);

    Timer t;
    double time{0.0};

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

    std::ofstream outcpu("raytracer_cpu.ppm", std::ios::binary);
    std::ofstream outgpu("raytracer_gpu.ppm", std::ios::binary);

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

    // Copy Sphere on the host to __constant__ device memory
    auto status =
        cudaMemcpyToSymbol(s_device, s_host, num_spheres * sizeof(Sphere), 0,
                           cudaMemcpyHostToDevice);
    cuda_check_status(status);
    free_host(s_host);

    image = malloc_host<char>(n);

    dim3 grid(width / 16, height / 16);
    dim3 threads(16, 16);

    cout << "raytracer (gpu)... " << flush;
    t.start();
    char* image_device = malloc_device<char>(n);
    cout << "DEBUG1" << endl;
    raytracer_kernel<<<grid, threads>>>(image_device, s_device, width, height,
                                        num_spheres);
    cout << "DEBUG2" << endl;
    copy_device_to_host(image_device, image, n);
    cout << "DEBUG3" << endl;
    time = t.stop();
    std::cout << time << " ms" << std::endl << std::flush;

    free_device(image_device);

    if (image != nullptr) {
        utils::write_ppm(image, width, height, outgpu);
        free_host(image);
    }

    return 0;
}