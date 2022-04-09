#include <fstream>
#include <iomanip>
#include <iostream>

#include "mandelbrot.h"

#include "mem.h"
#include "ppm.h"
#include "timing.h"

using namespace std;

int main() {

    size_t width{1000}, height{800};
    size_t n{width * height * 3};

    Timer t;
    double time{0.0};
    CUDATimer ct;
    double ctime{0.0};

    ofstream outcpu("mandelbrot_cpu.ppm", ios::binary);
    ofstream outgpu("mandelbrot_gpu.ppm", ios::binary);

    // Allocate image on the host
    char* image = malloc_host<char>(n);

    cout << "mandelbrot (cpu)... " << flush;
    t.start();
    mandelbrot(image, width, height);
    time = t.stop();
    cout << fixed << setprecision(0) << time << " ms" << endl << flush;

    if (image != nullptr) {
        utils::write_ppm(image, width, height, outcpu);
        free_host(image);
    }

    dim3 grid(width, height);
    image = malloc_host<char>(n);

    cout << "mandelbrot (gpu)... " << flush;
    ct.start();
    char* image_device = malloc_device<char>(n);
    mandelbrot_gpu<<<grid, 1>>>(image_device, width, height);
    copy_device_to_host(image_device, image, n);
    ctime = ct.stop();
    cout << fixed << setprecision(0) << ctime << " ms" << endl << flush;

    if (image != nullptr) {
        utils::write_ppm(image, width, height, outgpu);
    }

    free_device(image_device);

    return 0;
}