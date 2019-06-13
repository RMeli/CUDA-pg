#include <fstream>
#include <iostream>

#include "mandelbrot.h"

#include "mem.h"
#include "ppm.h"
#include "timing.h"

int main(){

    std::size_t width{1000}, height{800};
    std::size_t n{width * height * 3};

    Timer t;
    double time{0.0};

    std::ofstream outcpu("mandelbrot_cpu.ppm", std::ios::binary);
    std::ofstream outgpu("mandelbrot_gpu.ppm", std::ios::binary);

    // Allocate image
    char* image = malloc_host<char>(n);

    std::cout << "mandelbrot (cpu)... " << std::flush;
    t.start();
    mandelbrot(image, width, height);
    time = t.stop();
    std::cout << time << " ms" << std::endl << std::flush;

    if(image != nullptr){
        utils::write_ppm(image, width, height, outcpu);
        free_host(image); 
    }

    dim3 grid(width, height);
    image = malloc_host<char>(n);

    std::cout << "mandelbrot (gpu)... " << std::flush;
    t.start();
    char* image_device = malloc_device<char>(n);
    mandelbrot_gpu<<<grid,1>>>(image_device, width, height);
    copy_device_to_host(image_device, image, n);
    time = t.stop();
    std::cout << time << " ms" << std::endl << std::flush;

    if(image != nullptr){
        utils::write_ppm(image, width, height, outgpu);
    }

    
    free_device(image_device);

    return 0;
}