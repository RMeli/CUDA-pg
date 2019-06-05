#include "mandelbrot.h"

#include <complex>

void mandelbrot_serial(char* image, std::size_t width, std::size_t height, std::size_t max_iters){
    for(std::size_t x{0}; x < width; x++){
        for(std::size_t y{0}; y < height; y++){
            std::size_t offset = x + y * width;

            bool in = mandelbrot(x, y, width, height);

            image[offset * 3 + 0] = in ? 0 : 255;
            image[offset * 3 + 1] = in ? 0 : 255;
            image[offset * 3 + 2] = in ? 0 : 255;
        }
    }
}

bool mandelbrot(std::size_t x, std::size_t y, std::size_t width, std::size_t height, std::size_t max_iters){

    constexpr double xmin{-2}, xmax{1}, ymin{-1.0}, ymax{1.0};

    // Map [0, width] to [-2, 1]
    double cx = xmin + x * (xmax - xmin) / width;

    // Map [0, height] to [-1, 1]
    double cy = ymin + y * (ymax - ymin) / height;

    std::complex<double> c(cx, cy);
    std::complex<double> z(0.0, 0.0);

    bool in{true};
    for(std::size_t iter{0}; iter < max_iters; iter++){
        z = z * z + c;

        if(std::abs(z) > 1000){
            in = false;
            break;
        }
    }

    return in;
}
