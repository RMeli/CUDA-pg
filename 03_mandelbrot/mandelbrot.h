#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <cstddef>

void mandelbrot(char *image, std::size_t width, std::size_t height,
                std::size_t max_iters = 1000);

__global__ void mandelbrot_gpu(char *image, std::size_t width,
                               std::size_t height,
                               std::size_t max_iters = 1000);

#endif // MANDELBROT_H