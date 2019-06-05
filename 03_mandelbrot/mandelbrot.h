#ifndef MANDELBROT_H
#define MANDELBROT_H

void mandelbrot_serial(char* image, std::size_t width, std::size_t height, std::size_t max_iters = 1000);

bool mandelbrot(std::size_t x, std::size_t y, std::size_t width, std::size_t height, std::size_t max_iters = 1000);

#endif // MANDELBROT_H