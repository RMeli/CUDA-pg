#include "mandelbrot.h"

#include <complex>

bool in_mandelbrot(std::size_t x, std::size_t y, std::size_t width,
                   std::size_t height, std::size_t max_iters = 1000);

void mandelbrot(char *image, std::size_t width, std::size_t height,
                std::size_t max_iters) {
    for (std::size_t x{0}; x < width; x++) {
        for (std::size_t y{0}; y < height; y++) {
            std::size_t offset = x + y * width;

            bool in = in_mandelbrot(x, y, width, height);

            image[offset * 3 + 0] = in ? 0 : 255;
            image[offset * 3 + 1] = in ? 0 : 255;
            image[offset * 3 + 2] = in ? 0 : 255;
        }
    }
}

bool in_mandelbrot(std::size_t x, std::size_t y, std::size_t width,
                   std::size_t height, std::size_t max_iters) {

    constexpr double xmin{-2}, xmax{1}, ymin{-1.0}, ymax{1.0};

    // Map [0, width] to [-2, 1]
    double cx = xmin + x * (xmax - xmin) / width;

    // Map [0, height] to [-1, 1]
    double cy = ymin + y * (ymax - ymin) / height;

    std::complex<double> c(cx, cy);
    std::complex<double> z(0.0, 0.0);

    bool in{true};
    for (std::size_t iter{0}; iter < max_iters; iter++) {
        z = z * z + c;

        if (std::abs(z) > 1000) {
            in = false;
            break;
        }
    }

    return in;
}

template <typename T> class cuComplex {
  public:
    __device__ cuComplex(T real_, T imag_) : real(real_), imag(imag_) {}

    __device__ cuComplex<T> operator+(const cuComplex<T> &c) const {
        return cuComplex<T>(real + c.real, imag + c.imag);
    }

    __device__ cuComplex<T> operator*(const cuComplex<T> &c) const {
        return cuComplex<T>(real * c.real - imag * c.imag,
                            imag * c.real + real * c.imag);
    }

    __device__ T abs2() const { return real * real + imag * imag; }

  private:
    T real = 0.0;
    T imag = 0.0;
};

__device__ bool in_mandelbrot_gpu(std::size_t x, std::size_t y,
                                  std::size_t width, std::size_t height,
                                  std::size_t max_iters = 1000);

__global__ void mandelbrot_gpu(char *image, std::size_t width,
                               std::size_t height, std::size_t max_iters) {
    std::size_t x = blockIdx.x;
    std::size_t y = blockIdx.y;
    std::size_t offset = x + y * gridDim.x;

    bool in = in_mandelbrot_gpu(x, y, width, height);

    image[offset * 3 + 0] = in ? 0 : 255;
    image[offset * 3 + 1] = in ? 0 : 255;
    image[offset * 3 + 2] = in ? 0 : 255;
}

__device__ bool in_mandelbrot_gpu(std::size_t x, std::size_t y,
                                  std::size_t width, std::size_t height,
                                  std::size_t max_iters) {

    constexpr double xmin{-2}, xmax{1}, ymin{-1.0}, ymax{1.0};

    // Map [0, width] to [-2, 1]
    double cx = xmin + x * (xmax - xmin) / width;

    // Map [0, height] to [-1, 1]
    double cy = ymin + y * (ymax - ymin) / height;

    cuComplex<double> c(cx, cy);
    cuComplex<double> z(0.0, 0.0);

    bool in{true};
    for (std::size_t iter{0}; iter < max_iters; iter++) {
        z = z * z + c;

        if (z.abs2() > 1000 * 1000) {
            in = false;
            break;
        }
    }

    return in;
}