#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "mem.h"
#include "num.h"
#include "timing.h"

#include "axpy.cu"

using namespace std;

int main() {
    constexpr std::size_t n{1024};
    constexpr std::size_t reps{10000};
    double x{1.3}, y{1.6}, a{2.4};

    Timer t;
    double time{0.0};
    CUDATimer ct;
    double ctime{0.0};

    double* x_host{nullptr};
    double* y_host{nullptr};

    cout << "axpy (cpu)... ";
    for (std::size_t i{0}; i < reps; i++) {
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);

        t.start();
        axpy::axpy_cpu(y_host, x_host, a, n);
        time += t.stop();

        for (std::size_t i{0}; i < n; i++) {
            assert(nearly_equal(y_host[i], y + a * x));
        }

        free_host(x_host);
        free_host(y_host);
    }
    cout << fixed << setprecision(0) << time << " ms" << endl;

    double* x_device{nullptr};
    double* y_device{nullptr};

    cout << "axpy (gpu)... ";
    for (std::size_t i{0}; i < reps; i++) {
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);

        ct.start();
        x_device = malloc_device<double>(n);
        y_device = malloc_device<double>(n);
        copy_host_to_device(x_host, x_device, n);
        copy_host_to_device(y_host, y_device, n);
        axpy::axpy_kernel<<<n, 1>>>(y_device, x_device, a, n);
        copy_device_to_host(y_device, y_host, n);
        ctime += ct.stop();

        for (std::size_t i{0}; i < n; i++) {
            assert(nearly_equal(y_host[i], y + a * x));
        }

        free_host(x_host);
        free_host(y_host);
        free_device(x_device);
        free_device(y_device);
    }
    cout << fixed << setprecision(0) << ctime << " ms" << endl;

    return 0;
}