#include <cassert>
#include <chrono>
#include <iostream>

#include "mem.h"
#include "num.h"
#include "timing.h"

#include "axpy.cu"

using namespace std;

int main() {
    constexpr std::size_t n{33554432};
    constexpr std::size_t reps{10};
    double x{1.3}, y{1.6}, a{2.4};

    Timer t;
    double time{0.0};
    double timemalloch{0.0}, timemallocd{0.0};
    double timefreeh{0.0}, timefreed{0.0};
    double timecopyhd{0.0}, timecopydh{0.0};

    double *x_host{nullptr};
    double *y_host{nullptr};

    cout << "axpy (cpu)..." << endl;
    for (std::size_t i{0}; i < reps; i++) {
        t.start();
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);
        timemalloch += t.stop();

        t.start();
        axpy::axpy_cpu(y_host, x_host, a, n);
        time += t.stop();

        for (std::size_t i{0}; i < n; i++) {
            assert(nearly_equal(y_host[i], y + a * x));
        }

        t.start();
        free_host(x_host);
        free_host(y_host);
        timefreeh += t.stop();
    }
    cout << "  malloc: " << timemalloch << " ms" << endl;
    cout << "  axpy: " << time << " ms" << endl;
    cout << "  free: " << timefreeh << " ms" << endl;

    double *x_device{nullptr};
    double *y_device{nullptr};

    // Reset times
    time = 0.0;
    timemalloch = 0.0;
    timefreeh = 0.0;

    cout << "axpy (gpu)..." << endl;
    for (std::size_t i{0}; i < reps; i++) {
        t.start();
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);
        timemalloch += t.stop();

        t.start();
        x_device = malloc_device<double>(n);
        y_device = malloc_device<double>(n);
        timemallocd += t.stop();

        t.start();
        copy_host_to_device(x_host, x_device, n);
        copy_host_to_device(y_host, y_device, n);
        timecopyhd += t.stop();

        t.start();
        axpy::axpy_kernel<<<4096, 1024>>>(y_device, x_device, a, n);
        time += t.stop();

        t.start();
        copy_device_to_host(y_device, y_host, n);
        timecopydh += t.stop();

        for (std::size_t i{0}; i < n; i++) {
            assert(nearly_equal(y_host[i], y + a * x));
        }

        t.start();
        free_host(x_host);
        free_host(y_host);
        timefreeh += t.stop();

        t.start();
        free_device(x_device);
        free_device(y_device);
        timefreed += t.stop();
    }
    cout << "  malloc (host): " << timemalloch << " ms" << endl;
    cout << "  malloc (device): " << timemallocd << " ms" << endl;
    cout << "  copy (host to device): " << timecopyhd << " ms" << endl;
    cout << "  axpy: " << time << " ms" << endl;
    cout << "  copy (device to host): " << timecopydh << " ms" << endl;
    cout << "  free (host): " << timefreeh << " ms" << endl;
    cout << "  free (device): " << timefreed << " ms" << endl;

    return 0;
}