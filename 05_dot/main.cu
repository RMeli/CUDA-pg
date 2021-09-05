#include <cassert>
#include <iostream>

#include "mem.h"
#include "num.h"
#include "timing.h"

#include "dot.cu"

using namespace std;

int main() {
    // numBlocks should be rather small
    // The last part of the reduction operator is performed on the CPU
    constexpr std::size_t numBlocks{64};
    constexpr std::size_t numThreadsPerBlock{512};

    constexpr std::size_t n{33554432};
    constexpr std::size_t reps{10};

    double x{1.0}, y{2.0};

    Timer t;
    double time{0.0};
    double timemalloch{0.0};
    double timefreeh{0.0};

    double* x_host{nullptr};
    double* y_host{nullptr};
    double r{0.0};

    cout << "dot (cpu)..." << endl;
    for (std::size_t i{0}; i < reps; i++) {
        t.start();
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);
        timemalloch += t.stop();

        t.start();
        dot::dot_cpu(x_host, y_host, r, n);
        time += t.stop();

        assert(nearly_equal(r, static_cast<double>(2 * n)));

        t.start();
        free_host(x_host);
        free_host(y_host);
        timefreeh += t.stop();
    }
    cout << "  malloc: " << timemalloch << " ms" << endl;
    cout << "  dot: " << time << " ms" << endl;
    cout << "  free: " << timefreeh << " ms" << endl;

    // Reset times
    time = 0.0;
    timemalloch = 0.0;
    timefreeh = 0.0;

    cout << "dot (gpu)..." << endl;
    for (std::size_t i{0}; i < reps; i++) {
        t.start();
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);
        timemalloch += t.stop();

        t.start();
        dot::dot_gpu<numBlocks, numThreadsPerBlock>(x_host, y_host, r, n);
        time += t.stop();

        assert(nearly_equal(r, static_cast<double>(2 * n)));

        t.start();
        free_host(x_host);
        free_host(y_host);
        timefreeh += t.stop();
    }
    cout << "  malloc (host): " << timemalloch << " ms" << endl;
    cout << "  dot: " << time << " ms" << endl;
    cout << "  free (host): " << timefreeh << " ms" << endl;

    return 0;
}