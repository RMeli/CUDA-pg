#include <iostream>
#include <chrono>
#include <cassert>

#include "mem.h"
#include "num.h"
#include "timing.h"

#include "axpy.cu"

using namespace std;

int main(){
    constexpr std::size_t n{1024};

    constexpr std::size_t reps{10000};

    double x{1.3}, y{1.6}, a{2.4};

    Timer t;
    double time;

    double* x_host{nullptr};
    double* y_host{nullptr};

    cout << "axpy (cpu)... ";
    t.start();
    for(std::size_t i{0}; i < reps; i++){
        x_host = malloc_host(n, x);
        y_host = malloc_host(n, y);
        axpy::axpy_cpu(y_host, x_host, a, n);
        free_host(x_host);
        free_host(y_host);
    }
    time = t.stop();
    cout << time << " ms" << endl;

    for(std::size_t i{0}; i < n; i++){
        assert(nearly_equal(y_host[i], y + a * x));
    }

    return 0;
}