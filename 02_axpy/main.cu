#include <iostream>
#include <chrono>
#include <cassert>

#include "mem.h"
#include "num.h"

#include "axpy.cu"

using namespace std;

using duration = chrono::duration<double>;

int main(){
    constexpr std::size_t n{1024};

    double x{1.3}, y{1.6}, a{2.4};

    double* x_host = malloc_host(n, x);
    double* y_host = malloc_host(n, y);

    duration time;

    cout << "axpy (cpu)... ";
    auto ti = chrono::high_resolution_clock::now();
    axpy::axpy_cpu(y_host, x_host, a, n);
    auto tf = chrono::high_resolution_clock::now();
    time = chrono::duration_cast<duration>(tf - ti);
    cout << time.count() << " seconds" << endl;

    for(std::size_t i{0}; i < n; i++){
        assert(nearly_equal(y_host[i], y + a * x));
    }

    free_host(x_host);
    free_host(y_host);

    return 0;
}