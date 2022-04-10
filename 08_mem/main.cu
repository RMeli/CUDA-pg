#include <iomanip>
#include <iostream>

#include "mem.h"
#include "timing.h"

constexpr size_t n{100};              // Number of host/device copies
constexpr size_t N{10 * 1024 * 1024}; // Size of arrays to copy

using namespace std;
int main() {
    CUDATimer ct;
    double ctime{0.0};

    auto a_host_paged = malloc_host<int>(N);
    auto a_host_locked = cualloc_host<int>(N);
    auto a_device = malloc_device<int>(N);

    cout << "memcopy (host to device)... ";
    ct.start();
    for (size_t i{0}; i < n; ++i) {
        copy_host_to_device(a_host_paged, a_device, N);
    }
    ctime += ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    cout << "memcopy (device to host)... ";
    ctime = 0.0;
    ct.start();
    for (size_t i{0}; i < n; ++i) {
        copy_device_to_host(a_device, a_host_paged, N);
    }
    ctime += ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    cout << "memcopy (pinned to device)... ";
    ct.start();
    for (size_t i{0}; i < n; ++i) {
        copy_host_to_device(a_host_locked, a_device, N);
    }
    ctime += ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    cout << "memcopy (device to pinned)... ";
    ctime = 0.0;
    ct.start();
    for (size_t i{0}; i < n; ++i) {
        copy_device_to_host(a_device, a_host_locked, N);
    }
    ctime += ct.stop();
    cout << fixed << setprecision(0) << ctime << "ms" << endl;

    free_host(a_host_paged);
    free_cuhost(a_host_locked);
    free_device(a_device);

    return 0;
}