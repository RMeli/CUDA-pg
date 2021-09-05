#include <iostream>

using namespace std;

struct Foo {
    size_t bar;
};

constexpr size_t n{42};
__constant__ Foo foo_device[n];

void cuda_check_status(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

template <typename T> T* malloc_host(std::size_t n, T value = T()) {
    T* host_ptr = new T[n]; // Allocate n elements of type T
    std::fill(host_ptr, host_ptr + n,
              value); // Initialise elements with default value
    return host_ptr;
}

template <typename T> void free_host(T* host_ptr) {
    delete[] host_ptr;  // Delete memory associated to host_ptr
    host_ptr = nullptr; // Nullify ptr
}

template <typename T>
void copy_host_to_device_constant(T* host_ptr, T* device_ptr, std::size_t n) {
    auto status = cudaMemcpyToSymbol(device_ptr, host_ptr, n * sizeof(T), 0,
                                     cudaMemcpyHostToDevice);
    cuda_check_status(status);
}

int main() {
    Foo* foo_host = malloc_host<Foo>(n);
    for (size_t i{0}; i < n; i++) {
        foo_host[i].bar = i;
    }

    //copy_host_to_device_constant(foo_host, foo_device, n);
    auto status = cudaMemcpyToSymbol(foo_device, foo_host, n * sizeof(Foo), 0,
                                     cudaMemcpyHostToDevice);
    cuda_check_status(status);

    free_host(foo_host);

    return 0;
}