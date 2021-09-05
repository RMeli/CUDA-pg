namespace axpy {

void axpy_cpu(double *y, double *x, double a, std::size_t n) {
    for (std::size_t i{0}; i < n; i++) {
        y[i] = y[i] + a * x[i];
    }
}

__global__ void axpy_kernel(double *y, double *x, double a, std::size_t n) {
    // Each parallel thread starts on a different data index
    // Each block is split into blockDim.x threads
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n) {
        y[i] = y[i] + a * x[i];

        // blockDim.x * gridDim.x is the total number of parallel threads (in the grid)
        i += blockDim.x * gridDim.x;
    }
}

} // namespace axpy