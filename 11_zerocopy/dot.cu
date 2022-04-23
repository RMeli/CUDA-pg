void dot_cpu(double* x, double* y, double& r, std::size_t n) {
    r = 0.0;
    for (std::size_t i{0}; i < n; i++) {
        r += x[i] * y[i];
    }
}