namespace axpy{

void axpy_cpu(double* y, double* x, double a, std::size_t n){
    for(std::size_t i{0}; i < n; i++){
        y[i] = y[i] + a * x[i];
    }
}

__global__
void axpy_kernel(double* y, double* x, double a, std::size_t n){
    auto i = blockIdx.x;

    if(i < n){
        y[i] = y[i] + a * x[i];
    }
}

}