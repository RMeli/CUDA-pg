template<typename T>
void matmul_serial(T* A, T* B, size_t rows_a, size_t cols_a, size_t cols_b, T* C){
    for(size_t i = 0; i < rows_a; i++){
        for(size_t j = 0; j < cols_b; j++){
            T sum = T();
            for(size_t k = 0; k < cols_a; k++){
                sum += A[i * cols_a + k] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
}


template <typename T>
__global__
void matmul_simple(const T* A, const T* B, size_t rows_a, size_t cols_a, size_t cols_b, T* C){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < rows_a and j < cols_b){
        T sum = T();
        for(size_t k = 0; k < cols_a; k++){
            sum += A[i * cols_a + k] * B[k * cols_b + j];
        }
        C[i * cols_b + j] = sum;
    }
}