#include <iostream>
#include <cassert>

#include "mem.h"
#include "err.h"
#include "timing.h"

#include "matmul.h"

int main(){
    constexpr size_t n = 1000;
    constexpr size_t rows_a = n;
    constexpr size_t cols_a = n;
    constexpr size_t rows_b = n;
    constexpr size_t cols_b = n / 2;

    static_assert(cols_a == rows_b, "Matrix dimensions must agree");

    Timer t;

    auto A = malloc_host<int>(rows_a * cols_a, 1);
    auto B = malloc_host<int>(rows_b * cols_b);
    auto C = malloc_host<int>(rows_a * cols_b);
    auto C_serial = malloc_host<int>(rows_a * cols_b);

    for(size_t i = 0; i < rows_a; i++){
        for(size_t j = 0; j < cols_a; j++){
            A[i * cols_a + j] = i - j;
        }
    }

    for(size_t i = 0; i < rows_b; i++){
        for(size_t j = 0; j < cols_b; j++){
            B[i * cols_b + j] = i + j;
        }
    }

    t.start();
    matmul_serial(A, B, rows_a, cols_a, cols_b, C_serial);
    auto elapsed_time = t.stop();
    std::cout << "Serial: " << elapsed_time << " ms\n";
    
    constexpr size_t num_threads = 16; // 16x16 threads in total
    constexpr size_t num_blocks_x = (rows_a + num_threads - 1) / num_threads;
    constexpr size_t num_blocks_y = (cols_b + num_threads - 1) / num_threads;
    
    std::cout << "CUDA Blocks: (" << num_blocks_x << ',' << num_blocks_y << ')' << std::endl;
    std::cout << "CUDA Threads: (" << num_threads << ',' << num_threads << ')' << std::endl;

    dim3 numThreadsPerBlock(num_threads, num_threads);
    dim3 numBlocks(num_blocks_x, num_blocks_y);

    t.start();
    cudaDeviceSynchronize();
    auto A_dev = malloc_device<int>(rows_a * cols_a);
    auto B_dev = malloc_device<int>(rows_b * cols_b);
    auto C_dev = malloc_device<int>(rows_a * cols_b);
    copy_host_to_device(A, A_dev, rows_a * cols_a);
    copy_host_to_device(B, B_dev, rows_b * cols_b);
    matmul_simple<<<numBlocks, numThreadsPerBlock>>>(A_dev, B_dev, rows_a, cols_a, cols_b, C_dev);
    // Check that kernel has been executed correctly
    auto err = cudaGetLastError();
    if(cudaSuccess != err){
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    copy_device_to_host(C_dev, C, rows_a * cols_b);
    cudaDeviceSynchronize();
    elapsed_time = t.stop();
    std::cout << "CUDA: " << elapsed_time << " ms\n";

    // Check and reset
    for(size_t i = 0; i < rows_a; i++){
        for(size_t j = 0; j < cols_b; j++){
            assert(C[i * cols_b + j] == C_serial[i * cols_b + j]);
        }
    }

    free_host(A);
    free_host(B);
    free_host(C);
    free_device(A_dev);
    free_device(B_dev);
    free_device(C_dev);
}
