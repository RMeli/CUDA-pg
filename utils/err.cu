#include <cstdlib>
#include <iostream>

void cuda_check_status(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}
