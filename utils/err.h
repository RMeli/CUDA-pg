#ifndef ERR_H
#define ERR_H

#include <iostream>
#include <cstdlib>

void cuda_check_status(cudaError_t status){
    if(status != cudaSuccess){
        std::cerr << "ERROR: "
            << cudaGetErrorString(status)
            << std::endl;

        exit(1);
    }
}

#endif // ERR_H