# CUDA Playground

Playground for experimentation with CUDA.

## CUDA Concepts

* `01_info`: device information
* `02_axpy`: kernel functions, kernel calls, blocks, host/device memory
* `03_mandelbrot`: device functions, grid of blocks
* `04_axpy`: threads, blocks and threads
* `05_dot`: shared memory, thread synchronization

### Timings

CPU timings are measured for the kernel execution. GPU timings are measured for device memory allocation, copy of data from host to device, kernel execution, and copy of data from device to host.

## Build Experiments

```bash
mkdir build        # create build directory
cd build           # change to build directory
cmake ..           # generato Makefile
make               # compile project
```

### Singularity

Run [Singularity](https://singularity.hpcng.org/) container with CUDA and [CMake](https://cmake.org/) interactively:

```bash
singularity shell --nv singularity/
```

## References

* [CSCS HPC Summer School 2018](https://github.com/eth-cscs/SummerSchool2018)
* [CUDA by Example - An Introduction to General Purpose GPU Programming](https://developer.nvidia.com/cuda-example)
