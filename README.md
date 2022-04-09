# CUDA Playground

Playground for experimentation with CUDA.

## CUDA Concepts

* `01_info`: [device information](01_info/README.md#cuda-concepts)
* `02_axpy`: [kernel functions](02_axpy/README.md#kernel-function), [kernel calls](02_axpy/README.md#kernel-call), [blocks](02_axpy/README.md#blocks), [host/device memory](02_axpy/README.md#device-memory)
* `03_mandelbrot`: [device functions](03_mandelbrot/README.md#device-functions), [grid of blocks](03_mandelbrot/README.md#grid-of-blocks)
* `04_axpy`: [threads](04_axpy/README.md#threads), [blocks and threads](04_axpy/README.md#blocks-and-threads)
* `05_dot`: [shared memory](05_dot/README.md#shared-memory), [thread synchronization](05_dot/README.md#thread-synchronization)
* `06_raytracer`: [constant memory](06_raytracer#constant-memory), [warps](06_raytracer#warps), [host/device functions](06_raytracer#hostdevice-functions), [GPU timing and events]((06_raytracer#gpu-timing)
### Timings

CPU timings are measured for the kernel execution. GPU timings are measured for device memory allocation, copy of data from host to device, kernel execution, and copy of data from device to host (unless such timings are clearly disaggregated).

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
singularity shell --nv singularity/<CONTAINER>.sif
```

## References

* [CSCS HPC Summer School 2018](https://github.com/eth-cscs/SummerSchool2018)
* [CUDA by Example - An Introduction to General Purpose GPU Programming](https://developer.nvidia.com/cuda-example)
