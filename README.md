# CUDA Playground

Playground for experimentation with CUDA.

## Build Experiments

```bash
mkdir build        # create build directory
cd build           # change to build directory
cmake ..           # generato Makefile
make               # compile project
```

## Singularity

Run [Singularity](https://singularity.hpcng.org/) container with CUDA and [CMake](https://cmake.org/) interactively:

```bash
singularity shell --nv singularity/
```

## References

* [CSCS HPC Summer School 2018](https://github.com/eth-cscs/SummerSchool2018)
* [CUDA by Example - An Introduction to General Purpose GPU Programming](https://developer.nvidia.com/cuda-example)