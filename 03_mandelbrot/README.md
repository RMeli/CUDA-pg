# MANDELBROT

## Description

Produce a [bitmap image](https://en.wikipedia.org/wiki/Netpbm#File_formats) of the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set), defined as the set of points `c` of the complex plane for which the series

```text
z(0) = 0
z(n + 1) = z^2(n) + c
```

remains bounded for all `n > 0`.

## CUDA Concepts

### Device Functions

The `__device__` qualifier indicates to the CUDA compiler that the function only runs on the device and not the host.

`__device__` functions can be called only from other `__device__` functions or from kernel (`__global__`) functions.

### Grid of Blocks

It is possible to define a two-dimensional grid of blocks to launch the kernel. The grid is defined as

```cpp
dim3 grid(DIM_X, DIM_Y);
```

where `dim3` represent a three dimensional tuple (last dimension set to `1`) and the kernel is launched with

```cpp
kernel<<<grid,1>>>();
```

Blocks are indexed by `blockIdx.x` and `blockIdx.y`, while `gridDim.x` and `gridDim.y` store the dimensin of the grid of blocks that was launched.
