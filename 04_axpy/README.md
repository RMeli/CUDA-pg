# AXPY

## Description

Compute the vector operation

```text
a * x + y
```

and store the result of the operation in `y`.

## CUDA Concepts

### Threads

A parallel block can be split into `maxThreadsPerBlock` threads (the total number of blocks is limited to `65525`).

The second argument of the kernel call `kernel<<<1,N>>>()` represents the number `N` of threads per block that will be created. The thread index can be accessed with `threadIdx.x`.

### Blocks and Threads

Since threads are limited to `maxThreadsPerBlock` a combination of blocks and threads is usually needed; one can see the number of blocks as the rows of a computation matrix and the threads as the columns, so that a computation is indexed by

```cpp
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

`blockDim.x` indicates the number of threads per block (i.e. the number of columns).

`blockDim` is different from `griDim`, which stores the number of blocks along each dimension of a block grid. Additionally, `gridDim` is two-dimensional (2D array of blocks) while `blockDim` is three dimensional (3D array of threads per block).

Both blocks and threads are limited in number (`maxThreadsPerBlock` threads and `65535` blocks). This means that for larger computations one has to re-use the allocated number of threads, given by `blockDim.x * gridDim.x` (number of threads per block times the total number of blocks). Each thread can compute element `i`, `i + blockDim.x * gridDim.x`, `i + 2 * blockDim.x * gridDim.x`, and so on.
