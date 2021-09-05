# DOT

## Description

Compute dot (inner) product between two vectors.

## CUDA Concepts

### Shared Memory

`__shared__` allows to declare a variable in shared memory within a block of threads. Each block has its own copy of a `__shared__` variable, which is shared amongst the threads of the same block.

Shared memory is on GPU and therefore has a very low latency access.

### Thread Synchronization

`__syncthreads()` allows to synchronize threads within the same block. This allows to wait for all threads (within a block) to finish their computation before continuing with additional computations.

`__syncthreads()` needs to be executed by _all_ thre threads, otherwise the process will hang forever.

## Notes

### Static vs Dynamic Shared Memory Allocation

If `const std:;size_t numThreadsPerBlock` is passed to the kernel function, trying to allocate a shared memory array as

```cpp
__shared__ double cache[numThreadsPerBlock]
```

results in the following error:

```text
note: the value of parameter "numThreadsPerBlock": here cannot be used as a constant
```

A `__shared__` variable can be statically or dynamically allocated. For static allocation, one can define the number of threads per block as `#define NUM_THREADS_PER_BLOCK (256)` or as a template parameter for the kernel function `template<std::size_t numThreadsPerBlock=256>`. For dynamic allocation, one can use the `extern` keyword and add the size of allocation as an argument of the kernel call `kernel<<<numBlocks, numThreadsPerBlock, sizeof(double)*numThreadsPerBlock>>>()` (where `numBlocks` is `gridDim.x` and `numThreadsPerBlock` is `blockDim.x`).
