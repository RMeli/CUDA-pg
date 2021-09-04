# AXPY

## Description

Compute the vector operation
```text
a * x + y
```
and store the result of the operation in `y`.

## CUDA Concepts

### Kernel Function

The `__global__` qualifier indicates to the CUDA compiler that the function (kernel) needs to be compiled to run on the device instead of the host.

```cuda
__global__ void kernel() {
}
```

A kernel (`__global__`) function can be called from the host but runs on the device.

### Kernel Call

A kernel function is called with parameters within `<<< >>>`. Such parameters are passed to the runtime system (not the kernel function) and determine how the device code is lunched.

```cuda
kernel<<<1,1>>>();
```

### Block

The first argument of the kernel call `kernel<<<N,1>>>()` represent the number `N` of parallel blocks that will be created. Blocks represent parallel invocations of the kernel and the block index can be accessed with `blockIdx.x`.

### Device Memory

The device has his own memory that needs to be allocated and deallocated (using `cudaMalloc()` and `cudaFree()`). Data from the host can be transferred to the device and back using `cudaMemcpy()`.

**Danger**: There are no safeguards poreventing to dereference a device pointer from the host.
