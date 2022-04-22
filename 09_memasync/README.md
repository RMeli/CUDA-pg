# MEMASYNC

Use single CUDA stream for asyncronous memory copy.
## CUDA Concepts

### CUDA Streams

A CUDA stream is a queue for the GPU on which operations can be queued in a specific order. Operations on the same stream are guaranteed to be executed in order.

```cpp
cudaStream_t stream;
auto status = cudaStreamCreate(&stream);
cuda_check_status(status);
// Do stuff...
status = cudaStreamDestroy(stream);
cuda_check_status(status);
```

### Asyncronous Memory Copy

Host to device and device to host copies can happen asyncronously with `cudaMemcopyAsync()`. This allows to speedup computation if `prop.deviceOverlap == true` because the device can perform computations _while_ transferring data from/to memory.
