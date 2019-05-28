# INFO

## Description

Small application that prints to the standard output different properties of all the available GPUs.

## CUDA Concepts

### Number of GPUs

Count the number of available GPUs:
```cpp
int nGPUs;
cudaGetDeviceCount(&nGPUs);
```

### GPU Properties

Get the properties of a given GPU:
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, i);
```
