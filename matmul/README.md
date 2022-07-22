# MatMul

Materix-matrix multiplication:

$$
    c_{ij} = \sum_k a_{ik}b_{kj}
$$

## CUDA Concepts

### Last CUDA Error

It is possible to retrieve the last CUDA error with the following code:

```cpp
auto err = cudaGetLastError();
if(cudaSuccess != err){
    std::cout << cudaGetErrorString(err) << std::endl;
}
```

This allows easier debugging. For example, if the kernel fails launching because too many threads were selected, nothing happens but the last CUDA error becomes `invalid configuration argument`.
