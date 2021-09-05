# RAYTRACER

## Description

Simple ray tracer.

## CUDA Concepts

### Constant Memory

`__constant__`

### `cudaMemcopyToSymbol`

### Host/Device Functions

Some functions can be executed both on the host and on the device without modification. Such functions can be defined with `__host__ __device__`.

## Notes

### `constexpr` Functions

`constexpr` functions (such as `std::numeric_limits<T>::lowest()`) can't be called within a `__device__` function since they are host functions:

```text
error: calling a constexpr __host__ function from a __device__ function is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
```

One can either use the experimental flag `--expt-relaxed-constexpr` (as suggested by the compiler) or can pre-declare a `constexpr` variable and return that instead.

### `__constant__` with Classes

`__constant__` variables do not seem to play nicely with classes. In particular it is not possible to use default-initialized variables in classes. It is better to use `struct` instead of `class` with public members that are assigned explicitly in the host code, rather than via a constructor.

```cpp
class Foo{
    public:
        Foo();

    private:
        int bar{0};
}
```

results in an error when used with `__constant__ Foo foo[42]`:

```text
error: dynamic initialization is not supported for a __constant__ variable
```

It is better to use a structure and to initialize members explicitly in the host code:

```cpp
class Foo{
    int bar;
}

Foo f;
f.bar = 0;
```

### Type Casting on Device Code

Type casting using `std::static_cast<T>(foo)` does not seem to work on device code (kernel functions). C-style conversion does work (`(T) foo`).

### Wrapping `cudaMemcpyToSymbol` into a Function

Wrapping `cudaMemcpyToSymbol()` into a function

```cpp
template <typename T>
void copy_host_to_device_constant(T* host_ptr, T* device_ptr, std::size_t n) {
    auto status = cudaMemcpyToSymbol(device_ptr, host_ptr, n * sizeof(T), 0,
                                     cudaMemcpyHostToDevice);
    cuda_check_status(status);
}
```

as done for `cudaMemcpy()` seem to lead to a `invalid device symbol`. When `cudaMemcpyToSymbol()` is called in `main.cu` (where the `__constant__` variable is defined) the error disappears.
