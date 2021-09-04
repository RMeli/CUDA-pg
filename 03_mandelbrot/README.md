# MANDELBROT

## Description

Produce a [bitmap image](https://en.wikipedia.org/wiki/Netpbm#File_formats) of the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set), defined as the set of points `c` of the complex plane for which the series

```text
z(0) = 0
z(n + 1) = z^2(n) + c
```

remains bounded for all `n > 0`.

## CUDA Concepts

### Device Function

The `__device__` qualifier indicates to the CUDA compiler that the function only runs on the device and not the host.

`__device__` functions can be called only from other `__device__` functions or from kernel (`__global__`) functions.
