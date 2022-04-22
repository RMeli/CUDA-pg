# STREAMS

Use multiple CUDA stream for asyncronous memory copy.

## CUDA Concepts

### Using CUDA Streams Effectively

In order to use CUDA stream effectively, operations need to be queued breadth-first instead of depth-first.

Bouncing between two streams provides the desired effect (operation overlap):
```cpp
copy_host_to_device_async(a_host + i, a_dev_0 + i, n, stream0);
copy_host_to_device_async(b_host + i, b_dev_0 + i, n, stream0);

copy_host_to_device_async(a_host + i + n, a_dev_1 + i, n, stream1);
copy_host_to_device_async(b_host + i + n, b_dev_1 + i, n, stream1);

kernel<<<n / 256, 256, 0, stream0>>>(a_dev_0, b_dev_0, c_dev_0);
kernel<<<n / 256, 256, 0, stream1>>>(a_dev_1, b_dev_1, c_dev_1);

copy_device_to_host_async(c_dev_0 + i, c_host + i, n, stream0);
copy_device_to_host_async(c_dev_1 + i, c_host + i + n, n, stream1);
```

Queuing all operations on one stream first and on another second ,doen not produce the desired speedup:

```cpp
// stream 0

copy_host_to_device_async(a_host + i, a_dev_0 + i, n, stream0);
copy_host_to_device_async(b_host + i, b_dev_0 + i, n, stream0);

kernel<<<n / 256, 256, 0, stream0>>>(a_dev_0, b_dev_0, c_dev_0);

copy_device_to_host_async(c_dev_0 + i, c_host + i, n, stream0);

// stream1

copy_host_to_device_async(a_host + i + n, a_dev_1 + i, n, stream1);
copy_host_to_device_async(b_host + i + n, b_dev_1 + i, n, stream1);

kernel<<<n / 256, 256, 0, stream1>>>(a_dev_1, b_dev_1, c_dev_1);

copy_device_to_host_async(c_dev_1 + i, c_host + i + n, n, stream1);
```
