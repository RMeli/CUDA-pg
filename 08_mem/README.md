# MEM

Benchmark page-locked (pinned) memory against pageable memory.

## CUDA Concepts

### Page-Locked Memory

Page-locked memory is memory that is guaranteed to be never paged out to disk. Page-locked (pinned) memory always reside on physical memory, and it is therefore always safe to access the physical address of the memory. The GPU can use direct memory access (DMA) to access page-locked memory.

Page-locked memory can be allocated with `cudaHostAlloc()` and deallocaed with `cudaFreeHost()`.
