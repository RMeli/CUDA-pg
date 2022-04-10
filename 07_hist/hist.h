namespace hist {
void hist_cpu(unsigned char* buffer, const size_t n, unsigned int* hist);

__global__ void hist_kernel_global(unsigned char* buffer, const size_t n,
                                   unsigned int* hist);
__global__ void hist_kernel_shared(unsigned char* buffer, const size_t n,
                                   unsigned int* hist);

} // namespace hist