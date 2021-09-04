#include <iostream>

using namespace std;

int main() {

    int nGPUs;
    cudaGetDeviceCount(&nGPUs);

    for (int i{0}; i < nGPUs; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        cout << "Device #" << i << endl;
        cout << "   Name: " << prop.name << endl;
        cout << "   Integrated: " << prop.integrated << endl;
        cout << "   Clock Rate (MHz): " << prop.clockRate / 1e3 << endl;
        cout << "   Global Memory (Gb): " << prop.totalGlobalMem / 1e9 << endl;
        cout << "   Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "   Threads per Warp: " << prop.warpSize << endl;
    }

    return 0;
}
