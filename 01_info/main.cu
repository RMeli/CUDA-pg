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
        cout << "   Global Memory (GN): " << prop.totalGlobalMem / 1e9 << endl;
        cout << "   Clock Rate (MHz): " << prop.clockRate / 1e3 << endl;
    }

    return 0;
}
