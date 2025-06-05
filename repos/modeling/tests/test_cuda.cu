#include <stdio.h>
#include <cuda_runtime.h>

// nvcc tests/test_cuda.cu -o outputs/test_cuda -lcuda -lcudart
// ldd outputs/test_cuda
__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        
        // Launch kernel
        hello_cuda<<<1, 5>>>();
        cudaDeviceSynchronize();
    }
    
    return 0;
}