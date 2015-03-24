#include <stdio.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
  int i, nDevices;

  cudaGetDeviceCount(&nDevices);
  for (i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.2f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total Global Mem (KB): %.2f\n", prop.totalGlobalMem/1024.0);
    printf("  Shared Mem per Block (KB): %.2f\n", prop.sharedMemPerBlock/1024.0);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  MultiProcessor count: %d\n", prop.multiProcessorCount);
    printf("  L2 Cache Size (KB): %.2f\n", prop.l2CacheSize/1024.0);
    printf("  Max Threads per MultiProcessor: %d\n\n", prop.maxThreadsPerMultiProcessor);
  }
}
