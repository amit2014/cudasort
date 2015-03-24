#include <stdio.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
  int i, nDevices;
  FILE *fp;
  fp = fopen("/tmp/cs1110298_sjob.out", "w");

  cudaGetDeviceCount(&nDevices);
  for (i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    fprintf(fp, "Device Number: %d\n", i);
    fprintf(fp, "  Device name: %s\n", prop.name);
    fprintf(fp, "  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    fprintf(fp, "  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    fprintf(fp, "  Peak Memory Bandwidth (GB/s): %.2f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    fprintf(fp, "  Total Global Mem (KB): %.2f\n", prop.totalGlobalMem/1024.0);
    fprintf(fp, "  Shared Mem per Block (KB): %.2f\n", prop.sharedMemPerBlock/1024.0);
    fprintf(fp, "  Warp Size: %d\n", prop.warpSize);
    fprintf(fp, "  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    fprintf(fp, "  Cuda version: %d.%d\n", prop.major, prop.minor);
    fprintf(fp, "  MultiProcessor count: %d\n", prop.multiProcessorCount);
    fprintf(fp, "  L2 Cache Size (KB): %.2f\n", prop.l2CacheSize/1024.0);
    fprintf(fp, "  Max Threads per MultiProcessor: %d\n\n", prop.maxThreadsPerMultiProcessor);
  }
  fclose(fp);
}
