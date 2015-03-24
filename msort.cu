#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "sort.h"
#include "common.h"

__device__ void mSort_helper(dataType *data, int n, dataType *res)   {
    // printf("Thread %d\n", omp_get_thread_num());
    if(n == 1)  {
        res[0] = data[0];
        return;
    }
    if(n <= 0) {
        return;
    }
    
    mSort_helper(res, n/2, data);
 
    mSort_helper(res+n/2, n-n/2, data+n/2);

    merge(data, n/2, n, res);
}

__global__ void mSortKernel(dataType *data, int n, dataType *res)  {
    bottomUpMergeSort(data, n, res);
    // mSort_helper(res, n, data);
}

void mSort(dataType *data, int n)    {
    dataType *buf1, *buf2;
    cudaMalloc((void**)&buf1, n*sizeof(dataType));
    cudaMalloc((void**)&buf2, n*sizeof(dataType));

    cudaMemcpy(buf1, data, n*sizeof(dataType), cudaMemcpyHostToDevice);
    cudaMemcpy(buf2, data, n*sizeof(dataType), cudaMemcpyHostToDevice);
    //printf("%d threads\n", omp_get_num_threads());
    mSortKernel <<< 1,2 >>> (buf1, n, buf2);

    cudaMemcpy(data, buf1, n*sizeof(dataType), cudaMemcpyDeviceToHost);
    cudaFree(buf1);
    cudaFree(buf2);
}

