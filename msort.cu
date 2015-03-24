#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "sort.h"

const int ser_n = 1<<9;

__device__ void merge(dataType *data, int n1, int n2, dataType *res) {
    // printf("Thread %d doing merge.\n", omp_get_thread_num());
    int i = 0, j = n1, k = 0;

    while(i < n1 && j < n2)
        if((long long)data[i].key < (long long)data[j].key)
            res[k++] = data[i++];
        else
            res[k++] = data[j++];
    
    while(i < n1)
        res[k++] = data[i++];
    while(j < n2)
        res[k++] = data[j++];
}

__global__ void mSortKernel(dataType *data, int n, dataType *res)  {
    
}

void mSort(dataType *data, int n)    {
    dataType *buf1, *buf2;
    cudaMalloc((void**)&buf1, n*sizeof(dataType));
    cudaMalloc((void**)&buf2, n*sizeof(dataType));

    cudaMemcpy(buf1, data, n*sizeof(dataType), cudaMemcpyHostToDevice);
    cudaMemcpy(buf2, data, n*sizeof(dataType), cudaMemcpyHostToDevice);
    //printf("%d threads\n", omp_get_num_threads());
    mSortKernel <<< 2,2 >>> (buf1, n, buf2);

    cudaMemcpy(data, buf2, n*sizeof(dataType), cudaMemcpyDeviceToHost);
    cudaFree(buf1);
    cudaFree(buf2);
}
