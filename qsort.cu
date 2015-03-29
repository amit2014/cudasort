#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "sort.h"
#include "common.h"

// generate random long in [a,b)
__device__ long long range_rand(curandState *s, long long a, long long b)    {
    return (1.0f-curand_uniform(s)) * (b-a) + a;
}

__device__ int serialPartition(curandState *s, dataType *data, int n, dataType *buf)  {
    int i, j;
    i = range_rand(s, 0, n);
    dataType tmp = data[i];
    data[i] = data[0];
    data[0] = tmp;

    long long pivot = data[0].key;
    i = 0; j = n-1;
    for(int k = 1; k < n; ++k)
        if(data[k].key < pivot)
            buf[i++] = data[k];
        else
            buf[j--] = data[k];

    buf[j] = data[0];

    for(int k = 0; k < n; ++k)
        data[k] = buf[k];
    return j;
}

__device__ int serialInPlacePartition(curandState *s, dataType *data, int n) {
    int i, j;
    dataType tmp;
    i = range_rand(s, 0, n);
    tmp = data[i];
    data[i] = data[0];
    data[0] = tmp;

    long long pivot = data[0].key;
    i = 1; j = n-1;
    while(i <= j)   {
        tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;

        while(i <= j && data[i].key <= pivot)
                i++;
        while(i <= j && pivot < data[j].key)
                j--;
    }
    tmp = data[0];
    data[0] = data[j];
    data[j] = tmp;

    return j;
}

__global__ void qSortKernel(dataType *data, int n, dataType *buf, int *start, int *end)   {
    curandState s;
    curand_init(clock64(), threadIdx.x, 0, &s);

    int a = blockDim.x << 1; // must be a power of 2
    if(threadIdx.x == 0)    {
        start[0] = 0;
        end[0] = n;
    }
    while((a >>= 1) > 1)    {
        if(threadIdx.x % a == 0)    {
            int left  = start[threadIdx.x],
                right = end[threadIdx.x];
            if(right - left > 1)   {
                int i = serialInPlacePartition(&s, data + left,
                                                    right-left);
                i += left;
                end[threadIdx.x + a/2] = right;
                end[threadIdx.x] = i;
                start[threadIdx.x + a/2] = i+1;
            }
            else    {
                end[threadIdx.x + a/2] = 0;
                start[threadIdx.x + a/2] = 0;
            }
        }
        __syncthreads();
    }
    if(end[threadIdx.x]-start[threadIdx.x] > 1) {
        for(int i = start[threadIdx.x]; i < end[threadIdx.x]; ++i)
            buf[i] = data[i];
        bottomUpMergeSort(data + start[threadIdx.x], end[threadIdx.x]-start[threadIdx.x],
                            buf + start[threadIdx.x]);
    }
}

void qSort(dataType *data, int n)   {
    dataType *buf1, *buf2;
    int *start, *end;
    int nthreads = 32*32;  // must be a power of 2

    cudaMalloc((void**)&buf1,  n*sizeof(dataType));
    cudaMalloc((void**)&buf2,  n*sizeof(dataType));
    cudaMalloc((void**)&start, nthreads*sizeof(int));
    cudaMalloc((void**)&end,   nthreads*sizeof(int));

    cudaMemcpy(buf1, data, n*sizeof(dataType), cudaMemcpyHostToDevice);

    qSortKernel <<< 1, nthreads >>> (buf1, n, buf2, start, end);

    cudaMemcpy(data, buf1, n*sizeof(dataType), cudaMemcpyDeviceToHost);
    cudaFree(buf1);
    cudaFree(buf2);
    cudaFree(start);
    cudaFree(end);
}
