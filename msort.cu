#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "sort.h"
#include "common.h"

__device__ void serialMSort(dataType *data, int n, dataType *res)   {
    // printf("Thread %d\n", omp_get_thread_num());
    if(n == 1)  {
        res[0] = data[0];
        return;
    }
    if(n <= 0) {
        return;
    }
    
    serialMSort(res, n/2, data);
    serialMSort(res+n/2, n-n/2, data+n/2);
    merge(data, n/2, n, res);
}

// assumes sorted in data and sorts into res if c%2==1, data o/w.
__device__ int warpMerge(dataType *data, int n, dataType *res,
        int i, int &j, int gThreadIdx, int unitSize, int groupSize) {
    int s,c,_j, nthreads = gridDim.x * blockDim.x;

    dataType *tmp;
    for(s = 2, c = 0; (s/2)*unitSize < groupSize; s<<=1, c+=1)  {
        if(threadIdx.x % s == 0)    {
            _j = segend(gThreadIdx + s*unitSize - 1, nthreads, n);
            merge(data+i, j-i, _j-i, res+i);
            j = _j;

            // switch the roles of tmp and data
            tmp = data;
            data = res;
            res = tmp;
        }
    }
    return c;
}

__device__ int blocks = 0;

__global__ void mSortKernel(dataType *data, int n, dataType *res)  {
    if(gridDim.x > 16 || gridDim.x == 2 || (gridDim.x > 4 && gridDim.x <= 8)) {
        dataType *tmp = data;
        data = res;
        res = tmp;
    }

    int nthreads = gridDim.x * blockDim.x;
    int gThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = segstart(gThreadIdx, nthreads, n);
    int j = segend(gThreadIdx, nthreads, n);

    // sort segment in data
    serialMSort(res+i, j-i, data+i);
        // bottomUpMergeSort(data, n, res);
    
    warpMerge(data, n, res, i, j, gThreadIdx, 1, 32);
    // now we have res[i..j-1] sorted
    __syncthreads();
    // everyone done with their warp.
    // we have a maximum of 1024 threads per block => max 32 warps per block
    // => 1 warp alone can take charge
    
    if(threadIdx.x < 32)   {
        gThreadIdx = blockIdx.x * blockDim.x + threadIdx.x * 32;
        i = segstart(gThreadIdx,  nthreads, n);
        j = segend(gThreadIdx+31, nthreads, n);

        int c = warpMerge(res, n, data, i, j, gThreadIdx, 32, blockDim.x);
        // now we have data[i..j-1] sorted if blockDim was odd power of 2 (c%2==1), res o/w
    }
    // done with the block.

    __shared__ bool lastblock;
    if(threadIdx.x == 0)    {
        lastblock = false;
        if(atomicAdd(&blocks, 1) == gridDim.x-1) {
            // every block done.
            lastblock = true;
        }
    }
    __syncthreads();
    if(lastblock && gridDim.x > 1)   {
        // also, we have only 13 MPs, and 1024*2 threadss max per SM at a time
        // so let's just handle using 1 warp
        if(threadIdx.x < gridDim.x)    {
            gThreadIdx = threadIdx.x * blockDim.x;
            i = segstart(gThreadIdx, nthreads, n);
            j = segend(gThreadIdx + blockDim.x-1, nthreads, n);
            int c = warpMerge(data, n, res, i, j, gThreadIdx, blockDim.x, nthreads);
            // now we have data[i..j-1] sorted if blockDim was even power of 2 (c%2==0), res o/w
        }
    }
}

void mSort(dataType *data, int n)    {
    dataType *buf1, *buf2;
    cudaMalloc((void**)&buf1, n*sizeof(dataType));
    cudaMalloc((void**)&buf2, n*sizeof(dataType));

    int b = 0, *db;
    cudaGetSymbolAddress((void**)&db, blocks);
    cudaMemcpy(db, &b, sizeof(b), cudaMemcpyHostToDevice);
    cudaMemcpy(buf1, data, n*sizeof(dataType), cudaMemcpyHostToDevice);
    cudaMemcpy(buf2, data, n*sizeof(dataType), cudaMemcpyHostToDevice);

    // gridDim <= 32
    mSortKernel <<< 8, 16*32 >>> (buf1, n, buf2);

    cudaMemcpy(data, buf1, n*sizeof(dataType), cudaMemcpyDeviceToHost);
    cudaFree(buf1);
    cudaFree(buf2);
}

