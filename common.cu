#include <stdlib.h>
#include <queue>
#include "common.h"
using namespace std;

long long randull(unsigned int *seed)   {
    return ((long long)rand_r(seed) << ((sizeof(int) * 8 - 1) * 2)) | 
           ((long long)rand_r(seed) << ((sizeof(int) * 8 - 1) * 1)) |
           ((long long)rand_r(seed) << ((sizeof(int) * 8 - 1) * 0));
}

__device__ void Heapify(dataType* A, int i, int heapsize)
{
    dataType tmp;
    while(true)
    {
        int l = 2*i+1;
        int r = 2*i+2;

        int largest = (l < heapsize && A[i].key < A[l].key) ? l : i;

        if(r < heapsize && A[largest].key < A[r].key)
            largest = r;

        if(largest != i)
        {
            tmp = A[i];
            A[i] = A[largest];
            A[largest] = tmp;
            i = largest;
        }
        else break;
    }
}

__device__ void serialHeapSort(dataType* array, int size)
{
    for(int i = size/2; i > 0;) Heapify(array, --i, size);

    dataType tmp;
    for(int i = size-1; i>0; --i)
    {
        tmp = array[0];
        array[0] = array[i];
        array[i] = tmp;
        Heapify(array, 0, i);
    }
}

// merges A[0..n1-1] with A[n1..n2-1]
__device__ void merge(dataType *data, int n1, int n2, dataType *res) {
    int i = 0, j = n1, k = 0;

    while(i < n1 && j < n2)
        if(data[i].key < data[j].key)
            res[k++] = data[i++];
        else
            res[k++] = data[j++];
    
    while(i < n1)
        res[k++] = data[i++];
    while(j < n2)
        res[k++] = data[j++];
}

// assume res has same data as data
__device__ void bottomUpMergeSort(dataType *data, int n, dataType *res)  {
    dataType *tmp;

    // if we are going odd-many levels, swap res&data
    int c = 0;
    for(int s = 1; s < n; s*=2)
        c++;
    if(c&1) {
        tmp = data;
        data = res;
        res = tmp;
    }

    for(int s = 1; s < n; s*=2) {
        for(int i = 0; i+s < n; i+=2*s)  {
            merge(data+i, s, min(i+2*s, n), res+i);
        }
        // swap the roles of data and res
        tmp = data;
        data = res;
        res = tmp;
    }
    // now *data is same as original *data, and sorted
}
