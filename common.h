#ifndef _COMMON_H
#define _COMMON_H

#include "sort.h"

#ifdef DEBUG
#define _DEBUG_DEBUG 1
#else
#define _DEBUG_DEBUG 0
#endif

#define Dprintf(fmt, ...) \
        do { if (_DEBUG_DEBUG) \
                fprintf(stderr, "%s:%d: %s(): " fmt, \
                    __FILE__, \
                    __LINE__, \
                    __func__, \
                    ##__VA_ARGS__);\
        } while (0)

#define NDprintf(...) \
        do { if (!_DEBUG_DEBUG) \
                fprintf(stderr, ##__VA_ARGS__);\
        } while (0)

long long randull(unsigned int *seed);
void psum(int *data, int n, int *data2);
__device__ void serialHeapSort(dataType* array, int size);
__device__ void merge(dataType *data, int n1, int n2, dataType *res);
__device__ void bottomUpMergeSort(dataType *data, int n, dataType *res);

#endif // _COMMON_H
