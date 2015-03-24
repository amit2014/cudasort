#include <stdlib.h>
#include "common.h"

int num_procs, for_ser_n;

long long randull(unsigned int *seed)   {
    return ((long long)rand_r(seed) << ((sizeof(int) * 8 - 1) * 2)) | 
           ((long long)rand_r(seed) << ((sizeof(int) * 8 - 1) * 1)) |
           ((long long)rand_r(seed) << ((sizeof(int) * 8 - 1) * 0));
}

void serialSort(dataType *data, int n)  {
    
}
