#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#include "common.h"

int main()  {
    int lim = 1<<14;
    int i;
    bool failed = false;
    dataType *data = new dataType[lim];
    srand(time(NULL));

    for(int k = 0; k < 15; ++k)
    {
        int n = (rand() % lim) + 1;
        printf("Sorting n = %d numbers.. \t", n);
        #pragma omp parallel firstprivate(data)
        {
            srand(time(NULL) ^ omp_get_thread_num());
            unsigned int seed = rand();
            #pragma omp for
            for(i = 0; i < n; ++i)  {
                data[i].key = (long long)randull(&seed);
            }
        }
        failed = false;

        set<long long> ints;
        for(i = 0; i < n; ++i)
            ints.insert(data[i].key);
        
        pSort(data, n, QUICK);
        for(i = 0; i < n-1; ++i)
            if(data[i].key > data[i+1].key) {
                printf("(Unordered) ");
                failed = true;
                break;
            }

        if(!failed) {
            set<long long> rets;
            for(i = 0; i < n; ++i)  {
                rets.insert(data[i].key);
            }

            vector<long long> v;
            set_symmetric_difference(
                ints.begin(), ints.end(),
                rets.begin(), rets.end(),
                std::back_inserter(v));
            if(v.size() || ints.size() != rets.size())    {
                printf("(Numbers changed) ");
                failed = true;
            }
        }

        if(failed)  {
            printf("Sort failed!\n");
            #ifdef DEBUG
            for(i = 0; i < n; ++i)
                if(i > 0 && data[i].key < data[i-1].key)
                    printf("%lld <---\n", data[i].key);
                else
                    printf("%lld\n", data[i].key);
            #endif
            return -1;
        }

        printf("Sort succesful.\n");
    }
    return 0;
}
