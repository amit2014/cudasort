#include <stdio.h>
using namespace std;

#include "sort.h"
#include "common.h"

void mSort(dataType *data, int n);
void qSort(dataType *data, int n);
void rSort(dataType *data, int n);
void bSort(dataType *data, int n);

void pSort(dataType *data, int ndata, SortType sorter)	{
	switch(sorter)	{
		case BEST:  qSort(data, ndata); break;
		case MERGE: mSort(data, ndata); break;
		case QUICK: qSort(data, ndata); break;
                case RADIX: rSort(data, ndata); break;

		default: fprintf(stderr, "Not implemented.");
	}
}
