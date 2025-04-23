#include "utils.cuh"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void dataGenerator(int* data, int count, int first, int step) {
    for (int i = 0; i < count; ++i)
        data[i] = first + i * step;
    srand(time(NULL));
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % i;
        int tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

__device__ uint bfe(uint x, uint start, uint nbits) {
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}