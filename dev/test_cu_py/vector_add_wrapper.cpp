#include <cuda_runtime.h>

extern "C" {
    void launchVectorAdd(const float *A, const float *B, float *C, int numElements);
}