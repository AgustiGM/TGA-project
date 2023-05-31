#include <math.h>

__global__ void matMult(int N, int M, int P, float *A, float *B, float *C);

__global__ void transpose(int N, int M, float *input, float *output);

__global__ void globalSoftmaxPrimitive(int nOutput, int batchSize, float *input, float *output);