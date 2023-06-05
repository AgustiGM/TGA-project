#include <math.h>

__global__ void matMult(int N, int M, int P, float *A, float *B, float *C);

__global__ void transpose(int N, int M, float *input, float *output);

__global__ void globalSoftmaxPrimitive(int nOutput, int batchSize, float *input, float *output);

__global__ void elementWiseProd(int N, int M, float *A, float *B, float *C);

__global__ void subtractMat(int N, int M, float *A, float *B, float *C);

__global__ void scalarDivMat(int N, int M, float value, float *A, float *C);

__global__ void scalarProdMat(int N, int M, float value, float *A, float *C);

__global__ void derivativeReLu(int N, int M, float *A, float *C);