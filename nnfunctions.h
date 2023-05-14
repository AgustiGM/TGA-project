#include <math.h>

__global__ void matMult(int N, int M, int P, float *A, float *B, float *C);

__global__ void sigmoid(int N, float *A);

__global__ void reLU(int N, float *A);

__global__ void backprop(int N, float *A);

