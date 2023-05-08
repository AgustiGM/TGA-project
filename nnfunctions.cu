#include "nnfunctions.h"

#ifndef SIZE
#define SIZE 32
#endif

__global__ void matMult(int N, int M, int P, float *A, float *B, float *C) {

  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;
  int k, m;

  float tmp = 0.0;
  for (m=0; m < P-SIZE; m=m+SIZE) {
    if (row<N) sA[ty][tx] = A[row*P + m + tx];
    if (col<M) sB[ty][tx] = B[col + (m + ty)*M];
    __syncthreads();

    for (k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];

    __syncthreads();
  }
  if (row<N) sA[ty][tx] = A[row*P + m + tx];
  if (col<M) sB[ty][tx] = B[col + (m + ty)*M];
  __syncthreads();
  for (k=0; m<P; k++, m++)
    tmp += sA[ty][k] * sB[k][tx];

  if (row<N && col<M) C[row*M+col] = tmp;

}


__global__ void sigmoid(int N, float *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = 1 / (1 + exp(-1*A[i]));
    }

}

__global__ void reLU(int N, float *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (A[i] < 0) A[i] = 0;
    }

}

__global__ void backprop(int N, float *A) {

}