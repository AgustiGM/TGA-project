#include "primitives.h"
#include <math.h>
// C(N × M) ← A(N × P) · B (P × M)

#define SIZE 32

__global__ void matMult(int N, int M, int P, float *A, float *B, float *C)
{

    __shared__ float sA[SIZE][SIZE];
    __shared__ float sB[SIZE][SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * SIZE + ty;
    int col = bx * SIZE + tx;
    int k, m;

    float tmp = 0.0;
    for (m = 0; m < P - SIZE; m = m + SIZE)
    {
        if (row < N)
            sA[ty][tx] = A[row * P + m + tx];
        if (col < M)
            sB[ty][tx] = B[col + (m + ty) * M];
        __syncthreads();

        for (k = 0; k < SIZE; k++)
            tmp += sA[ty][k] * sB[k][tx];

        __syncthreads();
    }
    if (row < N)
        sA[ty][tx] = A[row * P + m + tx];
    if (col < M)
        sB[ty][tx] = B[col + (m + ty) * M];
    __syncthreads();
    for (k = 0; m < P; k++, m++)
        tmp += sA[ty][k] * sB[k][tx];

    if (row < N && col < M)
        C[row * M + col] = tmp;
}

__global__ void transpose(int N, int M, float *input, float *output)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
    {
        for (int j = 0; j < M; j++)
        {
            output[j * N + i] = input[i * M + j];
        }
    }
}

__global__ void globalSoftmaxPrimitive(int nOutput, int batchSize, float *input, float *output) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;

    if (bx < batchSize) {
        float max = input[bx * nOutput];
        for (int i = 1; i < nOutput; i++) {
            if (input[bx * nOutput + i] > max) {
                max = input[bx * nOutput + i];
            }
        }
        float sum = 0.0;
        for (int i = 0; i < nOutput; i++) {
            sum += expf(input[bx * nOutput + i] - max);
        }
        for (int i = 0; i < nOutput; i++) {
            output[bx * nOutput + i] = expf(input[bx * nOutput + i] - max) / sum;
        }
    }

}