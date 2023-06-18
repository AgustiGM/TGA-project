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

    // if (row < N && col < M && tmp != 0)
    //     printf("row: %d, col: %d, tmp: %f\n", row, col, tmp);

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batchSize) {
        // each column is the result for an input
        float max = input[tid];
        for (int i = 1; i < nOutput; i++) {
            if (input[i * batchSize + tid] > max) {
                max = input[i * batchSize + tid];
            }
        }
        float sum = 0.0;
        for (int i = 0; i < nOutput; i++) {
            sum += expf(input[i * batchSize + tid] - max);
        }
        for (int i = 0; i < nOutput; i++) {
            output[i * batchSize + tid] = expf(input[i * batchSize + tid] - max) / sum;
        }
    }

}

__global__ void elementWiseProd(int N, int M, float *A, float *B, float *C) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < N * M; i += blockDim.x * gridDim.x) {
        C[i] = A[i] * B[i];
    }
}

__global__ void subtractMat(int N, int M, float *A, float *B, float *C) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < N * M; i += blockDim.x * gridDim.x) {
        C[i] = A[i] - B[i];
    }
}

__global__ void scalarDivMat(int N, int M, float value, float *A, float *C) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < N * M; i += blockDim.x * gridDim.x) {
        C[i] = A[i] / value;
    }
}

__global__ void scalarProdMat(int N, int M, float value, float *A, float *C) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < N * M; i += blockDim.x * gridDim.x) {
        C[i] = A[i] * value;
    }
}


__global__ void derivativeReLu(int N, int M, float *A, float *C){

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < N * M; i += blockDim.x * gridDim.x) {
        if (A[i] > 0) {
            C[i] = 1.0f;
        } else {
            C[i] = 0.0f;
        }
    }
    
}

__global__ void accuracy(int batchSize, int nOutput, float *predictions, float *labels, float *d_accuracy) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    *d_accuracy = 0.0f;
    for (int i = tid; i < batchSize; i += blockDim.x * gridDim.x) {
        // find the max idx
        int maxIdx = 0;
        float max = predictions[i*nOutput];
        for (int j = 1; j < nOutput; j++) {
            if (predictions[i * nOutput + j] > max) {
                max = predictions[i * nOutput + j];
                maxIdx = j;
            }
        }
        if (labels[i * nOutput + maxIdx] != 0.0f) {
            atomicAdd(d_accuracy, 1.0f);
        }
    }
    __syncthreads();
    if (tid == 0){
        // printf("accuracy: %f\n", *d_accuracy);
        *d_accuracy /= (float)batchSize;
    }
}