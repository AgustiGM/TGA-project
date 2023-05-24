#include "nnfunctions.h"
#include <stdio.h>

#ifndef SIZE
#define SIZE 32
#endif

// struct Layer {
//     int nInput;
//     int nOutput;
//     float *weights;
//     float *biases;
//     float *activations;
// };

// C(N × M) ← A(N × P) · B (P × M)
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


__global__ void sigmoid(int N, float *input, float *output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < N; i += blockDim.x * gridDim.x)
  {
    output[i] = 1 / (1 + exp(-1 * input[i]));

  }
  
}


__device__ void reLU(int N, float *input, float *output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < N; i += blockDim.x * gridDim.x)
  {
    if (input[i] > 0)
    {
      output[i] = input[i];
    }
    else
    {
      output[i] = 0;
    }
  }
}

__global__ void globalReLU(int N, float *input, float *output)
{
  reLU(N, input, output);
}

__global__ void backprop(int N, float *A)
{
}

__device__ float localSigmoid(float x)
{
  return 1 / (1 + exp(-x));
}

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                            float *input, float *weights, float *weightsOutput, float *activationL1, float *result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < batchSize)
  {
    
    // Compute the activations of the hidden layer (Layer 1)
    for (int i = 0; i < nHiddenLayer; i++)
    {
      float hiddenSum = 0.0f;

      // Perform matrix multiplication between transposed input and weights
      for (int j = 0; j < nFeatures; j++)
      {
        hiddenSum += input[j * batchSize + tid] * weights[i * nFeatures + j];
      }

      // Store the activation of the hidden layer
      activationL1[tid * nHiddenLayer + i] = localSigmoid(hiddenSum);
    }

    // Compute the output layer activations
    for (int c = 0; c < nOutput; c++)
    {
      float sum = 0.0f;

      for (int i = 0; i < nHiddenLayer; i++)
      {
        sum += activationL1[tid * nHiddenLayer + i] * weightsOutput[i * nOutput + c];
      }

      // Apply activation function (e.g., sigmoid, ReLU, etc.) to the sum
      result[tid * nOutput + c] = exp(sum);
    }
    // Normalize the result to obtain probabilities using softmax
    float totalSum = 0.0f;
    for (int c = 0; c < nOutput; c++)
    {
      totalSum += result[tid * nOutput + c];
    }

    for (int c = 0; c < nOutput; c++)
    {
      result[tid * nOutput + c] /= totalSum;
    }
  }
}



__global__ void transpose(int N, int M, float *input, float *output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < N * M; i += blockDim.x * gridDim.x)
  {
    int row = i / M;
    int col = i % M;
    
    output[col * N + row] = input[i];
  }
}

__device__ void softmax(int nOutput, int batchSize, float *input) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batchSize) {
        float maxVal = input[tid * nOutput];  
        for (int i = 1; i < nOutput; i++) {
            maxVal = max(maxVal, input[tid * nOutput + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < nOutput; i++) {
            input[tid * nOutput + i] = exp(input[tid * nOutput + i] - maxVal);
            sum += input[tid * nOutput + i];
        }

        for (int i = 0; i < nOutput; i++) {
            input[tid * nOutput + i] /= sum;
        }
    }
}

__global__ void globalSoftmax(int nOutput, int batchSize, float *input)
{
  softmax(nOutput, batchSize, input);
}



__global__ void optimizedForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                            float *input, float *weights, float *weightsOutput, float *activationL1, float *result)
{
  int bx = blockIdx.x;
  int tid = threadIdx.x;
  
  
  // if (tid == 0) printf("blockfim: %d\n", blockDim.x);
  if (bx < batchSize)
  {
    for (int i = tid; i < nHiddenLayer; i += blockDim.x)
    {
      
      float hiddenSum = 0.0f;
      for (int j = 0; j < nFeatures; j++)
      { 
        hiddenSum += input[j * batchSize + bx] * weights[i * nFeatures + j];
      }
    
      activationL1[bx * nHiddenLayer + i] = localSigmoid(hiddenSum);
    }
    __syncthreads();
    for (int c = tid; c < nOutput; c += blockDim.x)
    {
      float sum = 0.0f;

      for (int i = 0; i < nHiddenLayer; i++)
      {
        sum += activationL1[bx * nHiddenLayer + i] * weightsOutput[i * nOutput + c];
      }
      result[bx * nOutput + c] = exp(sum);
    }
    __syncthreads();
    float totalSum = 0.0f;
    for (int c = 0; c < nOutput; ++c)
    {
      totalSum += result[bx * nOutput + c];
    }
    
    __syncthreads();
    for (int c = tid; c < nOutput; c += blockDim.x)
    {
      result[bx * nOutput + c] /= totalSum;
    }
  }
}
