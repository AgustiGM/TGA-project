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
/**__global__ void matMult(int N, int M, int P, float *A, float *B, float *C)
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
}**/

__global__ void sigmoid(int N, float *input, float *output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < N; i += blockDim.x * gridDim.x)
  {
    output[i] = 1.0f / (1.0f + exp(-1.0f * input[i]));
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

<<<<<<< HEAD
__global__ void costFunction(int N, int nFeatures, float *Z, float *Y, float *odata){

  __shared__ float tmpd[N];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  tmpd[tid] = (Y[idx] * logf(Z[idx])) + ((1 - Y[idx]) * logf(1 - Z[idx]));

  // Synchronize threads within the block
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // desenrrollamos el ultimo warp activo
  if (tid < 32) {
    volatile double *smem = sdata;

    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) odata[blockIdx.x] = sdata[0]/nFeatures;
}

__global__ void dotProd(int N, float *vec1, float *vec2, float *res){
  __shared__ float tmpd[N];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Perform the dot product calculation
  tmpd[tid] = vec1[idx] * vec2[idx];

  // Synchronize threads within the block
  __syncthreads();

  // Perform parallel reduction
  for (int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      tmpd[tid] += tmpd[tid + s];
    __syncthreads();
  }
  // desenrrollamos el ultimo warp activo
  if (tid < 32) {
    volatile double *smem = tmpd;

    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
  }
  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) res[blkId] = tmpd[0];
}
__global__ int derivative(float Z){
    //if (Z > 0) return 1; 
    //else return 0;
}

__device__ void transposeMatrix(float *A, float *B, int row, int col){


  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(idx < col && idy < row){
    B[idy + row*idx] = A[idx + col*idy];
  }
}
__global__ void updateLayers(){
    /*
    Updates weight, biases suing gradient descendent method using backprop
    Given:
      -derivative w
      -derivative b
      -alpha learning rate (own set)
    Output:
      Updates current layer's weights and biases 
    */

    //for each layer:
    //W_i = W_i - aplha * dW_i
    //b_i = b_i - alpha * db1
}
__global__ void backprop(int nFeatures, int batchSize, int nHiddenLayer, int nOutput, int nLayers,
                        float *hiddenWeights, float *outputWeights, float *actL1, float *actL2, float *Y,
                        float *dZ1, float dZ2, float dW1, float *dW2, float *db1, float *db2) {
    /*
    Given:
      - number of input data m
      - z that is z = w*a + b
      - w weight vector
      - b biases vector
    
    Output
      - derivative z respective to error
      - derivative w respective to error
      - derivative b respective to error
    */

    // the following lanes must be done trhough every layer
    //dZ[last_column] 2 = A[last_column] - Y
    //dW[last_column] 2= 1 / m * dZ[last_column] * transpose(A[last_column - 1])
    //db[last_column] 2= 1 / m * sum(dZ[last_column])
    
    //dZ[last_column - 1] 1= trasnpose(W[last_column]) *(dot product) dZ[last_column] * derivative(Z[last_column-1])
    //dW[last_column - 1] 1= 1 / m * dZ1 *(dot prod) transpose(X)
    //db[last_column - 1] 1= 1 / m * sum(dZ[last_column - 1])

    __shared__ float tmpdZ2[nOutput]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batchSize){
      
      int l2 = nLayers - 1;
      //Derivative Z output layer Out[2]
      for(int i = 0; i < nOutput; ++i){
        dZ2[idx*nOutput + i] = actL2[idx*nOutput + i] - Y[idx*nOutput + i];
      }
      //Derivative W
      for(int i = 0; i < nOutput; ++i){
        //dW[idx*nOutput + i] = 1/nFeatures * dZ[idx + nOutput*layer2]
        transpose(actL1, res);
        matMult(N, M, P, dZ, res, c);
        dW[idx*nOutput + i] = c;
      }
      //Derivative b
      
    }
=======
__global__ void backprop(int N, float *A)
{
}

__device__ float localSigmoid(float x)
{
  return 1 / (1 + exp(-x));
>>>>>>> simple-nn
}

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                            float *input, float *weights, float *weightsOutput, float *activationL1, float *result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < batchSize)
  {

<<<<<<< HEAD
}

=======
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

__device__ void softmax(int nOutput, int batchSize, float *input)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < batchSize)
  {
    float maxVal = input[tid * nOutput];
    for (int i = 1; i < nOutput; i++)
    {
      maxVal = max(maxVal, input[tid * nOutput + i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < nOutput; i++)
    {
      input[tid * nOutput + i] = exp(input[tid * nOutput + i] - maxVal);
      sum += input[tid * nOutput + i];
    }

    for (int i = 0; i < nOutput; i++)
    {
      input[tid * nOutput + i] /= sum;
    }
  }
}

__global__ void globalSoftmax(int nOutput, int batchSize, float *input)
{
  softmax(nOutput, batchSize, input);
}

__global__ void optimizedForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                                     float *input,
                                     float *weights, float * Z1, float *activationL1,
                                     float * weightsOutput, float *Z2, float *result)
{
  int bx = blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float activationL1_s[];

  for (int i = tid; i < nHiddenLayer; i += blockDim.x)
  {
    float hiddenSum = 0.0f;
    for (int j = 0; j < nFeatures; j++)
    {
      hiddenSum += input[j * batchSize + bx] * weights[i * nFeatures + j];
    }
    Z1[bx * nHiddenLayer + i] = hiddenSum;
    activationL1_s[i] = localReLU(hiddenSum);
  }
  __syncthreads();

  for (int c = tid; c < nOutput; c += blockDim.x)
  {
    float sum = 0.0f;

    for (int i = 0; i < nHiddenLayer; i++)
    {
      sum += activationL1_s[i] * weightsOutput[i * nOutput + c];
    }
    Z2[bx * nOutput + c] = sum;
    result[bx * nOutput + c] = exp(sum);
  }
  // __syncthreads();
  float totalSum = 0.0f;
  for (int c = 0; c < nOutput; ++c)
  {
    totalSum += result[bx * nOutput + c];
  }

  // __syncthreads();
  for (int c = tid; c < nOutput; c += blockDim.x)
  {
    result[bx * nOutput + c] /= totalSum;
  }
  for (int i = tid; i < nHiddenLayer; i += blockDim.x)
  {
    activationL1[bx * nHiddenLayer + i] = activationL1_s[i];
  }
}

__global__ void categoricalCrossEntropy(int nOutput, int batchSize, float *groundTruth, float *predictions, float *loss)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < batchSize)
  {
    float example_loss = 0.0f;

    for (int c = 0; c < nOutput; c++)
    {
      example_loss -= groundTruth[tid * nOutput + c] * log(predictions[tid * nOutput + c]);
    }

    loss[tid] = example_loss;
  }
}

__global__ void backpropagation(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                                     float *Z1, float *activationL1,
                                     float *Z2, float *result, float * weightsOutput,
                                     float *input, float *labels,
                                     float* dZ2, float* dW2, float* dZ1, float* dW1) {
  int bx = blockIdx.x;
  int tid = threadIdx.x;
  for (int i = tid; i < nOutput; i += blockDim.x)
  {
    dZ2[bx * nOutput + i] = result[bx * nOutput + i] - labels[bx * nOutput + i];
  }
  // dW2 = 1/m + dZ2 * activationL1.transpose()
  for (int i = tid; i < nHiddenLayer; i += blockDim.x)
  {
    float sum = 0.0f;
    for (int c = 0; c < nOutput; c++)
    {
      sum += dZ2[bx * nOutput + c] * activationL1[c*nHiddenLayer + i];
    }
    dW2[bx * nHiddenLayer + i] = sum / batchSize;
  }

  // dZ1 = weightsOutput.transpose() * dZ2 * localReLU'(Z1)
  for (int i = tid; i < nHiddenLayer; i += blockDim.x)
  {
    float sum = 0.0f;
    for (int c = 0; c < nOutput; c++)
    {
      sum += weightsOutput[c * nOutput + i] * dZ2[bx * nOutput + c];
    }
    dZ1[bx * nHiddenLayer + i] = sum * localReLUPrime(Z1[bx * nHiddenLayer + i]);
  }

  //dW1 = 1/m * dZ1 * input.transpose()
  for (int i = tid; i < nFeatures; i += blockDim.x)
  {
    float sum = 0.0f;
    for (int c = 0; c < nHiddenLayer; c++)
    {
      sum += dZ1[bx * nHiddenLayer + c] * input[c * nFeatures + i];
    }
    dW1[bx * nFeatures + i] = sum / nOutput;
  }
}

__device__ float localReLU(float x)
{
  return max(0.0f, x);
}

__device__ float localReLUPrime(float x)
{
  return x > 0.0f ? 1.0f : 0.0f;
}
>>>>>>> simple-nn
