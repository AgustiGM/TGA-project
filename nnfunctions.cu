#include "nnfunctions.h"

#ifndef SIZE
#define SIZE 32
#endif

__device__ void matMult(int N, int M, int P, float *A, float *B, float *C) {

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

__global__ void globalMatMult(int N, int M, int P, float *A, float *B, float *C) {
    matMult(N,M,P,A,B,C);
}


__device__ void sigmoid(int N, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = 1 / (1 + exp(-1*input[i]));
    }
}

__global__ void globalSigmoid(int N, float *input, float *output) {
  sigmoid(N,input, output);
}

__device__ void reLU(int N, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (input[i] < 0) output[i] = 0;
        else output[i] = input[i];
    }
}

__global__ void globalReLU(int N, float *input, float *output) {
  reLU(N,input,output);
}

__global__ void costFunction(){
    /*
    Given the output from forward stream, computes the prediction for the input. 
    */
}

__global__ void derivative(float Z){
    // if Z > 0 return 1 else return 0
}

__global__ void transposeMatrix(float *odata, const float *idata){

  int size = gridDim.y;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.x + threadIdx.y;

  for (int j = 0; j < blockDim.x; j+= size)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
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
__global__ void backprop(int N, int m, float *A, float *Z, float *W, float *Y, float *dZ, float *dW, float *dB) {
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
    //dZ[last_column] = A[last_column] - Y
    //dW[last_column] = 1 / m * dZ[last_column] *(dot product) transpose(A[last_column - 1])
    //db[last_column] = 1 / m * sum(dZ[last_column])
    
    //dZ[last_column - 1] = trasnpose(W[last_column]) *(dot product) dZ[last_column] * derivative(Z[last_column-1])
    //dW[last_column - 1] = 1 / m * dZ1 *(dot prod) transpose(X)
    //db[last_column - 1] = 1 / m * sum(dZ[last_column - 1])
}

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                            float *input, float *hiddenWeights, float *activationL1, float *outputWeights, float *result) {

  // matMult(batchSize, nHiddenLayer, nFeatures, input, hiddenWeights, activationL1);
  // sigmoid(batchSize * nHiddenLayer, activationL1);
  // matMult(batchSize, nOutput, nHiddenLayer, activationL1, outputWeights, result);
  // sigmoid(batchSize * nHiddenLayer, result);

}

