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

__global__ void costFunction(int N, int nFeatures, float *Z, float *Y, float *odata){
  /*
  Given the output from forward stream, computes the prediction for the input. 
  */

  //Store the parcial cost for each row:
  __shared__ double C[N];

  volatile double *sumC = C;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
    sumC[idx] = (Y[idx] * logf(Z[idx])) + ((1 - Y[idx]) * logf(1 - Z[idx]));
  }
  __syncthreads();

  //Sum of the parical cost
  __shared__ double sdata[N];
  unsigned int s;

  // Cada thread realiza la suma parcial de los datos que le
  // corresponden y la deja en la memoria compartida
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int gridSize = blockDim.x*2*gridDim.x;
  sdata[tid] = 0;
  while (i < N) {
    sdata[tid] += C[i] + C[i+blockDim.x];
    i += gridSize;
  }
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>32; s>>=1) {
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


__global__ int derivative(float Z){
    //if (Z > 0) return 1; 
    //else return 0;
}

__device__ void transposeMatrix(float *inMat, float *outMat, int sizex, int sizey){


  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i < sizex && j < sizey){
    int idxin = i + sizex * j;
    int idxout = j + sizey * i;
    outMat[idxout] = inMat[idxin];
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
__global__ void backprop(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                        float *hiddenWeights, float *outputWeights, float *activationL1, float *activationL2, float *Y,
                        float *dZ, float *dW, float *db) {
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
    //dW[last_column] 2= 1 / m * dZ[last_column] *(dot product) transpose(A[last_column - 1])
    //db[last_column] 2= 1 / m * sum(dZ[last_column])
    
    //dZ[last_column - 1] 1= trasnpose(W[last_column]) *(dot product) dZ[last_column] * derivative(Z[last_column-1])
    //dW[last_column - 1] 1= 1 / m * dZ1 *(dot prod) transpose(X)
    //db[last_column - 1] 1= 1 / m * sum(dZ[last_column - 1])
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < batchSize){
      // Compute the derivatives of the last layer
      /*for(int i = nHiddenLayer - 1; i >= 0 ; ++i){
        dZ[tid + nOutput*i] = activationL2[tid + nOutput*i] - Y[tid + nOutput*i];
      }*/
      int layer2 = nHiddenLayer - 1;
      //Derivative Z output layer
      dZ[tid + nOutput*layer2] = activationL2[tid + nOutput*layer2] - Y[tid + nOutput*layer2];

      //Derivative W
      
      for(int j = 0; j < nOutput; ++j){
        dW[tid + nFeatures*nOutput + j] = 1/batchSize * dZ[tid + nOutput*layer2] 
      }
      
    }
}

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                            float *input, float *hiddenWeights, float *activationL1, float *outputWeights, float *result) {

  // matMult(batchSize, nHiddenLayer, nFeatures, input, hiddenWeights, activationL1);
  // sigmoid(batchSize * nHiddenLayer, activationL1);
  // matMult(batchSize, nOutput, nHiddenLayer, activationL1, outputWeights, result);
  // sigmoid(batchSize * nHiddenLayer, result);

}

