// #include "nnfunctions.h"

// #ifndef SIZE
// #define SIZE 32
// #endif

// __device__ void matMult(int N, int M, int P, float *A, float *B, float *C) {

//   __shared__ float sA[SIZE][SIZE];
//   __shared__ float sB[SIZE][SIZE];

//   int bx = blockIdx.x;  int by = blockIdx.y;
//   int tx = threadIdx.x; int ty = threadIdx.y;
//   int row = by * SIZE + ty;
//   int col = bx * SIZE + tx;
//   int k, m;

//   float tmp = 0.0;
//   for (m=0; m < P-SIZE; m=m+SIZE) {
//     if (row<N) sA[ty][tx] = A[row*P + m + tx];
//     if (col<M) sB[ty][tx] = B[col + (m + ty)*M];
//     __syncthreads();

//     for (k=0; k<SIZE; k++)
//       tmp += sA[ty][k] * sB[k][tx];

//     __syncthreads();
//   }
//   if (row<N) sA[ty][tx] = A[row*P + m + tx];
//   if (col<M) sB[ty][tx] = B[col + (m + ty)*M];
//   __syncthreads();
//   for (k=0; m<P; k++, m++)
//     tmp += sA[ty][k] * sB[k][tx];

//   if (row<N && col<M) C[row*M+col] = tmp;

// }

// __global__ void globalMatMult(int N, int M, int P, float *A, float *B, float *C) {
//     matMult(N,M,P,A,B,C);
// }


// __device__ void sigmoid(int N, float *input, float *output) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         output[i] = 1 / (1 + exp(-1*input[i]));
//     }
// }

// __global__ void globalSigmoid(int N, float *input, float *output) {
//   sigmoid(N,input, output);
// }

// __device__ void reLU(int N, float *input, float *output) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         if (input[i] < 0) output[i] = 0;
//         else output[i] = input[i];
//     }
// }

// __global__ void globalReLU(int N, float *input, float *output) {
//   reLU(N,input,output);
// }

// __global__ void costFunction(int N, int nFeatures, float *Z, float *Y, float *odata){

//   __shared__ float tmpd[N];
//   int tid = threadIdx.x;
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;

//   tmpd[tid] = (Y[idx] * logf(Z[idx])) + ((1 - Y[idx]) * logf(1 - Z[idx]));

//   // Synchronize threads within the block
//   __syncthreads();

//   // Hacemos la reduccion en la memoria compartida
//   for (int s=blockDim.x/2; s>32; s>>=1) {
//     if (tid < s)
//       sdata[tid] += sdata[tid + s];
//     __syncthreads();
//   }
//   // desenrrollamos el ultimo warp activo
//   if (tid < 32) {
//     volatile double *smem = sdata;

//     smem[tid] += smem[tid + 32];
//     smem[tid] += smem[tid + 16];
//     smem[tid] += smem[tid + 8];
//     smem[tid] += smem[tid + 4];
//     smem[tid] += smem[tid + 2];
//     smem[tid] += smem[tid + 1];
//   }


//   // El thread 0 escribe el resultado de este bloque en la memoria global
//   if (tid == 0) odata[blockIdx.x] = sdata[0]/nFeatures;
// }

// __global__ void dotProd(int N, float *vec1, float *vec2, float *res){
//   __shared__ float tmpd[N];
//   int tid = threadIdx.x;
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;

//   // Perform the dot product calculation
//   tmpd[tid] = vec1[idx] * vec2[idx];

//   // Synchronize threads within the block
//   __syncthreads();

//   // Perform parallel reduction
//   for (int s=blockDim.x/2; s>32; s>>=1) {
//     if (tid < s)
//       tmpd[tid] += tmpd[tid + s];
//     __syncthreads();
//   }
//   // desenrrollamos el ultimo warp activo
//   if (tid < 32) {
//     volatile double *smem = tmpd;

//     smem[tid] += smem[tid + 32];
//     smem[tid] += smem[tid + 16];
//     smem[tid] += smem[tid + 8];
//     smem[tid] += smem[tid + 4];
//     smem[tid] += smem[tid + 2];
//     smem[tid] += smem[tid + 1];
//   }
//   // El thread 0 escribe el resultado de este bloque en la memoria global
//   if (tid == 0) res[blkId] = tmpd[0];
// }

// __device__ void transposeMatrix(float *A, float *B, int row, int col){


//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int idy = blockIdx.y * blockDim.y + threadIdx.y;

//   if(idx < col && idy < row){
//     B[idy + row*idx] = A[idx + col*idy];
//   }
// }
// __global__ void updateLayers( double alpha, 
//                               float* dW1, float *dW2
//                               float *W1, float *W2){
//     /*
//     Updates weight, biases suing gradient descendent method using backprop
//     Given:
//       -derivative w
//       -derivative b
//       -alpha learning rate (own set)
//     Output:
//       Updates current layer's weights and biases 
//     */

//     //for each layer:
//     //W_i = W_i - aplha * dW_i
//     //b_i = b_i - alpha * db1

//     scalarProdMat<<<grid, block>>>(nOutput, nOutput, alpha, dW1, dW1alpha);
//     substractMat<<<grid, block>>>(nHiddenLayer, nOutput, W1, dW1alpha);

//     scalarProdMat<<<grid, block>>>(nFeatures, nHiddenLayer, alpha, dW2, dW2alpha);
//     substractMat<<<grid, block>>>(nFeatures, nHiddenLayer, W2, dW2alpha);

// }
// __global__ void backprop(int nFeatures, int batchSize, int nHiddenLayer, int nOutput, int nLayers,
//                         float *X, float *hiddenWeights, float *outputWeights, float *actL1, float *actL2, float *Y,
//                         float *dZ1, float dZ2, float dW1, float *dW2, float *db1, float *db2) {
//     /*
//     Given:
//       - number of input data m
//       - z that is z = w*a + b
//       - w weight vector
//       - b biases vector
    
//     Output
//       - derivative z respective to error
//       - derivative w respective to error
//       - derivative b respective to error
//     */

//     // the following lanes must be done trhough every layer
//     //dZ[last_column] 2 = A[last_column] - Y
//     //dW[last_column] 2= 1 / m * dZ[last_column] * transpose(A[last_column - 1])
//     //db[last_column] 2= 1 / m * sum(dZ[last_column])
    
//     //dZ[last_column - 1] 1= trasnpose(W[last_column]) *(dot product) dZ[last_column] * derivative(Z[last_column-1])
//     //dW[last_column - 1] 1= 1 / m * dZ1 *(dot prod) transpose(X)
//     //db[last_column - 1] 1= 1 / m * sum(dZ[last_column - 1])

//     /*__shared__ float tmpdZ2[nOutput]

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < batchSize){
      
//       int l2 = nLayers - 1;
//       //Derivative Z output layer Out[2]
//       for(int i = 0; i < nOutput; ++i){
//         dZ2[idx*nOutput + i] = actL2[idx*nOutput + i] - Y[idx*nOutput + i];
//       }
//       //Derivative W
//       for(int i = 0; i < nOutput; ++i){
//         //dW[idx*nOutput + i] = 1/nFeatures * dZ[idx + nOutput*layer2]
//         transpose(actL1, res);
//         matMult(N, M, P, dZ, res, c);
//         dW[idx*nOutput + i] = c;
//       }
//       //Derivative b
      
//     }*/ç
//     // Derivative dZ2
//     substractMat<<<6, 10>>>(nOutput, batchSize, actL2, Y, dZ2);
//     // Derivative dW2
//     transpose<<<6, 10>>>(nOutput, batchSize, actL1, actL1T);
//     matMult<<<grid, block>>>(nOutput, nOutput, batchSize, dZ2, actL1T, res);
//     scalarDivMat<<<grid, block>>>(nOutput, nOutput, batchSize, res, dW2);
//     // Derivative db2


//     // Derivative Z1
//     derivativeReLu<<<grid, block>>>(nOutput, batchSize, Z1, gZ1);
//     transpose<<<6, 10>>>(nHiddenLayer, nOutput, w2, w2T);
//     matMult<<<grid, block>>>(nOutput, batchSize,nOutput, w2T, dZ2, aux2);
//     elementWiseProd<<<grid, block>>>(nOutput, batchSize, aux2, gZ1, dZ1);
    
//     //Derivative W1
//     transpose<<<6, 10>>>(batchSize, nFeatures, X, XT);
//     matMult<<<grid, block>>>(nOutput, nFeatures, batchSize, dZ1, XT, dW1aux);
//     scalarDivMat<<<grid, block>>>(nOutput, nFeatures, batchSize, dW1aux, dW1);
// }

// __global__ void elementWiseProd(int N, int M, float *A, float *B, float *C) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < N && j < M) {
//         C[i * M + j] = A[i * M + j] * B[i * M + j];
//     }
// }

// __global__ void subtractMat(int N, int M, float *A, float *B, float *C) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < N && j < M) {
//         C[i * M + j] = A[i * M + j] - B[i * M + j];
//     }
// }

// __global__ void scalarDivMat(int N, int M, float value, float *A, float *C) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < N && j < M) {
//         C[i * M + j] = A[i * M + j] / value;
//     }
// }

// __global__ void scalarProdMat(int N, int M, float value, float *A, float *C) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < N && j < M) {
//         C[i * M + j] = A[i * M + j] * value;
//     }
// }

// __global__ int derivative(int N, int M, float *A, float *C){
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < N && j < M) {
//         if (A[i*M + j] > 0) C[i*M + j] = 1;
//         else C[i*M + j] = 0;
//     }
    
// }

// __global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
//                             float *input, float *hiddenWeights, float *activationL1, float *outputWeights, float *result) {

//   // matMult(batchSize, nHiddenLayer, nFeatures, input, hiddenWeights, activationL1);
//   // sigmoid(batchSize * nHiddenLayer, activationL1);
//   // matMult(batchSize, nOutput, nHiddenLayer, activationL1, outputWeights, result);
//   // sigmoid(batchSize * nHiddenLayer, result);

// }

