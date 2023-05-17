#include <math.h>

// struct Layer {
//     int nInput;
//     int nOutput;
//     float *weights;
//     float *biases;
//     float *activations;
// };


__device__ void matMult(int N, int M, int P, float *A, float *B, float *C);

__global__ void globalMatMult(int N, int M, int P, float *A, float *B, float *C);

__global__ void globalSigmoid(int N, float *input, float *output);

__device__ void sigmoid(int N, float *input, float *output);

__device__ void reLU(int N, float *input, float *output);

__global__ void globalReLU(int N, float *input, float *output);

__global__ void backprop(int N, float *A);

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                        float *input, float *weights, float *weightsOutput, float* activationL1, float *result);


__device__ void softmax(int N, float *input, float *output);

__device__ void transpose(int N, int M, float *input, float *output);
