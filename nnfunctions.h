#include <math.h>

// struct Layer {
//     int nInput;
//     int nOutput;
//     float *weights;
//     float *biases;
//     float *activations;
// };


// __global__ void matMult(int N, int M, int P, float *A, float *B, float *C);

// __global__ void globalMatMult(int N, int M, int P, float *A, float *B, float *C);



__global__ void sigmoid(int N, float *input, float *output);

__device__ void reLU(int N, float *input, float *output);

__global__ void globalReLU(int N, float *input, float *output);

__global__ void backpropagation(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                                     float *Z1, float *activationL1,
                                     float *Z2, float *result, float * weightsOutput,
                                     float *input, float *labels,
                                     float* dZ2, float* dW2, float* dZ1, float* dW1);

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                        float *input, float *weights, float *weightsOutput, float* activationL1, float *result);

__device__ float localReLU(float x);

__device__ float localReLUPrime(float x);

<<<<<<< HEAD
__global__ void transposeMatrix(float *odata, const float *idata);

=======
__device__ void softmax(int nOutput, int batchSize, float *input);
>>>>>>> simple-nn

__global__ void globalSoftmax(int nOutput, int batchSize, float *input);

__global__ void optimizedForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                                     float *input,
                                     float *weights, float * Z1, float *activationL1,
                                     float * weightsOutput, float *Z2, float *result);

__global__ void categoricalCrossEntropy(int nOutput, int batchSize, float *groundTruth, float* predictions, float *loss);
