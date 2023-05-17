#include <math.h>

typedef struct {
    int nInput;
    int nOutput;
    float *weights;
    float *biases;
    float *activations;
} Layer;


__device__ void matMult(int N, int M, int P, float *A, float *B, float *C);

__global__ void globalMatMult(int N, int M, int P, float *A, float *B, float *C);

__device__ void sigmoid(int N, float *input, float *output);

__global__ void globalSigmoid(int N, float *input, float *output);

__device__ void reLU(int N, float *input, float *output);

__global__ void globalReLU(int N, float *input, float *output);

__global__ void backprop(int N, float *A);

__global__ void forwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                            float *input, float *hiddenWeights, float *outputWeights, float *result);


__device__ void softmax(int N, float *input, float *output);

__global__ void transposeMatrix(float *odata, const float *idata);


