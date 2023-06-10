#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <math.h>

#include "nnfunctions.h"
// #include "input_utils.h"
#include "primitives.h"
#include "utils.h"

#define INPUT_SIZE 50
#define OUTPUT_SIZE 10
#define HIDDEN_SIZE 10

#define LEARNING_RATE 0.1


// define headers
void seqSubstractMat(int n, int m, float *A, float *B, float *C);
void seqTranspose(int n, int m, float *A, float *B);
void seqMatMult(int n, int m, int p, float *A, float *B, float *C);
void seqDerivativeReLu(int n, float *A, float *B);
void seqElementWiseProduct(int n, float *A, float *B, float *C);
void seqScalarDivMat(int n, float *A, float scalar, float *B);



// test to see if dw1 computations are correct
int main()
{
    cudaError_t err;
    int nFeatures, batchSize, nOutput, nHiddenLayer;
    nFeatures = INPUT_SIZE;
    batchSize = 20;
    nOutput = OUTPUT_SIZE;
    nHiddenLayer = HIDDEN_SIZE;

    float alpha = LEARNING_RATE;

    // pointers to input, weights, zets, and activations
    float *d_input, *d_weights, *d_weightsOutput, *d_Z1, *d_activation, *d_Z2, *d_result;

    // pointers to future transposed matrices
    float *d_inputT, *d_weightsOutputT, *d_activationT;

    // pointers to derivatives
    float *d_dW1, *d_dW2, *d_dZ1, *d_dZ2;

    // pointers to learning rates
    float *d_dW1_alpha, *d_dW2_alpha;

    // pointers to labels and loss
    float *d_labels, *d_loss;

    float *d_gZ1;

    // pointers to host memory
    // float *h_dW1_alpha, *h_dW2_alpha;
    // float *h_dW1, *h_dW2;
    // float *h_Z1, *h_Z2;
    float *h_input, *h_weights, *h_weightsOutput, *h_Z1, *h_activation, *h_Z2, *h_result;
    float *h_inputT, *h_weightsOutputT, *h_activationT;
    float *h_labels, *h_loss;

    float *h_dW1, *h_dW2;

    float *h_dZ2;

    float *h_dZ1;
    // Allocate host memory for the input, layers, and result arrays
    h_input = (float *)malloc(sizeof(float) * nFeatures * batchSize);

    h_inputT = (float *)malloc(sizeof(float) * nFeatures * batchSize);

    h_weights = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);

    h_weightsOutput = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);

    h_activation = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    h_Z1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    h_activationT = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    h_result = (float *)malloc(sizeof(float) * nOutput * batchSize);

    h_loss = (float *)malloc(sizeof(float) * batchSize);

    h_labels = (float *)malloc(sizeof(float) * nOutput * batchSize);

    h_dZ2 = (float *)malloc(sizeof(float) * nOutput * batchSize);

    h_dZ1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    h_dW1 = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);




    // Allocate device memory for the input, layers, and result arrays
    cudaMalloc((void **)&d_input, sizeof(float) * nFeatures * batchSize);
    cudaMalloc((void **)&d_inputT, sizeof(float) * nFeatures * batchSize);

    cudaMalloc((void **)&d_weights, sizeof(float) * nFeatures * nHiddenLayer);
    cudaMalloc((void **)&d_weightsOutput, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **)&d_weightsOutputT, sizeof(float) * nHiddenLayer * nOutput);

    cudaMalloc((void **)&d_activation, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_activationT, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_result, sizeof(float) * nOutput * batchSize);

    cudaMalloc((void **)&d_labels, sizeof(float) * nOutput * batchSize);
    cudaMalloc((void **)&d_loss, sizeof(float) * batchSize);
    cudaMalloc((void **)&d_Z1, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_Z2, sizeof(float) * nOutput * batchSize);
    cudaMalloc((void **)&d_dW1, sizeof(float) * nFeatures * nHiddenLayer);
    cudaMalloc((void **)&d_dW2, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **)&d_dZ1, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_dZ2, sizeof(float) * nOutput * batchSize);
    cudaMalloc((void **)&d_dW1_alpha, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **)&d_dW2_alpha, sizeof(float) * nHiddenLayer * nFeatures);
    cudaMalloc((void **)&d_gZ1, sizeof(float) * nHiddenLayer * batchSize);

    // initialize weights and input at random
    for (int i = 0; i < nFeatures * nHiddenLayer; i++)
    {
        h_weights[i] = -1.0f + 2.0f * (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++)
    {
        h_weightsOutput[i] = -1.0f + 2.0f * (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nFeatures * batchSize; i++)
    {
        h_input[i] = -1.0f + 2.0f * (float)rand() / (float)RAND_MAX;
    }

    // initialize labels
    for (int i = 0; i < nOutput * batchSize; i++)
    {
        if (i % 10 == 0)
        {
            h_labels[i] = 1.0f;
        }
        else
        {
            h_labels[i] = 0.0f;
        }
    }

    // copy input and weights to device
    cudaMemcpy(d_input, h_input, sizeof(float) * nFeatures * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(float) * nFeatures * nHiddenLayer, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsOutput, h_weightsOutput, sizeof(float) * nHiddenLayer * nOutput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, sizeof(float) * nOutput * batchSize, cudaMemcpyHostToDevice);
    int nThreads = 32;
    int nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
    int nBlocksM = (batchSize + nThreads - 1) / nThreads;

    dim3 grid(nBlocksM, nBlocksN, 1);
    dim3 block(nThreads, nThreads, 1);
    // perform forward pass
    transpose<<<64, 1024>>>(batchSize, nFeatures, d_input, d_inputT);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error trans: %s\n", cudaGetErrorString(err));
    }
    // C(N × M) ← A(N × P) · B (P × M)
    matMult<<<grid, block>>>(nHiddenLayer, batchSize, nFeatures, d_weights, d_inputT, d_Z1);
    cudaMemcpy(h_Z1, d_Z1, sizeof(float) * nHiddenLayer * batchSize, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error nat: %s\n", cudaGetErrorString(err));
    }
    reLU<<<64, 1024>>>(nHiddenLayer * batchSize, d_Z1, d_activation);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error sig: %s\n", cudaGetErrorString(err));
    }

    nBlocksN = (nOutput + nThreads - 1) / nThreads;
    nBlocksM = (batchSize + nThreads - 1) / nThreads;
    dim3 grid1(nBlocksM, nBlocksN, 1);
    // C(N × M) ← A(N × P) · B (P × M)
    matMult<<<grid1, block>>>(nOutput, batchSize, nHiddenLayer, d_weightsOutput, d_activation, d_Z2);
    if (err != cudaSuccess)
    {
        printf("Error sig: %s\n", cudaGetErrorString(err));
    }
    globalSoftmaxPrimitive<<<batchSize, nOutput>>>(nOutput, batchSize, d_Z2, d_result);
    cudaMemcpy(h_result, d_result, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);

    // backward pass
    // dZ2 = A2 - Y
    // dW2 = dZ2 * A1T
    // dZ1 = W2T * dZ2
    // dW1 = dZ1 * X
    // dZ1 = dZ1 * g'(Z1)
    // dW1_alpha = dW1 * alpha
    // dW2_alpha = dW2 * alpha
    // W1 = W1 - dW1_alpha
    // W2 = W2 - dW2_alpha

    // dZ1 computation
    subtractMat<<<64, 1024>>>(nOutput, batchSize, d_Z2, d_labels, d_dZ2);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error sub: %s\n", cudaGetErrorString(err));
    }

    transpose<<<64, 1024>>>(batchSize, nHiddenLayer, d_weightsOutput, d_weightsOutputT);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error sub: %s\n", cudaGetErrorString(err));
    }
    
    nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
    nBlocksM = (batchSize + nThreads - 1) / nThreads;

    dim3 grid2(nBlocksM, nBlocksN, 1);
    matMult<<<grid2, block>>>(nHiddenLayer, batchSize, nOutput, d_weightsOutputT, d_dZ2, d_dZ1);

    derivativeReLu<<<64, 1024>>>(nHiddenLayer, batchSize, d_Z1, d_gZ1);

    elementWiseProd<<<64, 1024>>>(nHiddenLayer, batchSize, d_dZ1, d_gZ1, d_dZ1);

    nBlocksM = (nHiddenLayer + nThreads - 1) / nThreads;
    nBlocksN = (nFeatures + nThreads - 1) / nThreads;

    dim3 grid3(nBlocksM, nBlocksN, 1);

    matMult<<<grid3, block>>>(nHiddenLayer, nFeatures, batchSize, d_dZ1, d_input, d_dW1);

    scalarDivMat<<<64, 1024>>>(nHiddenLayer, nFeatures, batchSize, d_dW1, d_dW1);


    cudaMemcpy(h_dW1, d_dW1, sizeof(float) * nHiddenLayer * nFeatures, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dZ1, d_dZ1, sizeof(float) * nHiddenLayer * batchSize, cudaMemcpyDeviceToHost);

    // sequential computation
    printf("Sequential computation\n");
    //memory allocation
    float *seq_dZ1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);
    float *seq_dW1 = (float *)malloc(sizeof(float) * nHiddenLayer * nFeatures);
    float *seq_dZ2 = (float *)malloc(sizeof(float) * nOutput * batchSize);
    float *seq_weightsOutputT = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    float *seq_dgZ1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);


    seqTranspose(nFeatures, batchSize, h_input, h_inputT);
    // computations
    seqSubstractMat(nOutput, batchSize, h_result, h_labels, seq_dZ2);
    printf("Substract done\n");
    seqTranspose(nHiddenLayer, nOutput, h_weightsOutput, seq_weightsOutputT);
    printf("Transpose done\n");
    seqMatMult(nHiddenLayer, batchSize, nOutput, seq_weightsOutputT, seq_dZ2, seq_dZ1);
    printf("Mat mult done\n");
    seqDerivativeReLu(nHiddenLayer * batchSize, h_Z1, seq_dgZ1);
    printf("Derivative done\n");
    seqElementWiseProduct(nHiddenLayer * batchSize, seq_dgZ1, seq_dZ1, seq_dZ1);
    printf("Element wise done\n");
    seqMatMult(nHiddenLayer, nFeatures, batchSize, seq_dZ1, h_input, seq_dW1);
    printf("Mat mult done\n");
    seqScalarDivMat(nHiddenLayer * nFeatures, seq_dW1, batchSize, seq_dW1);
    printf("Scalar div done\n");

    //result comparison
    for (int i = 0; i < nHiddenLayer * nFeatures; i++)
    {
        if (abs(seq_dW1[i] - h_dW1[i]) > 0.0001)
        {
            printf("Error in dW1 at %d, %f != %f\n", i, seq_dW1[i], h_dW1[i]);
            break;
        }
    }

    for (int i = 0; i < nHiddenLayer * batchSize; i++)
    {
        if (abs(seq_dZ1[i] - h_dZ1[i]) > 0.0001)
        {
            printf("Error in dZ1 at %d, %f != %f\n", i, seq_dZ1[i], h_dZ1[i]);
            break;
        }
    }

    // free memory
    free(seq_dZ1);
    free(seq_dW1);
    free(seq_dZ2);
    free(seq_weightsOutputT);

     free(h_weights);
    free(h_weightsOutput);
    free(h_activation);
    free(h_result);
    free(h_loss);


    // free memory in device
    cudaFree(d_input);
    cudaFree(d_inputT);
    cudaFree(d_weights);

    cudaFree(d_weightsOutput);
    cudaFree(d_weightsOutputT);

    cudaFree(d_activation);
    cudaFree(d_activationT);
    cudaFree(d_result);
    cudaFree(d_labels);
    cudaFree(d_loss);

    cudaFree(d_Z1);
    cudaFree(d_Z2);
    cudaFree(d_dZ1);
    cudaFree(d_dZ2);

    cudaFree(d_dW1);
    cudaFree(d_dW2);
    cudaFree(d_dW1_alpha);
    cudaFree(d_dW2_alpha);

    cudaFree(d_gZ1);



}

void seqSubstractMat(int n, int m, float *A, float *B, float *C)
{
    for (int i = 0; i < n * m; i++)
    {
        C[i] = A[i] - B[i];
    }
}

void seqTranspose(int n, int m, float *A, float *B)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            B[j * n + i] = A[i * m + j];
        }
    }
}

void seqMatMult(int n, int m, int p, float *A, float *B, float *C)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < p; k++)
            {
                C[i * m + j] += A[i * p + k] * B[k * m + j];
            }
        }
    }
}

void seqDerivativeReLu(int n, float *A, float *B)
{
    for (int i = 0; i < n; i++)
    {
        if (A[i] > 0)
        {
            B[i] = 1.0f;
        }
        else
        {
            B[i] = 0.0f;
        }
    }
}

void seqElementWiseProduct(int n, float *A, float *B, float *C)
{
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] * B[i];
    }
}

void seqScalarDivMat(int n, float *A, float scalar, float *B)
{
    for (int i = 0; i < n; i++)
    {
        B[i] = A[i] / scalar;
    }
}