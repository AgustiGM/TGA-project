#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnfunctions.h"
#include "primitives.h"
#include "utils.h"

// Define the necessary CUDA headers and functions here
void seqForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput, float *input, float *weights, float *weightsOutput, float *activationL1, float *result);

float computeBatchCategoricalCrossEntropy(int nOutput, int batchSize, float *target, float *predicted);
float reLU(float x);

void seqSoftmax(int nOutput, int batchSize, float *input, float *output);
void seqMatMult(int N, int M, int P, float *A, float *B, float *C);
void seqTranspose(int N, int M, float *A, float *C);
void seqSigmoid(int N, float *A, float *C);

int main()
{
    // define pointers to data
    float *d_input, *d_weights, *d_weightsOutput, *d_activation, *d_result, *d_temp;
    float *d_inputT, *d_weightsOutputT, *d_activationT;
    float *d_Z1, *d_Z2;
    float *d_dW1, *d_dW2, *d_dZ1, *d_dZ2;
    float *d_dW1_alpha, *d_dW2_alpha;

    float *h_dW1_alpha, *h_dW2_alpha;
    float *h_dW1, *h_dW2;
    float *h_Z1, *h_Z2;
    float *h_input, *h_weights, *h_weightsOutput, *h_activation, *h_result;
    float *h_inputT, *h_weightsOutputT, *h_activationT;

    cudaEvent_t E0, E1, E2, E3;

    float *h_labels, *h_loss;
    float *d_labels, *d_loss;
    // srand(87);
    int nFeatures = 28 * 28;
    int batchSize = 128;
    int nOutput = 15;
    int nHiddenLayer = 3500;

    float totalTime;
    // float seqTime;

    // Allocate host memory for the input, layers, and result arrays
    h_input = (float *)malloc(sizeof(float) * nFeatures * batchSize);
    h_inputT = (float *)malloc(sizeof(float) * nFeatures * batchSize);
    h_weights = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);
    h_dW1 = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);
    h_dW2 = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_weightsOutput = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_weightsOutputT = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);

    h_activation = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);
    h_activationT = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    h_result = (float *)malloc(sizeof(float) * nOutput * batchSize);
    h_labels = (float *)malloc(sizeof(float) * nOutput * batchSize);
    h_loss = (float *)malloc(sizeof(float) * batchSize);
    h_Z1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);
    h_Z2 = (float *)malloc(sizeof(float) * nOutput * batchSize);

    h_dW1_alpha = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_dW2_alpha = (float *)malloc(sizeof(float) * nHiddenLayer * nFeatures);

    // Allocate device memory for the input, layers, and result arrays
    cudaMalloc((void **)&d_input, sizeof(float) * nFeatures * batchSize);
    cudaMalloc((void **)&d_inputT, sizeof(float) * nFeatures * batchSize);

    cudaMalloc((void **)&d_weights, sizeof(float) * nFeatures * nHiddenLayer);
    cudaMalloc((void **)&d_weightsOutput, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **)&d_weightsOutputT, sizeof(float) * nHiddenLayer * nOutput);

    cudaMalloc((void **)&d_activation, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_activationT, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_result, sizeof(float) * nOutput * batchSize);
    cudaMalloc((void **)&d_temp, sizeof(float) * nFeatures * batchSize);
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

    // Initialize the neural network weights with random values
    for (int i = 0; i < nFeatures * nHiddenLayer; i++)
    {

        h_weights[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++)
    {
        h_weightsOutput[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nFeatures * batchSize; i++)
    {

        h_input[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < nOutput * batchSize; i++)
    {
        if (i % nOutput == 0)
        {
            h_labels[i] = 1.0f;
        }
        else
        {
            h_labels[i] = 0.0f;
        }
    }
    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);

    // Copy the input and layer data to the device
    cudaMemcpy(d_input, h_input, sizeof(float) * nFeatures * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(float) * nFeatures * nHiddenLayer, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsOutput, h_weightsOutput, sizeof(float) * nHiddenLayer * nOutput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, sizeof(float) * nOutput * batchSize, cudaMemcpyHostToDevice);

    int nThreads = 32;
    int nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
    int nBlocksM = (batchSize + nThreads - 1) / nThreads;

    dim3 grid(nBlocksM, nBlocksN, 1);
    dim3 block(nThreads, nThreads, 1);

    // Define the grid and block sizes for the CUDA kernel launch
    // dim3 grid(32, 32, 1);
    // dim3 block(32, 32, 1);

    // Call the forwardPass CUDA kernel

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // forward pass
    transpose<<<64, 1024>>>(batchSize, nFeatures, d_input, d_inputT);
    matMult<<<grid, block>>>(nHiddenLayer, batchSize, nFeatures, d_weights, d_inputT, d_Z1);
    sigmoid<<<64, 1024>>>(nHiddenLayer * batchSize, d_Z1, d_activation);

    nBlocksN = (nOutput + nThreads - 1) / nThreads;
    nBlocksM = (batchSize + nThreads - 1) / nThreads;
    dim3 grid1(nBlocksM, nBlocksN, 1);

    matMult<<<grid1, block>>>(nOutput, batchSize, nHiddenLayer, d_weightsOutput, d_activation, d_Z2);
    globalSoftmaxPrimitive<<<batchSize, nOutput>>>(nOutput, batchSize, d_Z2, d_result);

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);

    // cudaMemcpy(h_Z2, d_Z2, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);
    // seqSoftmax(nOutput, batchSize, h_Z2, h_result);
    // cudaMemcpy(d_result, h_result, sizeof(float) * nOutput * batchSize, cudaMemcpyHostToDevice);
    categoricalCrossEntropy<<<batchSize, nOutput>>>(nOutput, batchSize, d_labels, d_result, d_loss);

    /*-----------------------------
            backpropagation
    -----------------------------*/
    // Derivative dZ2
    substractMat<<<6, 10>>>(nOutput, batchSize, d_result, d_labels, d_dZ2);
    // Derivative dW2
    transpose<<<6, 10>>>(nHiddenLayer, batchSize, d_activation, d_activationT);
    matMult<<<grid, block>>>(nOutput, batchSize, nHiddenLayer, d_dZ2, d_activationT, d_dZ2);
    scalarDivMat<<<grid, block>>>(nOutput, nHiddenLayer, batchSize, d_dZ2, d_dW2);


    // Derivative Z1
    derivativeReLu<<<grid, block>>>(nHiddenLayer, batchSize, d_Z1, d_gZ1);
    transpose<<<6, 10>>>(nOutput, nHiddenLayer, d_weightsOutput, d_weightsOutputT);
    matMult<<<grid, block>>>(nHiddenLayer, nOutput,batchSize, d_weightsOutputT, d_dZ2, d_dZ1); 
    elementWiseProd<<<grid, block>>>(nHiddenLayer, batchSize, d_dZ1, d_gZ1, d_dZ1);
    
    //Derivative W1
    transpose<<<6, 10>>>(nFeatures, batchSize, d_input, d_inputT);
    matMult<<<grid, block>>>(nHiddenLayer, batchSize, nFeatures, d_dZ1, d_inputT, d_dW1);
    scalarDivMat<<<grid, block>>>(nHiddenLayer, nFeatures, batchSize, d_dW1, d_dW1);

    /*-----------------------------
            update
    -----------------------------*/
    scalarProdMat<<<grid, block>>>(nHiddenLayer, nFeatures, alpha, d_dW1, d_dW1_alpha);
    substractMat<<<grid, block>>>(nHiddenLayer, nFeatures, d_weights, d_dW1_alpha);


    scalarProdMat<<<grid, block>>>(nFeatures, nHiddenLayer, alpha, d_dW2, d_dW2_alpha);
    substractMat<<<grid, block>>>(nHiddenLayer, nOutput, d_weightsOutput, d_dW2_alpha);

    cudaMemcpy(h_loss, d_loss, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalTime, E0, E1);

    // Copy the result data back to the host
    cudaMemcpy(h_result, d_result, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);


    float *h_temp = (float *)malloc(sizeof(float) * nOutput * batchSize);

    seqTranspose(batchSize, nFeatures, h_input, h_inputT);
    seqMatMult(nHiddenLayer, batchSize, nFeatures, h_weights, h_inputT, h_Z1);
    seqSigmoid(nHiddenLayer * batchSize, h_Z1, h_activation);
    seqMatMult(nOutput, batchSize, nHiddenLayer, h_weightsOutput, h_activation, h_Z2);
    seqSoftmax(nOutput, batchSize, h_Z2, h_temp);

    int count = 0;

    for (int i = 0; i < batchSize; i++)
    {
        float sum1 = 0;
        float sum2 = 0;

        for (int j = 0; j < nOutput; j++)
        {
            if (abs(h_result[i * nOutput + j] - h_temp[i * nOutput + j]) > 0.0001)
            {
                printf("batch %d, output %d: %f, %f\n", i, j, h_result[i * nOutput + j], h_temp[i * nOutput + j]);
                ++count;
            }

            sum1 += h_result[i * nOutput + j];
            sum2 += h_temp[i * nOutput + j];
        }

    }
    printf("count: %d\n", count);
    float ls = computeBatchCategoricalCrossEntropy(nOutput, batchSize, h_labels, h_temp);
    float ls_d = 0.0f;
    for (int i = 0; i < batchSize; i++)
    {
        ls_d += h_loss[i];
    }

    // backpropagation<<<batchSize, nOutput>>>(nFeatures, batchSize, nHiddenLayer, nOutput,
    //                                         d_Z1, d_activation, d_Z2, d_result, d_weightsOutput,
    //                                         d_input, d_labels,
    //                                         d_dZ2, d_dW2, d_dZ1, d_dW1);

    cudaMemcpy(h_activation, d_activation, sizeof(float) * nHiddenLayer * batchSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_Z1, d_Z1, sizeof(float) * nHiddenLayer * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z2, d_Z2, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_dW1, d_dW1, sizeof(float) * nHiddenLayer * nFeatures, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW2, d_dW2, sizeof(float) * nOutput * nHiddenLayer, cudaMemcpyDeviceToHost);

    float *h_dW1_seq = (float *)malloc(sizeof(float) * nHiddenLayer * nFeatures);
    float *h_dW2_seq = (float *)malloc(sizeof(float) * nOutput * nHiddenLayer);
    float *h_dZ2 = (float *)malloc(sizeof(float) * nOutput * batchSize);
    float *h_dZ1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    // seqBackPropagation(nFeatures, batchSize, nHiddenLayer, nOutput, h_Z1, h_activation, h_Z2, h_result, h_weightsOutput, h_input, h_labels, h_dZ2, h_dW2_seq, h_dZ1, h_dW1_seq);

    // for (int i = 0; i < nHiddenLayer * nFeatures; i++)
    // {
    //     if (abs(h_dW1[i] - h_dW1_seq[i]) > 0.0001)
    //     {
    //         printf("dW1 %d: %f, %f\n", i, h_dW1[i], h_dW1_seq[i]);
    //     }
    //     if (abs(h_dW2[i] - h_dW2_seq[i]) > 0.0001)
    //     {
    //         printf("dW2 %d: %f, %f\n", i, h_dW2[i], h_dW2_seq[i]);
    //     }
    // }

    int numMatrixMult1Ops = 2.0f * batchSize * nFeatures * nHiddenLayer; // input x weights
    int numMatrixMult2Ops = 2.0f * batchSize * nHiddenLayer * nOutput;   // activationL1 x weightsOutput

    // Estimate the floating-point operations for the additions
    // Each matrix multiplication involves (nFeatures - 1) additions

    int numAdditionOps1 = nHiddenLayer * batchSize;
    int numAdditionOps2 = batchSize * (nOutput + nOutput - 1 + nOutput);

    // Total floating-point operations
    int totalFloatingPointOps = numMatrixMult1Ops + numMatrixMult2Ops + numAdditionOps1 + numAdditionOps2;

    printf("loss cuda: %f\n", ls_d / batchSize);
    printf("loss seq: %f\n", ls);
    // double seqtime = ((double)(stop - start))/1000.0;
    printf("Total time: %4.6f milseg\n", totalTime);
    // printf("seq time: %f\n mils", seqtime);
    printf("Total floating-point operations: %d\n", totalFloatingPointOps);
    printf("GFLOPs: %4.6f\n", totalFloatingPointOps / (totalTime * 1000000.0));
    // printf("GFLOPs SEQ: %4.6f\n", totalFloatingPointOps / (seqtime * 1000000.0));
    // printf("Speedup: %4.6f\n", seqtime/ totalTime);

    // Free the device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_weightsOutput);
    cudaFree(d_activation);
    cudaFree(d_result);
    cudaFree(d_temp);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_Z1);
    cudaFree(d_Z2);
    cudaFree(d_dW1);
    cudaFree(d_dW2);
    cudaFree(d_dZ2);
    cudaFree(d_inputT);
    cudaFree(d_dW1_alpha);
    cudaFree(d_dW2_alpha);

    // Free the host memory
    free(h_input);
    free(h_weights);
    free(h_weightsOutput);
    free(h_activation);
    free(h_result);
    free(h_temp);
    free(h_labels);
    free(h_loss);
    free(h_Z1);
    free(h_Z2);
    free(h_dW1);
    free(h_dW2);
    free(h_dZ2);
    free(h_inputT);
    free(h_dZ1);
    free(h_dW1_seq);
    free(h_dW2_seq);
    free(h_dZ2);
    free(h_dZ1);
}
// C(N × M) ← A(N × P) · B (P × M)
void seqMatMult(int N, int M, int P, float *A, float *B, float *C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++) // C[i][j]
        {
            float sum = 0.0f;
            for (int k = 0; k < P; k++)
            {
                sum += A[i * P + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

void seqTranspose(int N, int M, float *A, float *C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            C[j * N + i] = A[i * M + j];
        }
    }
}

void seqSigmoid(int N, float *A, float *C)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = 1.0f / (1.0f + exp(-A[i]));
    }
}

float computeBatchCategoricalCrossEntropy(int nOutput, int batchSize, float *target, float *predicted)
{
    float totalLoss = 0.0f;

    for (int i = 0; i < batchSize; i++)
    {
        float loss = 0.0f;

        for (int c = 0; c < nOutput; c++)
        {
            loss -= target[i * nOutput + c] * log(predicted[i * nOutput + c]);
        }

        totalLoss += loss;
    }

    return totalLoss / batchSize;
}

float reLU(float x)
{
    return x > 0 ? x : 0;
}

void seqSoftmax(int nOutput, int batchSize, float *input, float *output)
{
    for (int i = 0; i < batchSize; i++)
    {
        float max = input[i * nOutput];
        for (int j = 1; j < nOutput; j++)
        {
            if (input[i * nOutput + j] > max)
            {
                max = input[i * nOutput + j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < nOutput; j++)
        {
            output[i * nOutput + j] = exp(input[i * nOutput + j] - max);
            sum += output[i * nOutput + j];
        }
        for (int j = 0; j < nOutput; j++)
        {
            output[i * nOutput + j] /= sum;
        }
    }
}

void seqElementWiseProd(int N, int M, float *A, float *B, float *C)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            C[i * M + j] = A[i * M + j] * B[i * M + j];
        }
    }
}

void seqSubstractMat(int N, int M, float *A, float *B, float *C)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            C[i * M + j] = A[i * M + j] - B[i * M + j];
        }
    }
}


void seqScalarProdMat(int N, int M, float value, float *A, float *C){
    for (int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            C[i*M + j] = A[i*M + j]*value;
        }
    }
}

void seqScalarDivMat(int N, int M, float value, float *A, float *C){
    for (int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            C[i*M + j] = A[i*M + j]/value;

        }
    }
}

void seqDerivativeReLU(int N, int M, float *A, float *C)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            if (A[i * M + j] > 0)
                C[i * M + j] = 1;
            else
                C[i * M + j] = 0;
        }
    }
}
