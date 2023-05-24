#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnfunctions.h"
#include "utils.h"

// Define the necessary CUDA headers and functions here
void seqForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput, float *input, float *weights, float *weightsOutput, float *activationL1, float *result);

float computeBatchCategoricalCrossEntropy(int nOutput, int batchSize, float *target, float *predicted);

int main()
{
    // define pointers to data
    float *d_input, *d_weights, *d_weightsOutput, *d_activation, *d_result, *d_temp;
    float *h_input, *h_weights, *h_weightsOutput, *h_activation, *h_result;
    cudaEvent_t E0, E1, E2, E3;

    float *h_labels, *h_loss;
    float *d_labels, *d_loss;
    // srand(87);
    int nFeatures = 28 * 28;
    int batchSize = 64;
    int nOutput = 15;
    int nHiddenLayer = 1024;

    float totalTime;
    float seqTime;

    // Allocate host memory for the input, layers, and result arrays
    h_input = (float *)malloc(sizeof(float) * nFeatures * batchSize);
    h_weights = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);
    h_weightsOutput = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_activation = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);
    h_result = (float *)malloc(sizeof(float) * nOutput * batchSize);
    h_labels = (float *)malloc(sizeof(float) * nOutput * batchSize);
    h_loss = (float *)malloc(sizeof(float) * batchSize);

    // Allocate device memory for the input, layers, and result arrays
    cudaMalloc((void **)&d_input, sizeof(float) * nFeatures * batchSize);
    cudaMalloc((void **)&d_weights, sizeof(float) * nFeatures * nHiddenLayer);
    cudaMalloc((void **)&d_weightsOutput, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **)&d_activation, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **)&d_result, sizeof(float) * nOutput * batchSize);
    cudaMalloc((void **)&d_temp, sizeof(float) * nFeatures * batchSize);
    cudaMalloc((void **)&d_labels, sizeof(float) * nOutput * batchSize);
    cudaMalloc((void **)&d_loss, sizeof(float) * batchSize);

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
   int nBlocksN = (nHiddenLayer+nThreads-1)/nThreads; 
  int nBlocksM = (batchSize+nThreads-1)/nThreads; 


  dim3 grid(nBlocksM, nBlocksN, 1);
  dim3 block(nThreads, nThreads, 1);

    // Define the grid and block sizes for the CUDA kernel launch
    // dim3 grid(32, 32, 1);
    // dim3 block(32, 32, 1);

    // Call the forwardPass CUDA kernel
    
    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);
    size_t sze = nHiddenLayer*sizeof(float);
    // forwardPass<<<batchSize, 512>>>(nFeatures, batchSize, nHiddenLayer, nOutput, d_input, d_weights, d_weightsOutput, d_activation, d_result);
    optimizedForwardPass<<<batchSize, 1024, sze>>>(nFeatures, batchSize, nHiddenLayer, nOutput, d_input, d_weights, d_weightsOutput, d_activation, d_result);
    cudaError_t error = cudaGetLastError();
    
    if (error != cudaSuccess) {
        printf("CUDA error occurred: %s\n",
            cudaGetErrorString(error));
    }


    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);
    categoricalCrossEntropy<<<batchSize, nOutput>>>(nOutput, batchSize, d_labels, d_result, d_loss);

    cudaMemcpy(h_loss, d_loss, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);


    // cudaEventRecord(E0, 0);
    // cudaEventSynchronize(E0);
    // // transpose<<<32,32>>>(batchSize, nFeatures, d_temp, d_input);

    // matMult<<<grid,block>>>(nHiddenLayer, batchSize, nFeatures, d_weights, d_input, d_activation);

    // // sigmoid<<<32,32>>>(nHiddenLayer*batchSize, d_activation, d_activation);

    // // matMult<<<grid,block>>>(nOutput, batchSize, nHiddenLayer, d_weightsOutput, d_activation, d_result);



    // // globalSoftmax<<<32,32>>>(nOutput,batchSize,d_result);
    // // cudaMemcpy(h_result, d_result, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);
    // cudaEventRecord(E1, 0);
    // cudaEventSynchronize(E1);
    
    
    cudaEventElapsedTime(&totalTime, E0, E1);
    int numMatrixMult1Ops = batchSize * nFeatures * nHiddenLayer; // input x weights
    int numMatrixMult2Ops = batchSize * nHiddenLayer * nOutput;   // activationL1 x weightsOutput

    // Estimate the floating-point operations for the additions
    // Each matrix multiplication involves (nFeatures - 1) additions

    int numAdditionOps1 = (batchSize * nFeatures * (nHiddenLayer - 1));
    int numAdditionOps2 = (batchSize * nHiddenLayer * (nOutput - 1));

    // Total floating-point operations
    int totalFloatingPointOps = numMatrixMult1Ops + numMatrixMult2Ops + numAdditionOps1 + numAdditionOps2;

    // Copy the result data back to the host
    // cudaMemcpy(h_result, d_result, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);

    //     // Print the results
    //     for(int i = 0; i < 1; i++){
    //     float sum = 0;
    //     for(int j = 0; j < nOutput; j++){
    //         printf("batch %d, output %d: %f\n", i, j, h_result[i * nOutput + j]);
    //         sum+=h_result[i * nOutput + j];
    //     }
    //     printf("total: %f\n", sum);
    // }

    // forwardPass<<<32, 32>>>(nFeatures, batchSize, nHiddenLayer, nOutput, d_input, d_weights, d_weightsOutput, d_activation, d_result);
    cudaMemcpy(h_result, d_result, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);
    float *h_temp = (float *)malloc(sizeof(float) * nOutput * batchSize);
    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);
    seqForwardPass(nFeatures, batchSize, nHiddenLayer, nOutput, h_input, h_weights, h_weightsOutput, h_activation, h_temp);
    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);
    cudaEventElapsedTime(&seqTime, E2, E3);

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
    printf("loss cuda: %f\n", ls_d / batchSize);
    printf("loss seq: %f\n", ls);

    printf("Total time: %4.6f milseg\n", totalTime);
    printf("Sequential time: %4.6f milseg\n", seqTime);
    printf("Total floating-point operations: %d\n", totalFloatingPointOps);
    printf("GFLOPs: %4.6f\n", totalFloatingPointOps / (totalTime * 1000000.0));
    printf("GFLOPs SEQ: %4.6f\n", totalFloatingPointOps / (seqTime * 1000000.0));
    printf("Speedup: %4.6f\n", seqTime / totalTime);

    // Free the device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_weightsOutput);
    cudaFree(d_activation);
    cudaFree(d_result);
    cudaFree(d_temp);

    // Free the host memory
    free(h_input);
    free(h_weights);
    free(h_weightsOutput);
    free(h_activation);
    free(h_result);
    free(h_temp);
}

void seqForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput, float *input, float *weights, float *weightsOutput, float *activationL1, float *result)
{
    for (int tid = 0; tid < batchSize; tid++)
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
            activationL1[tid * nHiddenLayer + i] = 1 / (1 + exp(-hiddenSum));
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
