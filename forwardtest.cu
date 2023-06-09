#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnfunctions.h"
#include "utils.h"

// Define the necessary CUDA headers and functions here
void seqForwardPass(int nFeatures, int batchSize, int nHiddenLayer, int nOutput, float *input, float *weights, float *weightsOutput, float *activationL1, float *result);

float computeBatchCategoricalCrossEntropy(int nOutput, int batchSize, float *target, float *predicted);
float reLU(float x);
void seqBackPropagation(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                                     float *Z1, float *activationL1,
                                     float *Z2, float *result, float * weightsOutput,
                                     float *input, float *labels,
                                     float* dZ2, float* dW2, float* dZ1, float* dW1);

int main()
{
    // define pointers to data
    float *d_input, *d_weights, *d_weightsOutput, *d_activation, *d_result, *d_temp;
    float *d_Z1, *d_Z2;
    float *h_Z1, *h_Z2;

    float *d_dW1, *d_dW2, *d_dZ1, *d_dZ2;
    float *h_dW1, *h_dW2;

    float *h_input, *h_weights, *h_weightsOutput, *h_activation, *h_result;
    
    cudaEvent_t E0, E1, E2, E3;

    float *h_labels, *h_loss;
    float *d_labels, *d_loss;
    // srand(87);
    int nFeatures = 28*28;
    int batchSize = 64;
    int nOutput = 15;
    int nHiddenLayer = 1500;

    float totalTime;
    // float seqTime;



    // Allocate host memory for the input, layers, and result arrays
    h_input = (float *)malloc(sizeof(float) * nFeatures * batchSize);
    h_weights = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);
    h_dW1 = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);
    h_dW2 = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_weightsOutput = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_activation = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);
    h_result = (float *)malloc(sizeof(float) * nOutput * batchSize);
    h_labels = (float *)malloc(sizeof(float) * nOutput * batchSize);
    h_loss = (float *)malloc(sizeof(float) * batchSize);
    h_Z1 = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);
    h_Z2 = (float *)malloc(sizeof(float) * nOutput * batchSize);

    // Allocate device memory for the input, layers, and result arrays
    cudaMalloc((void **)&d_input, sizeof(float) * nFeatures * batchSize);
    cudaMalloc((void **)&d_weights, sizeof(float) * nFeatures * nHiddenLayer);
    cudaMalloc((void **)&d_weightsOutput, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **)&d_activation, sizeof(float) * nHiddenLayer * batchSize);
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
    optimizedForwardPass<<<batchSize, 1024, sze>>>(nFeatures, batchSize, nHiddenLayer, nOutput, d_input, d_weights, d_Z1, d_activation, d_weightsOutput, d_Z2, d_result);
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
    int numMatrixMult1Ops = 2.0f*batchSize * nFeatures * nHiddenLayer; // input x weights
    int numMatrixMult2Ops = 2.0f*batchSize * nHiddenLayer * nOutput;   // activationL1 x weightsOutput

    // Estimate the floating-point operations for the additions
    // Each matrix multiplication involves (nFeatures - 1) additions

    int numAdditionOps1 = nHiddenLayer*batchSize;
    int numAdditionOps2 = batchSize * (nOutput + nOutput - 1 + nOutput);

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
    // clock_t start, stop;
    // start = clock();
    seqForwardPass(nFeatures, batchSize, nHiddenLayer, nOutput, h_input, h_weights, h_weightsOutput, h_activation, h_temp);
    // stop = clock();
    

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
    float* h_dZ2 = (float*)malloc(sizeof(float) * nOutput * batchSize);
    float* h_dZ1 = (float*)malloc(sizeof(float) * nHiddenLayer * batchSize);


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
            activationL1[tid * nHiddenLayer + i] = reLU(hiddenSum);
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

float reLU(float x)
{
    return x > 0 ? x : 0;
}

void seqBackPropagation(int nFeatures, int batchSize, int nHiddenLayer, int nOutput,
                                     float *Z1, float *activationL1,
                                     float *Z2, float *result, float * weightsOutput,
                                     float *input, float *labels,
                                     float* dZ2, float* dW2, float* dZ1, float* dW1) {
    for (int tid = 0; tid < batchSize; tid++)
    {
        // Compute the gradient of the loss function with respect to the output layer
        for (int c = 0; c < nOutput; c++)
        {
            dZ2[tid * nOutput + c] = result[tid * nOutput + c] - labels[tid * nOutput + c];
        }

        // Compute the gradient of the loss function with respect to the weights of the output layer
        for (int i = 0; i < nHiddenLayer; i++)
        {
            for (int c = 0; c < nOutput; c++)
            {
                dW2[i * nOutput + c] += activationL1[tid * nHiddenLayer + i] * dZ2[tid * nOutput + c];
                dW2[i * nOutput + c] /= batchSize;
            }
        }

        // Compute the gradient of the loss function with respect to the hidden layer
        for (int i = 0; i < nHiddenLayer; i++)
        {
            float sum = 0.0f;

            for (int c = 0; c < nOutput; c++)
            {
                sum += weightsOutput[i * nOutput + c] * dZ2[tid * nOutput + c];
            }

            dZ1[tid * nHiddenLayer + i] = sum * (Z1[tid * nHiddenLayer + i] > 0 ? 1 : 0);
        }

        // Compute the gradient of the loss function with respect to the weights of the hidden layer
        for (int j = 0; j < nFeatures; j++)
        {
            for (int i = 0; i < nHiddenLayer; i++)
            {
                dW1[j * nHiddenLayer + i] += input[i * batchSize + tid] * dZ1[tid * nHiddenLayer + i];
            }
        }
    }

}

