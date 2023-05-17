#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnfunctions.h"
#include "utils.h"

// Define the necessary CUDA headers and functions here

int main() {
    // define pointers to data
    float* d_input, *d_weights, *d_weightsOutput, *d_activation, *d_result;
    float* h_input, *h_weights, *h_weightsOutput, *h_activation, *h_result;

    int nFeatures = 10; int batchSize = 4; int nOutput = 5; int nHiddenLayer = 120;

    // Allocate host memory for the input, layers, and result arrays
    h_input = (float *) malloc(sizeof(float) * nFeatures * batchSize);
    h_weights = (float *) malloc(sizeof(float) * nFeatures * nHiddenLayer);
    h_weightsOutput = (float *) malloc(sizeof(float) * nHiddenLayer * nOutput);
    h_activation = (float *) malloc(sizeof(float) * nHiddenLayer * batchSize);
    h_result = (float *) malloc(sizeof(float) * nOutput * batchSize);

    // Allocate device memory for the input, layers, and result arrays
    cudaMalloc((void **) &d_input, sizeof(float) * nFeatures * batchSize);
    cudaMalloc((void **) &d_weights, sizeof(float) * nFeatures * nHiddenLayer);
    cudaMalloc((void **) &d_weightsOutput, sizeof(float) * nHiddenLayer * nOutput);
    cudaMalloc((void **) &d_activation, sizeof(float) * nHiddenLayer * batchSize);
    cudaMalloc((void **) &d_result, sizeof(float) * nOutput * batchSize);

    // Initialize the neural network weights with random values
    for (int i = 0; i < nFeatures * nHiddenLayer; i++) {
        
        h_weights[i] = rand() / (float) RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++) {
        h_weightsOutput[i] = rand() / (float) RAND_MAX;
    }
    for (int i = 0; i < nFeatures * batchSize; i++) {
        
        h_input[i] = rand() / (float) RAND_MAX;
        
    }

    // Copy the input and layer data to the device
    cudaMemcpy(d_input, h_input, sizeof(float) * nFeatures * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(float) * nFeatures * nHiddenLayer, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsOutput, h_weightsOutput, sizeof(float) * nHiddenLayer * nOutput, cudaMemcpyHostToDevice);


    // Define the grid and block sizes for the CUDA kernel launch
    dim3 grid(1,1,1);
    dim3 block(batchSize,1,1);

    // Call the forwardPass CUDA kernel
    forwardPass<<<grid, block>>>(nFeatures, batchSize, nHiddenLayer, nOutput, d_input, d_weights, d_weightsOutput, d_activation, d_result);

    // Copy the result data back to the host
    cudaMemcpy(h_result, d_result, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);


    // Print the results
    for (int i = 0; i < batchSize; i++) {
        float sum = 0;
        for (int j = 0; j < nOutput; j++) 
        {
            printf("batch %d, output %d: %f\n", i, j, h_result[i * nOutput + j]);
            sum += h_result[i * nOutput + j];
        }
        printf("sum: %f\n", sum);
    }

     // Free the device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_weightsOutput);
    cudaFree(d_activation);
    cudaFree(d_result);


    // Free the host memory
    free(h_input);
    free(h_weights);
    free(h_weightsOutput);
    free(h_activation);
    free(h_result);


}
