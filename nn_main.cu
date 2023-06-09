#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <math.h>

#include "nnfunctions.h"
// #include "input_utils.h"
#include "primitives.h"
#include "utils.h"

float *readImageData(char *filename, int size)
{
    int fd = open(filename, O_RDONLY);
    unsigned char buf[16];
    int n = read(fd, buf, 16);

    unsigned char *data = (unsigned char *)malloc(size);
    n = read(fd, data, size);
    float *fdata = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        fdata[i] = (float)data[i];
    }
    free(data);
    close(fd);
    return fdata;
}

float *readLabels(char *filename, int size)
{
    int fd = open(filename, O_RDONLY);
    unsigned char buf[8];
    int n = read(fd, buf, 8);

    unsigned char *data = (unsigned char *)malloc(size);
    n = read(fd, data, size);
    float *fdata = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        fdata[i] = (float)data[i];
    }
    free(data);
    close(fd);
    return fdata;
}

int main(int argc, char **argv)
{
     cudaSetDevice(0);
    int nFeatures, batchSize, nOutput, nHiddenLayer, training_size, testing_size, nEpochs;
    float learning_rate;
    char *filename_train, *train_labels, *filename_test;
    // todo parse arguments
    if (argc != 12)
    {
        printf("Usage: ./nn <nFeatures> <batchSize> <nOutput> <nHiddenLayer> <learning_rate> <training_size> <testing_size> <nEpochs> <filename_train> <train_labels> <filename_test>\n");
        exit(1);
    }
    nFeatures = atoi(argv[1]);
    batchSize = atoi(argv[2]);
    nOutput = atoi(argv[3]);
    nHiddenLayer = atoi(argv[4]);
    learning_rate = atoi(argv[5]);
    training_size = atoi(argv[6]);
    testing_size = atoi(argv[7]);
    nEpochs = atoi(argv[8]);
    filename_train = argv[9];
    train_labels = argv[10];
    filename_test = argv[11];

    // read input
    float *training_input, *training_labels, *testing_input, *testing_labels;
    training_input = readImageData(filename_train, training_size * nFeatures);
    training_labels = readLabels(train_labels, training_size);


    // srand(87);
    // nFeatures = 28 * 28;
    // batchSize = 128;
    // nOutput = 15;
    // nHiddenLayer = 3500;

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

    // Allocate host memory for the input, layers, and result arrays
    h_input = (float *)malloc(sizeof(float) * nFeatures * batchSize);

    h_weights = (float *)malloc(sizeof(float) * nFeatures * nHiddenLayer);

    h_weightsOutput = (float *)malloc(sizeof(float) * nHiddenLayer * nOutput);

    h_activation = (float *)malloc(sizeof(float) * nHiddenLayer * batchSize);

    h_result = (float *)malloc(sizeof(float) * nOutput * batchSize);

    h_loss = (float *)malloc(sizeof(float) * batchSize);

    h_labels = (float *)malloc(sizeof(float) * nOutput * batchSize);

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

    cudaEvent_t E0, E1, E2, E3;
    cudaEventCreate(&E0);

    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);


    float totalTime;

    // Initialize the neural network weights with random values
    for (int i = 0; i < nFeatures * nHiddenLayer; i++)
    {

        h_weights[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++)
    {
        h_weightsOutput[i] = -1.0f + 2.0f * rand() / (float)RAND_MAX;
    }

    int iterations = training_size / batchSize;
    cudaMemcpy(d_weights, h_weights, sizeof(float) * nFeatures * nHiddenLayer, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsOutput, h_weightsOutput, sizeof(float) * nHiddenLayer * nOutput, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        {
            printf("Error before iter: %s\n", cudaGetErrorString(err));
        }
    
         cudaEventRecord(E0, 0);
        cudaEventSynchronize(E0);

    for (int i = 0; i < 7; ++i)
    {
        int startIndex = i * nFeatures * batchSize;
        // copy the corresponding inputs into device memory
        cudaMemcpy(d_input, training_input + startIndex, sizeof(float) * nFeatures * batchSize, cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error copy: %s\n", cudaGetErrorString(err));
        }
        cudaMemcpy(d_labels, training_labels + i * nOutput * batchSize, sizeof(float) * nOutput * batchSize, cudaMemcpyHostToDevice);
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error copy: %s\n", cudaGetErrorString(err));
        }

        int nThreads = 32;
        int nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
        int nBlocksM = (batchSize + nThreads - 1) / nThreads;
        float alpha  = learning_rate;

        dim3 grid(nBlocksM, nBlocksN, 1);
        dim3 block(nThreads, nThreads, 1);

        // Define the grid and block sizes for the CUDA kernel launch
        // dim3 grid(32, 32, 1);
        // dim3 block(32, 32, 1);

        // Call the forwardPass CUDA kernel

   

        // forward pass
        transpose<<<64, 1024>>>(batchSize, nFeatures, d_input, d_inputT);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error trans: %s\n", cudaGetErrorString(err));
        }
        matMult<<<grid, block>>>(nHiddenLayer, batchSize, nFeatures, d_weights, d_inputT, d_Z1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error nat: %s\n", cudaGetErrorString(err));
        }
        sigmoid<<<64, 1024>>>(nHiddenLayer * batchSize, d_Z1, d_activation);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error sig: %s\n", cudaGetErrorString(err));
        }

        nBlocksN = (nOutput + nThreads - 1) / nThreads;
        nBlocksM = (batchSize + nThreads - 1) / nThreads;
        dim3 grid1(nBlocksM, nBlocksN, 1);

        matMult<<<grid1, block>>>(nOutput, batchSize, nHiddenLayer, d_weightsOutput, d_activation, d_Z2);
        if (err != cudaSuccess)
        {
            printf("Error sig: %s\n", cudaGetErrorString(err));
        }
        globalSoftmaxPrimitive<<<batchSize, nOutput>>>(nOutput, batchSize, d_Z2, d_result);
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        // backward
        
        // derivative dZ2
        subtractMat<<<grid, block>>>(nOutput, batchSize, d_result, d_labels, d_dZ2);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error subtract dZ2: %s\n", cudaGetErrorString(err));
        }
        // derivative dW2
        transpose<<<grid, block>>>(nHiddenLayer, batchSize, d_activation, d_activationT);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error transpose dW2: %s\n", cudaGetErrorString(err));
        }

        matMult<<<grid, block>>>(nOutput, batchSize, nHiddenLayer, d_dZ2, d_activationT, d_dZ2);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error matMult dW2: %s\n", cudaGetErrorString(err));
        }

        scalarDivMat<<<grid, block>>>(nOutput, nHiddenLayer, batchSize, d_dZ2, d_dW2);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error scalar division dW2: %s\n", cudaGetErrorString(err));
        }
        // derivative dZ1
        derivativeReLu<<<grid, block>>>(nHiddenLayer, batchSize, d_Z1, d_gZ1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error derivative dZ1: %s\n", cudaGetErrorString(err));
        }

        transpose<<<grid, block>>>(nOutput, nHiddenLayer, d_weightsOutput, d_weightsOutputT);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error transpose dZ1: %s\n", cudaGetErrorString(err));
        }

        matMult<<<grid, block>>>(nHiddenLayer, nOutput,batchSize, d_weightsOutputT, d_dZ2, d_dZ1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error matMult dZ1: %s\n", cudaGetErrorString(err));
        }

        elementWiseProd<<<grid, block>>>(nHiddenLayer, batchSize, d_dZ1, d_gZ1, d_dZ1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error element wise prod dZ1: %s\n", cudaGetErrorString(err));
        }

        // derivative dW1
        transpose<<<grid, block>>>(nFeatures, batchSize, d_input, d_inputT);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error transpose dW1: %s\n", cudaGetErrorString(err));
        }

        matMult<<<grid, block>>>(nHiddenLayer, batchSize, nFeatures, d_dZ1, d_inputT, d_dW1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error matMult dW1: %s\n", cudaGetErrorString(err));
        }

        scalarDivMat<<<grid, block>>>(nHiddenLayer, nFeatures, batchSize, d_dW1, d_dW1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error scalar division dW1: %s\n", cudaGetErrorString(err));
        }

        //update
        // w1 = new_w1
        scalarProdMat<<<grid, block>>>(nHiddenLayer, nFeatures, alpha, d_dW1, d_dW1_alpha);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error scalar prod update W1: %s\n", cudaGetErrorString(err));
        }

        subtractMat<<<grid, block>>>(nHiddenLayer, nFeatures, d_weights, d_dW1_alpha, d_weights);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error subtract update W1: %s\n", cudaGetErrorString(err));
        }
        // w2 = new_w2
        scalarProdMat<<<grid, block>>>(nFeatures, nHiddenLayer, alpha, d_dW2, d_dW2_alpha);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error scalar prod update W2: %s\n", cudaGetErrorString(err));
        }

        subtractMat<<<grid, block>>>(nHiddenLayer, nOutput, d_weightsOutput, d_dW2_alpha, d_weightsOutput);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Error subtract update W2: %s\n", cudaGetErrorString(err));
        }

        // cudaMemcpy(h_Z2, d_Z2, sizeof(float) * nOutput * batchSize, cudaMemcpyDeviceToHost);
        // seqSoftmax(nOutput, batchSize, h_Z2, h_result);
        // cudaMemcpy(d_result, h_result, sizeof(float) * nOutput * batchSize, cudaMemcpyHostToDevice);
        categoricalCrossEntropy<<<batchSize, nOutput>>>(nOutput, batchSize, d_labels, d_result, d_loss);
        if (err != cudaSuccess)
        {
            printf("Error sig: %s\n", cudaGetErrorString(err));
        }
        cudaMemcpy(h_loss, d_loss, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);
        printf("in iteration %d, loss is %f\n", i, h_loss[0]);
    }
          cudaEventRecord(E1, 0);
        cudaEventSynchronize(E1);

    //get total time
    cudaEventElapsedTime(&totalTime, E0, E1);
    printf("Total time is %f ms\n", totalTime);
    //gflops calculation

     int numMatrixMult1Ops = 2.0f * batchSize * nFeatures * nHiddenLayer; // input x weights
    int numMatrixMult2Ops = 2.0f * batchSize * nHiddenLayer * nOutput;   // activationL1 x weightsOutput

    // Estimate the floating-point operations for the additions
    // Each matrix multiplication involves (nFeatures - 1) additions

    int numAdditionOps1 = nHiddenLayer * batchSize;
    int numAdditionOps2 = batchSize * (nOutput + nOutput - 1 + nOutput);

    // Total floating-point operations
    int totalFloatingPointOps = 7* (numMatrixMult1Ops + numMatrixMult2Ops + numAdditionOps1 + numAdditionOps2);
     printf("GFLOPs: %4.6f\n", totalFloatingPointOps / (totalTime * 1000000.0));
    
    // free memory in host

    free(h_weights);
    free(h_weightsOutput);
    free(h_activation);
    free(h_result);
    free(h_loss);
    free(training_input);
    free(training_labels);

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