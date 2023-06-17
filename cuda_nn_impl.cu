#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <math.h>

#include "primitives.h"
#include "nnfunctions.h"
#include "seq_primitives.h"

float *readImageData(char *filename, int size)
{
    int fd = open(filename, O_RDONLY);
    unsigned char buf[16];
    int n = read(fd, buf, 16);

    unsigned char *data = (unsigned char *)malloc(size);
    n = read(fd, data, size*sizeof(unsigned char));
    float *fdata = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        fdata[i] = (float) (data[i]) / 255.0f;
        
    }
    free(data);
    close(fd);
    return fdata;
}

float *readLabels(char *filename, int size, int nLabels)
{
    int fd = open(filename, O_RDONLY);
    unsigned char buf[8];
    int n = read(fd, buf, 8);

    unsigned char *data = (unsigned char *)malloc(size);
    n = read(fd, data, size);
    float *fdata = (float *)malloc(size * nLabels * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < nLabels; j++)
        {
            if (j == data[i])
                fdata[i * nLabels + j] = 1.0f;
            else
                fdata[i * nLabels + j] = 0.0f;
        }
    }
    free(data);
    close(fd);
    return fdata;
}

// parallel nn implementation
int main(int argc, char **argv)
{
    int nFeatures, batchSize, nOutput, nHiddenLayer, training_size, testing_size, nEpochs;
    float learning_rate;
    int nThreads =32;
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
    learning_rate = atof(argv[5]);
    training_size = atoi(argv[6]);
    testing_size = atoi(argv[7]);
    nEpochs = atoi(argv[8]);
    filename_train = argv[9];
    train_labels = argv[10];
    filename_test = argv[11];

    // read input
    float *training_input, *training_labels, *testing_input, *testing_labels;
    training_input = readImageData(filename_train, training_size * nFeatures);
    training_labels = readLabels(train_labels, training_size, nOutput);

    // for (int i = 0; i < 28; i++)
    // {
    //     for (int j = 0; j < 28; j++)
    //     {
    //         printf("%d ", (int) (training_input[i*28  + j] * 255.0f));
    //     }
    //     printf("\n");
    // }

    // for (int i = 0; i < 4; i++)
    // {
    //     for (int j = 0; j < nOutput; j++)
    //     {
    //         printf("%f ", training_labels[i * nOutput + j]);
    //     }
    //     printf("\n");
    // }

    // define variables to hold weights

    float *h_input, *h_weights, *h_weightsOutput, *h_Z1, *h_activation, *h_Z2, *h_result;
    float *h_inputT, *h_weightsOutputT, *h_activationT;
    float *h_labels, *h_loss;
    float *h_resultT;

    float *h_dW1, *h_dW2;

    float *h_dZ2;
    

    float *h_dZ1;

    float *h_dgZ1;

    // define variables in device memory

    float *d_input, *d_weights, *d_weightsOutput, *d_Z1, *d_activation, *d_Z2, *d_result;
    float *d_inputT, *d_weightsOutputT, *d_activationT;
    float *d_labels, *d_loss;
    float *d_resultT;

    float *d_dW1, *d_dW2;
    // srand(time(NULL));

    float *d_dZ2;

    float *d_dZ1;

    float *d_dgZ1;

    float *d_labelsT;

    // allocate memory

    h_input = (float *)malloc(batchSize * nFeatures * sizeof(float));
    h_weights = (float *)malloc(nFeatures * nHiddenLayer * sizeof(float));
    h_weightsOutput = (float *)malloc(nHiddenLayer * nOutput * sizeof(float));
    h_Z1 = (float *)malloc(batchSize * nHiddenLayer * sizeof(float));
    h_activation = (float *)malloc(batchSize * nHiddenLayer * sizeof(float));
    h_Z2 = (float *)malloc(batchSize * nOutput * sizeof(float));
    h_result = (float *)malloc(batchSize * nOutput * sizeof(float));

    h_resultT = (float *)malloc(batchSize * nOutput * sizeof(float));

    float *h_labelsT = (float *)malloc(batchSize * nOutput * sizeof(float));

    
    h_inputT = (float *)malloc(nFeatures * batchSize * sizeof(float));
    h_weightsOutputT = (float *)malloc(nOutput * nHiddenLayer * sizeof(float));
    h_activationT = (float *)malloc(nHiddenLayer * batchSize * sizeof(float));
    h_labels = (float *)malloc(batchSize * nOutput * sizeof(float));
    h_loss = (float *)malloc(batchSize * nOutput * sizeof(float));
    h_dW1 = (float *)malloc(nFeatures * nHiddenLayer * sizeof(float));
    h_dW2 = (float *)malloc(nHiddenLayer * nOutput * sizeof(float));
    h_dZ2 = (float *)malloc(batchSize * nOutput * sizeof(float));
    h_dZ1 = (float *)malloc(batchSize * nHiddenLayer * sizeof(float));
    h_dgZ1 = (float *)malloc(batchSize * nHiddenLayer * sizeof(float));

    // allocate memory in device

    cudaMalloc((void **)&d_input, batchSize * nFeatures * sizeof(float));
    cudaMalloc((void **)&d_weights, nFeatures * nHiddenLayer * sizeof(float));
    cudaMalloc((void **)&d_weightsOutput, nHiddenLayer * nOutput * sizeof(float));
    cudaMalloc((void **)&d_Z1, batchSize * nHiddenLayer * sizeof(float));
    cudaMalloc((void **)&d_activation, batchSize * nHiddenLayer * sizeof(float));
    cudaMalloc((void **)&d_Z2, batchSize * nOutput * sizeof(float));
    cudaMalloc((void **)&d_result, batchSize * nOutput * sizeof(float));

    cudaMalloc((void **)&d_resultT, batchSize * nOutput * sizeof(float));

    cudaMalloc((void **)&d_labelsT, batchSize * nOutput * sizeof(float));

    cudaMalloc((void **)&d_inputT, nFeatures * batchSize * sizeof(float));
    cudaMalloc((void **)&d_weightsOutputT, nOutput * nHiddenLayer * sizeof(float));
    cudaMalloc((void **)&d_activationT, nHiddenLayer * batchSize * sizeof(float));
    cudaMalloc((void **)&d_labels, batchSize * nOutput * sizeof(float));
    cudaMalloc((void **)&d_loss, batchSize * nOutput * sizeof(float));
    cudaMalloc((void **)&d_dW1, nFeatures * nHiddenLayer * sizeof(float));
    cudaMalloc((void **)&d_dW2, nHiddenLayer * nOutput * sizeof(float));
    cudaMalloc((void **)&d_dZ2, batchSize * nOutput * sizeof(float));
    cudaMalloc((void **)&d_dZ1, batchSize * nHiddenLayer * sizeof(float));
    cudaMalloc((void **)&d_dgZ1, batchSize * nHiddenLayer * sizeof(float));

    // initialize weights
    for (int i = 0; i < nFeatures * nHiddenLayer; i++)
    {
        h_weights[i] = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++)
    {
        h_weightsOutput[i] = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
    }

    // copy weights to device
    cudaMemcpy(d_weights, h_weights, nFeatures * nHiddenLayer * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsOutput, h_weightsOutput, nHiddenLayer * nOutput * sizeof(float), cudaMemcpyHostToDevice);


    // compute number of iterations
    int nIterations = training_size / batchSize;
    for (int epoch = 0; epoch < nEpochs; ++epoch)
    {
        printf("Epoch %d\n", epoch);
        float e_accuracy = 0.0f;
        printf("eaccuracy %f\n", e_accuracy);

        for (int i = 0; i < nIterations; ++i)
        {
            
            memcpy(h_input, training_input + i * batchSize * nFeatures, batchSize * nFeatures * sizeof(float));
            memcpy(h_labels, training_labels + i * batchSize * nOutput, batchSize * nOutput * sizeof(float));

            // copy the the input and labels to device
            cudaMemcpy(d_input, h_input, batchSize * nFeatures * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, h_labels, batchSize * nOutput * sizeof(float), cudaMemcpyHostToDevice);

            //transpose labels
            transpose<<<batchSize,1>>>(batchSize, nOutput, d_labels, d_labelsT);
            
            transpose<<<batchSize,1>>>(batchSize, nFeatures, d_input, d_inputT);

            // transpose input
            

            // define nBlocksN and nBlocksM
            int nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
            int nBlocksM = (batchSize + nThreads - 1) / nThreads;

            dim3 dimGrid(nBlocksM, nBlocksN,1);
            dim3 dimBlock(nThreads, nThreads,1);

            // compute Z1 in device
            // C(N × M) ← A(N × P) · B (P × M)
            matMult<<<dimGrid, dimBlock>>>(nHiddenLayer, batchSize, nFeatures, d_weights, d_inputT, d_Z1);

            // copy dz1 to host
             cudaMemcpy(h_Z1, d_Z1, nHiddenLayer * batchSize * sizeof(float), cudaMemcpyDeviceToHost);

            //  if (i == 0 && epoch == 0) {
            //     for (int j = 0; j < nHiddenLayer; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_Z1[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            // }
            

            // compute activation in device
            reLU<<<64, 1024>>>(nHiddenLayer* batchSize, d_Z1, d_activation);
            

            // define nBlocksN and nBlocksM
            nBlocksN = (nOutput + nThreads - 1) / nThreads;
            nBlocksM = (batchSize + nThreads - 1) / nThreads;

            dimGrid = dim3(nBlocksM, nBlocksN,1);


            // compute Z2
            // C(N × M) ← A(N × P) · B (P × M)
            matMult<<<dimGrid, dimBlock>>>(nOutput, batchSize, nHiddenLayer, d_weightsOutput, d_activation, d_Z2);

            cudaMemcpy(h_Z2, d_Z2, batchSize * nOutput * sizeof(float), cudaMemcpyDeviceToHost);

            //print z2
            // if (i == 0 && epoch == 0) {
            //     for (int j = 0; j < nOutput; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_Z2[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            // }

            // transpose<<<512,1024>>>(nOutput, batchSize, d_Z2, d_resultT);

            // compute result
            globalSoftmaxPrimitive<<<batchSize, 1>>>(nOutput, batchSize, d_Z2, d_result);
            
            // // transpose result
            // transpose<<<nOutput,1024>>>(nOutput, batchSize, d_result, d_resultT);


            // copy result to host for checking
            cudaMemcpy(h_result, d_result, batchSize * nOutput * sizeof(float), cudaMemcpyDeviceToHost);

            // print h_result  
            // if (i == 0 && epoch == 0)
            // {
            //     for (int j = 0; j < nOutput; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_result[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            // }
            
            seqTranspose(nOutput, batchSize, h_result, h_resultT);

            // print results

            // compute loss

            seqCrossEntropy(batchSize, nOutput, h_resultT, h_labels, h_loss);

            float seqAcc = seqAccuracy(batchSize, nOutput, h_resultT, h_labels);
            e_accuracy += seqAcc;

            float avg_loss = 0.0;
            for (int j = 0; j < batchSize * nOutput; j++)
            {
                avg_loss += h_loss[j];
            }
            avg_loss /= batchSize * nOutput;

            // compute gradients
            // if (i == 0)
            //     printf("loss: %f; accuracy: %f\n", avg_loss, seqAcc);

            // compute dZ2 (nOutputxbatchSize) in device
            subtractMat<<<64, 1024>>>(nOutput, batchSize, d_result, d_labelsT, d_dZ2);

            cudaMemcpy(h_dZ2, d_dZ2, batchSize * nOutput * sizeof(float), cudaMemcpyDeviceToHost);

            
            // print h_dZ2

           

            // transpose h_activation to batchSize x nHiddenLayer in device
            transpose<<<nHiddenLayer,1024>>>(nHiddenLayer, batchSize, d_activation, d_activationT);

            // compute nBlocksN and nBlocksM
            nBlocksN = (nOutput + nThreads - 1) / nThreads;
            nBlocksM = (nHiddenLayer + nThreads - 1) / nThreads;

            dimGrid = dim3(nBlocksM, nBlocksN,1);
            
            // compute dW2 in device
            // C(N × M) ← A(N × P) · B (P × M)
            // h_dW2 nOutputxnHiddenLayer
            matMult<<<dimGrid, dimBlock>>>(nOutput, nHiddenLayer, batchSize, d_dZ2, d_activationT, d_dW2);

            // divide by batchsize in device

            scalarDivMat<<<64, 1024>>>(nOutput, nHiddenLayer, batchSize, d_dW2, d_dW2);

            // copy dW2 to host

            // cudaMemcpy(h_dW2, d_dW2, nOutput * nHiddenLayer * sizeof(float), cudaMemcpyDeviceToHost);

            // print h_dW2
            
            // if (i == nIterations/2 && epoch == 0)
            // {
            //     for (int j = 0; j < nOutput; j++)
            //     {
            //         for (int k = 0; k < nHiddenLayer; k++)
            //         {
            //             printf("%f ", h_dW2[j * nHiddenLayer + k]);
            //         }
            //         printf("\n");
            //     }
            // }

            // transpose h_weightsOutput to nHiddenLayer x nOutput

            transpose<<<nOutput,1024>>>(nOutput, nHiddenLayer, d_weightsOutput, d_weightsOutputT);
            
            // compute nBlocksN and nBlocksM
            nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
            nBlocksM = (batchSize + nThreads - 1) / nThreads;

            dimGrid = dim3(nBlocksM, nBlocksN);

            // compute dZ1 in device
            // C(N × M) ← A(N × P) · B (P × M)

            matMult<<<dimGrid, dimBlock>>>(nHiddenLayer, batchSize, nOutput, d_weightsOutputT, d_dZ2, d_dZ1);

            // // copy dZ1 to host

            // cudaMemcpy(h_dZ1, d_dZ1, batchSize * nHiddenLayer * sizeof(float), cudaMemcpyDeviceToHost);

            // // print h_dZ1

            // if (i == 0 && epoch == 0) {
            //     for (int j = 0; j < nHiddenLayer; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_dZ1[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }



            // compute derivative of z1 in device
            derivativeReLu<<<nHiddenLayer, batchSize>>>(nHiddenLayer, batchSize, d_Z1, d_dgZ1);

            // // copy d_dgZ1 to host

            // cudaMemcpy(h_dgZ1, d_dgZ1, batchSize * nHiddenLayer * sizeof(float), cudaMemcpyDeviceToHost);

            // // print h_dgZ1

            // if (i == 0 && epoch == 0) {
            //     for (int j = 0; j < nHiddenLayer; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_dgZ1[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }


            // compute dZ1 in device
            elementWiseProd<<<nHiddenLayer, 1024>>>(nHiddenLayer, batchSize, d_dZ1, d_dgZ1, d_dZ1);

            cudaMemcpy(h_dZ1, d_dZ1, batchSize * nHiddenLayer * sizeof(float), cudaMemcpyDeviceToHost);

            // print h_dZ1

            // if (i == 0 && epoch == 0)
            // {
            //     for (int j = 0; j < nHiddenLayer; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_dZ1[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            // }

 

            // compute nBlocksN and nBlocksM
            nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
            nBlocksM = (nFeatures + nThreads - 1) / nThreads;

            // dimGrid = dim3(nBlocksN, nBlocksM);

            dim3 dimGrid2(nBlocksM, nBlocksN);



            
            // compute dw1
            // C(N × M) ← A(N × P) · B (P × M)
            // printf("now\n");
            matMult<<<dimGrid2, dimBlock>>>(nHiddenLayer, nFeatures, batchSize, d_dZ1, d_input, d_dW1);
            // printf("stop\n");

            // // print input
            // if (i == 0 && epoch == 0)
            // {
            //     for (int j = 0; j < nFeatures; j++)
            //     {
            //         for (int k = 0; k < batchSize; k++)
            //         {
            //             printf("%f ", h_input[j * batchSize + k]);
            //         }
            //         printf("\n");
            //     }
            // }

            // copy dW1 to host for checking
            cudaMemcpy(h_dW1, d_dW1, nHiddenLayer * nFeatures * sizeof(float), cudaMemcpyDeviceToHost);

            // print h_dW1

            // if (i == 0 && epoch == 0) {
            //     printf("dw1\n");
            //     for (int j = 0; j < nHiddenLayer; j++)
            //     {
            //         for (int k = 0; k < nFeatures; k++)
            //         {
            //             if (h_dW1[j * nFeatures + k] != 0)
            //                 printf("%f ", h_dW1[j * nFeatures + k]);
            //         }
            //         printf("\n");
            //     }
            // }
            
            // divide by batchsize in device
            scalarDivMat<<<64, 1024>>>(nHiddenLayer, nFeatures, batchSize, d_dW1, d_dW1);

            // update params

            // compute dw1 * alpha in device
            scalarProdMat<<<64, 1024>>>(nHiddenLayer, nFeatures, learning_rate, d_dW1, d_dW1);
            
            // perform the substraction in the device
            subtractMat<<<64, 1024>>>(nHiddenLayer, nFeatures, d_weights, d_dW1, d_weights);

            
            // compute dw2 * alpha in device
            scalarProdMat<<<64, 1024>>>(nOutput, nHiddenLayer, learning_rate, d_dW2, d_dW2);
            
            // perform the substraction in the device
            subtractMat<<<64, 1024>>>(nOutput, nHiddenLayer, d_weightsOutput, d_dW2, d_weightsOutput);
            
            //error checking
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }


            // if (i == 0)
            // {
            //     for (int j = 0; j < nHiddenLayer; j++)
            //     {
            //         for (int k = 0; k < nOutput; k++)
            //         {
            //             printf("%f \t", h_weightsOutput[j * nHiddenLayer + k]);
            //         }
            //         printf("\n");
            //     }
            // }
        }
        printf("nIterations %d\n", nIterations);
        printf("eaccuracy %f\n", e_accuracy);
        printf("epoch %d accuracy %f\n", epoch, e_accuracy / nIterations);
    }

    // free memory
    free(h_input);
    free(h_inputT);
    free(h_labels);
    free(h_weights);
    free(h_weightsOutput);
    free(h_weightsOutputT);
    free(h_Z1);
    free(h_activation);
    free(h_activationT);
    free(h_Z2);
    free(h_result);
    free(h_loss);
    free(h_dZ2);
    free(h_dW2);
    free(h_dZ1);
    free(h_dgZ1);
    free(h_dW1);
    free(training_input);
    free(training_labels);
    free(h_resultT);

    //free device memory
    cudaFree(d_input);
    cudaFree(d_inputT);
    cudaFree(d_labels);
    cudaFree(d_labelsT);
    cudaFree(d_weights);
    cudaFree(d_weightsOutput);
    cudaFree(d_weightsOutputT);
    cudaFree(d_Z1);
    cudaFree(d_activation);
    cudaFree(d_activationT);
    cudaFree(d_Z2);
    cudaFree(d_result);
    cudaFree(d_resultT);
    cudaFree(d_loss);
    cudaFree(d_dZ2);
    cudaFree(d_dW2);
    cudaFree(d_dZ1);
    cudaFree(d_dgZ1);
    cudaFree(d_dW1);


    return 0;
}
