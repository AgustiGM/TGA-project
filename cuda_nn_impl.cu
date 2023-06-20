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
    n = read(fd, data, size * sizeof(unsigned char));
    float *fdata = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        fdata[i] = (float)(data[i]) / 255.0f;
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
    int nFeatures, batchSize, nOutput, nHiddenLayer, training_size, test_size, nEpochs;
    float learning_rate;
    int nThreads = 32;
    char *filename_train, *train_labels, *filename_test, *test_labels;
    // todo parse arguments
    if (argc != 13)
    {
        printf("Usage: ./nn <nFeatures> <batchSize> <nOutput> <nHiddenLayer> <learning_rate> <training_size> <testing_size> <nEpochs> <filename_train> <train_labels> <filename_test> <test_labels>\n");
        exit(1);
    }
    nFeatures = atoi(argv[1]);
    batchSize = atoi(argv[2]);
    nOutput = atoi(argv[3]);
    nHiddenLayer = atoi(argv[4]);
    learning_rate = atof(argv[5]);
    training_size = atoi(argv[6]);
    test_size = atoi(argv[7]);
    nEpochs = atoi(argv[8]);
    filename_train = argv[9];
    train_labels = argv[10];
    filename_test = argv[11];
    test_labels = argv[12];

    printf("Data read\n");
    // read input
    float *training_input, *training_labels, *testing_input, *testing_labels;
    training_input = readImageData(filename_train, training_size * nFeatures);
    training_labels = readLabels(train_labels, training_size, nOutput);

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

    float *d_accuracy;

    float *d_test, *d_testLabels;

    float *d_testT;

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

    float *h_accuracy = (float *)malloc(sizeof(float)*batchSize);

    printf("Memory allocated in host\n");
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

    cudaMalloc((void **)&d_accuracy, batchSize * sizeof(float));

    printf("Memory allocated\n");
    // initialize weights
    for (int i = 0; i < nFeatures * nHiddenLayer; i++)
    {
        h_weights[i] = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++)
    {
        h_weightsOutput[i] = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
    }
    printf("Weights initialized\n");
    // copy weights to device
    cudaMemcpy(d_weights, h_weights, nFeatures * nHiddenLayer * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsOutput, h_weightsOutput, nHiddenLayer * nOutput * sizeof(float), cudaMemcpyHostToDevice);

    // create cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // get first event time
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    float *d_training_input;
    float *d_training_labels;

    cudaMalloc((void **)&d_training_input, training_size * nFeatures * sizeof(float));
    cudaMalloc((void **)&d_training_labels, training_size * nOutput * sizeof(float));

    cudaMemcpy(d_training_input, training_input, training_size * nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_training_labels, training_labels, training_size * nOutput * sizeof(float), cudaMemcpyHostToDevice);

    printf("Training data copied to device\n");
    // compute number of iterations
    int nIterations = training_size / batchSize;
    for (int epoch = 0; epoch < nEpochs; ++epoch)
    {
        printf("Epoch %d\n", epoch);
        float e_accuracy = 0.0f;

        for (int i = 0; i < nIterations; ++i)
        {
            d_input = d_training_input + i * batchSize * nFeatures;
            d_labels = d_training_labels + i * batchSize * nOutput;
            // memcpy(h_input, training_input + i * batchSize * nFeatures, batchSize * nFeatures * sizeof(float));
            // memcpy(h_labels, training_labels + i * batchSize * nOutput, batchSize * nOutput * sizeof(float));

            // copy the the input and labels to device
            // cudaMemcpy(d_input, training_input + i * batchSize * nFeatures, batchSize * nFeatures * sizeof(float), cudaMemcpyHostToDevice);
            // cudaMemcpy(d_labels, training_labels + i * batchSize * nOutput, batchSize * nOutput * sizeof(float), cudaMemcpyHostToDevice);

            // transpose labels

            transpose<<<batchSize, 1>>>(batchSize, nOutput, d_labels, d_labelsT);

            transpose<<<32, 1024>>>(batchSize, nFeatures, d_input, d_inputT);

            // transpose input

            // define nBlocksN and nBlocksM
            int nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
            int nBlocksM = (batchSize + nThreads - 1) / nThreads;

            dim3 dimGrid(nBlocksM, nBlocksN, 1);
            dim3 dimBlock(nThreads, nThreads, 1);

            // compute Z1 in device
            // C(N × M) ← A(N × P) · B (P × M)
            matMult<<<dimGrid, dimBlock>>>(nHiddenLayer, batchSize, nFeatures, d_weights, d_inputT, d_Z1);

            // compute activation in device
            reLU<<<64, 1024>>>(nHiddenLayer * batchSize, d_Z1, d_activation);

            // define nBlocksN and nBlocksM
            nBlocksN = (nOutput + nThreads - 1) / nThreads;
            nBlocksM = (batchSize + nThreads - 1) / nThreads;

            dimGrid = dim3(nBlocksM, nBlocksN, 1);

            // C(N × M) ← A(N × P) · B (P × M)
            matMult<<<dimGrid, dimBlock>>>(nOutput, batchSize, nHiddenLayer, d_weightsOutput, d_activation, d_Z2);

            // compute result
            globalSoftmaxPrimitive<<<32, 1024>>>(nOutput, batchSize, d_Z2, d_result);

            transpose<<<nOutput, 1024>>>(nOutput, batchSize, d_result, d_resultT);

            accuracy<<<32, 1024>>>(batchSize, nOutput, d_resultT, d_labels, d_accuracy);

            cudaMemcpy(h_accuracy, d_accuracy, sizeof(float)*batchSize, cudaMemcpyDeviceToHost);
            float batch_acc = 0.0f;
            for (int j = 0; j < batchSize; ++j) {
                batch_acc += h_accuracy[j];
            }
            e_accuracy += batch_acc/batchSize;

            // compute gradients

            // compute dZ2 (nOutputxbatchSize) in device
            subtractMat<<<64, 1024>>>(nOutput, batchSize, d_result, d_labelsT, d_dZ2);

            // transpose h_activation to batchSize x nHiddenLayer in device
            transpose<<<nHiddenLayer, 1024>>>(nHiddenLayer, batchSize, d_activation, d_activationT);

            // compute nBlocksN and nBlocksM
            nBlocksN = (nOutput + nThreads - 1) / nThreads;
            nBlocksM = (nHiddenLayer + nThreads - 1) / nThreads;

            dimGrid = dim3(nBlocksM, nBlocksN, 1);

            // compute dW2 in device
            // C(N × M) ← A(N × P) · B (P × M)
            // h_dW2 nOutputxnHiddenLayer
            matMult<<<dimGrid, dimBlock>>>(nOutput, nHiddenLayer, batchSize, d_dZ2, d_activationT, d_dW2);

            // divide by batchsize in device

            scalarDivMat<<<64, 1024>>>(nOutput, nHiddenLayer, batchSize, d_dW2, d_dW2);

            transpose<<<nOutput, 1024>>>(nOutput, nHiddenLayer, d_weightsOutput, d_weightsOutputT);

            // compute nBlocksN and nBlocksM
            nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
            nBlocksM = (batchSize + nThreads - 1) / nThreads;

            dimGrid = dim3(nBlocksM, nBlocksN);

            // compute dZ1 in device
            // C(N × M) ← A(N × P) · B (P × M)

            matMult<<<dimGrid, dimBlock>>>(nHiddenLayer, batchSize, nOutput, d_weightsOutputT, d_dZ2, d_dZ1);

            // compute derivative of z1 in device
            derivativeReLu<<<nHiddenLayer, batchSize>>>(nHiddenLayer, batchSize, d_Z1, d_dgZ1);

            // compute dZ1 in device
            elementWiseProd<<<nHiddenLayer, 1024>>>(nHiddenLayer, batchSize, d_dZ1, d_dgZ1, d_dZ1);

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

            // error checking
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
        printf("nIterations %d\n", nIterations);
        printf("eaccuracy %f\n", e_accuracy);
        printf("epoch %d accuracy %f\n", epoch, e_accuracy / nIterations);
    }

    // get last event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calculate time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // synch events

    double gflop = (2 * nHiddenLayer * batchSize * nFeatures +
                    nHiddenLayer * batchSize +
                    2 * nOutput * batchSize * nHiddenLayer +
                    batchSize * nOutput +
                    nOutput * batchSize +
                    2 * nOutput * nHiddenLayer * batchSize +
                    nOutput * nHiddenLayer * batchSize +
                    2 * nHiddenLayer * batchSize * nOutput +
                    nHiddenLayer * batchSize +
                    nHiddenLayer * batchSize +
                    2 * nHiddenLayer * nFeatures * batchSize +
                    nHiddenLayer * nFeatures +
                    nHiddenLayer * nFeatures * batchSize +
                    nHiddenLayer * nFeatures +
                    nOutput * nHiddenLayer +
                    nOutput * nHiddenLayer

    );

    printf("GFLOP: %f\n", gflop);
    gflop /= 1e9;

    // print gflop

    printf("GFLOP: %f\n", gflop);

    printf("Elapsed time : %f s\n", elapsedTime / 1000);

    // print gflops
    printf("GFLOP/s: %f\n", gflop / (elapsedTime / (nIterations * nEpochs * 1000.0)));

    // // check accuracy for test set
    // float *d_Z1Test, *d_Z2Test, *d_activationTest, *d_resultTest, *d_resultTestT;

    // float *d_accuracyTest;
    // printf("Testing\n");

    // printf("Reading test data from file %s\n", filename_test);
    // printf("Reading test labels from file %s\n", test_labels);

    // testing_input = readImageData(filename_test, test_size * nFeatures);
    // testing_labels = readLabels(test_labels, test_size, nOutput);


    // cudaMalloc((void **)&d_test, test_size * nFeatures * sizeof(float));
    // cudaMalloc((void **)&d_testT, test_size * nFeatures * sizeof(float));
    // cudaMalloc((void **)&d_accuracyTest, test_size * sizeof(float));
    // cudaMalloc((void **)&d_testLabels, test_size * nOutput * sizeof(float));
    

    // cudaMalloc((void **)&d_Z1Test, test_size * nHiddenLayer * sizeof(float));
    // cudaMalloc((void **)&d_Z2Test, test_size * nOutput * sizeof(float));
    // cudaMalloc((void **)&d_activationTest, test_size * nOutput * sizeof(float));
    // cudaMalloc((void **)&d_resultTest, test_size * nOutput * sizeof(float));
    // cudaMalloc((void **)&d_resultTestT, test_size * nOutput * sizeof(float));

    // cudaMemcpy(d_test, testing_input, test_size * nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_testLabels, testing_labels, test_size * nOutput * sizeof(float), cudaMemcpyHostToDevice);

    // // transpose
    // transpose<<<32, 1024>>>(test_size, nFeatures, d_test, d_testT);
    
    // float* resultTest = (float*)malloc(test_size * nOutput * sizeof(float));

    // // perform forward propagation

    // // compute Z1 in device
    // // C(N × M) ← A(N × P) · B (P × M)
    // int nBlocksN = (nHiddenLayer + nThreads - 1) / nThreads;
    // int nBlocksM = (test_size + nThreads - 1) / nThreads;

    // dim3 dimGrid(nBlocksM, nBlocksN);
    // dim3 dimBlock(nThreads, nThreads);

    // matMult<<<dimGrid, dimBlock>>>(nHiddenLayer, test_size, nFeatures, d_weights, d_testT, d_Z1Test);

    // // compute activation in device
    // reLU<<<nHiddenLayer, 1024>>>(nHiddenLayer*test_size, d_Z1Test, d_activationTest);

    // // compute Z2 in device
    // // C(N × M) ← A(N × P) · B (P × M)
    // nBlocksN = (nOutput + nThreads - 1) / nThreads;
    // nBlocksM = (test_size + nThreads - 1) / nThreads;

    // dimGrid = dim3(nBlocksM, nBlocksN);

    // matMult<<<dimGrid, dimBlock>>>(nOutput, test_size, nHiddenLayer, d_weightsOutput, d_activationTest, d_Z2Test);

    // // compute result in device
    // globalSoftmaxPrimitive<<<32, 1024>>>(nOutput, test_size, d_Z2Test, d_resultTest);

    // transpose<<<32, 1024>>>(nOutput, test_size, d_resultTest, d_resultTestT);

    // cudaMemcpy(resultTest, d_resultTestT, test_size * nOutput * sizeof(float), cudaMemcpyDeviceToHost);

    // // print firsts 10 results

    // // for (int i = 0; i < 10; i++)
    // // {
    // //     printf("Result %d: ", i);
    // //     for (int j = 0; j < nOutput; j++)
    // //     {
    // //         printf("%f ", resultTest[i * nOutput + j]);
    // //     }
    // //     printf("\n");
    // // }

    // // // print first 10 labels
    // // for (int i = 0; i < 10; i++)
    // // {
    // //     printf("Label %d: ", i);
    // //     for (int j = 0; j < nOutput; j++)
    // //     {
    // //         printf("%f ", testing_labels[i * nOutput + j]);
    // //     }
    // //     printf("\n");
    // // }

    // // compute accuracy
    // accuracy<<<test_size, 1024>>>(test_size, nOutput, d_resultTestT, d_testLabels, d_accuracyTest);

    // float *h_accuracyTest = (float *)malloc(sizeof(float)*test_size);

    // cudaMemcpy(h_accuracyTest, d_accuracyTest, sizeof(float), cudaMemcpyDeviceToHost);
    // float test_acc = 0.0f;
    // for (int j = 0; j < test_size; j++)
    // {   
    //     if (h_accuracyTest[j] != 0.0f) {
    //         printf("Accuracy %d: %f\n", j, h_accuracyTest[j]);
    //     }
    //     test_acc += h_accuracyTest[j];
    // }
    // printf("Accuracy in the test set: %f\n", test_acc);
    // printf("Accuracy in the test set: %f\n", test_acc/(float)test_size);
    // cudaError_t err = cudaGetLastError();
    //         if (err != cudaSuccess)
    //         {
    //             printf("Error: %s\n", cudaGetErrorString(err));
    //             exit(-1);
    //         }

    // allocate memory for device

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
    free(testing_input);
    free(testing_labels);
    free(h_resultT);
    free(h_accuracy);
    

    // free device memory
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
    cudaFree(d_accuracy);
    // cudaFree(d_test);
    // cudaFree(d_testT);
    // cudaFree(d_testLabels);
    // cudaFree(d_Z1Test);
    // cudaFree(d_Z2Test);
    // cudaFree(d_activationTest);
    // cudaFree(d_resultTest);
    // cudaFree(d_resultTestT);

    
    

    return 0;
}
