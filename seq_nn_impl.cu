#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <math.h>

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

// sequential nn implementation
int main(int argc, char **argv)
{
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

    // initialize weights
    for (int i = 0; i < nFeatures * nHiddenLayer; i++)
    {
        h_weights[i] = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < nHiddenLayer * nOutput; i++)
    {
        h_weightsOutput[i] = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
    }

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
            seqTranspose(batchSize, nOutput, h_labels, h_labelsT);

            // transpose input
            seqTranspose(batchSize, nFeatures, h_input, h_inputT);

            // compute Z1
            // C(N × M) ← A(N × P) · B (P × M)
            seqMatMult(nHiddenLayer, batchSize, nFeatures, h_weights, h_inputT, h_Z1);

            // compute activation
            seqReLu(nHiddenLayer, batchSize, h_Z1, h_activation);

            // compute Z2
            // C(N × M) ← A(N × P) · B (P × M)
            seqMatMult(nOutput, batchSize, nHiddenLayer, h_weightsOutput, h_activation, h_Z2);

            // compute result
            seqSoftmax(nOutput, batchSize, h_Z2, h_result);

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

            // compute dZ2 (nOutputxbatchSize)
            seqSubstractMat(nOutput, batchSize, h_result, h_labelsT, h_dZ2);

            
            seqTranspose(nHiddenLayer, batchSize, h_activation, h_activationT);

        
            // compute dW2
            // C(N × M) ← A(N × P) · B (P × M)
            // h_dW2 nOutputxnHiddenLayer
            seqMatMult(nOutput, nHiddenLayer, batchSize, h_dZ2, h_activationT, h_dW2);

            // print the h_dwerights

            seqScalarDivMat(nOutput, nHiddenLayer, h_dW2, batchSize, h_dW2);

            // transpose h_weightsOutput to nHiddenLayer x nOutput
            seqTranspose(nOutput, nHiddenLayer, h_weightsOutput, h_weightsOutputT);

            // compute dZ1
            // C(N × M) ← A(N × P) · B (P × M)
            seqMatMult(nHiddenLayer, batchSize, nOutput, h_weightsOutputT, h_dZ2, h_dZ1);

            // compute derivative of z1
            seqDerivativeReLu(nHiddenLayer, batchSize, h_Z1, h_dgZ1);

            

            // compute dZ1

            seqElementWiseProduct(nHiddenLayer, batchSize, h_dZ1, h_dgZ1, h_dZ1);

            // print dz1

            // compute dw1
            seqMatMult(nHiddenLayer, nFeatures, batchSize, h_dZ1, h_input, h_dW1);

            // print non zeros of dw1

            // if (i == 0 && epoch == 0)
            // {
            //     printf("dw1\n");
            //     for (int j = 0; j < nHiddenLayer * nFeatures; j++)
            //     {
            //         if (h_dW1[j] != 0)
            //         {
            //             printf("%f\n", h_dW1[j]);
            //         }
            //     }
            // }

            seqScalarDivMat(nHiddenLayer, nFeatures, h_dW1, batchSize, h_dW1);

            // print dw1

            // update params

            // compute dw1 * alpha
            seqScalarProdMat(nHiddenLayer, nFeatures, h_dW1, learning_rate, h_dW1);


            seqSubstractMat(nHiddenLayer, nFeatures, h_weights, h_dW1, h_weights);

            // compute dw2 * alpha
            seqScalarProdMat(nOutput, nHiddenLayer, h_dW2, learning_rate, h_dW2);

            seqSubstractMat(nOutput, nHiddenLayer, h_weightsOutput, h_dW2, h_weightsOutput);
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
}
