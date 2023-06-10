#include "seq_primitives.h"
#include <math.h>
#include <stdio.h>

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
            C[i * m + j] = 0.0f;
            for (int k = 0; k < p; k++)
            {
                C[i * m + j] += A[i * p + k] * B[k * m + j];
            }
        }
    }
}

void seqDerivativeReLu(int n, int m, float *A, float *B)
{
    for (int i = 0; i < n*m; i++)
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

void seqElementWiseProduct(int n, int m, float *A, float *B, float *C)
{
    for (int i = 0; i < n * m; i++)
    {
        C[i] = A[i] * B[i];
    }
    
}

void seqScalarProdMat(int n, int m, float *A, float scalar, float *B) {
    for (int i = 0; i < n*m; i++)
    {
        B[i] = A[i] * scalar;
    }
}

void seqScalarDivMat(int n, int m, float *A, float scalar, float *B)
{
    for (int i = 0; i < n*m; i++)
    {
        B[i] = A[i] / scalar;
    }
}

void seqSoftmax(int nOutput, int batchSize, float *A, float *B)
{
   // a matrix and b matrix are nOutputxbatchSize
    for (int i = 0; i < batchSize; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < nOutput; j++)
        {
            sum += exp(A[j * batchSize + i]);
        }
        
        for (int j = 0; j < nOutput; j++)
        {
            B[j * batchSize + i] = exp(A[j * batchSize + i]) / sum;
        
            
        }
        
    }
}

void seqReLu(int n, int m, float *A, float *B)
{
    for (int i = 0; i < n*m; i++)
    {
        if (A[i] > 0)
        {
            B[i] = A[i];
        }
        else
        {
            B[i] = 0.0f;
        }
    }
}

void seqCrossEntropy(int batchSize, int nOutput, float *h_result, float *h_labels, float *h_loss) {
    for (int i = 0; i < batchSize; i++)
    {   
        h_loss[i] = 0.0f;
        for (int j = 0; j < nOutput; j++)
        {
            h_loss[i] -= h_labels[i * nOutput + j] * log(h_result[i * nOutput + j] != 0 ? h_result[i * nOutput + j] : 1e-30f);
        }
    }
}

int seqMaxIdx(int n, float *A) {
    int maxIdx = 0;
    float maxVal = A[0];
    for (int i = 1; i < n; i++)
    {
        if (A[i] > maxVal)
        {
            maxVal = A[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

float seqAccuracy(int batchSize, int nOutput, float *h_result, float *h_labels) {
    int correct = 0;
    for (int i = 0; i < batchSize; i++)
    {
        int maxIdx = seqMaxIdx(nOutput, &h_result[i * nOutput]);
        if (h_labels[i * nOutput + maxIdx] == 1.0f)
        {
            // printf("In example %d, the network predicted %d, which is correct\n", i, maxIdx);
            // printf("h_result: %f\n, h_labels: %f\n", h_result[i * nOutput + maxIdx], h_labels[i * nOutput + maxIdx]);
            // printf("maxIdx: %d\n", maxIdx);
            correct++;
        }
    }
    // printf("Correct: %d\n", correct);
    // printf("Batch size: %d\n", batchSize);
    return (float)correct / (float)batchSize;
}

