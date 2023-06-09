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

// C(N × M) ← A(N × P) · B (P × M)
void seqMatMult(int n, int m, int p, float *A, float *B, float *C)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {   
            float sum = 0.0f;
            for (int k = 0; k < p; k++)
            {
                sum += A[i * p + k] * B[k * m + j];
            }
            
            C[i * m + j] = sum;
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
            h_loss[i] -= h_labels[i * nOutput + j] * log(h_result[i * nOutput + j]);
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
        if (h_labels[i * nOutput + maxIdx] != 0.0f)
        {
            correct++;
        }
    }
    return (float)correct / (float)batchSize;
}

void seqSigmoid(int n, int m, float *A, float *B)
{
    for (int i = 0; i < n*m; i++)
    {
        B[i] = 1.0f / (1.0f + exp(-A[i]));
    }
}

void seqSigmoidDerivative(int n, int m, float *A, float *B)
{
    for (int i = 0; i < n*m; i++)
    {
        B[i] = A[i] * (1.0f - A[i]);
    }
}

