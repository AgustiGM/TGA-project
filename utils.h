#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void InitM(int N, int M, float *Mat);

int error(float a, float b);

int TestMM(int N, int M, int P, float *A, float *B, float *C);

int testSigmoid(int N, float *S, float *T);

int testReLU(int N, float *S, float *T);