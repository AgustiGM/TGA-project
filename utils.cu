#include "utils.h"

void InitM(int N, int M, float *Mat) {
   int i;
   for (i=0; i<N*M; i++) 
     Mat[i] = rand() / (float) RAND_MAX;
   
}

int error(float a, float b) {
  float tmp;

  tmp = abs(a-b) / abs(min(a,b));

  if (isnan(tmp) || tmp > 0.0001) return 1;
  else  return 0;

}

int TestMM(int N, int M, int P, float *A, float *B, float *C) {
   int i, j, k;
   float tmp;
   for (i=0; i<N; i++)
     for (j=0; j<M; j++) {
       tmp = 0.0;
       for (k=0; k<P; k++) 
         tmp = tmp + A[i*P+k] * B[k*M+j]; 
       if (error(tmp, C[i*M+j])) {
         printf ("%d:%d: %f - %f = %f \n", i, j, tmp, C[i*M+j], abs(tmp - C[i*M+j]));
         return 0;
       }
     }
   
   return 1;
}

int testSigmoid(int N, float *S, float *T) {
    float tmp;
    for (int i = 0; i < N; ++i) {
        tmp = 1 / (1 + exp(-1*S[i]));
        if (error(tmp, T[i])) {
         printf ("%d: %f - %f = %f \n", i, tmp, T[i], abs(tmp - T[i]));
         return 0;
       }
    }
    return 1;
}

int testReLU(int N, float *S, float *T) {
    float tmp;
    for (int i = 0; i < N; ++i) {
        tmp = S[i] < 0 ? 0 : S[i];
        if (error(tmp, T[i])) {
         printf ("%d: %f - %f = %f \n", i, tmp, T[i], abs(tmp - T[i]));
         return 0;
       }
    }
    return 1;
}