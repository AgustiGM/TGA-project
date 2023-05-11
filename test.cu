#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnfunctions.h"
#include "utils.h"

#ifndef PINNED
#define PINNED 0
#endif

int main(int argc, char** argv)
{
  unsigned int N; N = 78;
  unsigned int numBytes;
  unsigned int nBlocks, nThreads;


  float* h, *r;
  float* origin;
  float *d, *dr, *outputs, *outputr;

     
  // numero de Threads en cada dimension 
  nThreads = 32;

  // numero de Blocks en cada dimension 
  nBlocks = (N+nThreads-1)/nThreads; 
    
  numBytes = N * sizeof(float);

  dim3 dimGrid(nBlocks, 1, 1);
  dim3 dimBlock(nThreads, 1, 1);



  h = (float*) malloc(numBytes); 
  r = (float*) malloc(numBytes); 
  origin = (float*) malloc(numBytes); 

  InitM(N,1,h);
  memcpy(origin,h,numBytes);
  memcpy(r,h,numBytes);
//   for (int i = 0; i < N; ++i) {or[i] = h[i];}


  // Inicialitzem les matrius dels pesos amb nombres aleatoris.

 
  // Obtener Memoria en el device
  cudaMalloc((float**)&d, numBytes); 
  cudaMalloc((float**)&dr, numBytes);
  cudaMalloc((float**)&outputs, numBytes); 
  cudaMalloc((float**)&outputr, numBytes);  


  // Copiar datos desde el host en el device 
  cudaMemcpy(d, h, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dr, h, numBytes, cudaMemcpyHostToDevice);
  


  
  // Ejecutar el kernel 
  globalSigmoid<<<dimGrid, dimBlock>>>(N, d, outputs);
  globalReLU<<<dimGrid, dimBlock>>>(N, dr, outputr);


  // Obtener el resultado desde el host 
  cudaMemcpy(h, outputs, numBytes, cudaMemcpyDeviceToHost); 
  cudaMemcpy(r, outputr, numBytes, cudaMemcpyDeviceToHost); 

  // Liberar Memoria del device 
  cudaFree(d);
  cudaFree(dr);
  cudaFree(outputs);
  cudaFree(outputr);


  

  if (testSigmoid(N, origin, h))
    printf ("SIGMOID TEST PASS\n");
  else
    printf ("SIGMOID TEST FAIL\n");

  if (testReLU(N, origin, r))
    printf ("ReLU TEST PASS\n");
  else
    printf ("ReLU TEST FAIL\n");



  free(h); free(r); free(origin);
 

}




