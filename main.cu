#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnfunctions.h"
#include "utils.h"

#ifndef PINNED
#define PINNED 0
#endif

// Invocacion:
// ./ejecutable N M P test
// TAM es el la dimension de las matrices
// test == 'Y', comprueba que el resultado sea correcto
// test == 'N', NO comprueba que el resultado (Util para tomar tiempos)
// Por defecto, N = 639, M = 641, P = 1023, test == 'N'

int main(int argc, char** argv)
{
  unsigned int N, M, P;
  unsigned int numBytesC, numBytesA, numBytesB;
  unsigned int nBlocksN, nBlocksM, nThreads;
 
  float TiempoTotal, TiempoKernel;
  cudaEvent_t E0, E1, E2, E3;

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;

  char test;

  // Dimension de las matrices NxM, NxP, PxM y comprobacion resultado
  if (argc == 5) { 
     N = atoi(argv[1]); 
     M = atoi(argv[2]); 
     P = atoi(argv[3]); 
     test = *argv[4]; 
  }
  else { printf("Usage: ./exe N M P test\n"); exit(0); }

  int count, gpu;
  // Buscar GPU de forma aleatoria
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  gpu = (rand()>>3) % count;
  cudaSetDevice(gpu);
       
  // numero de Threads en cada dimension 
  nThreads = SIZE;

  // numero de Blocks en cada dimension 
  nBlocksN = (N+nThreads-1)/nThreads; 
  nBlocksM = (M+nThreads-1)/nThreads; 
  
  numBytesC = N * M * sizeof(float);
  numBytesA = N * P * sizeof(float);
  numBytesB = P * M * sizeof(float);

  dim3 dimGrid(nBlocksM, nBlocksN, 1);
  dim3 dimBlock(nThreads, nThreads, 1);

  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);
  
  // Matrius en calen tantes com nombres de hidden layers + la d'outputs.
  // Mida de l'input: nFeatures+1xnSamples. (Trasposada de nSamplesxnFeatures+1)
  // Mida de les matrius dels pesos. Primera: HiddenSizexnFeatures+1. i així fins a totes les hiddens (per simplificar totes tindran la mateixa mida)
  // Mida output: nHiddenSizexnOutputs
  // seq operacions: outputxhiddennxhiddenn-1...xhidden1xinputs. (ordre de multiplicació de matrius)

  if (PINNED) {
    // Obtiene Memoria [pinned] en el host
    cudaMallocHost((float**)&h_A, numBytesA); 
    cudaMallocHost((float**)&h_B, numBytesB); 
    cudaMallocHost((float**)&h_C, numBytesC); 
  }
  else {
    // Obtener Memoria en el host
    h_A = (float*) malloc(numBytesA); 
    h_B = (float*) malloc(numBytesB); 
    h_C = (float*) malloc(numBytesC); 
  }

  // Inicialitzem les matrius dels pesos amb nombres aleatoris.

  InitM(N, P, h_A);
  InitM(P, M, h_B);

  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);
  
  // Obtener Memoria en el device
  cudaMalloc((float**)&d_A, numBytesA); 
  cudaMalloc((float**)&d_B, numBytesB); 
  cudaMalloc((float**)&d_C, numBytesC); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(d_A, h_A, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, numBytesB, cudaMemcpyHostToDevice);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);
  
  // Ejecutar el kernel 
  // matMult<<<dimGrid, dimBlock>>>(N, M, P, d_A, d_B, d_C);

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host 
  cudaMemcpy(h_C, d_C, numBytesC, cudaMemcpyDeviceToHost); 

  // Liberar Memoria del device 
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  printf("\nKERNEL 11\n");
  printf("GPU utilizada: %d\n", gpu);
  printf("Dimensiones: %dx%d <- %dx%d * %dx%d\n", N, M, N, P, P, M);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocksM, nBlocksN, nBlocksN*nBlocksM);
  if (PINNED) printf("Usando Pinned Memory\n");
         else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoTotal));
  printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  if (test == 'N')
    printf ("NO TEST\n");
  else  if (TestMM(N, M, P, h_A, h_B, h_C))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

  if (PINNED) {
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
  }
  else {
    free(h_A); free(h_B); free(h_C);
  }

}




