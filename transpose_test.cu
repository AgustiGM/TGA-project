#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>

#include <math.h>

#include "primitives.h"
#include "nnfunctions.h"
#include "seq_primitives.h"


int main(int argc, char **argv) {
    int N,M;

    //check that there are 2 input arguments
    if (argc != 3) {
        printf("Usage: %s <N> <M>\n", argv[0]);
        exit(1);
    }

    //read N and M from input arguments
    N = atoi(argv[1]);
    M = atoi(argv[2]);

    // generate random NxM matrix input
    float *input = (float *) malloc(N * M * sizeof(float));
    for (int i = 0; i < N * M; i++) {
        input[i] = (float) rand() / (float) RAND_MAX;
    }

    // allocate memory for output
    float *output = (float *) malloc(N * M * sizeof(float));

    // allocate memory for seq output
    float *seq_output = (float *) malloc(N * M * sizeof(float));

    // allocate memory device

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * M * sizeof(float));
    cudaMalloc(&d_output, N * M * sizeof(float));

    // copy input to device
    cudaMemcpy(d_input, input, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // call kernel

    transpose<<<N, 1024>>>(N, M, d_input, d_output);

    // copy output to host
    cudaMemcpy(output, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // call sequential function
    seqTranspose(N, M, input, seq_output);

    // check if the output is correct

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++){
            if (fabs(output[i * N + j] - seq_output[i * N + j]) > 0.0001) {
                printf("Error at position %d,%d\n", i, j);
                printf("Output: %f\n", output[i * N + j]);
                printf("Expected: %f\n", seq_output[i * N + j]);
                exit(1);
            }
        }
    }

    printf("Success!\n");



}