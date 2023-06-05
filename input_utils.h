#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

__host__ float* readImageData(char* filename, int size);

__host__ float* readLabels(char* filename, int size);