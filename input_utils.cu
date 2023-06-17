#include "input_utils.h"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

__host__ float* readImageData(char *filename, int size) {
    int fd = open(filename, O_RDONLY);
    unsigned char buf[16];
    int n = read(fd, buf, 16);

    unsigned char *data = (unsigned char*)malloc(size);
    n = read(fd, data, size);
    float *fdata = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        fdata[i] = (float)data[i];
    }
    free(data);
    return fdata;
}

__host__ float* readLabels(char* filename, int size) {
    int fd = open(filename, O_RDONLY);
    unsigned char buf[8];
    int n = read(fd, buf, 8);

    unsigned char *data  = (unsigned char*)malloc(size);
    n = read(fd, data, size);
    float *fdata = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        fdata[i] = (float)data[i];
    }
    free(data);
    return fdata;

}
