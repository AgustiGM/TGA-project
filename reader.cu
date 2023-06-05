// #include "input_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// read a idx format file
int main()
{
    float* img = (float*)malloc(28 * 28 * sizeof(float));
    printf("image data read and converted\n");
    for (int k = 0; k < 1; k++)
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                char c = 0;
                if (img[k * 28 * 28 + i * 28 + j] != 0)
                    c = 1;
                printf("%d ", c);
            }
            printf("\n");
        }
        printf("\n");
    }
    
}