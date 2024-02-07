#include <stdlib.h>
#include "matrix_utility.h"

float **matrix_init(int n, int m)
{
    float matrix[n][m];
    float **indexed_matrix;
    int i;

    indexed_matrix = (float **)malloc(n * sizeof(float *));
    if (indexed_matrix == NULL)
    {
        return NULL;
    }

    for (i = 0; i < n; i++)
    {
        indexed_matrix[i] = matrix[i];
    }

    return indexed_matrix;
}