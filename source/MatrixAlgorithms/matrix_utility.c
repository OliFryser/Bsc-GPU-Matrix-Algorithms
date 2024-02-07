#include <stdlib.h>
#include <stdio.h>
#include "matrix_utility.h"

Matrix *matrix_init(int rows, int columns)
{
    if (rows <= 0 || columns <= 0)
        return NULL;

    Matrix *matrix;
    float values[rows][columns];
    int i;

    matrix = malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        return NULL;
    }
    matrix->values = malloc(rows * sizeof(float *));
    if (matrix->values == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix->values[i] = values[i];
    }
    matrix->rows = rows;
    matrix->columns = columns;
}

void matrix_free(Matrix *matrix)
{
    int i;
    if (matrix == NULL)
    {
        printf("Matrix is null... returning.");
        return;
    }
    printf("Matrix is not null.\n");
    if (matrix->values != NULL)
    {
        for (i = 0; i < matrix->rows; i++)
        {
            if (matrix->values[i] == NULL)
                continue;
            free(matrix->values[i]);
        }
        free(matrix->values);
    }
    free(matrix);
}