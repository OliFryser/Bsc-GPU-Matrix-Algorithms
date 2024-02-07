#include <stdlib.h>
#include "matrix_utility.h"

Matrix *matrix_init(int rows, int columns)
{
    if (rows <= 0 || columns <= 0)
        return NULL;

    float values[rows][columns];
    float **indexed_values;
    int i;

    indexed_values = (float **)malloc(rows * sizeof(float *));
    if (indexed_values == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        indexed_values[i] = values[i];
    }

    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->values = indexed_values;
    matrix->rows = rows;
    matrix->columns = columns;
}

void matrix_free(Matrix *matrix)
{
    int i;
    if (matrix == NULL)
        return;

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