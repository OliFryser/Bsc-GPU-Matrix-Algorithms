#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix_utility.h"
#include "csv_utility.h"

Matrix *matrix_init(int rows, int columns)
{
    if (rows <= 0 || columns <= 0)
        return NULL;

    Matrix *matrix;
    int i;

    matrix = malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        return NULL;
    }
    matrix->values = malloc(rows * sizeof(float *));
    if (matrix->values == NULL)
    {
        free(matrix);
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix->values[i] = malloc(columns * sizeof(float));
    }
    matrix->rows = rows;
    matrix->columns = columns;
    return matrix;
}

void matrix_free(Matrix *matrix)
{
    int i;
    if (matrix == NULL)
    {
        return;
    }
    if (matrix->values != NULL)
    {
        for (i = 0; i < matrix->rows; i++)
        {
            if (matrix->values[i] != NULL)
                free(matrix->values[i]);
        }
        free(matrix->values);
    }
    free(matrix);
}

void matrix_print(Matrix *matrix)
{
    int i, j;

    printf("# PRINTING MATRIX #\n");
    for (i = 0; i < matrix->rows; i++)
    {
        for (j = 0; j < matrix->columns; j++)
        {
            printf("%.2f ", matrix->values[i][j]);
        }
        printf("\n");
    }
}

Matrix *matrix_init_from_csv(FILE *csv_file)
{
    Matrix *matrix;
    char line[100];
    char *token;
    float value;
    int row_count;
    int column_count;
    int row = 0;
    int column;

    if (csv_file == NULL) {
        printf("File is NULL.\n");
        return NULL;
    }

    // read dimensions
    fgets(line, sizeof(line), csv_file);
    token = strtok(line, ",");
    if (token == NULL) return NULL;
    row_count = atoi(token);
    token = strtok(NULL, ",");
    if (token == NULL) return NULL;
    column_count = atoi(token);

    matrix = matrix_init(row_count, column_count);
    if (matrix == NULL) return NULL;

    // read values
    while(fgets(line, sizeof(line), csv_file) != NULL) {
        column = 0;
        if (line[0] == '\n') continue;
        
        token = strtok(line, ",");
        while(token != NULL) {
            value = atof(token);
            matrix->values[row][column] = value;
            token = strtok(NULL, ",");
            column++;
        }
        row++;

        if (column != column_count) {
            printf("Wrong column count. Expected %d but got %d", column_count, column);
            return NULL;
        }
    }

    if (row != row_count) {
        printf("Wrong row count. Expected %d but got %d", row_count, row);
        return NULL;
    }

    return matrix;
}

bool matrix_equal_dimensions(Matrix *matrix1, Matrix *matrix2) {
    return matrix1->columns == matrix2->columns && matrix1->rows == matrix2->rows;
}

bool matrix_equal(Matrix *matrix1, Matrix *matrix2) {
    if (matrix1 == NULL) return false;
    if (matrix2 == NULL) return false;
    if (!matrix_equal_dimensions(matrix1, matrix2)) return false;

    int rows = matrix1->rows;
    int columns = matrix1->columns;
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            if (matrix1->values[i][j] != matrix2->values[i][j]) return false;
        }
    }

    return true;
}

void matrix_copy(Matrix *original, Matrix *copy) {
    return;
}

bool matrix_addition(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
    if (matrix1 == NULL) return false;
    if (matrix2 == NULL) return false;
    if (result == NULL) return false;
    if (!matrix_equal_dimensions(matrix1, matrix2)) return false;
    if (!matrix_equal_dimensions(matrix1, result)) return false;
    int i;
    int j;
    int rows = matrix1->rows;
    int columns = matrix1->columns;
    
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            result->values[i][j] = matrix1->values[i][j] + matrix2->values[i][j];
        }
    }
    
    return true;
}