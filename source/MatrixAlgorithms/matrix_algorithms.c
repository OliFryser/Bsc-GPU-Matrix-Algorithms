#include "matrix_algorithms.h"

bool matrix_addition_cpu(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
    if (matrix1 == NULL) return false;
    if (matrix2 == NULL) return false;
    if (result == NULL) return false;
    if (!matrix_equal_dimensions(matrix1, matrix2)) return false;
    if (!matrix_equal_dimensions(matrix1, result)) return false;
    int i;
    int j;
    int rows = matrix1->rows;
    int columns = matrix1->columns;

    for (i = 0; i < rows * columns; i++)
        result->values[i] = matrix1->values[i] + matrix2->values[i];

    return true;
}

bool matrix_multiplication_cpu(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
    if (matrix1 == NULL) return false;
    if (matrix2 == NULL) return false;
    if (result == NULL) return false;

    if (matrix1->columns != matrix2->rows) return false;
    int common_dimension_length = matrix1->columns;
    int result_rows = matrix1->rows;
    int result_columns = matrix2->columns;

    float product;
    float sum_of_products;

    for (int i = 0; i < result_rows; i++)
    {
        for (int j = 0; j < result_columns; j++)
        {
            sum_of_products = 0.0f;
            for (int k = 0; k < common_dimension_length; k++)
            {
                product = matrix1->values[INDEX(i,k,matrix1->columns)] * matrix2->values[INDEX(k,j,matrix2->columns)];
                sum_of_products += product;
            }
            result->values[INDEX(i, j, result_columns)] = sum_of_products;
        }
    }
    return true;
}

bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
    return false;
}