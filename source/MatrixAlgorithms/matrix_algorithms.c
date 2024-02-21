#include "matrix_algorithms.h"

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

    for (i = 0; i < rows * columns; i++)
        result->values[i] = matrix1->values[i] + matrix2->values[i];

    return true;
}

bool matrix_multiplication(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
    return false;
}

bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
    return false;
}