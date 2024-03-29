#ifndef MATRIX_ALGORITHMS_H_2D
#define MATRIX_ALGORITHMS_H_2D

#include <stdbool.h>

#include "2d_matrix_utility.h"

bool matrix_addition(matrix_t *matrix1, matrix_t *matrix2, matrix_t *result);
bool matrix_multiplication(
    matrix_t *matrix1, matrix_t *matrix2, matrix_t *result);
bool matrix_inverse(matrix_t *matrix1, matrix_t *matrix2, matrix_t *result);

#endif