#ifndef MATRIX_ALGORITHMS_H_2D
#define MATRIX_ALGORITHMS_H_2D

#include <stdbool.h>

#include "2d_matrix_utility.h"

bool matrix_addition(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_multiplication(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result);

#endif