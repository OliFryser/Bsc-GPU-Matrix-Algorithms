#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <stdbool.h>
#include "matrix_utility.h"

bool matrix_addition_cpu(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_multiplication(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result);

#endif