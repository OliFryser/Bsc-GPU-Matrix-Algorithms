#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <stdbool.h>

#include "matrix_utility.h"

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a));

typedef union {
    matrix_t *matrix;
    float *vector;
} algorithm_arg_t;

// Adapter used for benchmark runner
bool matrix_addition_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool matrix_addition(matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);
bool matrix_multiplication(matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);
bool matrix_inverse(matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);
bool matrix_qr_decomposition(matrix_t *matrix, float *diagonal, float *c);
#endif