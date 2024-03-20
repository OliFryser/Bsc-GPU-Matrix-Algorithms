#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "cuda_matrix_utility.h"
#include "matrix_utility.h"

bool cuda_matrix_addition_single_core(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_addition_multi_core(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_addition_multi_core2(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_single_core(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_multi_core_unwrapping_i(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

#endif