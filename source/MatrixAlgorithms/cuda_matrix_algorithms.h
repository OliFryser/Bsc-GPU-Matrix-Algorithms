#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "cuda_matrix_utility.h"
#include "matrix_utility.h"

bool cuda_matrix_addition_single_core(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool cuda_matrix_addition_multi_core(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool cuda_matrix_addition_multi_core2(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool cuda_matrix_multiplication_single_core(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool cuda_matrix_multiplication_multi_core_unwrapping_i(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

#endif