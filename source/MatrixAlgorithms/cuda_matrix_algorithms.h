#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "cuda_matrix_utility.h"
#include "matrix_utility.h"

bool matrix_addition_gpu_single_core(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool matrix_addition_gpu_multi_core(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool matrix_addition_gpu_multi_core2(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool matrix_multiplication_gpu_single_core(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool matrix_multiplication_gpu_multi_core_unwrapping_i(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

bool matrix_multiplication_gpu_multi_core_unwrapping_i_and_j(
    Matrix *matrix1, Matrix *matrix2, Matrix *result);

#endif