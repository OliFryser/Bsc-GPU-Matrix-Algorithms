#ifndef CUDA_MATRIX_ALGORITHMS_H_2D
#define CUDA_MATRIX_ALGORITHMS_H_2D

#include "2d_cuda_matrix_utility.h"
#include "2d_matrix_utility.h"

bool matrix_addition_gpu_single_core(
    matrix_t *matrix1, matrix_t *matrix2, matrix_t *result);

bool matrix_addition_gpu_multi_core(
    matrix_t *matrix1, matrix_t *matrix2, matrix_t *result);

#endif