#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "matrix_utility.h"
#include "cuda_matrix_utility.h"

void matrix_addition_gpu_single_core(DEVICE_MATRIX matrix1, DEVICE_MATRIX matrix2,
                                     DEVICE_MATRIX result, int rows, int columns);

#endif