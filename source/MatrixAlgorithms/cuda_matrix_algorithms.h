#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "matrix_utility.h"

void matrix_addition_gpu_single_core(Matrix *matrix1, Matrix *matrix2,
                                     Matrix *result);

#endif