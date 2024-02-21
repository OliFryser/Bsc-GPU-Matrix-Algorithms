#ifndef CUDA_MATRIX_UTILITY_H
#define CUDA_MATRIX_UTILITY_H
#define DEVICE_MATRIX float *

#include "matrix_utility.h"

DEVICE_MATRIX cuda_matrix_init(int rows, int columns);
bool cuda_matrix_free(DEVICE_MATRIX device_matrix);
bool cuda_matrix_host_to_device(DEVICE_MATRIX dst, Matrix *src);
bool cuda_matrix_device_to_host(Matrix *dst, DEVICE_MATRIX src);

#endif