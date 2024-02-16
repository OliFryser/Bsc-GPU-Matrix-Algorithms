#ifndef CUDA_MATRIX_UTILITY_H
#define CUDA_MATRIX_UTILITY_H
#define DEVICE_MATRIX float *
#define INDEX(row_index, column_index, columns) \
    ((row_index) * (columns) + (column_index))

#include "matrix_utility.h"

DEVICE_MATRIX cuda_matrix_init(int rows, int columns);
void cuda_matrix_free(DEVICE_MATRIX device_matrix);
void cuda_matrix_host_to_device(DEVICE_MATRIX dst, Matrix *src);
void cuda_matrix_device_to_host(Matrix *dst, DEVICE_MATRIX src);

#endif