#ifndef CUDA_MATRIX_UTILITY_H
#define CUDA_MATRIX_UTILITY_H
#define INDEX(row_index, column_index, columns) \
    ((row_index) * (columns) + (column_index))

#include "matrix_utility.h"

typedef float *device_matrix_t;

device_matrix_t cuda_matrix_init(int rows, int columns);
void cuda_matrix_free(device_matrix_t device_matrix);
void cuda_matrix_host_to_device(device_matrix_t dst, Matrix *src);
void cuda_matrix_device_to_host(Matrix *dst, device_matrix_t src);

#endif