#ifndef CUDA_MATRIX_UTILITY_H
#define CUDA_MATRIX_UTILITY_H

#include "matrix_utility.h"

typedef float *device_matrix_t;

#define gpuErrorcheck(function) { gpuAssert((function), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

device_matrix_t cuda_matrix_init(int rows, int columns);
bool cuda_matrix_free(device_matrix_t device_matrix);
bool cuda_matrix_host_to_device(device_matrix_t dst, matrix_t *src);
bool cuda_matrix_device_to_host(matrix_t *dst, device_matrix_t src);

#endif