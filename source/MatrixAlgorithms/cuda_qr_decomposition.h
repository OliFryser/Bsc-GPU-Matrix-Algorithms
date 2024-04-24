#ifndef CUDA_QR_DECOMPOSITION_H
#define CUDA_QR_DECOMPOSITION_H

#include "cuda_matrix_utility.h"
#include "matrix_utility.h"
#include "matrix_algorithms.h"

bool cuda_matrix_qr_decomposition_single_core(matrix_t* matrix, float* diagonal, float* c);

bool cuda_matrix_qr_decomposition_parallel_max(matrix_t* matrix, float* diagonal, float* c);

#endif