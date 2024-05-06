#ifndef CUDA_QR_DECOMPOSITION_H
#define CUDA_QR_DECOMPOSITION_H

#include "cuda_matrix_utility.h"
#include "matrix_algorithms.h"
#include "matrix_utility.h"

typedef float (*reducer_t)(float, float);

bool cuda_matrix_qr_decomposition_parallel_max_adapter(
    algorithm_arg_t* matrix, algorithm_arg_t* diagonal, algorithm_arg_t* c);

bool cuda_matrix_qr_decomposition_single_core_adapter(
    algorithm_arg_t* matrix, algorithm_arg_t* diagonal, algorithm_arg_t* c);

bool cuda_matrix_qr_decomposition_single_core(
    matrix_t* matrix, float* diagonal, float* c);

bool cuda_matrix_qr_decomposition_parallel_max(
    matrix_t* matrix, float* diagonal, float* c);

#endif