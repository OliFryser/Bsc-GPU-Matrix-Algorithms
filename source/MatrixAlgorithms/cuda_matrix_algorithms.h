#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "cuda_matrix_utility.h"
#include "matrix_algorithms.h"
#include "matrix_utility.h"

bool cuda_matrix_addition_single_core_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_addition_multi_core_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_addition_multi_core2_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_addition_blocks_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_multiplication_single_core_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_multiplication_multi_core_unwrapping_i_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_multiplication_multi_core_unwrapping_i_and_j_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_multiplication_multi_core_shared_memory_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

bool cuda_matrix_addition_single_core(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_addition_multi_core(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_addition_multi_core2(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_addition_blocks(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_single_core(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_multi_core_unwrapping_i(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_multi_core_shared_memory(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c);

bool cuda_matrix_qr_decomposition_single_core(
    matrix_t *matrix, float *diagonal, float *c);

bool cuda_matrix_qr_decomposition_parallel_max(
    matrix_t *matrix, float *diagonal, float *c);

#endif