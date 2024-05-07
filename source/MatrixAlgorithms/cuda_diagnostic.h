#ifndef CUDA_MATRIX_ALGORITHMS_H
#define CUDA_MATRIX_ALGORITHMS_H

#include "matrix_algorithms.h"
#include "cuda_matrix_utility.h"
#include "array_algorithms.h"
typedef float (*reducer_t)(float, float);

void launch_kernel_1_block_1_thread();
bool launch_kernel_1_block_1_thread_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool launch_kernel_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool malloc_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool memcpy_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool memcpy_and_kernel_launch_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool memcpy_and_larger_kernel_launch_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool launch_x_kernels_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool launch_x_kernels_sequentially_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool write_managed_vector_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool write_vector_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool parallel_max_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool sequential_max_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
#endif