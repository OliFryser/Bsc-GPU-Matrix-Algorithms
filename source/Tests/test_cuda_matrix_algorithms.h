#ifndef TEST_CUDA_MATRIX_ALGORITHMS_H
#define TEST_CUDA_MATRIX_ALGORITHMS_H

#include <CUnit/Basic.h>
#include <CUnit/Console.h>

#include "../MatrixAlgorithms/cuda_matrix_algorithms.h"
#include "../MatrixAlgorithms/matrix_algorithms.h"

int init_cuda_matrix_suite(void);
int clean_cuda_matrix_suite(void);
void test_cuda_matrix_utility(void);
void test_matrix_addition_gpu_single_core(void);
void test_matrix_addition_gpu_multi_core(void);
void test_matrix_addition_gpu_multi_core2(void);
void test_matrix_addition_gpu_multi_core2_larger_matrices(void);
void test_matrix_multiplication_gpu_single_core(void);

#endif