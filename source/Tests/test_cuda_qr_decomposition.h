#ifndef TEST_CUDA_QR_DECOMPOSITION_H
#define TEST_CUDA_QR_DECOMPOSITION_H

#include <CUnit/Basic.h>
#include <CUnit/Console.h>

#include "../MatrixAlgorithms/array_algorithms.h"
#include "../MatrixAlgorithms/cuda_matrix_algorithms.h"
#include "../MatrixAlgorithms/matrix_algorithms.h"

int init_cuda_qr_decomposition_suite(void);
int clean_cuda_qr_decomposition_suite(void);
void test_matrix_qr_single_core(void);
void test_matrix_qr_parallel_max(void);
void test_matrix_qr_parallel_max_larger_matrices(void);
void test_matrix_qr_multi_core_single_kernel(void);

#endif