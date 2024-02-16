#ifndef TEST_CUDA_MATRIX_ALGORITHMS_H
#define TEST_CUDA_MATRIX_ALGORITHMS_H

#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include "../MatrixAlgorithms/cuda_matrix_utility.h"

int init_cuda_matrix_suite(void);
int clean_cuda_matrix_suite(void);
void test_cuda_matrix_utility(void);

#endif