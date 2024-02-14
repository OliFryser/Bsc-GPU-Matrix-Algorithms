#ifndef TEST_MATRIX_ALGORITHMS_H
#define TEST_MATRIX_ALGORITHMS_H

#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include <stdlib.h>
#include <string.h>

#include "../MatrixAlgorithms/matrix_algorithms.h"
#include "../MatrixAlgorithms/cuda_matrix_algorithms.h"

void test_init_matrix(void);
void test_init_matrix_0_values(void);
void test_init_matrix_2x2_from_csv(void);
void test_init_matrix_4x1_from_csv(void);
void test_init_matrix_2x2_doubled_from_csv(void);
void test_matrix_equal_dimensions(void);
void test_matrix_not_equal_dimensions(void);
void test_matrix_equal(void);
void test_matrix_not_equal(void);
void test_matrix_copy(void);
void test_matrix_addition(void);
void test_matrix_addition_gpu_single_core(void);
void test_matrix_random_fill(void);
int init_matrix_suite(void);
int clean_matrix_suite(void);

#endif