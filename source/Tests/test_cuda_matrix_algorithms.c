#include "test_cuda_matrix_algorithms.h"

matrix_t *cuda_matrix_2x2;
matrix_t *cuda_matrix_doubled_2x2;
matrix_t *cuda_matrix_multiplication1;
matrix_t *cuda_matrix_multiplication2;
matrix_t *cuda_matrix_multiplication_expected_result;

int init_cuda_matrix_suite(void) {
    char *csv_path;
    FILE *csv_file;

    csv_path = "./source/Tests/csv_test_matrix_2x2.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_2x2 = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_doubled_2x2.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_doubled_2x2 = matrix_init_from_csv(csv_file);

    if (cuda_matrix_2x2 == NULL || cuda_matrix_doubled_2x2 == NULL) return -1;

    // Multiplication matrices
    csv_path = "./source/Tests/csv_test_matrix_multiplication_1.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_multiplication1 = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_multiplication_2.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_multiplication2 = matrix_init_from_csv(csv_file);

    csv_path =
        "./source/Tests/csv_test_matrix_multiplication_expected_result.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_multiplication_expected_result = matrix_init_from_csv(csv_file);

    if (cuda_matrix_multiplication1 == NULL ||
        cuda_matrix_multiplication2 == NULL ||
        cuda_matrix_multiplication_expected_result == NULL)
        return -1;

    return 0;
}

int clean_cuda_matrix_suite(void) {
    matrix_free(cuda_matrix_2x2);
    matrix_free(cuda_matrix_doubled_2x2);
    return 0;
}

void test_cuda_matrix_utility(void) {
    matrix_t *src, *dst;
    device_matrix_t device_matrix;
    src = matrix_init(5, 5);
    dst = matrix_init(5, 5);
    device_matrix = cuda_matrix_init(5, 5);
    if (src == NULL || dst == NULL) return;
    CU_ASSERT_PTR_NOT_NULL_FATAL(device_matrix);
    matrix_random_fill(0.0F, 10.0F, src);
    cuda_matrix_host_to_device(device_matrix, src);
    cuda_matrix_device_to_host(dst, device_matrix);
    CU_ASSERT_TRUE(matrix_equal(src, dst));
    matrix_free(src);
    matrix_free(dst);
    cuda_matrix_free(device_matrix);
}

void test_matrix_addition_gpu_single_core(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(cuda_matrix_2x2);
    matrix_t *result =
        matrix_init(cuda_matrix_2x2->rows, cuda_matrix_2x2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(result);
    CU_ASSERT_TRUE(cuda_matrix_addition_single_core(
        cuda_matrix_2x2, cuda_matrix_2x2, result));
    CU_ASSERT_TRUE(matrix_equal(result, cuda_matrix_doubled_2x2));
    matrix_free(result);
}

void test_matrix_addition_gpu_multi_core(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(cuda_matrix_2x2);
    matrix_t *result =
        matrix_init(cuda_matrix_2x2->rows, cuda_matrix_2x2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(result);
    CU_ASSERT_TRUE(cuda_matrix_addition_multi_core(
        cuda_matrix_2x2, cuda_matrix_2x2, result));
    CU_ASSERT_TRUE(matrix_equal(result, cuda_matrix_doubled_2x2));
    matrix_free(result);
}

void test_matrix_addition_gpu_multi_core2(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(cuda_matrix_2x2);
    matrix_t *result =
        matrix_init(cuda_matrix_2x2->rows, cuda_matrix_2x2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(result);
    CU_ASSERT_TRUE(cuda_matrix_addition_multi_core2(
        cuda_matrix_2x2, cuda_matrix_2x2, result));
    CU_ASSERT_TRUE(matrix_equal(result, cuda_matrix_doubled_2x2));
    matrix_free(result);
}

void test_matrix_addition_gpu_multi_core2_larger_matrices(void) {
    matrix_t *matrixA, *matrix_b, *cpu_result, *gpu_result;
    int rows = 100;
    int cols = 100;

    matrixA = matrix_init(rows, cols);
    matrix_b = matrix_init(rows, cols);
    cpu_result = matrix_init(rows, cols);
    gpu_result = matrix_init(rows, cols);

    CU_ASSERT_PTR_NOT_NULL_FATAL(matrixA);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_b);
    CU_ASSERT_PTR_NOT_NULL_FATAL(cpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(gpu_result);

    CU_ASSERT_TRUE(matrix_random_fill(10.0f, 100.0f, matrixA));
    CU_ASSERT_TRUE(matrix_random_fill(10.0f, 100.0f, matrix_b));

    matrix_addition(matrixA, matrix_b, cpu_result);

    CU_ASSERT_TRUE(
        cuda_matrix_addition_multi_core2(matrixA, matrix_b, gpu_result));

    CU_ASSERT_TRUE(matrix_equal(gpu_result, cpu_result));
    matrix_free(matrixA);
    matrix_free(matrix_b);
    matrix_free(cpu_result);
    matrix_free(gpu_result);
}

void test_matrix_multiplication_gpu_single_core(void) {
    matrix_t *actual_result = matrix_init(cuda_matrix_multiplication1->rows,
        cuda_matrix_multiplication2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(
        cuda_matrix_multiplication_expected_result, actual_result));
    CU_ASSERT_TRUE_FATAL(
        cuda_matrix_multiplication_single_core(cuda_matrix_multiplication1,
            cuda_matrix_multiplication2, actual_result));
    CU_ASSERT_TRUE(matrix_equal(
        cuda_matrix_multiplication_expected_result, actual_result));
    matrix_free(actual_result);
}

void test_matrix_multiplication_gpu_multi_core_unwrapping_i(void) {
    matrix_t *actual_result = matrix_init(cuda_matrix_multiplication1->rows,
        cuda_matrix_multiplication2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(
        cuda_matrix_multiplication_expected_result, actual_result));
    CU_ASSERT_TRUE_FATAL(cuda_matrix_multiplication_multi_core_unwrapping_i(
        cuda_matrix_multiplication1, cuda_matrix_multiplication2,
        actual_result));
    CU_ASSERT_TRUE(matrix_equal(
        cuda_matrix_multiplication_expected_result, actual_result));
    matrix_free(actual_result);
}

void test_matrix_multiplication_gpu_multi_core_unwrapping_i_and_j(void) {
    matrix_t *actual_result = matrix_init(cuda_matrix_multiplication1->rows,
        cuda_matrix_multiplication2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(
        cuda_matrix_multiplication_expected_result, actual_result));
    CU_ASSERT_TRUE_FATAL(
        cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(
            cuda_matrix_multiplication1, cuda_matrix_multiplication2,
            actual_result));
    CU_ASSERT_TRUE(matrix_equal(
        cuda_matrix_multiplication_expected_result, actual_result));
    matrix_free(actual_result);
}

void test_matrix_multiplication_gpu_multi_core_unwrapping_i_and_j_larger_matrices(
    void) {
    matrix_t *matrix_a, *matrix_b, *cpu_result, *gpu_result;
    int m = 10;
    int l = 5;
    int n = 6;

    matrix_a = matrix_init(l, m);
    matrix_b = matrix_init(m, n);
    cpu_result = matrix_init(l, n);
    gpu_result = matrix_init(l, n);

    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_a);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_b);
    CU_ASSERT_PTR_NOT_NULL_FATAL(cpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(gpu_result);

    CU_ASSERT_TRUE(matrix_random_fill(10.0f, 20.0f, matrix_a));
    CU_ASSERT_TRUE(matrix_random_fill(10.0f, 20.0f, matrix_b));

    matrix_multiplication(matrix_a, matrix_b, cpu_result);

    CU_ASSERT_TRUE(cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(
        matrix_a, matrix_b, gpu_result));

    // CU_ASSERT_TRUE(matrix_equal(gpu_result, cpu_result));
    CU_ASSERT_TRUE(matrix_almost_equal(gpu_result, cpu_result));
    matrix_free(matrix_a);
    matrix_free(matrix_b);
    matrix_free(cpu_result);
    matrix_free(gpu_result);
}

void test_matrix_multiplication_gpu_multi_core_shared_memory(void) {
    matrix_t *actual_result = matrix_init(cuda_matrix_multiplication1->rows,
        cuda_matrix_multiplication2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(
        cuda_matrix_multiplication_expected_result, actual_result));
    CU_ASSERT_TRUE_FATAL(cuda_matrix_multiplication_multi_core_shared_memory(
        cuda_matrix_multiplication1, cuda_matrix_multiplication2,
        actual_result));
    CU_ASSERT_TRUE(matrix_equal(actual_result, cuda_matrix_multiplication_expected_result));
    matrix_free(actual_result);
}

void test_matrix_multiplication_gpu_multi_core_shared_memory_larger_matrices(
    void) {
    matrix_t *matrix_a, *matrix_b, *cpu_result, *gpu_result;
    int m = 5;
    int l = 199;
    int n = 73;

    matrix_a = matrix_init(l, m);
    matrix_b = matrix_init(m, n);
    cpu_result = matrix_init(l, n);
    gpu_result = matrix_init(l, n);

    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_a);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_b);
    CU_ASSERT_PTR_NOT_NULL_FATAL(cpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(gpu_result);

    CU_ASSERT_TRUE(matrix_random_fill(1.0f, 2.0f, matrix_a));
    CU_ASSERT_TRUE(matrix_random_fill(1.0f, 2.0f, matrix_b));

    matrix_multiplication(matrix_a, matrix_b, cpu_result);

    CU_ASSERT_TRUE(cuda_matrix_multiplication_multi_core_shared_memory(
        matrix_a, matrix_b, gpu_result));

    // CU_ASSERT_TRUE(matrix_equal(gpu_result, cpu_result));
    CU_ASSERT_TRUE(matrix_almost_equal(gpu_result, cpu_result));
    matrix_free(matrix_a);
    matrix_free(matrix_b);
    matrix_free(cpu_result);
    matrix_free(gpu_result);
}