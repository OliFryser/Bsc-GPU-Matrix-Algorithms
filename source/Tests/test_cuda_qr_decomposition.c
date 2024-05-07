#include "test_cuda_qr_decomposition.h"

matrix_t *cuda_matrix_qr_input;
matrix_t *cuda_matrix_qr_expected_r;
matrix_t *cuda_matrix_qr_expected_q;

int init_cuda_qr_decomposition_suite(void) {
    char *csv_path;
    FILE *csv_file;

    csv_path = "./source/Tests/csv_test_matrix_qr_2_input.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_qr_input = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_qr_2_r.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_qr_expected_r = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_qr_2_q.csv";
    csv_file = read_csv(csv_path);
    cuda_matrix_qr_expected_q = matrix_init_from_csv(csv_file);

    if (cuda_matrix_qr_input == NULL || cuda_matrix_qr_expected_r == NULL ||
        cuda_matrix_qr_expected_q == NULL)
        return -1;

    return 0;
}

int clean_cuda_qr_decomposition_suite(void) {
    matrix_free(cuda_matrix_qr_expected_q);
    matrix_free(cuda_matrix_qr_expected_r);
    matrix_free(cuda_matrix_qr_input);

    return 0;
}

void test_matrix_qr_single_core(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(cuda_matrix_qr_input);
    matrix_t *actual_result =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    matrix_copy(cuda_matrix_qr_input, actual_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);

    float *diagonal, *c;
    diagonal = malloc(sizeof(float) * cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(diagonal);
    c = malloc(sizeof(float) * cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(c);

    CU_ASSERT_FALSE_FATAL(
        cuda_matrix_qr_decomposition_single_core(actual_result, diagonal, c));

    matrix_t *r =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(r);
    matrix_extract_r(actual_result, diagonal, r);

    matrix_t *q =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(q);

    matrix_t *q_j =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(q);

    extract_q(q, actual_result, c, q_j);

    matrix_t *multiplication_result =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(multiplication_result);

    CU_ASSERT_TRUE(matrix_almost_equal(q, cuda_matrix_qr_expected_q));
    CU_ASSERT_TRUE(matrix_almost_equal(r, cuda_matrix_qr_expected_r));
    matrix_multiplication(q, r, multiplication_result);

    CU_ASSERT_TRUE(
        matrix_almost_equal(multiplication_result, cuda_matrix_qr_input));
    free(c);
    free(diagonal);
    matrix_free(r);
    matrix_free(q);
    matrix_free(q_j);
    matrix_free(actual_result);
}

void test_matrix_qr_parallel_max(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(cuda_matrix_qr_input);
    matrix_t *actual_result =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    matrix_copy(cuda_matrix_qr_input, actual_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);

    float *diagonal, *c;
    diagonal = malloc(sizeof(float) * cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(diagonal);
    c = malloc(sizeof(float) * cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(c);

    CU_ASSERT_FALSE_FATAL(
        cuda_matrix_qr_decomposition_parallel_max(actual_result, diagonal, c));

    matrix_t *r =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(r);
    matrix_extract_r(actual_result, diagonal, r);

    matrix_t *q =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(q);

    matrix_t *q_j =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(q);

    extract_q(q, actual_result, c, q_j);

    matrix_t *multiplication_result =
        matrix_init(cuda_matrix_qr_input->rows, cuda_matrix_qr_input->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(multiplication_result);

    CU_ASSERT_TRUE(matrix_almost_equal(q, cuda_matrix_qr_expected_q));
    CU_ASSERT_TRUE(matrix_almost_equal(r, cuda_matrix_qr_expected_r));
    matrix_multiplication(q, r, multiplication_result);

    CU_ASSERT_TRUE(
        matrix_almost_equal(multiplication_result, cuda_matrix_qr_input));
    free(c);
    free(diagonal);
    matrix_free(r);
    matrix_free(q);
    matrix_free(q_j);
    matrix_free(actual_result);
}

void test_matrix_qr_parallel_max_larger_matrices(void) {
    matrix_t *cpu_result, *gpu_result;
    float *c_cpu, *c_gpu;
    float *diagonal_cpu, *diagonal_gpu;
    int rows = 32;
    int cols = rows;

    cpu_result = matrix_init(rows, cols);
    gpu_result = matrix_init(rows, cols);

    diagonal_cpu = malloc(sizeof(float) * cpu_result->columns);
    diagonal_gpu = malloc(sizeof(float) * cpu_result->columns);
    c_cpu = malloc(sizeof(float) * cpu_result->columns);
    c_gpu = malloc(sizeof(float) * cpu_result->columns);

    CU_ASSERT_PTR_NOT_NULL_FATAL(cpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(gpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(diagonal_cpu);
    CU_ASSERT_PTR_NOT_NULL_FATAL(diagonal_gpu);
    CU_ASSERT_PTR_NOT_NULL_FATAL(c_cpu);
    CU_ASSERT_PTR_NOT_NULL_FATAL(c_gpu);

    CU_ASSERT_TRUE(matrix_random_fill(1.0f, 10.0f, cpu_result));
    CU_ASSERT_TRUE(matrix_copy(cpu_result, gpu_result));

    CU_ASSERT_FALSE(cuda_matrix_qr_decomposition_single_core(cpu_result, diagonal_cpu, c_cpu));

    CU_ASSERT_FALSE(cuda_matrix_qr_decomposition_parallel_max(
        gpu_result, diagonal_gpu, c_gpu));

    CU_ASSERT_TRUE(matrix_almost_equal(gpu_result, cpu_result));
    CU_ASSERT_TRUE(
        array_almost_equal(diagonal_cpu, diagonal_gpu, cpu_result->columns))
    CU_ASSERT_TRUE(array_almost_equal(c_cpu, c_gpu, cpu_result->columns))

    free(c_cpu);
    free(c_gpu);
    free(diagonal_cpu);
    free(diagonal_gpu);
    matrix_free(cpu_result);
    matrix_free(gpu_result);
}

void test_matrix_qr_multi_core_single_kernel(void)
{
    matrix_t *cpu_result, *gpu_result;
    float *c_cpu, *c_gpu;
    float *diagonal_cpu, *diagonal_gpu;
    int rows = 4;
    int cols = rows;

    cpu_result = matrix_init(rows, cols);
    gpu_result = matrix_init(rows, cols);

    diagonal_cpu = malloc(sizeof(float) * cpu_result->columns);
    diagonal_gpu = malloc(sizeof(float) * cpu_result->columns);
    c_cpu = malloc(sizeof(float) * cpu_result->columns);
    c_gpu = malloc(sizeof(float) * cpu_result->columns);

    CU_ASSERT_PTR_NOT_NULL_FATAL(cpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(gpu_result);
    CU_ASSERT_PTR_NOT_NULL_FATAL(diagonal_cpu);
    CU_ASSERT_PTR_NOT_NULL_FATAL(diagonal_gpu);
    CU_ASSERT_PTR_NOT_NULL_FATAL(c_cpu);
    CU_ASSERT_PTR_NOT_NULL_FATAL(c_gpu);

    CU_ASSERT_TRUE(matrix_random_fill(1.0f, 10.0f, cpu_result));
    CU_ASSERT_TRUE(matrix_copy(cpu_result, gpu_result));

    CU_ASSERT_FALSE(cuda_matrix_qr_decomposition_single_core(cpu_result, diagonal_cpu, c_cpu));

    CU_ASSERT_FALSE(cuda_matrix_qr_decomposition_multi_core_single_kernel(
        gpu_result, diagonal_gpu, c_gpu));

    CU_ASSERT_TRUE(matrix_almost_equal(gpu_result, cpu_result));
    CU_ASSERT_TRUE(
        array_almost_equal(diagonal_cpu, diagonal_gpu, cpu_result->columns))
    CU_ASSERT_TRUE(array_almost_equal(c_cpu, c_gpu, cpu_result->columns))

    free(c_cpu);
    free(c_gpu);
    free(diagonal_cpu);
    free(diagonal_gpu);
    matrix_free(cpu_result);
    matrix_free(gpu_result);
}
