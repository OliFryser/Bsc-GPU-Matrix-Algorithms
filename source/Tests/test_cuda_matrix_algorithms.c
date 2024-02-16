#include "test_cuda_matrix_algorithms.h"

Matrix *matrix_2x2;
Matrix *matrix_doubled_2x2;

int init_cuda_matrix_suite(void) { 
    char *csv_path;
    FILE *csv_file;
    
    csv_path = "./source/Tests/csv_test_matrix_2x2.csv";
    csv_file = read_csv(csv_path);
    matrix_2x2 = matrix_init_from_csv(csv_file);
    
    csv_path = "./source/Tests/csv_test_matrix_doubled_2x2.csv";
    csv_file = read_csv(csv_path);
    matrix_doubled_2x2 = matrix_init_from_csv(csv_file);

    if (matrix_2x2 == NULL || matrix_doubled_2x2 == NULL) return -1;
    return 0;
 }

int clean_cuda_matrix_suite(void) { 
    matrix_free(matrix_2x2);
    matrix_free(matrix_doubled_2x2);
    return 0; 
}

void test_cuda_matrix_utility(void) {
    Matrix *src, *dst;
    DEVICE_MATRIX device_matrix;
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
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_2x2);
    Matrix *result = matrix_init(matrix_2x2->rows, matrix_2x2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(matrix_2x2, result));
    matrix_addition_gpu_single_core(matrix_2x2, matrix_2x2, result);
    CU_ASSERT_TRUE(matrix_equal(result, matrix_doubled_2x2));
    matrix_free(result);
}