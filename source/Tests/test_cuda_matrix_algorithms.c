#include "test_cuda_matrix_algorithms.h"

int init_cuda_matrix_suite(void)
{
    return 0;
}

int clean_cuda_matrix_suite(void)
{
    return 0;
}

void test_cuda_matrix_utility(void)
{
    Matrix *src, *dst, *device_matrix;
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