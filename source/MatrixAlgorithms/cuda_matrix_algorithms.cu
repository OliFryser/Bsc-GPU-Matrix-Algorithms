extern "C" {
#include "cuda_matrix_algorithms.h"
}

__global__ void matrix_addition_gpu_single_core_kernel(
    DEVICE_MATRIX matrix1,
    DEVICE_MATRIX matrix2,
    DEVICE_MATRIX result,
    int size) {
    
    for (int i = 0; i < size; i++)
    {
        result[i] = matrix1[i] + matrix2[i];
    }
}

extern "C" bool matrix_addition_gpu_single_core(Matrix *matrix1,
                                                Matrix *matrix2,
                                                Matrix *result) {

    if (matrix1 == NULL || matrix2 == NULL || result == NULL) return false;

    DEVICE_MATRIX device_matrix1 = cuda_matrix_init(matrix1->rows, matrix1->columns);
    DEVICE_MATRIX device_matrix2 = cuda_matrix_init(matrix2->rows, matrix2->columns);
    DEVICE_MATRIX device_result = cuda_matrix_init(result->rows, result->columns);

    if (device_matrix1 == NULL || device_matrix2 == NULL || device_result == NULL) return false;

    cuda_matrix_host_to_device(device_matrix1, matrix1);
    cuda_matrix_host_to_device(device_matrix2, matrix2);
    cuda_matrix_host_to_device(device_result, result);
    
    matrix_addition_gpu_single_core_kernel<<<1,1>>>(device_matrix1, device_matrix2, device_result, result->rows * result->columns);

    cuda_matrix_device_to_host(result, device_result);

    cuda_matrix_free(device_matrix1);
    cuda_matrix_free(device_matrix2);
    cuda_matrix_free(device_result);

    return true;
}