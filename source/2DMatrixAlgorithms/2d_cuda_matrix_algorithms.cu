extern "C" {
#include "2d_cuda_matrix_algorithms.h"
}

__global__ void matrix_addition_gpu_single_core_kernel(device_matrix_t matrix1,
    device_matrix_t matrix2, device_matrix_t result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
}

__global__ void matrix_addition_gpu_multi_core_kernel(device_matrix_t matrix1,
    device_matrix_t matrix2, device_matrix_t result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    result[index] = matrix1[index] + matrix2[index];
}

bool gpu_matrix_algorithm_runner(matrix_t *matrix1, matrix_t *matrix2,
    matrix_t *result,
    void (*kernel)(device_matrix_t, device_matrix_t, device_matrix_t, int),
    int grid_size, int block_size) {
    if (matrix1 == NULL || matrix2 == NULL || result == NULL) return false;

    device_matrix_t device_matrix1 =
        cuda_matrix_init(matrix1->rows, matrix1->columns);
    device_matrix_t device_matrix2 =
        cuda_matrix_init(matrix2->rows, matrix2->columns);
    device_matrix_t device_result =
        cuda_matrix_init(result->rows, result->columns);

    if (device_matrix1 == NULL || device_matrix2 == NULL ||
        device_result == NULL)
        return false;

    cuda_matrix_host_to_device(device_matrix1, matrix1);
    cuda_matrix_host_to_device(device_matrix2, matrix2);
    cuda_matrix_host_to_device(device_result, result);

    kernel<<<grid_size, block_size>>>(device_matrix1, device_matrix2,
        device_result, result->rows * result->columns);

    cuda_matrix_device_to_host(result, device_result);

    cuda_matrix_free(device_matrix1);
    cuda_matrix_free(device_matrix2);
    cuda_matrix_free(device_result);

    return true;
}

extern "C" bool matrix_addition_gpu_single_core(
    matrix_t *matrix1, matrix_t *matrix2, matrix_t *result) {
    return gpu_matrix_algorithm_runner(matrix1, matrix2, result,
        &(matrix_addition_gpu_single_core_kernel), 1, 1);
}

bool matrix_addition_gpu_multi_core(
    matrix_t *matrix1, matrix_t *matrix2, matrix_t *result) {
    bool success;
    int grid_size, block_size;
    grid_size = matrix1->rows;
    block_size = matrix1->columns;

    success = gpu_matrix_algorithm_runner(matrix1, matrix2, result,
        &(matrix_addition_gpu_multi_core_kernel), grid_size, block_size);

    return success;
}