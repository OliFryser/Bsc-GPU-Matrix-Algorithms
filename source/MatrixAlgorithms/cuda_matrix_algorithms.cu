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

extern "C" void matrix_addition_gpu_single_core(DEVICE_MATRIX matrix1,
                                                DEVICE_MATRIX matrix2,
                                                DEVICE_MATRIX result,
                                                int rows,
                                                int columns) {
    matrix_addition_gpu_single_core_kernel<<<1,1>>>(matrix1, matrix2, result, rows * columns);
    cudaDeviceSynchronize();
}