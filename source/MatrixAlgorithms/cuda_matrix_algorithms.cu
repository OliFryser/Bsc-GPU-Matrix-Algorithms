extern "C" {
#include "cuda_matrix_algorithms.h"
}

__global__ void matrix_addition_gpu_single_core_kernel(DEVICE_MATRIX matrix1,
                                                       DEVICE_MATRIX matrix2,
                                                       DEVICE_MATRIX result,
                                                       int size) {}

extern "C" void matrix_addition_gpu_single_core(DEVICE_MATRIX matrix1,
                                                DEVICE_MATRIX matrix2,
                                                DEVICE_MATRIX result,
                                                int rows,
                                                int columns) {
    cudaDeviceSynchronize();
}