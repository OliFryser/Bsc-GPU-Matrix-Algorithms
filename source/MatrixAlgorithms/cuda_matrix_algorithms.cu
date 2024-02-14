extern "C" {
#include "cuda_matrix_algorithms.h"
}

__global__ void matrix_addition_gpu_single_core_kernel(Matrix *matrix1,
                                                       Matrix *matrix2,
                                                       Matrix *result) {}

extern "C" void matrix_addition_gpu_single_core(Matrix *matrix1,
                                                Matrix *matrix2,
                                                Matrix *result) {
    cudaDeviceSynchronize();
}