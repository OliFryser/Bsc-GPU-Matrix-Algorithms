extern "C" {
#include "cuda_matrix_utility.h"
}

extern "C" Matrix *cuda_matrix_init(int rows, int columns) {
    Matrix *device_matrix;
    cudaMalloc(&device_matrix, sizeof(Matrix));
    if (device_matrix != NULL) return NULL;
    device_matrix->rows = rows;
    device_matrix->columns = columns;

    cudaMalloc(&device_matrix->values, rows * sizeof(float *));

    for (int i = 0; i < rows; i++) {
        cudaMalloc(&device_matrix->values[i], sizeof(columns * sizeof(float)));
    }

    return device_matrix;
}

extern "C" void cuda_matrix_free(Matrix *device_matrix) {
    if (device_matrix == NULL) return;
    if (device_matrix->values != NULL) {
        for (int i = 0; i < device_matrix->rows; i++) {
            if (device_matrix->values[i] != NULL)
                cudaFree(device_matrix->values[i]);
        }
        cudaFree(device_matrix->values);
    }
    cudaFree(device_matrix);
}

extern "C" void cuda_matrix_host_to_device(Matrix *dst, Matrix *src) {}
extern "C" void cuda_matrix_device_to_host(Matrix *dst, Matrix *src) {}