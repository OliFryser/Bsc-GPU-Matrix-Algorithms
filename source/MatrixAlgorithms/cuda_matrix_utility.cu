extern "C" {
#include "cuda_matrix_utility.h"
}

extern "C" Matrix *cuda_matrix_init(int rows, int columns) {
    Matrix *host_matrix, *device_matrix;
    float **device_array;
    host_matrix = matrix_init(rows, columns);

    cudaMalloc(&device_matrix, sizeof(Matrix));

    cudaMemcpy(device_matrix, host_matrix, sizeof(Matrix),
               cudaMemcpyHostToDevice);

    cudaMalloc(&device_array, rows * sizeof(float *));

    for (int i = 0; i < rows; i++) {
        float *row;
        cudaMalloc(&row, columns * sizeof(float));
        cudaMalloc(&(device_array[i]), columns * sizeof(float));
    }

    // cudaMemcpy(device_matrix->values, device_array, rows * sizeof(float *),
    // cudaMemcpyDeviceToDevice);
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