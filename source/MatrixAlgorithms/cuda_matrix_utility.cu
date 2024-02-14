extern "C" {
#include "cuda_matrix_utility.h"
}
// Deep copy:
// https://forums.developer.nvidia.com/t/clean-way-of-copying-a-struct-with-pointers-to-the-gpu/225833

// Arguments for why this is bad:
// https://stackoverflow.com/questions/6137218/how-can-i-add-up-two-2d-pitched-arrays-using-nested-for-loops/6137517#6137517
extern "C" Matrix *cuda_matrix_init(int rows, int columns) {
    Matrix *host_matrix, *device_matrix;
    host_matrix = matrix_init(rows, columns);

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