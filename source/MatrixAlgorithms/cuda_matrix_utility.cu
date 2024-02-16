extern "C" {
#include "cuda_matrix_utility.h"
}

// Deep copy:
// https://forums.developer.nvidia.com/t/clean-way-of-copying-a-struct-with-pointers-to-the-gpu/225833

// Arguments for why this is bad:
// https://stackoverflow.com/questions/6137218/how-can-i-add-up-two-2d-pitched-arrays-using-nested-for-loops/6137517#6137517
extern "C" DEVICE_MATRIX cuda_matrix_init(int rows, int columns) {
    DEVICE_MATRIX device_array;
    cudaMalloc(&device_array, rows * columns * sizeof(float));
    return device_array;
}

extern "C" void cuda_matrix_free(DEVICE_MATRIX device_matrix) {
    if (device_matrix == NULL) return;
    cudaFree(device_matrix);
}

void cuda_matrix_2d_to_1d(float *dst, Matrix *src) {
    for (int i = 0; i < src->rows; i++)
        for (int j = 0; j < src->columns; j++)
            dst[INDEX(i, j, src->columns)] = src->values[i][j];
}

void cuda_matrix_1d_to_2d(Matrix *dst, float *src) {
    for (int i = 0; i < dst->rows; i++)
        for (int j = 0; j < dst->columns; j++)
            dst->values[i][j] = src[INDEX(i, j, dst->columns)];
}

extern "C" void cuda_matrix_host_to_device(DEVICE_MATRIX dst, Matrix *src) {
    float *cpu_values;
    size_t size = src->rows * src->columns * sizeof(float);
    cpu_values = (float *)malloc(size);
    cuda_matrix_2d_to_1d(cpu_values, src);
    cudaMemcpy(dst, cpu_values, size, cudaMemcpyHostToDevice);
    free(cpu_values);
}

extern "C" void cuda_matrix_device_to_host(Matrix *dst, DEVICE_MATRIX src) {
    float *cpu_values;
    size_t size = dst->rows * dst->columns * sizeof(float);
    cpu_values = (float *)malloc(size);
    cudaMemcpy(cpu_values, src, size, cudaMemcpyDeviceToHost);
    cuda_matrix_1d_to_2d(dst, cpu_values);
    free(cpu_values);
}
