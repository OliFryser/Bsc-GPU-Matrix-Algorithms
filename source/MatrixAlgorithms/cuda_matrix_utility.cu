extern "C" {
#include "cuda_matrix_utility.h"
}

// Deep copy:
// https://forums.developer.nvidia.com/t/clean-way-of-copying-a-struct-with-pointers-to-the-gpu/225833

// Arguments for why this is bad:
// https://stackoverflow.com/questions/6137218/how-can-i-add-up-two-2d-pitched-arrays-using-nested-for-loops/6137517#6137517
extern "C" DEVICE_MATRIX cuda_matrix_init(int rows, int columns) {
    DEVICE_MATRIX device_array;
    cudaError_t error =
        cudaMalloc(&device_array, rows * columns * sizeof(float));
    if (error != cudaSuccess) {
        printf("\n%d\n", error);
        return NULL;
    }
    return device_array;
}

extern "C" bool cuda_matrix_free(DEVICE_MATRIX device_matrix) {
    if (device_matrix == NULL) return false;
    cudaError_t error = cudaFree(device_matrix);
    if (error != cudaSuccess) {
        printf("\n%d\n", error);
        return false;
    }
    return true;
}

extern "C" bool cuda_matrix_host_to_device(DEVICE_MATRIX dst, Matrix *src) {
    size_t size;
    size = src->rows * src->columns * sizeof(float);
    cudaError_t error =
        cudaMemcpy(dst, src->values, size, cudaMemcpyHostToDevice);

    if (error != cudaSuccess) {
        printf("\n%d\n", error);
        return false;
    }

    return true;
}

extern "C" bool cuda_matrix_device_to_host(Matrix *dst, DEVICE_MATRIX src) {
    size_t size;
    size = dst->rows * dst->columns * sizeof(float);
    cudaError_t error =
        cudaMemcpy(dst->values, src, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        printf("\n%d\n", error);
        return false;
    }

    return true;
}
