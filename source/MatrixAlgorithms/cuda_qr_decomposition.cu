extern "C" {
#include "cuda_qr_decomposition.h"
}

__global__ void cuda_matrix_qr_decomposition_single_core_kernel(
    device_matrix_t matrix, float *diagonal, float *c, int dimension,
    bool *is_singular) {
    float column_length;  // sigma in book
    float column_length_squared, element;
    int n = dimension;
    float scale;
    *is_singular = false;

    // for every column
    for (int k = 0; k < n; k++) {
        scale = 0.0f;
        // scale is the max absolute value of the column
        for (int i = k; i < n; i++)
            scale = fmaxf(scale, fabsf(matrix[INDEX(i, k, n)]));

        if (scale == 0.0) {
            *is_singular = true;
            c[k] = diagonal[k] = 0.0f;
            continue;
        }
        // Normalize column
        for (int i = k; i < n; i++) matrix[INDEX(i, k, n)] /= scale;

        // column length below diagonal
        column_length_squared = 0.0f;  // sum in book.
        for (int i = k; i < n; i++) {
            element = matrix[INDEX(i, k, n)];
            column_length_squared += element * element;
        }

        // column length below diagonal, with the sign of diagonal k
        column_length =
            SIGN(sqrtf(column_length_squared), matrix[INDEX(k, k, n)]);

        // add column length to diagonal k
        matrix[INDEX(k, k, n)] += column_length;

        c[k] = matrix[INDEX(k, k, n)] * column_length;

        diagonal[k] = -scale * column_length;

        // Calculate Q[k] = I - (u[k] (x) u[k]) / c[k]
        for (int j = k + 1; j < n; j++) {
            // inner product for column j below diagonal
            float inner_product = 0.0f;
            for (int i = k; i < n; i++) {
                inner_product +=
                    matrix[(INDEX(i, k, n))] * matrix[(INDEX(i, j, n))];
            }

            // division
            float tau = inner_product / c[k];

            for (int i = k; i < n; i++) {
                matrix[(INDEX(i, j, n))] -= tau * matrix[(INDEX(i, k, n))];
            }
        }
    }

    if (!*is_singular) *is_singular = diagonal[n - 1] == 0.0f;
}

bool cuda_qr_decomposition_runner(matrix_t *matrix, float *diagonal, float *c,
    void (*kernel)(device_matrix_t, float *, float *, int, bool *),
    dim3 grid_size, dim3 block_size) {
    device_matrix_t device_matrix =
        cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);

    size_t diagonal_size = sizeof(float) * matrix->columns;

    float *device_diagonal;
    cudaMalloc(&device_diagonal, diagonal_size);

    float *device_c;
    cudaMalloc(&device_c, diagonal_size);

    bool *device_is_singular;
    cudaMalloc(&device_is_singular, sizeof(bool));

    kernel<<<grid_size, block_size>>>(device_matrix, device_diagonal, device_c,
        matrix->columns, device_is_singular);

    bool is_singular;
    cudaMemcpy(
        &is_singular, device_is_singular, sizeof(bool), cudaMemcpyDeviceToHost);
    cuda_matrix_device_to_host(matrix, device_matrix);
    cudaMemcpy(
        diagonal, device_diagonal, diagonal_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, device_c, diagonal_size, cudaMemcpyDeviceToHost);

    cuda_matrix_free(device_matrix);
    cudaFree(device_diagonal);
    cudaFree(device_c);
    cudaFree(device_is_singular);

    return is_singular;
}

bool cuda_matrix_qr_decomposition_single_core(
    matrix_t *matrix, float *diagonal, float *c) {
    return cuda_qr_decomposition_runner(matrix, diagonal, c,
        &(cuda_matrix_qr_decomposition_single_core_kernel), dim3(1), dim3(1));
}


__global__ void cuda_matrix_qr_decomposition_parallel_max_kernel(
    device_matrix_t matrix, float *diagonal, float *c, int dimension,
    bool *is_singular, int k) {
    float column_length;  // sigma in book
    float column_length_squared, element;
    int n = dimension;
    float scale;

    scale = 0.0f;
    // scale is the max absolute value of the column
    for (int i = k; i < n; i++)
        scale = fmaxf(scale, fabsf(matrix[INDEX(i, k, n)]));

    if (scale == 0.0) {
        *is_singular = true;
        c[k] = diagonal[k] = 0.0f;
    }

    // Normalize column
    for (int i = k; i < n; i++) matrix[INDEX(i, k, n)] /= scale;

    // column length below diagonal
    column_length_squared = 0.0f;  // sum in book.
    for (int i = k; i < n; i++) {
        element = matrix[INDEX(i, k, n)];
        column_length_squared += element * element;
    }

    // column length below diagonal, with the sign of diagonal k
    column_length =
        SIGN(sqrtf(column_length_squared), matrix[INDEX(k, k, n)]);

    // add column length to diagonal k
    matrix[INDEX(k, k, n)] += column_length;

    c[k] = matrix[INDEX(k, k, n)] * column_length;

    diagonal[k] = -scale * column_length;

    // Calculate Q[k] = I - (u[k] (x) u[k]) / c[k]
    for (int j = k + 1; j < n; j++) {
        // inner product for column j below diagonal
        float inner_product = 0.0f;
        for (int i = k; i < n; i++) {
            inner_product +=
                matrix[(INDEX(i, k, n))] * matrix[(INDEX(i, j, n))];
        }

        // division
        float tau = inner_product / c[k];

        for (int i = k; i < n; i++) {
            matrix[(INDEX(i, j, n))] -= tau * matrix[(INDEX(i, k, n))];
        }
    }

}

bool cuda_matrix_qr_decomposition_parallel_max(matrix_t* matrix, float* diagonal, float* c) {
    device_matrix_t device_matrix = cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);

    size_t diagonal_size = sizeof(float) * matrix->columns;

    float *device_diagonal;
    cudaMalloc(&device_diagonal, diagonal_size);

    float *device_c;
    cudaMalloc(&device_c, diagonal_size);

    bool *device_is_singular;
    cudaMalloc(&device_is_singular, sizeof(bool));

    for (int k = 0; k < matrix->columns; k++)
    {
        cuda_matrix_qr_decomposition_parallel_max_kernel<<<dim3(1), dim3(1)>>>(device_matrix, device_diagonal, device_c,
        matrix->columns, device_is_singular, k);
    }

    bool is_singular;
    cudaMemcpy(
        &is_singular, device_is_singular, sizeof(bool), cudaMemcpyDeviceToHost);
    cuda_matrix_device_to_host(matrix, device_matrix);
    cudaMemcpy(
        diagonal, device_diagonal, diagonal_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, device_c, diagonal_size, cudaMemcpyDeviceToHost);

    cuda_matrix_free(device_matrix);
    cudaFree(device_diagonal);
    cudaFree(device_c);
    cudaFree(device_is_singular);

    return is_singular;
}