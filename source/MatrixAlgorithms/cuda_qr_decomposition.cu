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

bool cuda_matrix_qr_decomposition_parallel_max_adapter(
    algorithm_arg_t *matrix, algorithm_arg_t *diagonal, algorithm_arg_t *c) {
    return cuda_matrix_qr_decomposition_parallel_max(
        matrix->matrix, diagonal->vector, c->vector);
}

bool cuda_matrix_qr_decomposition_single_core_adapter(
    algorithm_arg_t *matrix, algorithm_arg_t *diagonal, algorithm_arg_t *c) {
    return cuda_matrix_qr_decomposition_single_core(
        matrix->matrix, diagonal->vector, c->vector);
}

bool cuda_matrix_qr_decomposition_single_core(
    matrix_t *matrix, float *diagonal, float *c) {
    return cuda_qr_decomposition_runner(matrix, diagonal, c,
        &(cuda_matrix_qr_decomposition_single_core_kernel), dim3(1), dim3(1));
}

__global__ void cuda_setup_column_kernel(
    device_matrix_t matrix, int column, int dimension, float *destination) {
    int column_index = 0;
    for (int i = column; i < dimension; i++) {
        destination[column_index] = matrix[INDEX(i, column, dimension)];
        column_index++;
    }
}

#define ELEMENTS_PR_THREAD 4
#define BLOCK_SIZE 4

__global__ void cuda_parallel_max_kernel(
    float *blocks, float *column, int column_length) {
    __shared__ float cache[BLOCK_SIZE];  // blockDim.x
    int i = blockIdx.x * ELEMENTS_PR_THREAD * blockDim.x + threadIdx.x;
    int cache_index = threadIdx.x;
    float thread_max = fabsf(column[0]);
    for (int j = 0; j < ELEMENTS_PR_THREAD; j++) {
        if (i >= column_length) break;
        if (fabsf(column[i]) > thread_max) thread_max = fabsf(column[i]);
        i += blockDim.x;
    }

    cache[cache_index] = thread_max;  // set the cache value

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int split_index = blockDim.x / 2;
    while (split_index != 0) {
        if (cache_index < split_index &&
            cache[cache_index + split_index] > cache[cache_index])
            cache[cache_index] = cache[cache_index + split_index];

        __syncthreads();

        split_index /= 2;
    }

    if (cache_index == 0) blocks[blockIdx.x] = cache[0];
}

enum FolderType { MAX, SUM };

typedef float (*folder_t)(float*, float, float);

__device__ float cuda_max_absolute(float *result, float a, float b) {
    *result = fmaxf(fabsf(a), fabsf(b));
}

__device__ float cuda_accumulate_sum_of_products(float *result, float a, float b) {
    *result += a * b;
}

__global__ void cuda_parallel_reduction_kernel(
    float *blocks, float *column, int column_length, FolderType folder_type) {
    __shared__ float cache[BLOCK_SIZE];  // blockDim.x
    int i = blockIdx.x * ELEMENTS_PR_THREAD * blockDim.x + threadIdx.x;
    int cache_index = threadIdx.x;
    float thread_max = fabsf(column[0]);
    folder_t folder;
    if (folder_type == MAX) folder = cuda_max_absolute;
    else if (folder_type == SUM) folder = cuda_accumulate_sum_of_products;
    for (int j = 0; j < ELEMENTS_PR_THREAD; j++) {
        if (i >= column_length) break;
        folder(&thread_max, column[i], thread_max);
        // if (fabsf(column[i]) > thread_max) thread_max = fabsf(column[i]);
        i += blockDim.x;
    }

    cache[cache_index] = thread_max;  // set the cache value

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int split_index = blockDim.x / 2;
    while (split_index != 0) {
        if (cache_index < split_index &&
            cache[cache_index + split_index] > cache[cache_index])
            cache[cache_index] = cache[cache_index + split_index];

        __syncthreads();

        split_index /= 2;
    }
    printf("Max: %f", cache[0]);
    if (cache_index == 0) blocks[blockIdx.x] = cache[0];
}

__global__ void cuda_matrix_qr_decomposition_kernel(device_matrix_t matrix,
    float *diagonal, float *c, int dimension, bool *is_singular, int k,
    float *scale_in_memory) {
    float column_length;  // sigma in book
    float column_length_squared, element;
    int n = dimension;
    float scale = *scale_in_memory;
    *is_singular = false;

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
    column_length = SIGN(sqrtf(column_length_squared), matrix[INDEX(k, k, n)]);

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

__global__ void cuda_max_value(
    float *device_scale, const float *values, int grid_size) {
    float max = values[0];
    for (int i = 1; i < grid_size; i++)
        if (values[i] > max) max = values[i];
    *device_scale = max;
}

__global__ void test_kernel(int number) { printf("\nTesting: %d", number); }

bool cuda_matrix_qr_decomposition_parallel_max(
    matrix_t *matrix, float *diagonal, float *c) {
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

    float *device_scale;
    cudaMalloc(&device_scale, sizeof(float));

    int dimension = matrix->columns;
    int grid_size = (dimension + ELEMENTS_PR_THREAD * BLOCK_SIZE - 1) /
                    (ELEMENTS_PR_THREAD * BLOCK_SIZE);

    float *device_blocks;
    cudaMalloc(&device_blocks, sizeof(float) * grid_size);

    float *column_after_k;
    cudaMalloc(&column_after_k, sizeof(float) * dimension);

    for (int k = 0; k < dimension; k++) {
        grid_size = (dimension - k + ELEMENTS_PR_THREAD * BLOCK_SIZE - 1) /
                    (ELEMENTS_PR_THREAD * BLOCK_SIZE);

        cuda_setup_column_kernel<<<1, 1>>>(
            device_matrix, k, dimension, column_after_k);

        cudaDeviceSynchronize();

        cuda_parallel_reduction_kernel<<<grid_size, BLOCK_SIZE>>>(
            device_blocks, column_after_k, (dimension - k), MAX);

        cudaDeviceSynchronize();

        cuda_max_value<<<1, 1>>>(device_scale, device_blocks, grid_size);

        cudaDeviceSynchronize();

        cuda_matrix_qr_decomposition_kernel<<<1, 1>>>(device_matrix,
            device_diagonal, device_c, dimension, device_is_singular, k,
            device_scale);

        cudaDeviceSynchronize();
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