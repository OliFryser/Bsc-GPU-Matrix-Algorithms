extern "C" {
#include "cuda_matrix_algorithms.h"
}

__global__ void cuda_matrix_addition_single_core_kernel(
    device_matrix_t matrix_a, device_matrix_t matrix_b,
    device_matrix_t matrix_c, int size, int rows, int columns) {
    for (int i = 0; i < size; i++) {
        matrix_c[i] = matrix_a[i] + matrix_b[i];
    }
}

__global__ void cuda_matrix_addition_multi_core_kernel(device_matrix_t matrix_a,
    device_matrix_t matrix_b, device_matrix_t matrix_c, int size, int rows,
    int columns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    matrix_c[index] = matrix_a[index] + matrix_b[index];
}

__global__ void cuda_matrix_addition_multi_core_kernel2(
    device_matrix_t matrix_a, device_matrix_t matrix_b,
    device_matrix_t matrix_c, int size, int rows, int columns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= columns) return;

    matrix_c[INDEX(i, j, columns)] =
        matrix_a[INDEX(i, j, columns)] + matrix_b[INDEX(i, j, columns)];
}

__global__ void cuda_matrix_multiplication_single_core_kernel(
    device_matrix_t matrix_a, device_matrix_t matrix_b,
    device_matrix_t matrix_c, int l, int n, int m) {
    float sum_of_products;

    for (int i = 0; i < l; i++)
        for (int j = 0; j < n; j++) {
            sum_of_products = 0.0f;
            for (int k = 0; k < m; k++)
                sum_of_products +=
                    matrix_a[INDEX(i, k, m)] * matrix_b[INDEX(k, j, n)];
            matrix_c[INDEX(i, j, n)] = sum_of_products;
        }
}

__global__ void cuda_matrix_multiplication_multicore_unwrapping_i_kernel(
    device_matrix_t matrix_a, device_matrix_t matrix_b,
    device_matrix_t matrix_c, int l, int n, int m) {
    int i = blockIdx.x;
    float sum_of_products;
    
    for (int j = 0; j < n; j++) {
        sum_of_products = 0.0f;
        for (int k = 0; k < m; k++)
            sum_of_products +=
                matrix_a[INDEX(i, k, m)] * matrix_b[INDEX(k, j, n)];
        matrix_c[INDEX(i, j, n)] = sum_of_products;
    }
}

__global__ void cuda_matrix_multiplication_multicore_unwrapping_i_and_j_kernel(
    device_matrix_t matrix_a, device_matrix_t matrix_b,
    device_matrix_t matrix_c, int l, int n, int m) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    float sum_of_products = 0.0f;

    for (int k = 0; k < m; k++)
        sum_of_products += matrix_a[INDEX(i, k, m)] * matrix_b[INDEX(k, j, n)];

    matrix_c[INDEX(i, j, n)] = sum_of_products;
}

#define BLOCK_SIZE 16

__device__ device_matrix_t get_sub_matrix(
    device_matrix_t matrix, int row, int column, int width) {
    return &matrix[INDEX(row * BLOCK_SIZE, column * BLOCK_SIZE, width)];
}

__device__ void print_device_matrix(
    device_matrix_t matrix, int row, int column, int width) {
    printf("\n\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            printf("%f ", matrix[INDEX(i, j, width)]);
        }
        printf("\n");
    }
    printf("Done");
}

__device__ void print_shared_matrix(float shared_matrix[BLOCK_SIZE][BLOCK_SIZE]) {
    printf("\n");
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            printf("%f ", shared_matrix[j][i]);
        }
        printf("\n");
    }
}

__global__ void cuda_matrix_multiplication_multi_core_shared_memory_kernel(
    device_matrix_t matrix_a, device_matrix_t matrix_b,
    device_matrix_t matrix_c, int l, int n, int m) {
    
    int block_row = blockIdx.y;
    int block_column = blockIdx.x;
    int row = threadIdx.y;
    int column = threadIdx.x;
    float c_value = .0f;

    int subs_in_m = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int k = 0; k < subs_in_m; k++) {
        device_matrix_t a_sub = get_sub_matrix(matrix_a, block_row, k, m);
        __shared__ float shared_a_sub[BLOCK_SIZE][BLOCK_SIZE];
        shared_a_sub[row][column] = a_sub[INDEX(row, column, m)];

        device_matrix_t b_sub = get_sub_matrix(matrix_b, k, block_column, n);
        __shared__ float shared_b_sub[BLOCK_SIZE][BLOCK_SIZE];
        shared_b_sub[row][column] = b_sub[INDEX(row, column, n)];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) 
            c_value += shared_a_sub[row][i] * shared_b_sub[i][column];
        __syncthreads();
    }

    if (row + BLOCK_SIZE * block_row < l && column + BLOCK_SIZE * block_column  < n) {
        device_matrix_t c_sub = get_sub_matrix(matrix_c, block_row, block_column, n);
        c_sub[INDEX(row, column, n)] = c_value;
    } 
}

__global__ void cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses_kernel(
    device_matrix_t matrix_a, device_matrix_t matrix_b, device_matrix_t matrix_c, int l, int n, int m) {
    
    int block_row = blockIdx.y;
    int block_column = blockIdx.x;
    int row = threadIdx.y;
    int column = threadIdx.x;
    float c_value = .0f;

    // Find the top left corner of the sub matrix
    // Then find the row inside the sub matrix
    // Then find the column inside the sub matrix
    device_matrix_t a_sub = &matrix_a[block_row * BLOCK_SIZE * m + row * m + column];
    device_matrix_t b_sub = &matrix_b[block_column * BLOCK_SIZE + row * n + column];

    int subs_in_m = m + BLOCK_SIZE - 1;
    for (int k = 0; k < subs_in_m; k += BLOCK_SIZE) {
        __shared__ float shared_a_sub[BLOCK_SIZE][BLOCK_SIZE];
        shared_a_sub[row][column] = a_sub[k];

        __shared__ float shared_b_sub[BLOCK_SIZE][BLOCK_SIZE];
        shared_b_sub[row][column] = b_sub[k * n];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) 
            c_value += shared_a_sub[row][i] * shared_b_sub[i][column];
        __syncthreads();
    }

    if (row + BLOCK_SIZE * block_row < l && column + BLOCK_SIZE * block_column  < n) {
        device_matrix_t c_sub = get_sub_matrix(matrix_c, block_row, block_column, n);
        c_sub[INDEX(row, column, n)] = c_value;
    } 
}

__global__ void cuda_matrix_qr_decomposition_single_core_kernel(device_matrix_t matrix, float *diagonal, float *c, 
    int dimension, bool *is_singular) {

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
                inner_product += matrix[(INDEX(i, k, n))] *
                                 matrix[(INDEX(i, j, n))];
            }

            // division
            float tau = inner_product / c[k];

            for (int i = k; i < n; i++) {
                matrix[(INDEX(i, j, n))] -=
                    tau * matrix[(INDEX(i, k, n))];
            }
        }
    }

    if (!*is_singular) *is_singular = diagonal[n - 1] == 0.0f;
}

bool cuda_qr_decomposition_runner(matrix_t *matrix, float *diagonal, float *c, 
    void (*kernel)(device_matrix_t, float *, float *, int, bool *), dim3 grid_size, dim3 block_size) {

    device_matrix_t device_matrix = cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);

    size_t diagonal_size = sizeof(float) * matrix->columns;
    
    float *device_diagonal;
    cudaMalloc(&device_diagonal, diagonal_size);

    float *device_c;
    cudaMalloc(&device_c, diagonal_size);

    bool *device_is_singular;
    cudaMalloc(&device_is_singular, sizeof(bool));

    kernel<<<grid_size, block_size>>>(device_matrix, device_diagonal, device_c, matrix->columns, device_is_singular);

    bool is_singular;
    cudaMemcpy(&is_singular, device_is_singular, sizeof(bool), cudaMemcpyDeviceToHost);
    cuda_matrix_device_to_host(matrix, device_matrix);
    cudaMemcpy(diagonal, device_diagonal, diagonal_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, device_c, diagonal_size, cudaMemcpyDeviceToHost);

    cuda_matrix_free(device_matrix);
    cudaFree(device_diagonal);
    cudaFree(device_c);
    cudaFree(device_is_singular);

    return is_singular;
}

bool cuda_matrix_algorithm_runner(matrix_t* matrix_a, matrix_t* matrix_b,
    matrix_t* matrix_c, int kernel_param1, int kernel_param2, int kernel_param3,
    void (*kernel)(
        device_matrix_t, device_matrix_t, device_matrix_t, int, int, int),
    dim3 grid_size, dim3 block_size) {
    if (matrix_a == NULL || matrix_b == NULL || matrix_c == NULL) return false;

    device_matrix_t device_matrix_a =
        cuda_matrix_init(matrix_a->rows, matrix_a->columns);
    device_matrix_t device_matrix_b =
        cuda_matrix_init(matrix_b->rows, matrix_b->columns);
    device_matrix_t device_matrix_c =
        cuda_matrix_init(matrix_c->rows, matrix_c->columns);

    if (device_matrix_a == NULL || device_matrix_b == NULL ||
        device_matrix_c == NULL)
        return false;

    cuda_matrix_host_to_device(device_matrix_a, matrix_a);
    cuda_matrix_host_to_device(device_matrix_b, matrix_b);

    kernel<<<grid_size, block_size>>>(device_matrix_a, device_matrix_b,
        device_matrix_c, kernel_param1, kernel_param2, kernel_param3);

    cuda_matrix_device_to_host(matrix_c, device_matrix_c);

    cuda_matrix_free(device_matrix_a);
    cuda_matrix_free(device_matrix_b);
    cuda_matrix_free(device_matrix_c);

    return true;
}

bool cuda_matrix_addition_single_core_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_addition_single_core(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_addition_single_core(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    return cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_c->rows * matrix_c->columns, matrix_c->rows, matrix_c->columns,
        &(cuda_matrix_addition_single_core_kernel), dim3(1), dim3(1));
}

bool cuda_matrix_addition_multi_core_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_addition_multi_core(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_addition_multi_core(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    bool success;
    dim3 grid_size, block_size;
    grid_size = dim3(matrix_a->rows);
    block_size = dim3(matrix_a->columns);

    success = cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_c->rows * matrix_c->columns, matrix_c->rows, matrix_c->columns,
        &(cuda_matrix_addition_multi_core_kernel), grid_size, block_size);

    return success;
}

bool cuda_matrix_addition_multi_core2_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_addition_multi_core2(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_addition_multi_core2(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    bool success;
    dim3 grid_size, block_size;
    int threads_per_block_dim = 16;

    block_size = dim3(threads_per_block_dim, threads_per_block_dim);
    grid_size = dim3((matrix_a->rows + block_size.x - 1) / block_size.x,
        (matrix_a->columns + block_size.y - 1) / block_size.y);

    success = cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_c->rows * matrix_c->columns, matrix_c->rows, matrix_c->columns,
        &(cuda_matrix_addition_multi_core_kernel2), grid_size, block_size);

    return success;
}

bool cuda_matrix_multiplication_single_core_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_multiplication_single_core(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_multiplication_single_core(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    return cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_a->rows, matrix_b->columns, matrix_a->columns,
        &cuda_matrix_multiplication_single_core_kernel, dim3(1), dim3(1));
}

bool cuda_matrix_multiplication_multi_core_unwrapping_i_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_multiplication_multi_core_unwrapping_i(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_multiplication_multi_core_unwrapping_i(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    return cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_a->rows, matrix_b->columns, matrix_a->columns,
        &cuda_matrix_multiplication_multicore_unwrapping_i_kernel,
        dim3(matrix_a->rows), dim3(1));
}

bool cuda_matrix_multiplication_multi_core_unwrapping_i_and_j_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_multiplication_multi_core_unwrapping_i_and_j(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    return cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_a->rows, matrix_b->columns, matrix_a->columns,
        &cuda_matrix_multiplication_multicore_unwrapping_i_and_j_kernel,
        dim3(matrix_a->rows), dim3(matrix_b->columns));
}

bool cuda_matrix_multiplication_multi_core_shared_memory_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_multiplication_multi_core_shared_memory(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

bool cuda_matrix_multiplication_multi_core_shared_memory(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    bool success;
    dim3 block_dim, grid_dim;

    block_dim = dim3(BLOCK_SIZE, BLOCK_SIZE);

    grid_dim = dim3((matrix_b->columns + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (matrix_a->rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    success = cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_a->rows, matrix_b->columns, matrix_a->columns,
        &(cuda_matrix_multiplication_multi_core_shared_memory_kernel), grid_dim,
        block_dim);

    return success;
}

bool cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses(
    matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    bool success;
    dim3 block_dim, grid_dim;

    block_dim = dim3(BLOCK_SIZE, BLOCK_SIZE);

    grid_dim = dim3((matrix_b->columns + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (matrix_a->rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    success = cuda_matrix_algorithm_runner(matrix_a, matrix_b, matrix_c,
        matrix_a->rows, matrix_b->columns, matrix_a->columns,
        &(cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses_kernel), grid_dim,
        block_dim);

    return success;
}

bool cuda_matrix_qr_decomposition_single_core(matrix_t *matrix, float *diagonal, float *c)
{
    return cuda_qr_decomposition_runner(matrix, diagonal, c, 
    &(cuda_matrix_qr_decomposition_single_core_kernel), dim3(1), dim3(1));
}
