extern "C" {
#include "cuda_diagnostic.h"
}

__global__ void noop_kernel() {}

void launch_kernel_1_block_1_thread() {
    // cudaDeviceProp result;
    // int device;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&result, device);
    // printf(
    //     "pageableMemoryAccess: %d, hostNativeAtomicSupported: %d, "
    //     "pageableMemoryAccessUsesHostPageTables: %d, "
    //     "directManagedMemAccessFromHost: %d\n",
    //     result.pageableMemoryAccess, result.hostNativeAtomicSupported,
    //     result.pageableMemoryAccessUsesHostPageTables,
    //     result.directManagedMemAccessFromHost);
    // printf("concurrentManagedAccess: %d, pageableMemoryAccess: %d\n",
    //     result.concurrentManagedAccess, result.pageableMemoryAccess);
    // printf("managedMemory: %d, concurrentManagedAccess: %d\n",
    //     result.managedMemory, result.concurrentManagedAccess);

    noop_kernel<<<1, 1>>>();
}

bool launch_kernel_1_block_1_thread_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    launch_kernel_1_block_1_thread();
    return true;
}

void launch_kernel_scaling_with_dimension(int dimension) {
    noop_kernel<<<dimension, 1024>>>();
}

bool launch_kernel_scaling_with_dimension_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    launch_kernel_scaling_with_dimension(arg_a->matrix->rows);
    return true;
}

void malloc_scaling_with_dimension(matrix_t *matrix) {
    device_matrix_t device_matrix =
        cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_free(device_matrix);
}

bool malloc_scaling_with_dimension_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    malloc_scaling_with_dimension(arg_a->matrix);
    return true;
}

void memcpy_scaling_with_dimension(matrix_t *matrix) {
    device_matrix_t device_matrix =
        cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);
    cuda_matrix_device_to_host(matrix, device_matrix);
    cuda_matrix_free(device_matrix);
}

bool memcpy_scaling_with_dimension_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    memcpy_scaling_with_dimension(arg_a->matrix);
    return true;
}

void memcpy_and_kernel_launch(matrix_t *matrix) {
    device_matrix_t device_matrix =
        cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);
    noop_kernel<<<1, 1>>>();
    cuda_matrix_device_to_host(matrix, device_matrix);
    cuda_matrix_free(device_matrix);
}

bool memcpy_and_kernel_launch_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    memcpy_and_kernel_launch(arg_a->matrix);
    return true;
}

void memcpy_and_larger_kernel_launch(matrix_t *matrix) {
    device_matrix_t device_matrix =
        cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);
    noop_kernel<<<matrix->columns, 1024>>>();
    cuda_matrix_device_to_host(matrix, device_matrix);
    cuda_matrix_free(device_matrix);
}

bool memcpy_and_larger_kernel_launch_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    memcpy_and_larger_kernel_launch(arg_a->matrix);
    return true;
}

__global__ void write_managed_vector_kernel(device_matrix_t matrix, int size) {
    matrix[blockIdx.x] = 4.0f;
}

void write_managed_vector(matrix_t *matrix) {
    device_matrix_t vector;
    int size = matrix->columns * matrix->rows;
    cudaMallocManaged(&vector, size * sizeof(float));
    write_managed_vector_kernel<<<size, 1>>>(vector, size);
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
        printf("\n%f", vector[i]);
    }
}

bool write_managed_vector_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    write_managed_vector(arg_a->matrix);
    return true;
}

// only memcpy, dont launch kernel
// memcpy more and more data and launch kernel
// memcpy more and more data dont launch kernel
// launch larger kernels