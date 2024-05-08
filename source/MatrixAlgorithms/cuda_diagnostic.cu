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

bool launch_x_kernels(int dimension) {
    for (int i = 0; i < dimension; i++)
    {
        noop_kernel<<<1, 1>>>();
    }
    return true;
}

bool launch_x_kernels_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return launch_x_kernels(arg_a->matrix->rows);
}

bool launch_x_kernels_sequentially(int dimension) {
    for (int i = 0; i < dimension; i++)
    {
        noop_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    return true;
}

bool launch_x_kernels_sequentially_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return launch_x_kernels_sequentially(arg_a->matrix->rows);
}

void write_managed_vector(matrix_t *matrix) {
    device_matrix_t vector;
    int size = matrix->columns * matrix->rows;
    cudaMallocManaged(&vector, size * sizeof(float));
    write_managed_vector_kernel<<<size, 1>>>(vector, size);
    cudaDeviceSynchronize();
    cudaFree(vector);
}

bool write_managed_vector_adapter(
    algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    write_managed_vector(arg_a->matrix);
    return true;
}

void write_vector(matrix_t *matrix) {
    device_matrix_t vector = cuda_matrix_init(matrix->rows, matrix->columns);
    int size = matrix->columns * matrix->rows;
    write_managed_vector_kernel<<<size, 1>>>(vector, size);
    cuda_matrix_device_to_host(matrix, vector);
    cuda_matrix_free(vector);
}

bool write_vector_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    write_vector(arg_a->matrix);
    return true;
}

#define ELEMENTS_PR_THREAD 4
#define BLOCK_SIZE 4

// copied
__device__ float cuda_max_absolute_copied(float a, float b) {
    return fmaxf(fabsf(a), fabsf(b));
}

// copied
__device__ void cuda_parallel_reduction_copied(
    float *cache, int cache_index, reducer_t reduce) {
    int split_index = blockDim.x;
    while (split_index != 0) {
        split_index /= 2;
        if (cache_index < split_index)
            cache[cache_index] =
                reduce(cache[cache_index], cache[cache_index + split_index]);

        __syncthreads();
    }
}

// modified
__global__ void cuda_parallel_max_kernel_copied(float *blocks, device_matrix_t matrix,
    int element_count) {
    __shared__ float cache[BLOCK_SIZE];
    int cache_index = threadIdx.x;
    int starting_index = blockIdx.x * ELEMENTS_PR_THREAD * BLOCK_SIZE + threadIdx.x * ELEMENTS_PR_THREAD;
    float thread_max = fabsf(matrix[starting_index]);

    for (int e = 1; e < ELEMENTS_PR_THREAD; e++) {
        if (e >= element_count) break;
        thread_max = cuda_max_absolute_copied(thread_max, matrix[starting_index + e]);
    }

    cache[cache_index] = thread_max;
    __syncthreads();
    cuda_parallel_reduction_copied(cache, cache_index, cuda_max_absolute_copied);
    if (cache_index == 0) blocks[blockIdx.x] = cache[0];
}

// copied
__global__ void cuda_max_value_copied(
    float *max_value, const float *values, int grid_size) {
    float max = values[0];
    for (int i = 1; i < grid_size; i++) {
        if (values[i] > max) max = values[i];
    }
    *max_value = max;
}


void parallel_max(int element_count, float *vector) {
    float *device_max_value;
    cudaMalloc(&device_max_value, sizeof(float));

    float *device_vector;
    cudaMalloc(&device_vector, sizeof(float) * element_count);
    cudaMemcpy(device_vector, vector, sizeof(float) * element_count, cudaMemcpyHostToDevice);

    int grid_size = (element_count + ELEMENTS_PR_THREAD * BLOCK_SIZE - 1) /
                    (ELEMENTS_PR_THREAD * BLOCK_SIZE);
    float *device_blocks;
    cudaMalloc(&device_blocks, sizeof(float) * grid_size);

    cuda_parallel_max_kernel_copied<<<grid_size, BLOCK_SIZE>>>(device_blocks, device_vector, element_count);
    cuda_max_value_copied<<<1, 1>>>(device_max_value, device_blocks, grid_size);

    float *max_value = (float *)malloc(sizeof(float));
    cudaMemcpy(max_value, device_max_value, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_max_value);
    cudaFree(device_vector);
    cudaFree(device_blocks);
    free(max_value);
}

bool parallel_max_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c)
{
    parallel_max(arg_a->matrix->rows, arg_b->vector); 
    return true;
}

void sequential_max(int element_count, float *vector) {
    float max = vector[0];
    for (int i = 0; i < element_count; i++)
    {
        max = fmaxf(fabsf(max), fabsf(vector[i]));
    }
}

bool sequential_max_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c)
{
    sequential_max(arg_a->matrix->rows, arg_b->vector); 
    return false;
}

// only memcpy, dont launch kernel
// memcpy more and more data and launch kernel
// memcpy more and more data dont launch kernel
// launch larger kernels