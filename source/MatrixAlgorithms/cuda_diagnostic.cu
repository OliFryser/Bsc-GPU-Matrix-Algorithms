extern "C" {
    #include "cuda_diagnostic.h"
}

__global__ void noop_kernel() {

}

void launch_kernel_1_block_1_thread() {
    noop_kernel<<<1, 1>>>();
}

bool launch_kernel_1_block_1_thread_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    launch_kernel_1_block_1_thread();
    return true;
}

void launch_kernel_scaling_with_dimension(int dimension) {
    noop_kernel<<<dimension, 1024>>>();
}

bool launch_kernel_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    launch_kernel_scaling_with_dimension(arg_a->matrix->rows);
    return true;
}

void malloc_scaling_with_dimension(matrix_t *matrix) {
    device_matrix_t device_matrix = cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_free(device_matrix);
}

bool malloc_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    malloc_scaling_with_dimension(arg_a->matrix);
    return true;
}

void memcpy_scaling_with_dimension(matrix_t *matrix) {
    device_matrix_t device_matrix = cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);
    cuda_matrix_device_to_host(matrix, device_matrix);
    cuda_matrix_free(device_matrix);
}

bool memcpy_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c)
{
    memcpy_scaling_with_dimension(arg_a->matrix);
    return true;
}

void memcpy_and_kernel_launch(matrix_t *matrix) {
    device_matrix_t device_matrix = cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);
    noop_kernel<<<1, 1>>>();
    cuda_matrix_device_to_host(matrix, device_matrix);
    cuda_matrix_free(device_matrix);
}

bool memcpy_and_kernel_launch_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c)
{
    memcpy_and_kernel_launch(arg_a->matrix);
    return true;
}

void memcpy_and_larger_kernel_launch(matrix_t *matrix) {
    device_matrix_t device_matrix = cuda_matrix_init(matrix->rows, matrix->columns);
    cuda_matrix_host_to_device(device_matrix, matrix);
    noop_kernel<<<matrix->columns, 1024>>>();
    cuda_matrix_device_to_host(matrix, device_matrix);
    cuda_matrix_free(device_matrix);
}

bool memcpy_and_larger_kernel_launch_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c)
{
    memcpy_and_larger_kernel_launch(arg_a->matrix);
    return true;
}


// only memcpy, dont launch kernel
// memcpy more and more data and launch kernel
// memcpy more and more data dont launch kernel
// launch larger kernels