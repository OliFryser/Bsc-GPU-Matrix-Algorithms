extern "C" {
    #include "cuda_diagnostic.h"
}

__global__ void noop_kernel() {

}

// Forgot cuda malloc
void only_memcpy() {
    int *ptr = (int *)malloc(sizeof(int));
    *ptr = 10;
    int *device_ptr = (int *)malloc(sizeof(int));
    cudaMemcpy(device_ptr, ptr, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr, device_ptr, sizeof(int), cudaMemcpyDeviceToHost);
}

bool only_memcpy_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    only_memcpy();
    return true;
}

void launch_kernel_without_memcpy() {
    noop_kernel<<<dim3(1), dim3(1)>>>();
}

bool launch_kernel_without_memcpy_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    launch_kernel_without_memcpy();
    return true;
}

// Forgot cuda malloc
void launch_kernel_with_memcpy() {
    int *ptr = (int *)malloc(sizeof(int));
    *ptr = 10;
    int *device_ptr = (int *)malloc(sizeof(int));
    cudaMemcpy(device_ptr, ptr, sizeof(int), cudaMemcpyHostToDevice);
    noop_kernel<<<dim3(1), dim3(1)>>>();
    cudaMemcpy(ptr, device_ptr, sizeof(int), cudaMemcpyDeviceToHost);
}

bool launch_kernel_with_memcpy_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c)
{
    launch_kernel_with_memcpy();
    return true;
}

void launch_kernel_scaling_with_dimension(int dimension) {
    noop_kernel<<<dim3(dimension), dim3(dimension)>>>();
}

bool launch_kernel_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    launch_kernel_scaling_with_dimension(arg_a->matrix->rows);
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

// only memcpy, dont launch kernel
// memcpy more and more data and launch kernel
// memcpy more and more data dont launch kernel
// launch larger kernels