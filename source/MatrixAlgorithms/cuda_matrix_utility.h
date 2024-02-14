#include "matrix_utility.h"

Matrix *cuda_matrix_init(int rows, int columns);
void cuda_matrix_host_to_device(Matrix *dst, Matrix* src);
void cuda_matrix_device_to_host(Matrix *dst, Matrix* src);