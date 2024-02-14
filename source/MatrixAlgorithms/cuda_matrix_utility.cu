extern "C" {
#include "cuda_matrix_utility.h"
}

extern "C" Matrix *cuda_matrix_init(int rows, int columns) {
    return NULL;
}

extern "C" void cuda_matrix_free(Matrix *matrix) {

}

extern "C" void cuda_matrix_host_to_device(Matrix *dst, Matrix* src) {

}
extern "C" void cuda_matrix_device_to_host(Matrix *dst, Matrix* src) {
    
}