#include <stdbool.h>

#include "matrix_utility.h"

bool matrix_addition(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_multiplication(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_addition_gpu_single_core(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result);