#ifndef MATRIX_UTILITY_H_2D
#define MATRIX_UTILITY_H_2D

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "2d_csv_utility.h"

typedef struct {
    int rows;
    int columns;
    float **values;
} Matrix;

Matrix *matrix_init(int rows, int columns);
bool matrix_random_fill(float min_value, float max_value, Matrix *matrix);
void matrix_free(Matrix *matrix);
void matrix_print(Matrix *matrix);
Matrix *matrix_init_from_csv(FILE *csv_file);
bool matrix_equal_dimensions(Matrix *matrix1, Matrix *matrix2);
bool matrix_equal(Matrix *matrix1, Matrix *matrix2);
bool matrix_copy(Matrix *original, Matrix *copy);

#endif