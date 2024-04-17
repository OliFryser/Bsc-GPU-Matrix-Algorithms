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
} matrix_t;

matrix_t *matrix_init(int rows, int columns);
bool matrix_random_fill(float min_value, float max_value, matrix_t *matrix);
void matrix_free(matrix_t *matrix);
void matrix_print(matrix_t *matrix);
matrix_t *matrix_init_from_csv(FILE *csv_file);
bool matrix_equal_dimensions(matrix_t *matrix1, matrix_t *matrix2);
bool matrix_equal(matrix_t *matrix1, matrix_t *matrix2);
bool matrix_copy(matrix_t *original, matrix_t *copy);

#endif