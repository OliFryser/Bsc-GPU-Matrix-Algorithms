#ifndef MATRIX_UTILITY_H
#define MATRIX_UTILITY_H
#define INDEX(row_index, column_index, columns) \
    ((row_index) * (columns) + (column_index))

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csv_utility.h"

typedef struct {
    int rows;
    int columns;
    float *values;
} matrix_t;

matrix_t *matrix_init(int rows, int columns);
bool matrix_random_fill(float min_value, float max_value, matrix_t *matrix);
void matrix_free(matrix_t *matrix);
void matrix_print(matrix_t *matrix);
matrix_t *matrix_init_from_csv(FILE *csv_file);
bool matrix_equal_dimensions(matrix_t *matrix_a, matrix_t *matrix_b);
bool almost_equal(float x, float y);
bool matrix_equal(matrix_t *matrix_a, matrix_t *matrix_b);
bool matrix_almost_equal(matrix_t *matrix_a, matrix_t *matrix_b);
bool matrix_copy(matrix_t *original, matrix_t *copy);
bool matrix_r_equal(matrix_t *r, matrix_t *composite, float *diagonal);
void matrix_extract_r(matrix_t *composite, float *d, matrix_t *r_result);
bool matrix_extract_q_j(
    matrix_t *composite, float *c, int j, matrix_t *q_result);
float random_float(float min_value, float max_value);

#endif