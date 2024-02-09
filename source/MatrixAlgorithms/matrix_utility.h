#include <stdio.h>
#include <stdbool.h>

typedef struct
{
    int rows;
    int columns;
    float **values;
} Matrix;

Matrix *matrix_init(int rows, int columns);
bool matrix_random(Matrix *matrix, float min_value, float max_value);
void matrix_free(Matrix *matrix);
void matrix_print(Matrix *matrix);
Matrix *matrix_init_from_csv(FILE *csv_file);
bool matrix_equal_dimensions(Matrix *matrix1, Matrix *matrix2);
bool matrix_equal(Matrix *matrix1, Matrix *matrix2);
bool matrix_copy(Matrix *original, Matrix *copy);
bool matrix_addition(Matrix *matrix1, Matrix *matrix2, Matrix *result);