#include "matrix_algorithms.h"

bool matrix_addition(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c) {
    if (matrix_a == NULL) return false;
    if (matrix_b == NULL) return false;
    if (matrix_c == NULL) return false;
    if (!matrix_equal_dimensions(matrix_a, matrix_b)) return false;
    if (!matrix_equal_dimensions(matrix_a, matrix_c)) return false;

    int rows = matrix_a->rows;
    int columns = matrix_a->columns;

    for (int i = 0; i < rows * columns; i++)
        matrix_c->values[i] = matrix_a->values[i] + matrix_b->values[i];

    return true;
}

bool matrix_multiplication(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c) {
    if (matrix_a == NULL) return false;
    if (matrix_b == NULL) return false;
    if (matrix_c == NULL) return false;

    if (matrix_a->columns != matrix_b->rows) return false;
    int common_dimension_length = matrix_a->columns;
    int result_rows = matrix_a->rows;
    int result_columns = matrix_b->columns;

    int matrix_a_columns = matrix_a->columns;
    int matrix_b_columns = matrix_b->columns;

    float sum_of_products;

    for (int i = 0; i < result_rows; i++)
        for (int j = 0; j < result_columns; j++) {
            sum_of_products = 0.0f;
            for (int k = 0; k < common_dimension_length; k++)
                sum_of_products +=
                    matrix_a->values[INDEX(i, k, matrix_a_columns)] *
                    matrix_b->values[INDEX(k, j, matrix_b_columns)];
            matrix_c->values[INDEX(i, j, result_columns)] = sum_of_products;
        }
    return true;
}

bool matrix_inverse(
    matrix_t *matrix_a, matrix_t *matrix_b, matrix_t *matrix_c) {
    return false;
}

bool matrix_qr_decomposition(matrix_t *matrix, float *diagonal, float *c) {
    float column_length;  // sigma in book
    int n = matrix->columns;
    for (int k = 0; k < n; k++) {
        float column_length_squared;  // sum in book
        for (int i = 0; i < n; i++) {
            float element = matrix->values[INDEX(i, k, n)];
            column_length_squared += element * element;
        }

        column_length =
            SIGN(sqrtf(column_length_squared), matrix->values[INDEX(k, k, n)]);
        matrix->values[INDEX(k, k, n)] += column_length;
        c[k] = matrix->values[INDEX(k, k, n)] * column_length;
        diagonal[k] = column_length;

        float outer_product;
        for (int j = k + 1; j < n; j++) {
            for (int i = k; i < n; i++) {
                outer_product += matrix->values[(INDEX(i, k, n))] *
                                 matrix->values[(INDEX(i, j, n))];
            }

            float tau = outer_product / c[k];
            for (int i = k; i < n; i++) {
                matrix->values[(INDEX(i, j, n))] -=
                    tau * matrix->values[(INDEX(i, k, n))];
            }
        }
    }

    diagonal[n - 1] = matrix->values[(INDEX(n - 1, n - 1, n))];

    return diagonal[n - 1] != 0.0f;  // Not Singular ?
}