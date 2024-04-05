#include "matrix_algorithms.h"

bool matrix_addition_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return matrix_addition(arg_a->matrix, arg_b->matrix, arg_c->matrix);
}

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

bool matrix_multiplication_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return matrix_multiplication(arg_a->matrix, arg_b->matrix, arg_c->matrix);
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

bool matrix_qr_decomposition_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c) {
    return matrix_qr_decomposition(arg_a->matrix, arg_b->vector, arg_c->vector);
}

// returns true if the matrix is singular
bool matrix_qr_decomposition(matrix_t *matrix, float *diagonal, float *c) {
    float column_length;  // sigma in book
    float column_length_squared, element;
    int n = matrix->columns;
    float scale;
    bool is_singular = false;
    // for every column
    for (int k = 0; k < n; k++) {
        scale = 0.0f;
        // scale is the max absolute value of the column
        for (int i = k; i < n; i++)
            scale = fmaxf(scale, fabsf(matrix->values[INDEX(i, k, n)]));

        if (scale == 0.0) {
            is_singular = true;
            c[k] = diagonal[k] = 0.0f;
            continue;
        }
        // Normalize column
        for (int i = k; i < n; i++) matrix->values[INDEX(i, k, n)] /= scale;

        // column length below diagonal
        column_length_squared = 0.0f;  // sum in book.
        for (int i = k; i < n; i++) {
            element = matrix->values[INDEX(i, k, n)];
            column_length_squared += element * element;
        }

        // column length below diagonal, with the sign of diagonal k
        column_length =
            SIGN(sqrtf(column_length_squared), matrix->values[INDEX(k, k, n)]);

        // add column length to diagonal k
        matrix->values[INDEX(k, k, n)] += column_length;

        c[k] = matrix->values[INDEX(k, k, n)] * column_length;

        diagonal[k] = -scale * column_length;

        // Calculate Q[k] = I - (u[k] (x) u[k]) / c[k]
        for (int j = k + 1; j < n; j++) {
            // inner product for column j below diagonal
            float inner_product = 0.0f;
            for (int i = k; i < n; i++) {
                inner_product += matrix->values[(INDEX(i, k, n))] *
                                 matrix->values[(INDEX(i, j, n))];
            }

            // division
            float tau = inner_product / c[k];

            for (int i = k; i < n; i++) {
                matrix->values[(INDEX(i, j, n))] -=
                    tau * matrix->values[(INDEX(i, k, n))];
            }
        }
    }

    if (!is_singular) is_singular = diagonal[n - 1] == 0.0f;
    return is_singular;
}