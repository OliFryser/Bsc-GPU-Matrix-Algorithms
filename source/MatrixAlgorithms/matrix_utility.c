#include "matrix_utility.h"

matrix_t *matrix_init(int rows, int columns) {
    if (rows <= 0 || columns <= 0) return NULL;

    matrix_t *matrix;

    matrix = (matrix_t *)malloc(sizeof(matrix_t));
    if (matrix == NULL) {
        return NULL;
    }
    matrix->values = (float *)malloc(rows * columns * sizeof(float *));
    if (matrix->values == NULL) {
        free(matrix);
        return NULL;
    }

    matrix->rows = rows;
    matrix->columns = columns;
    return matrix;
}

float random_float(float min_value, float max_value) {
    max_value -= min_value;
    return (float)(((float)rand() / (float)RAND_MAX) * max_value) + min_value;
}

bool matrix_random_fill(float min_value, float max_value, matrix_t *matrix) {
    if (matrix == NULL) return false;
    if (matrix->values == NULL) return false;

    for (int i = 0; i < matrix->rows * matrix->columns; i++)
        matrix->values[i] = random_float(min_value, max_value);

    return true;
}

void matrix_free(matrix_t *matrix) {
    if (matrix == NULL) return;
    if (matrix->values != NULL) free(matrix->values);
    free(matrix);
}

void matrix_print(matrix_t *matrix) {
    int i, j;

    printf("\n# PRINTING MATRIX #\n");
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->columns; j++) {
            printf("%.6f ", matrix->values[INDEX(i, j, matrix->columns)]);
        }
        printf("\n");
    }
}

matrix_t *matrix_init_from_csv(FILE *csv_file) {
    matrix_t *matrix;
    char line[100];
    char *token;
    float value;
    int rows;
    int columns;
    int row = 0;
    int column;

    if (csv_file == NULL) {
        printf("File is NULL.\n");
        return NULL;
    }

    // read dimensions
    fgets(line, sizeof(line), csv_file);
    token = strtok(line, ",");
    if (token == NULL) return NULL;
    rows = atoi(token);
    token = strtok(NULL, ",");
    if (token == NULL) return NULL;
    columns = atoi(token);

    matrix = matrix_init(rows, columns);
    if (matrix == NULL) return NULL;

    // read values
    while (fgets(line, sizeof(line), csv_file) != NULL) {
        column = 0;
        if (line[0] == '\n') continue;

        token = strtok(line, ",");
        while (token != NULL) {
            value = atof(token);
            matrix->values[INDEX(row, column, columns)] = value;
            token = strtok(NULL, ",");
            column++;
        }
        row++;

        if (column != columns) {
            printf(
                "Wrong column count. Expected %d but got %d", columns, column);
            return NULL;
        }
    }

    if (row != rows) {
        printf("Wrong row count. Expected %d but got %d", rows, row);
        return NULL;
    }

    return matrix;
}

bool matrix_equal_dimensions(matrix_t *matrix_a, matrix_t *matrix_b) {
    return matrix_a->columns == matrix_b->columns &&
           matrix_a->rows == matrix_b->rows;
}

bool matrix_equal(matrix_t *matrix_a, matrix_t *matrix_b) {
    if (matrix_a == NULL) return false;
    if (matrix_b == NULL) return false;
    if (!matrix_equal_dimensions(matrix_a, matrix_b)) return false;

    int rows = matrix_a->rows;
    int columns = matrix_a->columns;

    for (int i = 0; i < rows * columns; i++)
        if (matrix_a->values[i] != matrix_b->values[i]) {
            printf("\nFOUND ERROR AT %d,%d\n", i / columns, i % columns);
            printf("\nPrinting matrix 1:\n");
            matrix_print(matrix_a);
            printf("\nPrinting matrix 2:\n");
            matrix_print(matrix_b);
            return false;
        }

    return true;
}

bool almost_equal(float x, float y) {
    int maxUlps = 2048;
    int xBits = *(int *)&x;  // Evil bit hack from Quake III Q_sqrt
    int yBits = *(int *)&y;  // Evil bit hack from Quake III Q_sqrt
    int minValue = 1 << 31;
    if (xBits < 0) xBits = minValue - xBits;
    if (yBits < 0) yBits = minValue - yBits;

    int difference = xBits - yBits;
    return difference != minValue && fabsf(difference) <= maxUlps;
}

bool matrix_almost_equal(matrix_t *matrix_a, matrix_t *matrix_b) {
    if (matrix_a == NULL) return false;
    if (matrix_b == NULL) return false;
    if (!matrix_equal_dimensions(matrix_a, matrix_b)) return false;

    int rows = matrix_a->rows;
    int columns = matrix_a->columns;
    bool equal;

    for (int i = 0; i < rows * columns; i++) {
        equal = almost_equal(matrix_a->values[i],
            matrix_b->values[i]);  // fabsf(matrix_a->values[i] -
                                   // matrix_b->values[i]) < 0.01f;
        if (!equal) {
            printf("\nFOUND ERROR AT %d,%d\n", i / columns, i % columns);
            printf("\nPrinting matrix 1:\n");
            matrix_print(matrix_a);
            printf("\nPrinting matrix 2:\n");
            matrix_print(matrix_b);
            return false;
        }
    }
    return true;
}

bool matrix_r_equal(matrix_t *r, matrix_t *composite, float *diagonal) {
    bool equal;
    for (int i = 0; i < r->rows; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                equal = almost_equal(
                    r->values[INDEX(i, j, r->columns)], diagonal[i]);
            } else {
                equal = almost_equal(r->values[INDEX(i, j, r->columns)],
                    composite->values[INDEX(i, j, r->columns)]);
            }
            if (!equal) return false;
        }
    }
    return true;
}

void matrix_extract_r(matrix_t *composite, float *d, matrix_t *r_result) {
    for (int i = 0; i < composite->rows; i++)
        for (int j = 0; j < composite->columns; j++) {
            float value;
            if (i == j)
                value = d[i];
            else if (i > j)
                value = 0.0f;
            else
                value = composite->values[INDEX(i, j, composite->columns)];

            r_result->values[INDEX(i, j, r_result->columns)] = value;
        }
}

bool matrix_extract_u_j(matrix_t *composite, int j, float *u) {
    if (composite == NULL || u == NULL) return false;
    if (j > composite->columns) return false;
    for (int i = 0; i < composite->rows; i++) {
        float value;
        if (i < j)
            value = 0.0f;
        else
            value = composite->values[INDEX(i, j, composite->columns)];
        u[i] = value;
    }
    return true;
}

bool vector_outer_product(
    float *vector1, float *vector2, int n, matrix_t *result) {
    if (vector1 == NULL || vector2 == NULL || result == NULL) return false;
    if (result->rows != n || result->columns != n) return false;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result->values[INDEX(i, j, n)] = vector1[i] * vector2[j];

    return true;
}

bool matrix_subtract_from_identity(matrix_t *matrix) {
    if (matrix == NULL) return false;
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->columns; j++) {
            float element = matrix->values[INDEX(i, j, matrix->columns)];

            if (i == j) {
                element = 1 - element;
                // if (i == matrix->rows - 1) {
                //     matrix->values[INDEX(i, j, matrix->columns)] *= -1;
                // }
            } else {
                element *= -1;
            }
            matrix->values[INDEX(i, j, matrix->columns)] = element;
        }
    return true;
}

bool matrix_extract_q_j(
    matrix_t *composite, float *c, int j, matrix_t *q_result) {
    if (composite == NULL || c == NULL || q_result == NULL) return false;
    float *u_j = malloc(sizeof(float) * composite->rows);
    if (u_j == NULL) return false;

    if (!matrix_extract_u_j(composite, j, u_j)) return false;

    if (!vector_outer_product(u_j, u_j, composite->columns, q_result))
        return false;

    for (int i = 0; i < q_result->rows * q_result->columns; i++)
        q_result->values[i] /= c[j];

    if (!matrix_subtract_from_identity(q_result)) return false;

    return true;
}

bool matrix_copy(matrix_t *original, matrix_t *copy) {
    if (original == NULL) return false;
    if (copy == NULL) return false;
    if (!matrix_equal_dimensions(original, copy)) {
        printf("Cannot copy. Matrices are of different dimensions.\n");
        return false;
    }

    int rows = original->rows;
    int columns = original->columns;

    for (int i = 0; i < rows * columns; i++)
        copy->values[i] = original->values[i];

    return true;
}