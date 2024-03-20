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