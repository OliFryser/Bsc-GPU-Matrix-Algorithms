#include "2d_matrix_utility.h"

matrix_t *matrix_init(int rows, int columns) {
    if (rows <= 0 || columns <= 0) return NULL;

    matrix_t *matrix;
    int i;

    matrix = (matrix_t *)malloc(sizeof(matrix_t));
    if (matrix == NULL) {
        return NULL;
    }
    matrix->values = (float **)malloc(rows * sizeof(float *));
    if (matrix->values == NULL) {
        free(matrix);
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        matrix->values[i] = (float *)malloc(columns * sizeof(float));
    }
    matrix->rows = rows;
    matrix->columns = columns;
    return matrix;
}

float random_float(float min_value, float max_value) {
    max_value -= min_value;
    return (float)rand() / (float)(RAND_MAX * max_value) + min_value;
}

bool matrix_random_fill(float min_value, float max_value, matrix_t *matrix) {
    if (matrix == NULL) return false;
    if (matrix->values == NULL) return false;

    int i, j;

    for (i = 0; i < matrix->rows; i++)
        for (j = 0; j < matrix->columns; j++)
            matrix->values[i][j] = random_float(min_value, max_value);

    return true;
}

void matrix_free(matrix_t *matrix) {
    int i;
    if (matrix == NULL) {
        return;
    }
    if (matrix->values != NULL) {
        for (i = 0; i < matrix->rows; i++) {
            if (matrix->values[i] != NULL) free(matrix->values[i]);
        }
        free(matrix->values);
    }
    free(matrix);
}

void matrix_print(matrix_t *matrix) {
    int i, j;

    printf("\n# PRINTING MATRIX #\n");
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->columns; j++) {
            printf("%.2f ", matrix->values[i][j]);
        }
        printf("\n");
    }
}

matrix_t *matrix_init_from_csv(FILE *csv_file) {
    matrix_t *matrix;
    char line[100];
    char *token;
    float value;
    int row_count;
    int column_count;
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
    row_count = atoi(token);
    token = strtok(NULL, ",");
    if (token == NULL) return NULL;
    column_count = atoi(token);

    matrix = matrix_init(row_count, column_count);
    if (matrix == NULL) return NULL;

    // read values
    while (fgets(line, sizeof(line), csv_file) != NULL) {
        column = 0;
        if (line[0] == '\n') continue;

        token = strtok(line, ",");
        while (token != NULL) {
            value = atof(token);
            matrix->values[row][column] = value;
            token = strtok(NULL, ",");
            column++;
        }
        row++;

        if (column != column_count) {
            printf("Wrong column count. Expected %d but got %d", column_count,
                column);
            return NULL;
        }
    }

    if (row != row_count) {
        printf("Wrong row count. Expected %d but got %d", row_count, row);
        return NULL;
    }

    return matrix;
}

bool matrix_equal_dimensions(matrix_t *matrix1, matrix_t *matrix2) {
    return matrix1->columns == matrix2->columns &&
           matrix1->rows == matrix2->rows;
}

bool matrix_equal(matrix_t *matrix1, matrix_t *matrix2) {
    if (matrix1 == NULL) return false;
    if (matrix2 == NULL) return false;
    if (!matrix_equal_dimensions(matrix1, matrix2)) return false;

    int rows = matrix1->rows;
    int columns = matrix1->columns;
    int i, j;

    for (i = 0; i < rows; i++)
        for (j = 0; j < columns; j++)
            if (matrix1->values[i][j] != matrix2->values[i][j]) return false;

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
    int i, j;

    for (i = 0; i < rows; i++)
        for (j = 0; j < columns; j++)
            copy->values[i][j] = original->values[i][j];

    return true;
}