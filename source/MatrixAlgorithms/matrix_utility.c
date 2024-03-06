#include "matrix_utility.h"

Matrix *matrix_init(int rows, int columns) {
    if (rows <= 0 || columns <= 0) return NULL;

    Matrix *matrix;

    matrix = (Matrix *)malloc(sizeof(Matrix));
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

bool matrix_random_fill(float min_value, float max_value, Matrix *matrix) {
    if (matrix == NULL) return false;
    if (matrix->values == NULL) return false;

    for (int i = 0; i < matrix->rows * matrix->columns; i++)
        matrix->values[i] = random_float(min_value, max_value);

    return true;
}

void matrix_free(Matrix *matrix) {
    if (matrix == NULL) return;
    if (matrix->values != NULL) free(matrix->values);
    free(matrix);
}

void matrix_print(Matrix *matrix) {
    int i, j;

    printf("\n# PRINTING MATRIX #\n");
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->columns; j++) {
            printf("%.6f ", matrix->values[INDEX(i, j, matrix->columns)]);
        }
        printf("\n");
    }
}

Matrix *matrix_init_from_csv(FILE *csv_file) {
    Matrix *matrix;
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

bool matrix_equal_dimensions(Matrix *matrix1, Matrix *matrix2) {
    return matrix1->columns == matrix2->columns &&
           matrix1->rows == matrix2->rows;
}

bool matrix_equal(Matrix *matrix1, Matrix *matrix2) {
    if (matrix1 == NULL) return false;
    if (matrix2 == NULL) return false;
    if (!matrix_equal_dimensions(matrix1, matrix2)) return false;

    int rows = matrix1->rows;
    int columns = matrix1->columns;

    for (int i = 0; i < rows * columns; i++)
        if (matrix1->values[i] != matrix2->values[i]) {
            printf("\nFOUND ERROR AT %d,%d\n", i / columns, i % columns);
            printf("\nPrinting matrix 1:\n");
            matrix_print(matrix1);
            printf("\nPrinting matrix 2:\n");
            matrix_print(matrix2);
            return false;
        }

    return true;
}

bool matrix_copy(Matrix *original, Matrix *copy) {
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