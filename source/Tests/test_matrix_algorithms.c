#include "test_matrix_algorithms.h"

int n = 4;
int m = 4;
matrix_t *empty_matrix;
matrix_t *matrix_2x2;
matrix_t *matrix_4x1;
matrix_t *matrix_doubled_2x2;
matrix_t *matrix_multiplication1;
matrix_t *matrix_multiplication2;
matrix_t *matrix_multiplication_expected_result;

/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int init_matrix_suite(void) {
    char *csv_path;
    FILE *csv_file;

    empty_matrix = matrix_init(n, m);

    csv_path = "./source/Tests/csv_test_matrix_2x2.csv";
    csv_file = read_csv(csv_path);
    matrix_2x2 = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_4x1.csv";
    csv_file = read_csv(csv_path);
    matrix_4x1 = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_doubled_2x2.csv";
    csv_file = read_csv(csv_path);
    matrix_doubled_2x2 = matrix_init_from_csv(csv_file);

    // Multiplication matrices
    csv_path = "./source/Tests/csv_test_matrix_multiplication_1.csv";
    csv_file = read_csv(csv_path);
    matrix_multiplication1 = matrix_init_from_csv(csv_file);

    csv_path = "./source/Tests/csv_test_matrix_multiplication_2.csv";
    csv_file = read_csv(csv_path);
    matrix_multiplication2 = matrix_init_from_csv(csv_file);

    csv_path =
        "./source/Tests/csv_test_matrix_multiplication_expected_result.csv";
    csv_file = read_csv(csv_path);
    matrix_multiplication_expected_result = matrix_init_from_csv(csv_file);
    return 0;
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int clean_matrix_suite(void) {
    matrix_free(empty_matrix);
    matrix_free(matrix_2x2);
    matrix_free(matrix_4x1);
    matrix_free(matrix_doubled_2x2);
    matrix_free(matrix_multiplication1);
    matrix_free(matrix_multiplication2);
    matrix_free(matrix_multiplication_expected_result);
    return 0;
}

/* Simple test of fprintf().
 * Writes test data to the temporary file and checks
 * whether the expected number of bytes were written.
 */
void test_init_matrix(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(empty_matrix);
    CU_ASSERT_PTR_NOT_NULL_FATAL(empty_matrix->values)
}

void test_init_matrix_0_values(void) {
    matrix_t *null_matrix;
    null_matrix = matrix_init(0, 0);
    CU_ASSERT_PTR_NULL(null_matrix);

    null_matrix = matrix_init(0, 1);
    CU_ASSERT_PTR_NULL(null_matrix);

    null_matrix = matrix_init(1, 0);
    CU_ASSERT_PTR_NULL(null_matrix);

    null_matrix = matrix_init(-1, -1);
    CU_ASSERT_PTR_NULL(null_matrix);
}

void test_init_matrix_2x2_from_csv(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_2x2);
    CU_ASSERT_EQUAL_FATAL(matrix_2x2->rows, 2);
    CU_ASSERT_EQUAL_FATAL(matrix_2x2->columns, 2);
    int columns = matrix_2x2->columns;
    CU_ASSERT_EQUAL(matrix_2x2->values[(INDEX(0, 0, columns))], 0.0F);
    CU_ASSERT_EQUAL(matrix_2x2->values[(INDEX(0, 1, columns))], 1.0F);
    CU_ASSERT_EQUAL(matrix_2x2->values[(INDEX(1, 0, columns))], 2.0F);
    CU_ASSERT_EQUAL(matrix_2x2->values[(INDEX(1, 1, columns))], 3.0F);
}

void test_init_matrix_4x1_from_csv(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_4x1);
    CU_ASSERT_EQUAL_FATAL(matrix_4x1->rows, 4);
    CU_ASSERT_EQUAL_FATAL(matrix_4x1->columns, 1);
    int columns = matrix_4x1->columns;
    CU_ASSERT_EQUAL(matrix_4x1->values[(INDEX(0, 0, columns))], 0.0F);
    CU_ASSERT_EQUAL(matrix_4x1->values[(INDEX(1, 0, columns))], 1.0F);
    CU_ASSERT_EQUAL(matrix_4x1->values[(INDEX(2, 0, columns))], 2.0F);
    CU_ASSERT_EQUAL(matrix_4x1->values[(INDEX(3, 0, columns))], 3.0F);
}

void test_init_matrix_2x2_doubled_from_csv(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_doubled_2x2);
    CU_ASSERT_EQUAL_FATAL(matrix_doubled_2x2->rows, 2);
    CU_ASSERT_EQUAL_FATAL(matrix_doubled_2x2->columns, 2);
    int columns = matrix_doubled_2x2->columns;
    CU_ASSERT_EQUAL(matrix_doubled_2x2->values[(INDEX(0, 0, columns))], 0.0F);
    CU_ASSERT_EQUAL(matrix_doubled_2x2->values[(INDEX(0, 1, columns))], 2.0F);
    CU_ASSERT_EQUAL(matrix_doubled_2x2->values[(INDEX(1, 0, columns))], 4.0F);
    CU_ASSERT_EQUAL(matrix_doubled_2x2->values[(INDEX(1, 1, columns))], 6.0F);
}

void test_matrix_equal_dimensions(void) {
    CU_ASSERT_TRUE(matrix_equal_dimensions(matrix_2x2, matrix_doubled_2x2));
}

void test_matrix_not_equal_dimensions(void) {
    CU_ASSERT_FALSE(matrix_equal_dimensions(matrix_2x2, matrix_4x1));
}

void test_matrix_equal(void) {
    CU_ASSERT_TRUE(matrix_equal(matrix_2x2, matrix_2x2));
}

void test_matrix_not_equal(void) {
    CU_ASSERT_FALSE(matrix_equal(matrix_2x2, matrix_4x1));
    CU_ASSERT_FALSE(matrix_equal(matrix_2x2, matrix_doubled_2x2));
}

void test_matrix_almost_equal(void) {
    CU_ASSERT_TRUE(matrix_almost_equal(matrix_2x2, matrix_2x2));
}

void test_matrix_copy(void) {
    matrix_t *destination = matrix_init(matrix_2x2->rows, matrix_2x2->columns);
    CU_ASSERT_TRUE_FATAL(matrix_copy(matrix_2x2, destination));
    CU_ASSERT_TRUE(matrix_equal(matrix_2x2, destination));
    matrix_free(destination);
}

void test_matrix_addition(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_2x2);
    matrix_t *result = matrix_init(matrix_2x2->rows, matrix_2x2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(matrix_2x2, result));
    CU_ASSERT_TRUE_FATAL(matrix_addition(matrix_2x2, matrix_2x2, result));
    CU_ASSERT_TRUE(matrix_equal(result, matrix_doubled_2x2));
    matrix_free(result);
}

void test_matrix_multiplication(void) {
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_multiplication1);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_multiplication2);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix_multiplication_expected_result);
    matrix_t *actual_result = matrix_init(
        matrix_multiplication1->rows, matrix_multiplication2->columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_result);
    CU_ASSERT_TRUE_FATAL(matrix_equal_dimensions(
        matrix_multiplication_expected_result, actual_result));
    CU_ASSERT_TRUE_FATAL(matrix_multiplication(
        matrix_multiplication1, matrix_multiplication2, actual_result));
    CU_ASSERT_TRUE(
        matrix_equal(matrix_multiplication_expected_result, actual_result));
    matrix_free(actual_result);
}

bool in_range(float value, float min, float max) {
    return value >= min && value <= max;
}

void test_matrix_random_fill(void) {
    matrix_t *random_matrix;
    float min, max;
    int rows, columns;

    rows = 100;
    columns = 100;

    random_matrix = matrix_init(rows, columns);
    CU_ASSERT_PTR_NOT_NULL_FATAL(random_matrix);

    min = -10.0f;
    max = 10.0f;

    CU_ASSERT_TRUE_FATAL(matrix_random_fill(min, max, random_matrix));

    for (int i = 0; i < rows * columns; i++)
        CU_ASSERT_TRUE(in_range(random_matrix->values[i], min, max));

    matrix_free(random_matrix);
}