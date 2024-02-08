#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include "../MatrixAlgorithms/matrix_utility.h"
#include "../MatrixAlgorithms/csv_utility.h"

int n = 4;
int m = 4;
Matrix *empty_matrix;

/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int init_matrix_suite(void)
{
    empty_matrix = matrix_init(n, m);
    return 0;
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int clean_matrix_suite(void)
{
    matrix_free(empty_matrix);
    return 0;
}

/* Simple test of fprintf().
 * Writes test data to the temporary file and checks
 * whether the expected number of bytes were written.
 */
void test_init_matrix(void)
{
    CU_ASSERT_PTR_NOT_NULL_FATAL(empty_matrix);
    CU_ASSERT_PTR_NOT_NULL_FATAL(empty_matrix->values)
    int i;
    for (i = 0; i < n; i++)
    {
        CU_ASSERT_PTR_NOT_NULL(empty_matrix->values[i])
    }
}

void test_init_matrix_0_values(void)
{
    Matrix *null_matrix;
    null_matrix = matrix_init(0, 0);
    CU_ASSERT_PTR_NULL(null_matrix);

    null_matrix = matrix_init(0, 1);
    CU_ASSERT_PTR_NULL(null_matrix);

    null_matrix = matrix_init(1, 0);
    CU_ASSERT_PTR_NULL(null_matrix);

    null_matrix = matrix_init(-1, -1);
    CU_ASSERT_PTR_NULL(null_matrix);
}

void test_init_matrix_2x2_from_csv(void)
{
    char *csv_path = "./Tests/csv_test_matrix_2x2.csv";
    FILE *csv_file = read_csv(csv_path);
    CU_ASSERT_PTR_NOT_NULL_FATAL(csv_file);
    Matrix *matrix = matrix_init_from_csv(csv_file);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix);
    CU_ASSERT_EQUAL_FATAL(matrix->rows, 2);
    CU_ASSERT_EQUAL_FATAL(matrix->columns, 2);
    CU_ASSERT_EQUAL(matrix->values[0][0], 0);
    CU_ASSERT_EQUAL(matrix->values[0][1], 1);
    CU_ASSERT_EQUAL(matrix->values[1][0], 2);
    CU_ASSERT_EQUAL(matrix->values[1][1], 3);
}

void test_init_matrix_4x1_from_csv(void)
{
    char *csv_path = "./Tests/csv_test_matrix_4x1.csv";
    FILE *csv_file = read_csv(csv_path);
    CU_ASSERT_PTR_NOT_NULL_FATAL(csv_file);
    Matrix *matrix = matrix_init_from_csv(csv_file);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix);
    CU_ASSERT_EQUAL_FATAL(matrix->rows, 4);
    CU_ASSERT_EQUAL_FATAL(matrix->columns, 1);
    CU_ASSERT_EQUAL(matrix->values[0][0], 0);
    CU_ASSERT_EQUAL(matrix->values[1][0], 1);
    CU_ASSERT_EQUAL(matrix->values[2][0], 2);
    CU_ASSERT_EQUAL(matrix->values[3][0], 3);
}

void test_matrix_equal(void) {
    char *csv_path = "./Tests/csv_test_matrix_2x2.csv";
    FILE *csv_file = read_csv(csv_path);
    CU_ASSERT_PTR_NOT_NULL_FATAL(csv_file);

    Matrix *matrix1 = matrix_init_from_csv(csv_file);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix1);
    rewind(csv_file);
    Matrix *matrix2 = matrix_init_from_csv(csv_file);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix2);

    CU_ASSERT_TRUE(matrix_equal(matrix1, matrix2));
}

void test_matrix_not_equal(void) {
    char *csv_path1 = "./Tests/csv_test_matrix_2x2.csv";
    FILE *csv_file1 = read_csv(csv_path1);
    CU_ASSERT_PTR_NOT_NULL_FATAL(csv_file1);
    char *csv_path2 = "./Tests/csv_test_matrix_4x1.csv";
    FILE *csv_file2 = read_csv(csv_path2);
    CU_ASSERT_PTR_NOT_NULL_FATAL(csv_file2);

    Matrix *matrix1 = matrix_init_from_csv(csv_file1);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix1);
    Matrix *matrix2 = matrix_init_from_csv(csv_file2);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix2);

    CU_ASSERT_FALSE(matrix_equal(matrix1, matrix2));
}