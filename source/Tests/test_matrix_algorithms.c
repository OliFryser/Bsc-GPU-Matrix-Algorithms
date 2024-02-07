#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include "../MatrixAlgorithms/matrix_utility.h"

int n = 4;
int m = 4;
Matrix *matrix;

/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int init_suite1(void)
{
    matrix = matrix_init(n, m);
    return 0;
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int clean_suite1(void)
{
    matrix_free(matrix);
    return 0;
}

/* Simple test of fprintf().
 * Writes test data to the temporary file and checks
 * whether the expected number of bytes were written.
 */
void test_init_matrix(void)
{
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix);
    CU_ASSERT_PTR_NOT_NULL_FATAL(matrix->values)
    int i;
    for (i = 0; i < n; i++)
    {
        CU_ASSERT_PTR_NOT_NULL(matrix->values[i])
    }
}

void test_free_matrix(void)
{
    matrix_free(matrix);
    CU_ASSERT_PTR_NULL(matrix);
}

int main()
{
    CU_pSuite pSuite = NULL;

    /* initialize the CUnit test registry */
    if (CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    /* add a suite to the registry */
    pSuite = CU_add_suite("Matrix Tests", init_suite1, clean_suite1);
    if (NULL == pSuite)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add the tests to the suite */
    /* NOTE - ORDER IS IMPORTANT - MUST TEST fread() AFTER fprintf() */
    if ((NULL == CU_add_test(pSuite, "Matrix Init Test", test_init_matrix)) || (NULL == CU_add_test(pSuite, "Matrix Free Test", test_free_matrix)))
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Run all tests using the CUnit Basic interface */
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}