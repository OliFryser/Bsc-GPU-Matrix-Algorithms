#include <CUnit/Basic.h>
#include "../MatrixAlgorithms/matrix_utility.h"

float **matrix;
int n = 4;
int m = 4;

/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int init_suite1(void)
{
    // matrix = matrix_init(n, m);
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int clean_suite1(void)
{
    // matrix_free(matrix);
}

/* Simple test of fprintf().
 * Writes test data to the temporary file and checks
 * whether the expected number of bytes were written.
 */
void test_init_matrix(void)
{
    float **actual_matrix = matrix_init(n, m);
    CU_ASSERT_PTR_NOT_NULL_FATAL(actual_matrix);
    int i;
    for (i = 0; i < n; i++)
    {
        CU_ASSERT_PTR_NOT_NULL(actual_matrix[i])
    }
}

int main()
{
    CU_pSuite pSuite = NULL;

    /* initialize the CUnit test registry */
    CU_ErrorCode errorCode = CU_initialize_registry();
    if (CUE_SUCCESS != errorCode)
        return CU_get_error();

    /* add a suite to the registry */
    if (NULL == CU_add_suite("Init Matrix", init_suite1, clean_suite1))
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add the tests to the suite */
    CU_pTest test = CU_add_test(pSuite, "test matrix init function", test_init_matrix);
    if (test == NULL)
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