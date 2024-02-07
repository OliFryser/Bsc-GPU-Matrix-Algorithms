#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include "test_matrix_algorithms.h"
#include "test_csv_utility.h"

int main()
{
    CU_pSuite matrix_suite = NULL;
    CU_pSuite csv_suite = NULL;

    /* initialize the CUnit test registry */
    if (CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    /* add a suite to the registry */
    matrix_suite = CU_add_suite("Matrix Tests", init_matrix_suite, clean_matrix_suite);
    if (NULL == matrix_suite)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add a suite to the registry */
    csv_suite = CU_add_suite("CSV Tests", init_csv_suite, clean_csv_suite);
    if (NULL == csv_suite)
    {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add the tests to the suite */
    /* NOTE - ORDER IS IMPORTANT - MUST TEST fread() AFTER fprintf() */
    if ((NULL == CU_add_test(matrix_suite, "Matrix Init Test", test_init_matrix)) || 
        (NULL == CU_add_test(matrix_suite, "Matrix Init From CSV Test", test_init_matrix_from_csv)) || 
        (NULL == CU_add_test(matrix_suite, "Matrix Init Test Bad Values", test_init_matrix_0_values)))
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