#include <CUnit/Basic.h>
#include <CUnit/Console.h>

#include "test_csv_utility.h"
#include "test_cuda_matrix_algorithms.h"
#include "test_matrix_algorithms.h"

int main() {
    CU_pSuite matrix_suite = NULL;
    CU_pSuite csv_suite = NULL;
    CU_pSuite cuda_matrix_suite = NULL;

    /* initialize the CUnit test registry */
    if (CUE_SUCCESS != CU_initialize_registry()) return CU_get_error();

    /* add a suite to the registry */
    matrix_suite =
        CU_add_suite("matrix_t Tests", init_matrix_suite, clean_matrix_suite);
    if (NULL == matrix_suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add a suite to the registry */
    csv_suite = CU_add_suite("CSV Tests", init_csv_suite, clean_csv_suite);
    if (NULL == csv_suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add a suite to the registry */
    cuda_matrix_suite = CU_add_suite(
        "Cuda matrix_t Tests", init_cuda_matrix_suite, clean_cuda_matrix_suite);
    if (NULL == cuda_matrix_suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* add the tests to the suite */
    /* NOTE - ORDER IS IMPORTANT - MUST TEST fread() AFTER fprintf() */
    if ((NULL == CU_add_test(
                     matrix_suite, "matrix_t Init Test", test_init_matrix)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t Init 2x2 From CSV Test",
                     test_init_matrix_2x2_from_csv)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t Init 4x1 From CSV Test",
                     test_init_matrix_4x1_from_csv)) ||
        (NULL == CU_add_test(matrix_suite,
                     "matrix_t Init 2x2 doubled From CSV Test",
                     test_init_matrix_2x2_doubled_from_csv)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t dimension equality",
                     test_matrix_equal_dimensions)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t dimension inequality",
                     test_matrix_not_equal_dimensions)) ||
        (NULL == CU_add_test(
                     matrix_suite, "matrix_t equality", test_matrix_equal)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t almost equal",
                     test_matrix_almost_equal)) ||
        (NULL ==
            CU_add_test(matrix_suite, "matrix_t copy", test_matrix_copy)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t addition",
                     test_matrix_addition)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t multiplication cpu",
                     test_matrix_multiplication)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t qr decomposition cpu",
                     test_matrix_qr_decomposition)) ||
        (NULL == CU_add_test(cuda_matrix_suite,
                     "matrix_t addition gpu single core",
                     test_matrix_addition_gpu_single_core)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t Init Test Bad Values",
                     test_init_matrix_0_values)) ||
        (NULL == CU_add_test(matrix_suite, "matrix_t Random Fill Test In Range",
                     test_matrix_random_fill)) ||
        (NULL == CU_add_test(cuda_matrix_suite, "Test cuda matrix utility",
                     test_cuda_matrix_utility)) ||
        (NULL == CU_add_test(cuda_matrix_suite,
                     "matrix_t addition gpu multi core",
                     test_matrix_addition_gpu_multi_core)) ||
        (NULL == CU_add_test(cuda_matrix_suite,
                     "matrix_t addition gpu multi core 2",
                     test_matrix_addition_gpu_multi_core2)) ||
        (NULL == CU_add_test(cuda_matrix_suite,
                     "matrix_t addition gpu multi core 2 larger matrices",
                     test_matrix_addition_gpu_multi_core2_larger_matrices)) ||
        (NULL == CU_add_test(cuda_matrix_suite,
                     "matrix_t multiplication gpu single core",
                     test_matrix_multiplication_gpu_single_core)) ||
        (NULL == CU_add_test(cuda_matrix_suite,
                     "matrix_t multiplication gpu multi core: unwrapping i",
                     test_matrix_multiplication_gpu_multi_core_unwrapping_i)) ||
        (NULL ==
            CU_add_test(cuda_matrix_suite,
                "matrix_t multiplication gpu multi core: unwrapping i and j",
                test_matrix_multiplication_gpu_multi_core_unwrapping_i_and_j)) ||
        (NULL ==
            CU_add_test(cuda_matrix_suite,
                "matrix_t multiplication gpu multi core: unwrapping i and j "
                "for "
                "larger matrices",
                test_matrix_multiplication_gpu_multi_core_unwrapping_i_and_j_larger_matrices)) ||
        (NULL ==
            CU_add_test(cuda_matrix_suite,
                "matrix_t multiplication gpu multi core: shared memory",
                test_matrix_multiplication_gpu_multi_core_shared_memory)) ||
        (NULL ==
            CU_add_test(cuda_matrix_suite,
                "matrix_t multiplication gpu multi core: shared memory for "
                "larger matrices",
                test_matrix_multiplication_gpu_multi_core_shared_memory_larger_matrices))) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Run all tests using the CUnit Basic interface */
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}