#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "array_algorithms.h"
#include "csv_utility.h"
#include "cuda_matrix_algorithms.h"
#include "matrix_algorithms.h"
#include "cuda_diagnostic.h"
#define NANOSECS_PER_SEC 1e9

void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[],
    double mean_run_time, double standard_deviation, int iterations);
double mean(double array[], int size_of_array);
double standard_deviation(double array[], int size_of_array, double mean);
bool launch_kernel_1_block_1_thread_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool launch_kernel_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool malloc_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool memcpy_scaling_with_dimension_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool memcpy_and_kernel_launch_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool memcpy_and_larger_kernel_launch_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool write_managed_vector_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);
bool write_vector_adapter(algorithm_arg_t *arg_a, algorithm_arg_t *arg_b, algorithm_arg_t *arg_c);

int main(int argc, char *argv[]) {
    // Command Line Arguments
    char *algorithm;
    char str_dimension[64];
    int dimension;
    char *save_file_name;

    // Program Variables
    FILE *file;
    matrix_t *matrix_a, *matrix_b, *matrix_c;
    bool (*matrix_algorithm)(
        algorithm_arg_t *, algorithm_arg_t *, algorithm_arg_t *);
    struct timespec start, end;
    double elapsed, elapsed_accumulative;
    double running_times_mean, running_times_standard_deviation;
    int iterations = 2;
    const double minimum_accumulative = 0.5;
    double *running_times;
    char *header;

    if (argc < 4) {
        printf("Usage: %s <algorithm> <dimension> <save_file_path.csv>\n",
            argv[0]);
        return 0;
    }

    // Initialize Command Line Argument Variables
    algorithm = argv[1];
    strcpy(str_dimension, argv[2]);
    dimension = atoi(str_dimension);
    save_file_name = argv[3];

    if (strcmp(algorithm, "addition cpu") == 0)
        matrix_algorithm = &matrix_addition_adapter;
    else if (strcmp(algorithm, "addition gpu single core") == 0)
        matrix_algorithm = &cuda_matrix_addition_single_core_adapter;
    else if (strcmp(algorithm, "addition gpu multi core") == 0)
        matrix_algorithm = &cuda_matrix_addition_multi_core_adapter;
    else if (strcmp(algorithm, "addition gpu multi core 2") == 0)
        matrix_algorithm = &cuda_matrix_addition_multi_core2_adapter;
    else if (strcmp(algorithm, "addition gpu blocks") == 0)
        matrix_algorithm = &cuda_matrix_addition_blocks_adapter;
    else if (strcmp(algorithm, "multiplication cpu") == 0)
        matrix_algorithm = &matrix_multiplication_adapter;
    else if (strcmp(algorithm, "multiplication gpu single core") == 0)
        matrix_algorithm = &cuda_matrix_multiplication_single_core_adapter;
    else if (strcmp(algorithm, "multiplication gpu multi core unwrapping i") ==
             0)
        matrix_algorithm =
            &cuda_matrix_multiplication_multi_core_unwrapping_i_adapter;
    else if (strcmp(algorithm,
                 "multiplication gpu multi core unwrapping i and j") == 0)
        matrix_algorithm =
            &cuda_matrix_multiplication_multi_core_unwrapping_i_and_j_adapter;
    else if (strcmp(algorithm, "shared memory multiplication") == 0)
        matrix_algorithm =
            &cuda_matrix_multiplication_multi_core_shared_memory_adapter;
    else if (strcmp(algorithm, "shared memory fewer accesses") == 0)
        matrix_algorithm =
            &cuda_matrix_multiplication_multi_core_shared_memory_fewer_accesses_adapter;
    else if (strcmp(algorithm, "qr cpu") == 0)
        matrix_algorithm = &matrix_qr_decomposition_adapter;
    else if (strcmp(algorithm, "diagnostic: launch kernel 1 block 1 thread") == 0)
        matrix_algorithm = &launch_kernel_1_block_1_thread_adapter;
    else if (strcmp(algorithm, "diagnostic: launch kernel scaling grid and blocks") == 0)
        matrix_algorithm = &launch_kernel_scaling_with_dimension_adapter;
    else if (strcmp(algorithm, "diagnostic: cudaMalloc") == 0)
        matrix_algorithm = &malloc_scaling_with_dimension_adapter;
    else if (strcmp(algorithm, "diagnostic: cudaMemcpy") == 0)
        matrix_algorithm = &memcpy_scaling_with_dimension_adapter;
    else if (strcmp(algorithm, "diagnostic: cudaMemcpy & launch kernel 1 block 1 thread") == 0)
        matrix_algorithm = &memcpy_and_kernel_launch_adapter;
    else if (strcmp(algorithm, "diagnostic: cudaMemcpy & launch larger kernel") == 0)
        matrix_algorithm = &memcpy_and_larger_kernel_launch_adapter;
    else if (strcmp(algorithm, "diagnostic: write managed") == 0)
        matrix_algorithm = &write_managed_vector_adapter;
    else if (strcmp(algorithm, "diagnostic: write vector") == 0)
        matrix_algorithm = &write_vector_adapter;

    matrix_a = matrix_init(dimension, dimension);
    matrix_b = matrix_init(dimension, dimension);
    matrix_c = matrix_init(dimension, dimension);
    if (matrix_a == NULL || matrix_b == NULL || matrix_c == NULL) return -1;

    matrix_random_fill(0.0f, 3.0f, matrix_a);
    matrix_random_fill(0.0f, 3.0f, matrix_b);

    algorithm_arg_t *arg_a = (algorithm_arg_t *)malloc(sizeof(algorithm_arg_t));
    algorithm_arg_t *arg_b = (algorithm_arg_t *)malloc(sizeof(algorithm_arg_t));
    algorithm_arg_t *arg_c = (algorithm_arg_t *)malloc(sizeof(algorithm_arg_t));

    if (strstr(algorithm, "diagnostic") != NULL) { 
        arg_a->matrix = matrix_a;
    } 
    else if (strstr(algorithm, "qr") != NULL) { 
        arg_a->matrix = matrix_a;
        arg_b->vector = (float *)malloc(sizeof(float) * dimension);
        arg_c->vector = (float *)malloc(sizeof(float) * dimension);
    } 
    else {
        arg_a->matrix = matrix_a;
        arg_b->matrix = matrix_b;
        arg_c->matrix = matrix_c;
    }

    file = append_csv(save_file_name);
    if (file == NULL) return -1;

    header = "Algorithm,\tDimensions,\tMean,\tStandard Deviation,\tIterations";
    write_header_to_csv(file, header);

    // Actual benchmarkings
    do {
        running_times = (double *)malloc(sizeof(double) * iterations);
        if (running_times == NULL) return -1;

        elapsed_accumulative = 0.0;

        for (int i = 0; i < iterations; i++) {
            timespec_get(&start, TIME_UTC);
            matrix_algorithm(arg_a, arg_b, arg_c);
            timespec_get(&end, TIME_UTC);

            elapsed = (end.tv_sec - start.tv_sec) +
                      (end.tv_nsec - start.tv_nsec) / NANOSECS_PER_SEC;
            running_times[i] = elapsed;
            elapsed_accumulative += elapsed;
        }
        running_times_mean = mean(running_times, iterations);
        running_times_standard_deviation =
            standard_deviation(running_times, iterations, running_times_mean);

        write_to_csv(file, algorithm, str_dimension, running_times_mean,
            running_times_standard_deviation, iterations);
        iterations *= 2;
    } while (elapsed_accumulative < minimum_accumulative);

    fclose(file);
    return 0;
}