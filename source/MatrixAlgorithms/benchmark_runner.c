#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "array_algorithms.h"
#include "csv_utility.h"
#include "cuda_matrix_algorithms.h"
#include "matrix_algorithms.h"
#define NANOSECS_PER_SEC 1e9

void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[],
                  double mean_run_time, double standard_deviation,
                  int iterations);
bool matrix_addition(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_addition_gpu_single_core(Matrix *matrix1, Matrix *matrix2,
                                     Matrix *result);
bool matrix_multiplication(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_addition_gpu_multi_core(Matrix *matrix1, Matrix *matrix2,
                                    Matrix *result);
bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result);
double mean(double array[], int size_of_array);
double standard_deviation(double array[], int size_of_array, double mean);

int main(int argc, char *argv[]) {
    // Command Line Arguments
    char *algorithm;
    char str_dimension[64];
    int dimension;
    char *save_file_name;

    // Program Variables
    FILE *file;
    Matrix *matrix1, *matrix2, *result;
    bool (*matrix_algorithm)(Matrix *, Matrix *, Matrix *);
    struct timespec start, end;
    double elapsed, elapsed_accumulative;
    double running_times_mean, running_times_standard_deviation;
    int iterations = 2;
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

    if (strcmp(algorithm, "addition") == 0)
        matrix_algorithm = &matrix_addition;
    else if (strcmp(algorithm, "addition gpu single core") == 0)
        matrix_algorithm = &matrix_addition_gpu_single_core;
    else if (strcmp(algorithm, "addition gpu multi core") == 0)
        matrix_algorithm = &matrix_addition_gpu_multi_core;
    else if (strcmp(algorithm, "multiplication") == 0)
        matrix_algorithm = &matrix_multiplication;
    else if (strcmp(algorithm, "inverse") == 0)
        matrix_algorithm = &matrix_inverse;

    matrix1 = matrix_init(dimension, dimension);
    matrix2 = matrix_init(dimension, dimension);
    result = matrix_init(dimension, dimension);
    if (matrix1 == NULL || matrix2 == NULL || result == NULL) return -1;

    file = append_csv(save_file_name);
    if (file == NULL) return -1;

    header = "Algorithm\tDimensions\tMean\tStandard Deviation\tIterations";
    write_header_to_csv(file, header);

    do {
        running_times = (double *)malloc(sizeof(double) * iterations);
        if (running_times == NULL) return -1;

        elapsed_accumulative = 0.0;

        for (int i = 0; i < iterations; i++) {
            timespec_get(&start, TIME_UTC);
            matrix_algorithm(matrix1, matrix2, result);
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
    } while (elapsed_accumulative < 0.5);

    //
    //
    // write_to_csv(file, "CPU Sum of numbers", number_count_string,
    // elapsed_time_string, "");
    fclose(file);
    return 0;
}