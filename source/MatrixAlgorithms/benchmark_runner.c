#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "array_algorithms.h"
#include "csv_utility.h"
#include "matrix_algorithms.h"
#define NANOSECS_PER_SEC 1e9

void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[], char mean_run_time[], char standard_deviation[]);
bool matrix_addition(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_multiplication(Matrix *matrix1, Matrix *matrix2, Matrix *result);
bool matrix_inverse(Matrix *matrix1, Matrix *matrix2, Matrix *result);

int main(int argc, char *argv[])
{
    // Command Line Arguments
    char *algorithm;
    char str_dimension[64];
    int dimension;
    char *save_file_name;

    // Program Variables
    FILE *file;
    int (*matrix_algorithm)(Matrix *, Matrix *, Matrix *);
    struct timespec start, end;
    double elapsed;
    int iterations = 2;
    double *running_times;

    if (argc < 4)
    {
        printf("Usage: %s <algorithm> <dimension> <save_file_path.csv>\n", argv[0]);
        return 0;
    }

    // Initialize Command Line Argument Variables
    algorithm = argv[1];
    strcpy(str_dimension, argv[2]);
    dimension = atoi(str_dimension);
    save_file_name = argv[3];

    if (strcmp(algorithm, "addition") == 0)
        matrix_algorithm = &matrix_addition;
    else if (strcmp(algorithm, "multiplication") == 0)
        matrix_algorithm = &matrix_multiplication;
    else if (strcmp(algorithm, "inverse") == 0)
        matrix_algorithm = &matrix_inverse;

    file = append_csv(save_file_name);
    if (file == NULL)
        return -1;

    do
    {
        running_times = malloc(sizeof(double) * iterations);
        if (running_times == NULL)
            return -1;

        for (int i = 0; i < iterations; i++)
        {
            timespec_get(&start, TIME_UTC);

            timespec_get(&end, TIME_UTC);

            elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / NANOSECS_PER_SEC;
            running_times[i] = elapsed
        }
        write_to_csv(file, "CPU Sum of numbers", "", "", "");

    } while ()

        //
        //
        // write_to_csv(file, "CPU Sum of numbers", number_count_string, elapsed_time_string, "");
        fclose(file);
    return 0;
}