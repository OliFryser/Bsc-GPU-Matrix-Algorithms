#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "array_algorithms.h"
#include "csv_utility.h"

void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[], char mean_run_time[], char standard_deviation[]);

int main(int argc, char *argv[])
{
    // Command Line Arguments
    char *algorithm;
    char str_dimension[64];
    int dimension;
    char *save_file_name;

    // Program Variables
    FILE *file;

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

    if (strcmp(algorithm, "addition") == 0) {
        printf("Benchmarking addition...");
    } 
    else if (strcmp(algorithm, "multiplication") == 0) {
        printf("Benchmarking multiplication...");
    }
    else if (strcmp(algorithm, "inverse") == 0) {
        printf("Benchmarking inverse...");
    }

    // file = append_csv(save_file_name);
    // if (file == NULL)
    //     return 0;
    // write_to_csv(file, "CPU Sum of numbers", number_count_string, elapsed_time_string, "");
    // fclose(file);

    return 0;
}