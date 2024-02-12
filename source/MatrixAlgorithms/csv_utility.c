#include <stdio.h>
#include <stdlib.h>
#include "csv_utility.h"

FILE *write_csv(char *csv_path) {
    FILE *csv_file;
    csv_file = fopen(csv_path, "w");
    if (csv_file == NULL) {
        fprintf(stderr, "Unable to open file for writing.\n");
    }
    return csv_file;
}

FILE *append_csv(char *csv_path) {
    FILE *csv_file;
    csv_file = fopen(csv_path, "a");
    if (csv_file == NULL) {
        fprintf(stderr, "Unable to open file for appending.\n");
    }
    return csv_file;
}

FILE *read_csv(char *csv_path) {
    FILE *csv_file;
    csv_file = fopen(csv_path, "r");
    if (csv_file == NULL) {
        fprintf(stderr, "Unable to open file for reading.\n");
    }
    return csv_file;
}

void close_file(FILE *file) {
    fclose(file);
}

void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[], double mean_run_time, double standard_deviation, int iterations) {    
    fprintf(file, "\n%s,\tMatrix dimensions %s,\tmean %f,\tstandard deviation %f,\tIterations %d", algorithm_name, matrix_dimensions, mean_run_time, standard_deviation, iterations);
}