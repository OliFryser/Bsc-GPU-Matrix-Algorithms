#include <stdio.h>
#include <stdlib.h>
#include "csv_utility.h"

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

void write_to_csv(FILE *file, char algorithm_name[], char input_size[], char run_time[]) {    
    fprintf(file, "\n%s,%s,%s", algorithm_name, input_size, run_time);
}