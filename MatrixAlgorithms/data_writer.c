#include <stdio.h>
#include <stdlib.h>
#include "data_writer.h"

FILE *create_csv(char *csvPath) {
    FILE *csvFile;
    csvFile = fopen(csvPath, "w");
    if (csvFile == NULL) {
        fprintf(stderr, "Unable to open file for writing.\n");
    }
    return csvFile;
}

FILE *get_csv(char *csvPath) {
    FILE *csvFile;
    csvFile = fopen(csvPath, "a");
    if (csvFile == NULL) {
        fprintf(stderr, "Unable to open file for appending.\n");
    }
    return csvFile;
}

void write_to_csv(FILE *file, char algorithm_name[], char input_size[], char run_time[]) {    
    fprintf(file, "\n%s,%s,%s", algorithm_name, input_size, run_time);
}