#include <stdio.h>

FILE *append_csv(char *csvPath);
FILE *read_csv(char *csv_path);
void close_file(FILE *file);
void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[], char mean_run_time[], char standard_deviation[]);