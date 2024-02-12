#include <stdio.h>

FILE *write_csv(char *csv_path);
FILE *append_csv(char *csvPath);
FILE *read_csv(char *csv_path);
void close_file(FILE *file);
void write_to_csv(FILE *file, char algorithm_name[], char matrix_dimensions[],
                  double mean_run_time, double standard_deviation,
                  int iterations);
void write_header_to_csv(FILE *file, char *header);