#include <stdio.h>

FILE *get_csv(char *csvPath);
void write_to_csv(FILE *file, char algorithm_name[], char input_size[], char run_time[]);