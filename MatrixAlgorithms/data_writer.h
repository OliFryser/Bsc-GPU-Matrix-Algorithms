#include <stdio.h>

FILE *createCSV(char *csvPath);
void write_to_csv(FILE *file, int data[], int size_of_array);