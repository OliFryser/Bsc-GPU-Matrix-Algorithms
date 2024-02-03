#include <stdio.h>

FILE *createCSV(char *csvPath);
void writeToCSV(FILE *file, int data[], int size_of_array);