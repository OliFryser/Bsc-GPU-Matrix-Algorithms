#include <stdio.h>
#include <stdlib.h>
#include "data_writer.h"

FILE *createCSV(char *csvPath) {
    FILE *csvFile;
    csvFile = fopen(csvPath, "w");
    if (csvFile == NULL) {
        fprintf(stderr, "Unable to open file for writing.\n");
    }
    return csvFile;
}

void write_to_csv(FILE *file, int data[], int size_of_array) {    
    for (int i = 0; i < size_of_array; i++)
    {
        int entry = data[i];
        fprintf(file, "%d", entry);
        if (i < size_of_array - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");
}

int date_writer_example() {
    char *csvPath = "DummyData/RandomData.csv";
    FILE *csvFile = createCSV(csvPath);
    int data[] = { 1, 2, 3, 4, 5 };
    int size_of_array = sizeof(data) / sizeof(data[0]);
    write_to_csv(csvFile, data, size_of_array);
}