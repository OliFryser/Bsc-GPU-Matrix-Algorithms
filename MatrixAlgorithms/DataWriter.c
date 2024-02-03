#include <stdio.h>
#include <stdlib.h>

FILE *createCSV(char *csvPath) {
    FILE *csvFile;
    csvFile = fopen(csvPath, "w");
    if (csvFile == NULL) {
        fprintf(stderr, "Unable to open file for writing\n");
        exit(1);
    }
    return csvFile;
}

void writeToCSV(FILE *file, int data[], int size_of_array) {    
    for (int i = 0; i < size_of_array; i++)
    {
        int entry = data[i];
        fprintf(file, "%d", entry);
        if (i < size_of_array - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");
}

int main() {
    char *csvPath = "DummyData/RandomData.csv";
    FILE *csvFile = createCSV(csvPath);
    int data[] = { 1, 2, 3, 4, 5 };
    int size_of_array = sizeof(data) / sizeof(data[0]);
    writeToCSV(csvFile, data, size_of_array);
}