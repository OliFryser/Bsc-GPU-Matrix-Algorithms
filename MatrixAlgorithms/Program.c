#include <stdio.h>
#include <stdlib.h>
#include "ArrayAlgorithms.h"
#include "DataWriter.h"

int *random_numbers(int size, int max_value);
FILE *createCSV(char *csvPath);
void writeToCSV(FILE *file, int data[], int size_of_array);

int main(int argc, char *argv[]) {
    char *save_file_name;
    FILE *file;
    int number_count;
    int *numbers;

    if (argc < 3) {
        printf("Usage: %s <file_path.csv> <number_count>\n", argv[0]);
        return 0;
    }

    save_file_name = argv[1];
    number_count = atoi(argv[2]);

    numbers = random_numbers(number_count, 10);
    file = createCSV(save_file_name);
    if (file == NULL) return 0;
    writeToCSV(file, numbers, number_count);

    return 0;
}