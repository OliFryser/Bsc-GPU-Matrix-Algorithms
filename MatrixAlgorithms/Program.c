#include <stdio.h>
#include "ArrayAlgorithms.h"
#include "DataWriter.h"

int *random_numbers(int size, int max_value);
FILE *createCSV(char *csvPath);
void writeToCSV(FILE *file, int data[], int size_of_array);

int main() {
    int *numbers;
    FILE *file;
    const int number_count = 5;

    numbers = random_numbers(number_count, 10);
    file = createCSV("DummyData/RandomData.csv");
    if (file == NULL) return 0;
    writeToCSV(file, numbers, number_count);

    return 0;
}