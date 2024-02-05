#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_algorithms.h"
#include "data_writer.h"

int sum_of_array(int array[], int size_of_array);
int *random_numbers(int size, int max_value);
FILE *createCSV(char *csvPath);
void writeToCSV(FILE *file, int data[], int size_of_array);

int main(int argc, char *argv[]) {
    char *save_file_name;
    FILE *file;
    int number_count;
    int *numbers;
    int sum;

    if (argc < 3) {
        printf("Usage: %s <file_path.csv> <number_count>\n", argv[0]);
        return 0;
    }

    save_file_name = argv[1];
    number_count = atoi(argv[2]);

    numbers = random_numbers(number_count, 10);

    clock_t start_time = clock();
    sum = sum_of_array(numbers, number_count);
    clock_t end_time = clock();
    printf("SUM: %d\n", sum);
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("TIME TO CALCULATE: %f\n", elapsed_time);

    file = createCSV(save_file_name);
    if (file == NULL) return 0;
    write_to_csv(file, numbers, number_count);

    return 0;
}