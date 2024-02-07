#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "array_algorithms.h"
#include "data_writer.h"

int sum_of_array(int array[], int size_of_array);
int *random_numbers(int size, int max_value);
FILE *get_csv(char *csvPath);
void write_to_csv(FILE *file, char algorithm_name[], char input_size[], char run_time[]);

int main(int argc, char *argv[])
{
    char *save_file_name;
    FILE *file;
    int number_count;
    int *numbers;
    int sum;
    char number_count_string[64];
    char elapsed_time_string[64];

    if (argc < 3)
    {
        printf("Usage: %s <save_file_path.csv> <number_count>\n", argv[0]);
        return 0;
    }

    save_file_name = argv[1];
    strcpy(number_count_string, argv[2]);
    number_count = atoi(number_count_string);

    numbers = random_numbers(number_count, 10);

    clock_t start_time = clock();
    sum = sum_of_array(numbers, number_count);
    clock_t end_time = clock();
    printf("SUM: %d\n", sum);
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    sprintf(elapsed_time_string, "%f", elapsed_time);
    printf("TIME TO CALCULATE: %f\n", elapsed_time);

    file = get_csv(save_file_name);
    if (file == NULL)
        return 0;
    write_to_csv(file, "CPU Sum of numbers", number_count_string, elapsed_time_string);
    fclose(file);

    return 0;
}