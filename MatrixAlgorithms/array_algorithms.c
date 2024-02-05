#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_algorithms.h"

int sum_of_array(int array[], int size_of_array) {
    int sum;
    sum = 0;
    for (int i = 0; i < size_of_array; i++)
    {
        sum += array[i];
    }
    return sum;
}

int *random_numbers(int number_count, int max_value) {
    int *numbers = malloc(number_count * sizeof(int));
    if (numbers == NULL) {
        fprintf(stderr, "Could not allocate array for random numbers.");
        return NULL;
    }
    
    srand((unsigned int)time(NULL));
    max_value += 1;
    for (int i = 0; i < number_count; i++)
    {
        numbers[i] = rand() % max_value;
    }

    return numbers;
}

void print_numbers(int array[], int size_of_array) {
    for (int i = 0; i < size_of_array; i++)
    {
        printf("%d", array[i]);
        if (i < size_of_array - 1) printf(", ");
        printf("\n");
    }
}