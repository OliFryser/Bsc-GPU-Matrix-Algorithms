#include <stdlib.h>
#include <stdio.h>

int sum_of_array(int array[], int size_of_array) {
    int sum;
    sum = 0;
    for (int i = 0; i < size_of_array; i++)
    {
        sum += array[i];
    }
    return sum;
}

int *random_numbers(int size, int max_value) {
    int *numbers = malloc(size * sizeof(int));
    if (numbers == NULL) {
        fprintf(stderr, "Could not allocate array for random numbers.");
        exit(1);
    }

    max_value += 1;
    for (int i = 0; i < size; i++)
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

int main(int argc, char *argv[]) {
    int size_of_array;
    if (argc > 1) size_of_array = atoi(argv[1]);
    else size_of_array = 10;

    int max_random_value = 10;
    int *numbers = random_numbers(size_of_array, max_random_value);
    print_numbers(numbers, size_of_array);
    int sum = sum_of_array(numbers, size_of_array);
    printf("SUM: %d\n", sum);
}