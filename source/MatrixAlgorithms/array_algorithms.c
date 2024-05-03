#include "array_algorithms.h"

int sum_of_array(int array[], int size_of_array) {
    int sum;
    sum = 0;
    for (int i = 0; i < size_of_array; i++) {
        sum += array[i];
    }
    return sum;
}

int *random_numbers(int number_count, int max_value) {
    int *numbers = (int *)malloc(number_count * sizeof(int));
    if (numbers == NULL) {
        fprintf(stderr, "Could not allocate array for random numbers.");
        return NULL;
    }

    srand((unsigned int)time(NULL));
    max_value += 1;
    for (int i = 0; i < number_count; i++) {
        numbers[i] = rand() % max_value;
    }

    return numbers;
}

void print_numbers(int array[], int size_of_array) {
    for (int i = 0; i < size_of_array; i++) {
        printf("%d", array[i]);
        if (i < size_of_array - 1) printf(", ");
        printf("\n");
    }
}

double mean(double array[], int size_of_array) {
    double sum = 0.0f;
    for (int i = 0; i < size_of_array; i++) {
        sum += array[i];
    }
    return sum / size_of_array;
}

double standard_deviation(double array[], int size_of_array, double mean) {
    double accumulative_deviation = 0.0;
    double entry_deviation;
    for (int i = 0; i < size_of_array; i++) {
        entry_deviation = pow(array[i] - mean, 2.0f);
        accumulative_deviation += entry_deviation;
    }
    return sqrt(accumulative_deviation / size_of_array);
}

bool array_almost_equal(float array1[], float array2[], int length) {
    if (array1 == NULL) return false;
    if (array2 == NULL) return false;
    bool equal;

    for (int i = 0; i < length; i++) {
        equal = almost_equal(array1[i], array2[i]);
        
        if (!equal) {
            printf("\nFOUND ERROR AT %d\n", i);
            printf("\nPrinting array 1:\n");
            print_numbers(array1, length);
            printf("\nPrinting array 2:\n");
            print_numbers(array2, length);
            return false;
        }
    }
    return true;
}