#ifndef ARRAY_ALGORITHMS_H
#define ARRAY_ALGORITHMS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix_utility.h"

int sum_of_array(int array[], int size_of_array);
int *random_numbers(int number_count, int max_value);
void print_numbers(int array[], int size_of_array);
bool array_almost_equal(float array1[], float array2[], int length);
double mean(double array[], int size_of_array);
double standard_deviation(double array[], int size_of_array, double mean);
void array_random_fill(float *array, int length);
void print_floats(float *array, int size_of_array);

#endif