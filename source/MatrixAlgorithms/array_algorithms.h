#ifndef ARRAY_ALGORITHMS_H
#define ARRAY_ALGORITHMS_H

#include "matrix_utility.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int sum_of_array(int array[], int size_of_array);
int *random_numbers(int number_count, int max_value);
void print_numbers(int array[], int size_of_array);
double mean(double array[], int size_of_array);
double standard_deviation(double array[], int size_of_array, double mean);

#endif