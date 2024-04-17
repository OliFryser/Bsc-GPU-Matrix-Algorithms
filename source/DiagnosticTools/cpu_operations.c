#include "cpu_operations.h"

void cpu_integer_addition(int a, int b, int iterations) {
    int c;
    for (int i = 0; i < iterations; i++) {
        c = a + b;
    }
}

void cpu_integer_multiplication(int a, int b, int iterations) {
    int c;
    for (int i = 0; i < iterations; i++) {
        c = a * b;
    }
}

void cpu_integer_division(int a, int b, int iterations) {
    int c;

    for (int i = 0; i < iterations; i++) {
        c = a / b;
    }
}

void cpu_no_operation(int a, int b, int iterations) {
    for (int i = 0; i < iterations; i++) {
    }
}

void cpu_indexing_1d(int i, int j, int iterations) {
    int c;
    int columns = 128;
    int address = 1024;
    for (int k = 0; k < iterations; k++) {
        c = address + i + j * columns;
    }
}

void cpu_indexing_2d(int i, int j, int iterations) {
    int c;
    int columns = 128;
    int address = 1024;
    for (int k = 0; k < iterations; k++) {
        c = address + i + j;
    }
}