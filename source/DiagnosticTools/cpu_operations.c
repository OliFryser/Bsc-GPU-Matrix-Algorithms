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