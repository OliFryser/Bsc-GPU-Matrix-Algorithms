#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cpu_operations.h"

#define NANOSECS_PER_SEC 1e9

double run_operation(int iterations, void (*operation)(int, int, int));

int main(int argc, char *argv[]) {
    if (argc < 1) {
        printf("Usage: %s (takes no arguments)", argv[0]);
        return -1;
    }

    printf("### RUNNING CPU DIAGNOSTIC... ###\n");

    double iterations;
    double elapsed_clocks;
    double cycles_per_operation;

    double minimum_elapsed = 0.5 * CLOCKS_PER_SEC;
    int operation_count = 4;

    char *operation_names[operation_count];
    void (*operations[operation_count])(int, int, int);

    // Setup operations
    operations[0] = &cpu_integer_addition;
    operation_names[0] = "CPU Integer Addition";

    operations[1] = &cpu_integer_multiplication;
    operation_names[1] = "CPU Integer Multiplication";

    operations[2] = &cpu_integer_division;
    operation_names[2] = "CPU Integer Division";

    operations[3] = &cpu_no_operation;
    operation_names[3] = "CPU No Operation";

    for (int i = 0; i < operation_count; i++) {
        iterations = 1024;
        do {
            elapsed_clocks = run_operation(iterations, operations[i]);
            iterations *= 2;
        } while (elapsed_clocks < minimum_elapsed);

        cycles_per_operation = elapsed_clocks / iterations;

        printf(
            "Cycles per <%s>: %lf\n", operation_names[i], cycles_per_operation);
    }

    printf("### CPU DIAGNOSTIC DONE! ###\n");

    return 0;
}

/**
 * Runs an operation a given iterations
 * Returns elapsed accumulative
 */
double run_operation(int iterations, void (*operation)(int, int, int)) {
    double elapsed_clocks;
    clock_t start, end;

    elapsed_clocks = 0.0;

    int a = 2;
    int b = 3;
    start = clock();
    operation(a, b, iterations);
    end = clock();

    elapsed_clocks += end - start;

    return elapsed_clocks;
}