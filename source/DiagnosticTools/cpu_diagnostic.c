#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "cpu_operations.h"

#define NANOSECS_PER_SEC 1e9

int run_operation(int iterations, void (*operation)(int, int, int));

int main(int argc, char *argv[]) {
    if (argc < 1) {
        printf("Usage: %s (takes no arguments)", argv[0]);
        return -1;
    }

    printf("### RUNNING CPU DIAGNOSTIC... ###\n");

    double iterations;
    int elapsed_clocks;
    double cycles_per_operation;
    double no_operation_time;

    double minimum_elapsed = 0.5 * 1e9 * 2;
    int operation_count = 5;

    char *operation_names[operation_count];
    void (*operations[operation_count])(int, int, int);

    // Setup operations
    operations[0] = &cpu_integer_addition;
    operation_names[0] = "CPU Integer Addition";

    operations[1] = &cpu_integer_multiplication;
    operation_names[1] = "CPU Integer Multiplication";

    operations[2] = &cpu_integer_division;
    operation_names[2] = "CPU Integer Division";

    operations[3] = &cpu_indexing_1d;
    operation_names[3] = "CPU Indexing 1d";

    operations[4] = &cpu_indexing_2d;
    operation_names[4] = "CPU Indexing 2d";

    iterations = 1048576;
    do {
        elapsed_clocks = run_operation(iterations, &cpu_no_operation);
        iterations *= 2;
    } while (elapsed_clocks < minimum_elapsed);

    no_operation_time = ((double)elapsed_clocks) / iterations;
    printf("Cycles per <%s>: %lf\n", "CPU No Operation", no_operation_time);

    for (int i = 0; i < operation_count; i++) {
        iterations = 1048576;
        do {
            elapsed_clocks = run_operation(iterations, operations[i]);
            iterations *= 2;
        } while (elapsed_clocks < minimum_elapsed);

        cycles_per_operation = ((double)elapsed_clocks) / iterations;
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
int run_operation(int iterations, void (*operation)(int, int, int)) {
    unsigned int elapsed_clocks;
    unsigned int start, end;

    elapsed_clocks = 0;

    int a = 2;
    int b = 3;

    // __rdtsc reads from the register that holds the cpu time
    start = __rdtsc();
    operation(a, b, iterations);
    end = __rdtsc();

    elapsed_clocks = end - start;

    return elapsed_clocks;
}