#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#define NANOSECS_PER_SEC 1e9

int main()
{
    struct timespec start, end;
    double elapsed_time;

    timespec_get(&start, TIME_UTC);
    // Benchmarking
    sleep(5);
    timespec_get(&end, TIME_UTC);

    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / NANOSECS_PER_SEC;
    printf("TIME TO CALCULATE: %f\n", elapsed_time);
}