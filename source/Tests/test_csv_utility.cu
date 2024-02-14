#include "test_csv_utility.h"

FILE *csv_file;

int init_csv_suite(void) {
    char *csv_path;
    csv_path = "./source/Tests/csv_test_matrix_2x2.csv";
    csv_file = read_csv(csv_path);
    if (csv_file == NULL) return -1;
    return 0;
}
int clean_csv_suite(void) {
    close_file(csv_file);
    return 0;
}