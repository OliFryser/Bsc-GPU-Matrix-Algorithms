typedef struct
{
    int rows;
    int columns;
    float **values;
} Matrix;

Matrix *matrix_init(int rows, int columns);
void matrix_free(Matrix *matrix);
void matrix_print(Matrix *matrix);