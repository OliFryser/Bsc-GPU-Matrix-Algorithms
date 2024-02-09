# Bsc-GPU-Matrix-Algorithms

Bachelor's project

## Prerequisites

- sudo apt-get install libcunit1 libcunit1-doc libcunit1-dev

## Testing

In the source folder, run ``make test``, then run ``./bin/tests``
To remove all build files run ``make clean``.

## Matrix CSV-format

Because C does not easily support dynamic data structures, we need to know the dimenions in advance, 
so we can allocate the right amount of memory. Because of this, the first line of the CSV file should be two integers:

    row_count, column_count
followed by the entries of the matrix as floats. It is important that the entries match the row and column count
    
    value00, value01
    value10, value11

A csv might contain the following 3x3 matrix: 

    3,3
    1.0,2.0,3.0
    4.0,5.0,6.0
    7.0,8.0,9.0

## Core-algorithms considerations

A matrix-calculating function can look in two ways. 
In the first one, a new Matrix is created and returns a pointer to the new matrix: 

    *Matrix calculate(Matrix *matrix1, Matrix *matrix2);
    
In the second case, the result matrix is also passed to the function to be modified and the method returns void. 

    void calculate(Matrix *matrix1, Matrix *matrix2, Matrix *matrix_result);


Things to consider:
- Is the size of the resulting matrix known in advance, and can be initialized within the function?
- Ease of use.
- How will each approach affect benchmarking. I.e. will the allocation of memory also be measured as part of the calculation. 