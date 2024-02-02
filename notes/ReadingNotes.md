# Reading Notes

## Matrices

A set of linear algebraic equations looks can be written in matrix form as:
```A * x = b```
Where A is the matrix of coefficients, so a_11 is at column 1 row 1. b is the column vector. We want to solve for x. The * is matrix multiplication.
God forklaring af Linear Algebraic Equations:
[Inverse matrices, column space and null space | Chapter 7, Essence of linear algebra](https://youtu.be/uQhTuRlWMxw?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

Hvis determinanten af ``A != 0``, så vil der eksistere en invers til matrisen A. Hvis determinanten af ``A = 0``, eksisterer der ikke en invers. (ville det give mening først at udregne determinanten til A, før vi prøver at beregne en invers?). Bemærk, linearly dependent implies ``det(A) = 0``.

In C we can access a matrix like ``a[i][j]``, where i is row and j is column. This hides a subtle detail with ``a[i]`` actually being a pointer to a whole row, so a is a pointer to an array of pointers. Construction like this enables us to strength reduce, since we do not multiply at instruction level (see §1.3, store tykke bog).

Matrix multiplikations video:
[Matrix multiplication as composition | Chapter 4, Essence of linear algebra](https://youtu.be/XkY2DOUCWMU?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)