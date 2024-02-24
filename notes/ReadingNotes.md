# Reading Notes

## Matrices

A set of linear algebraic equations looks can be written in matrix form as:
`A * x = b`
Where A is the matrix of coefficients, so a_11 is at column 1 row 1. b is the column vector. We want to solve for x. The \* is matrix multiplication.
God forklaring af Linear Algebraic Equations:
[Inverse matrices, column space and null space | Chapter 7, Essence of linear algebra](https://youtu.be/uQhTuRlWMxw?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

Hvis determinanten af `A != 0`, så vil der eksistere en invers til matrisen A. Hvis determinanten af `A = 0`, eksisterer der ikke en invers. (ville det give mening først at udregne determinanten til A, før vi prøver at beregne en invers?). Bemærk, linearly dependent implies `det(A) = 0`.

In C we can access a matrix like `a[i][j]`, where i is row and j is column. This hides a subtle detail with `a[i]` actually being a pointer to a whole row, so a is a pointer to an array of pointers. Construction like this enables us to strength reduce, since we do not multiply at instruction level (see §1.3, store tykke bog).

Matrix multiplikations video:
[Matrix multiplication as composition | Chapter 4, Essence of linear algebra](https://youtu.be/XkY2DOUCWMU?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

[Rigtig god visuel forklaring på QR-decomposition af chat-gpt](https://chat.openai.com/share/c3d4a5bb-2d47-453d-83ab-be67e55bf617)

## Cuda C

[Easy introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)

[Deep copy](https://forums.developer.nvidia.com/t/clean-way-of-copying-a-struct-with-pointers-to-the-gpu/225833/2)

[The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)
