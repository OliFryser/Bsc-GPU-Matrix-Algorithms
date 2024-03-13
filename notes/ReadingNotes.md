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

[Karl Rupp papers](https://www.karlrupp.net/publications/journal-articles/)

## QR-Decomposition

Page 98 and 99 in the book Numerical Recipes in C

Householder method page 470- in the same book

Decompose en matrice til en orthogonal matrice og en upper triangular matrice.

En orthogonal matrice er defineret ved at dens transpose er identitetsmatricen. Det vil også sige at dens transpose også er dens inverse. Den fortæller om rotation og reflection.

En upper triangular matrix er en matrice hvor alle elementer under diagonalen er 0. Den fortæller om sheering og scaling.

QR decomposition består af en række Householder Transformations på hinanden følgende.

Bogens QR-decomposition har signaturen:

    void qrdcmp(float **a, int n, float *c, float *d, int *sing)

- n er dimensionen på matricen
- a vil blive lavet om til den upper triangular matrix, dog uden diagonalen
- d bliver alle elementerne i diagonalen
- sing er en boolean (1 eller 0) som fortæller om matricen (transformationen) er singular. Singular betyder at matricen går fra n dimensioner til mindre end n-dimensioner. Dvs. den ikke kan have en inverse. Selvom det viser sig at matricen er singular vil QR-decompositionen stadig blive færdigudregnet.

Algoritmen looper over matricens n rækker.
For hver række starter den med at udregne om den Qi'te orthogonale matrice er singular, for hvilket vil betyde at hele matricen vil blive singular.

    scale=0.0;
    for (i=k;i<=n;i++) 
        scale=FMAX(scale,fabs([i][k]));
    if (scale == 0.0) {
        *sing=1;
        c[k]=d[k]=0.0;
    }

Hvis ikke den er singular, vil Qk og Qk * A blive udregnet

### Symetric matrix

En symetrisk matrice er en matrice hvor hvert element A(i,j) = A(j,i). Dvs. at dens transpose er den selv. Dette kræver at matricen er kvadratisk.

### Tridiagonal matrix

En matrice vis diagonal, upper diagonal og lower diagonal vis entries er de eneste entries i matricen som er non-zero.
En upper diagonal er diagonalen lige over main diagonalen. En lower diagonal er diagonalen lige under main diagonalen.

### Householder algorithm

Householder alogitmen reducerer en symmetrisk matrice til en tridiagonal matrice ved at lave n-2 othogonale transformationer (rotation + reflection) hvor n = dimensionsstørrelsen. 


[Very simple explanation of QR-decomposition using the Gram-Schmidt algorithm](https://www.codingdrills.com/tutorial/matrix-data-structure/qr-decomposition)