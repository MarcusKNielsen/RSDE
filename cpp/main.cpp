// main.cpp
#include "DenseMatrix.h"
#include "DiagonalMatrix.h"
#include "SymmetricTriDiagonalMatrix.h"

int main() {

    int n = 3;

    // Dense matrix (3x3)
    DenseMatrix A(n, n);
    A.set(0, 0, 1.0);
    A.set(1, 0, 2.0);
    A.set(2, 0, 3.0);
    A.set(1, 1, 4.0);
    A.set(2, 2, 5.0);

    std::cout << "Dense Matrix:\n";
    A.print();

    // Diagonal matrix (3x3)
    DiagonalMatrix diagonal(3);
    diagonal.set(0, 0, 1.0);
    diagonal.set(1, 1, 2.0);
    diagonal.set(2, 2, 3.0);

    std::cout << "\nDiagonal Matrix:\n";
    diagonal.print();

    SymmetricTriDiagonalMatrix B(4);
    B.set(0, 0, 1.0);
    B.set(1, 1, 2.0);
    B.set(2, 2, 3.0);
    B.set(3, 3, 3.0);

    B.set(1, 0, 1.0);
    B.set(2, 1, 2.0);
    B.set(3, 2, 2.0);

    std::cout << "\nSymmetric Tri-Diagonal Matrix: \n";
    B.print();

    return 0;
}
