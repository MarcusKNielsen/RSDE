// main.cpp
#include "DenseMatrix.h"
#include "DiagonalMatrix.h"
#include "SymmetricTriDiagonalMatrix.h"
#include "LAPACKWrapper.h"

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


    const int N = 4;                    // Size of the matrix
    double D[N] = {4.0, 3.0, 2.0, 1.0}; // Diagonal elements
    double E[N - 1] = {0.1, 0.2, 0.3};  // Subdiagonal elements
    double Z[N * N] = {0};              // Space for eigenvectors
    const int LDZ = N;                  // Leading dimension

    try {
        // Compute eigenvalues (and optionally eigenvectors)
        computeEigenvalues(N, D, E, Z, LDZ, true);

        std::cout << "Eigenvalues:\n";
        for (int i = 0; i < N; ++i) {
            std::cout << D[i] << " ";
        }
        std::cout << "\n";

        std::cout << "Eigenvectors (stored column-wise):\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << Z[i + j * N] << " ";
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }


    return 0;
}
