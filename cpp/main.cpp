// main.cpp
#include "DenseMatrix.h"
#include "DiagonalMatrix.h"
#include "SymmetricTriDiagonalMatrix.h"
#include "LAPACKWrapper.h"
#include "Hermite.h"
#include <iomanip>

void printMatrix(double* matrix, int rows, int cols, int precision = 6) {
    int width = precision + 3;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(precision)
                      << std::setw(width) << matrix[i + rows * j] << " ";
        }
        std::cout << "\n";
    }
}

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

    std::cout << "Hermite Nodes First Print" << std::endl;
    int N = 5;
    double* x = Nodes(N);
    for (int i=0; i < N; i++){
        std::cout << x[i] << std::endl;
    }

    // Vandermonde
    // Inputs: nodes x, number of nodes N, number of basis functions M
    int M = 5;

    double* V  = Vander(x,N,M);
    double* Vx = VanderDiff(x,V,N,M);
    std::cout << "Matrix: V" << std::endl;
    printMatrix(V,N,M);
    std::cout << "Matrix: Vx" << std::endl;
    printMatrix(Vx,N,M);

    free(x);
    free(V);
    free(Vx);

    return 0;
}
