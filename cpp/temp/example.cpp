#include <iostream>

extern "C" {
    // Fortran signature of DGEMM
    void dgemm_(const char* TRANSA, const char* TRANSB,
                const int* M, const int* N, const int* K,
                const double* ALPHA, const double* A, const int* LDA,
                const double* B, const int* LDB,
                const double* BETA, double* C, const int* LDC);
}

extern "C" {
    // LAPACK FORTRAN SUBROUTINE: dstev
    // d  = double precision
    // st = symmetric tridiagnal matrix
    // ev = eigenvalue problem solver
    // https://www.netlib.org/lapack/explore-html-3.6.1/d7/d48/dstev_8f_aaa6df51cfd92c4ab08d41a54bf05c3ab.html
    void dstev_(const char* JOBZ, const int* N, double* D,
                double* E, double* Z, const int* LDZ,
                double* WORK, int* INOF);
}

int main() {
    // Matrix dimensions
    const int m = 2, n = 2, k = 2;
    const double alpha = 1.0, beta = 0.0;

    // Matrices
    double A[] = {1.0, 2.0, 3.0, 4.0}; // 2x2 matrix
    double B[] = {5.0, 6.0, 7.0, 8.0}; // 2x2 matrix
    double C[] = {0.0, 0.0, 0.0, 0.0}; // 2x2 result matrix

    // Call dgemm
    dgemm_("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);

    // Print the result matrix C
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i + j * m] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
