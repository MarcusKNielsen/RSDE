#include "LAPACKWrapper.h"
#include <iostream>
#include <stdexcept>

void computeEigenvalues(const int N, double* D, double* E, double* Z, const int LDZ, bool computeEigenvectors) {
    char JOBZ = computeEigenvectors ? 'V' : 'N'; // 'V' for eigenvectors, 'N' for eigenvalues only
    double* WORK = new double[2 * N];            // Workspace array
    int INFO = 0;                                // Output status

    // Call the LAPACK Fortran subroutine
    dstev_(&JOBZ, &N, D, E, Z, &LDZ, WORK, &INFO);

    delete[] WORK; // Clean up workspace

    if (INFO != 0) {
        throw std::runtime_error("LAPACK dstev_ failed with INFO = " + std::to_string(INFO));
    }
}
