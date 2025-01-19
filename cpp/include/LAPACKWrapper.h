#ifndef LAPACKWRAPPER_H
#define LAPACKWRAPPER_H

extern "C" {
    // Declare the LAPACK function
    void dstev_(const char* JOBZ, const int* N, double* D,
                double* E, double* Z, const int* LDZ,
                double* WORK, int* INFO);
}

// Wrapper function for easier usage
void computeEigenvalues(const int N, double* D, double* E, double* Z, const int LDZ, bool computeEigenvectors);

#endif // LAPACKWRAPPER_H
