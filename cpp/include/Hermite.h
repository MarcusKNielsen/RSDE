#ifndef HERMITE_H
#define HERMITE_H

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <numbers>

extern "C" {
    void dstev_(const char* JOBZ, const int* N, double* D, double* E, 
                double* Z, const int* LDZ, double* WORK, int* INFO);
}

double* Nodes(int N);
double* Vander(double* x, int N, int M);
double* VanderDiff(double* x, double* V, int N, int M);

#endif