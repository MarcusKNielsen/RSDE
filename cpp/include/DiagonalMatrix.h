// DiagonalMatrix.h
#ifndef DIAGONALMATRIX_H
#define DIAGONALMATRIX_H

#include "Matrix.h"
#include <stdexcept>
#include <cstdlib>

class DiagonalMatrix : public Matrix {
private:
    double* diagonal;

public:
    DiagonalMatrix(int size);
    ~DiagonalMatrix() override;

    double get(int row, int col) const override;
    void set(int row, int col, double value) override;
};

#endif // DIAGONALMATRIX_H
