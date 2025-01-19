// SymmetricTriDiagonalMatrix.h
#ifndef SYMMETRICTRIDIAGONALMATRIX_H
#define SYMMETRICTRIDIAGONALMATRIX_H

#include "Matrix.h"
#include <stdexcept>
#include <cstdlib>

class SymmetricTriDiagonalMatrix : public Matrix {
private:
    double* diagonal;
    double* subdiagonal;

public:
    SymmetricTriDiagonalMatrix(int size);
    ~SymmetricTriDiagonalMatrix() override;

    double get(int row, int col) const override;
    void set(int row, int col, double value) override;
};

#endif // SYMMETRICTRIDIAGONALMATRIX_H
