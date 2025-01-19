// DenseMatrix.h
#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include "Matrix.h"
#include <stdexcept>
#include <cstdlib>

class DenseMatrix : public Matrix {
private:
    double* data;
    int index(int row, int col) const;

public:
    DenseMatrix(int rows, int cols);
    ~DenseMatrix() override;

    double get(int row, int col) const override;
    void set(int row, int col, double value) override;
};

#endif // DENSEMATRIX_H
