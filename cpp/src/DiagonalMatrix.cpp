// DiagonalMatrix.cpp
#include "DiagonalMatrix.h"

DiagonalMatrix::DiagonalMatrix(int size) : Matrix(size, size) {
    diagonal = (double*)malloc(size * sizeof(double));
    if (!diagonal) throw std::runtime_error("Failed to allocate memory for DiagonalMatrix");
}

DiagonalMatrix::~DiagonalMatrix() {
    free(diagonal);
}

double DiagonalMatrix::get(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) throw std::out_of_range("Index out of bounds");
    return (row == col) ? diagonal[row] : 0.0;
}

void DiagonalMatrix::set(int row, int col, double value) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) throw std::out_of_range("Index out of bounds");
    if (row != col) throw std::invalid_argument("Cannot set non-diagonal elements in DiagonalMatrix");
    diagonal[row] = value;
}
