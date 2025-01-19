// SymmetricTriDiagonalMatrix.cpp
#include "SymmetricTriDiagonalMatrix.h"

SymmetricTriDiagonalMatrix::SymmetricTriDiagonalMatrix(int size) : Matrix(size, size) {
    diagonal = (double*)malloc(size * sizeof(double));
    if (!diagonal) throw std::runtime_error("Failed to allocate memory for diagonal of SymmetricTriDiagonalMatrix");

    subdiagonal = (double*)malloc((size - 1) * sizeof(double));
    if (!subdiagonal) throw std::runtime_error("Failed to allocate memory for subdiagonal of SymmetricTriDiagonalMatrix");
}

SymmetricTriDiagonalMatrix::~SymmetricTriDiagonalMatrix() {
    free(diagonal);
    free(subdiagonal);
}

double SymmetricTriDiagonalMatrix::get(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) throw std::out_of_range("Index out of bounds");
    if (row == col) return diagonal[row];
    if (row == col + 1) return subdiagonal[col];
    if (col == row + 1) return subdiagonal[row];
    return 0.0;
}

void SymmetricTriDiagonalMatrix::set(int row, int col, double value) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) throw std::out_of_range("Index out of bounds");
    if (row == col) diagonal[row] = value;
    else if (row == col + 1) subdiagonal[col] = value;
    else if (col == row + 1) subdiagonal[row] = value;
    else throw std::invalid_argument("Cannot set non-diagonal / non-subdiagonal elements in SymmetricTriDiagonalMatrix");
}
