// DenseMatrix.cpp
#include "DenseMatrix.h"

DenseMatrix::DenseMatrix(int rows, int cols) : Matrix(rows, cols) {
    data = (double*)malloc(rows * cols * sizeof(double));
    if (!data) throw std::runtime_error("Failed to allocate memory for DenseMatrix");
    for (int i = 0; i < rows * cols; ++i) data[i] = 0.0;
}

DenseMatrix::~DenseMatrix() {
    free(data);
}

double DenseMatrix::get(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) throw std::out_of_range("Index out of bounds");
    return data[index(row, col)];
}

void DenseMatrix::set(int row, int col, double value) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) throw std::out_of_range("Index out of bounds");
    data[index(row, col)] = value;
}

int DenseMatrix::index(int row, int col) const {
    return col * rows + row;
}

