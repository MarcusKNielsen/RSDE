#include "Matrix.h"

// Constructor
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {}

// Accessor for rows
int Matrix::getRows() const {
    return rows;
}

// Accessor for cols
int Matrix::getCols() const {
    return cols;
}

// Print function
void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << get(i, j) << " ";
        }
        std::cout << "\n";
    }
}
