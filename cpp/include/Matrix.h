#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>

class Matrix {
protected:
    int rows, cols; // Dimensions of the matrix

public:
    // Constructor
    Matrix(int rows, int cols);

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Matrix() = default;

    // Pure virtual functions for matrix operations
    virtual double get(int row, int col) const = 0;
    virtual void set(int row, int col, double value) = 0;

    // Accessor methods for dimensions
    int getRows() const;
    int getCols() const;

    // Print the matrix
    void print() const;
};

#endif // MATRIX_H
