#include <iostream>
#include <stdexcept>

// Base class for all matrix types
class Matrix {
protected:
    int rows, cols; // Dimensions of the matrix

public:
    // Constructor
    Matrix(int rows, int cols) : rows(rows), cols(cols) {}

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Matrix() = default;

    // Pure virtual functions for matrix operations
    virtual double get(int row, int col) const = 0;
    virtual void set(int row, int col, double value) = 0;

    // Accessor methods for dimensions
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // Print the matrix
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << get(i, j) << " ";
            }
            std::cout << "\n";
        }
    }    
    
};


class DenseMatrix : public Matrix {
private:
    double* data; // Pointer to the matrix data

    // Helper to calculate the index for column-major order
    int index(int row, int col) const {
        return col * rows + row;
    }

public:
    // Constructor
    DenseMatrix(int rows, int cols) : Matrix(rows, cols) {
        data = (double*)malloc(rows * cols * sizeof(double));
        if (!data) {
            throw std::runtime_error("Failed to allocate memory for DenseMatrix");
        }
        // Initialize all elements to 0.0
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = 0.0;
        }
    }

    // Destructor
    ~DenseMatrix() override {
        free(data); // Free allocated memory
    }

    // Get an element
    double get(int row, int col) const override {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[index(row, col)];
    }

    // Set an element
    void set(int row, int col, double value) override {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        data[index(row, col)] = value;
    }

};


class DiagonalMatrix : public Matrix {
private:
    double* diagonal; // Pointer to the diagonal elements

public:
    // Constructor
    DiagonalMatrix(int size) : Matrix(size, size) {
        diagonal = (double*)malloc(size * sizeof(double));
        if (!diagonal) {
            throw std::runtime_error("Failed to allocate memory for DiagonalMatrix");
        }
        // Initialize all diagonal elements to 0.0
        for (int i = 0; i < size; ++i) {
            diagonal[i] = 0.0;
        }
    }

    // Destructor
    ~DiagonalMatrix() override {
        free(diagonal); // Free allocated memory
    }

    // Get an element
    double get(int row, int col) const override {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        return (row == col) ? diagonal[row] : 0.0; // Non-diagonal elements are 0
    }

    // Set an element
    void set(int row, int col, double value) override {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        if (row != col) {
            throw std::invalid_argument("Cannot set non-diagonal elements in DiagonalMatrix");
        }
        diagonal[row] = value;
    }

};

class SymmetricTriDiagonalMatrix : public Matrix {
private:

    double* diagonal;
    double* subdiagonal;

public:

    // Constructor
    SymmetricTriDiagonalMatrix(int size) : Matrix(size, size) {

        diagonal = (double *)malloc(size*sizeof(double));
        if (!diagonal) {
            throw std::runtime_error("Failed to allocate memory for diagonal of SymmetricTriDiagonalMatrix");
        }

        subdiagonal = (double *)malloc((size-1)*sizeof(double));
        if (!subdiagonal){
            throw std::runtime_error("Failed to allocate memory for subdiagonal of SymmetricTriDiagonalMatrix");
        }
    }

    // Destructor
    ~SymmetricTriDiagonalMatrix() override {
        free(diagonal);
        free(subdiagonal);
    }

    // Get an element
    double get(int row, int col) const override {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        } else if (row == col){
            return diagonal[row];
        } else if (row == col+1){
            return subdiagonal[col];
        } else if (col == row+1){
            return subdiagonal[row];
        } else {
            return 0.0;
        }
    }

    // Set an element
    void set(int row, int col, double value) override {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        } else if (row == col){
            diagonal[row] = value;
        } else if (row == col+1){
            subdiagonal[col] = value;
        } else if (col == row+1){
            subdiagonal[row] = value;
        } else {
            throw std::invalid_argument("Cannot set non-diagonal / non-subdiagonal elements in SymmetricTriDiagonalMatrix");
        }
    }

};

int main() {

    int n = 3;

    // Dense matrix (3x3)
    DenseMatrix A(n, n);
    A.set(0, 0, 1.0);
    A.set(1, 0, 2.0);
    A.set(2, 0, 3.0);
    A.set(1, 1, 4.0);
    A.set(2, 2, 5.0);

    std::cout << "Dense Matrix:\n";
    A.print();

    // Diagonal matrix (3x3)
    DiagonalMatrix diagonal(3);
    diagonal.set(0, 0, 1.0);
    diagonal.set(1, 1, 2.0);
    diagonal.set(2, 2, 3.0);

    std::cout << "\nDiagonal Matrix:\n";
    diagonal.print();

    SymmetricTriDiagonalMatrix B(4);
    B.set(0, 0, 1.0);
    B.set(1, 1, 2.0);
    B.set(2, 2, 3.0);
    B.set(3, 3, 3.0);

    B.set(1, 0, 1.0);
    B.set(2, 1, 2.0);
    B.set(3, 2, 2.0);

    std::cout << "\nSymmetric Tri-Diagonal Matrix: \n";
    B.print();

    return 0;
}
