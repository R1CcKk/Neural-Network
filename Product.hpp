#ifndef PRODUCT_HPP
#define PRODUCT_HPP

#include <iostream>
#include "Matrix.hpp"

//prodotto classico tra matrici 
template<typename T> class Matrix;
template <typename T>

/**
 * @brief Performs standard matrix multiplication using a cache-friendly approach.
 * * This function implements the classical O(n^3) matrix multiplication algorithm.
 * It utilizes an "ikj" loop ordering instead of the traditional "ijk" to optimize
 * CPU cache performance by ensuring row-major contiguous memory access in the
 * innermost loop.
 * * @tparam T The numeric type of the matrix elements.
 * @param A The left-hand side matrix of dimensions (rowsA x colsA).
 * @param B The right-hand side matrix of dimensions (colsA x colsB).
 * @return A new Matrix object containing the product A * B.
 */

Matrix<T> matrixMultiply(const Matrix<T> &A, const Matrix<T> &B)
{

    Matrix<T> result(A.getRows(), B.getCols());
    for (int i = 0; i < A.getRows(); ++i) {
        for (int k = 0; k < A.getCols(); ++k) {
            T tmp = A(i,k);
            for (int j = 0; j < B.getCols(); ++j) {
                result(i,j) += tmp * B(k,j);
            }
        }
    }
    return result;
}

/**
 * @brief Performs matrix multiplication using Strassen's Divide and Conquer algorithm.
 * * Strassen's algorithm reduces the asymptotic complexity of matrix multiplication
 * from O(n^3) to approximately O(n^2.807). It works by recursively partitioning
 * the matrices into four sub-blocks and calculating seven specific products (M1-M7).
 * * @note This implementation uses a hybrid approach: when the matrix size falls
 * below a predefined threshold, it switches to the classical ikj multiplication
 * to avoid the overhead of recursive calls and temporary matrix allocations.
 * * @tparam T The numeric type of the matrix elements.
 * @param A The left-hand side square matrix.
 * @param B The right-hand side square matrix.
 * @return A new Matrix object containing the product A * B.
 */
template<typename T>

Matrix<T> strassenMultiply(const Matrix<T>& A, const Matrix<T>& B) {
    int n = A.getRows();

    // Base case: switch to classical multiplication for small matrices to improve performance
    int treshold = 64;
    int newSize = n / 2;
    if (n <= treshold) {
        return matrixMultiply(A, B);
    }else{

        Matrix<T> A11 = A.getSubMatrix(0, 0, newSize);
        Matrix<T> A12 = A.getSubMatrix(0, newSize, newSize);
        Matrix<T> A21 = A.getSubMatrix(newSize, 0, newSize);
        Matrix<T> A22 = A.getSubMatrix(newSize, newSize, newSize);

        Matrix<T> B11 = B.getSubMatrix(0, 0, newSize);
        Matrix<T> B12 = B.getSubMatrix(0, newSize, newSize);
        Matrix<T> B21 = B.getSubMatrix(newSize, 0, newSize);
        Matrix<T> B22 = B.getSubMatrix(newSize, newSize, newSize);

        Matrix<T> M1 = strassenMultiply(A11 + A22, B11 + B22);
        Matrix<T> M2 = strassenMultiply(A21 + A22, B11);
        Matrix<T> M3 = strassenMultiply(A11, B12 - B22);
        Matrix<T> M4 = strassenMultiply(A22, B21 - B11);
        Matrix<T> M5 = strassenMultiply(A11 + A12, B22);
        Matrix<T> M6 = strassenMultiply(A21 - A11, B11 + B12);
        Matrix<T> M7 = strassenMultiply(A12 - A22, B21 + B22);

        Matrix<T> C11 = M1 + M4 - M5 + M7;
        Matrix<T> C12 = M3 + M5;
        Matrix<T> C21 = M2 + M4;
        Matrix<T> C22 = M1 - M2 + M3 + M6;

        Matrix<T> C(n, n);
        C.setSubMatrix(0, 0, C11);
        C.setSubMatrix(0, newSize, C12);
        C.setSubMatrix(newSize, 0, C21);
        C.setSubMatrix(newSize, newSize, C22);
        return C;
    }
    
}

#endif // PRODUCT_HPP