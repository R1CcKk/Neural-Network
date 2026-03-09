#ifndef HELPER_HPP
#define HELPER_HPP

#include<cmath>
#include<algorithm>
#include <iostream>
#include "Matrix.hpp"

/**
 * @brief Computes the smallest power of two greater than or equal to a given value.
 * * This utility is primarily used for padding matrices to dimensions that are powers of two,
 * which is a requirement for certain recursive algorithms like Strassen's multiplication.
 * It employs bitwise operations for efficiency: first checking if the number is already
 * a power of two using the (n & (n - 1)) idiom, then using bit-shifting to calculate
 * the next power if necessary.
 * * @param n The base size to evaluate.
 * @return The next power of two size_t value. Returns 1 if input is 0.
 */

inline size_t nextPowerOfTwo(size_t n){

    if(n == 0) return 1;
    // Check if n is already a power of two using bitwise AND
    // Example: 4 (100) & 3 (011) == 0
    if((n & (n - 1)) == 0) return n; 
    size_t power = 1;
    while(power < n){
        power <<= 1; 
    }
    return power;
}

/**
 * @brief Verifies the correctness of a matrix multiplication result.
 * * Performs a validation check by computing the product of matrices A and X
 * and comparing the result element-wise against matrix B. This is typically
 * used to verify that an optimized multiplication (like Strassen) matches
 * the expected theoretical outcome.
 * * @tparam T The numeric type of the matrix elements.
 * @param A The first factor matrix.
 * @param X The second factor matrix.
 * @param B The expected result matrix to verify against.
 * @return true if (A * X) is identical to B, false otherwise.
 */
template <typename T>
inline bool check(const Matrix<T> &A, const Matrix<T> &X, const Matrix<T> &B)
{
    Matrix<T> result = A * X;
    for (auto i = 0; i < result.getRows(); ++i)
    {
        for (auto j = 0; j < result.getCols(); ++j)
        {
            if (result(i, j) != B(i, j))
            {
                std::cout << "Verification failed: Matrices are different." << std::endl;
                return false;
            }
        }
    }
    std::cout << "Verification successful: Matrices are identical." << std::endl;
    return true;
}

/**
 * @brief Reads a vector from a structured text file.
 * * The file should start with an integer indicating the size,
 * followed by the vector elements.
 * * @tparam T The numeric type of the elements.
 * @param filename Path to the vector file.
 * @return A std::vector containing the loaded data.
 */
template <typename T>
inline std::vector<T> loadVectorFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: Could not open vector file " + filename);
    }

    int size;
    file >> size;

    std::vector<T> vec(size);
    for (int i = 0; i < size; ++i)
    {
        if (!(file >> vec[i]))
        {
            throw std::runtime_error("Error: Insufficient data in vector file " + filename);
        }
    }
    return vec;
}


#endif // HELPER_HPP