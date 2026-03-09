#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <cmath>

/**
 * @class Matrix
 * @brief Matrix class optimized for CPU cache locality.
 * * Uses a single 1D std::vector instead of a 2D vector (std::vector<std::vector<T>>).
 * This ensures that all data resides in a contiguous memory block, maximizing
 * L1/L2 cache locality during computationally heavy operations like multiplication.
 */
template <typename T>
class Matrix
{
private:
    int m_rows;
    int m_cols;
    std::vector<T> m_data;

public:
    Matrix(int rows, int cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0) {}
    Matrix() : m_rows(0), m_cols(0), m_data(0, 0) {}

    /**
     * @brief Constructs a Matrix from a standard vector.
     */
    Matrix(const std::vector<T> &vec, bool asColumn = true)
    {
        int n = vec.size();
        if (asColumn)
        {
            m_rows = n;
            m_cols = 1;
        }
        else
        {
            m_rows = 1;
            m_cols = n;
        }
        m_data = vec;
    }

    int getRows() const { return m_rows; }
    int getCols() const { return m_cols; }

    /**
     * @brief Overloads operator() to map 2D coordinates to the 1D array.
     */
    T &operator()(int row, int col)
    {
        return m_data[row * m_cols + col];
    }

    const T &operator()(int row, int col) const
    {
        return m_data[row * m_cols + col];
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Dimension mismatch for addition.");

        Matrix<T> result(m_rows, m_cols);
        for (int i = 0; i < m_rows * m_cols; ++i)
        {
            result.m_data[i] = m_data[i] + other.m_data[i];
        }
        return result;
    }

    Matrix<T> operator-(const Matrix<T> &other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Dimension mismatch for subtraction.");

        Matrix<T> result(m_rows, m_cols);
        for (int i = 0; i < m_rows * m_cols; ++i)
        {
            result.m_data[i] = m_data[i] - other.m_data[i];
        }
        return result;
    }

    /**
     * @brief Cache-friendly O(n^3) matrix multiplication.
     * * Implements an 'ikj' loop order rather than the standard 'ijk'.
     * By keeping 'k' in the outer loop and 'j' in the inner loop,
     * the matrix B is accessed sequentially in memory, significantly
     * reducing cache misses.
     */
    Matrix<T> operator*(const Matrix<T> &other) const
    {
        if (m_cols != other.m_rows)
            throw std::invalid_argument("Dimension mismatch for multiplication.");

        Matrix<T> result(m_rows, other.m_cols);
        for (int i = 0; i < m_rows; ++i)
        {
            for (int k = 0; k < m_cols; ++k)
            {
                T tmp = (*this)(i, k);
                for (int j = 0; j < other.m_cols; ++j)
                {
                    result(i, j) += tmp * other(k, j);
                }
            }
        }
        return result;
    }

    Matrix<T> operator*(T scalar) const
    {
        Matrix<T> result(m_rows, m_cols);
        for (int i = 0; i < m_rows * m_cols; ++i)
        {
            result.m_data[i] = m_data[i] * scalar;
        }
        return result;
    }

    Matrix<T> transpose() const
    {
        Matrix<T> transposed(m_cols, m_rows);
        for (int i = 0; i < m_rows; ++i)
        {
            for (int j = 0; j < m_cols; ++j)
            {
                transposed(j, i) = (*this)(i, j);
            }
        }
        return transposed;
    }

    /**
     * @brief Modifies the current matrix by applying a given function element-wise.
     */
    Matrix<T> &apply(double (*func)(double))
    {
        for (int i = 0; i < m_rows * m_cols; ++i)
        {
            m_data[i] = func(m_data[i]);
        }
        return *this;
    }

    /**
     * @brief Returns a new matrix after applying a function element-wise.
     * Required during backpropagation to prevent overwriting the pre-activation Z values.
     */
    Matrix<T> map(double (*func)(double)) const
    {
        Matrix<T> result(m_rows, m_cols);
        for (int i = 0; i < m_rows * m_cols; ++i)
        {
            result.m_data[i] = func(m_data[i]);
        }
        return result;
    }

    /**
     * @brief Element-wise (Hadamard) product.
     */
    Matrix<T> hadamard(const Matrix<T> &other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Dimensions must match for Hadamard product.");
        Matrix<T> result(m_rows, m_cols);
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            result.m_data[i] = m_data[i] * other.m_data[i];
        }
        return result;
    }

    /**
     * @brief Sums matrix columns into a single column vector.
     * Primarily used to compute the bias gradient during backpropagation.
     */
    Matrix<T> sumColumns() const
    {
        Matrix<T> result(m_rows, 1);
        for (int i = 0; i < m_rows; ++i)
        {
            T sum = 0;
            for (int j = 0; j < m_cols; ++j)
            {
                sum += (*this)(i, j);
            }
            result(i, 0) = sum;
        }
        return result;
    }

    /**
     * @brief Xavier Initialization.
     * Scales the initial random weights based on the matrix dimensions
     * to prevent initial gradients from vanishing or exploding.
     */
    void randomize()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (m_rows + m_cols));
        std::uniform_real_distribution<> dis(-limit, limit);

        for (int i = 0; i < m_rows * m_cols; ++i)
        {
            m_data[i] = dis(gen);
        }
    }

    /**
     * @brief Binary serialization method.
     * Writes the contiguous memory block directly to disk to minimize I/O overhead.
     */
    void save(std::ofstream &out) const
    {
        out.write(reinterpret_cast<const char *>(&m_rows), sizeof(m_rows));
        out.write(reinterpret_cast<const char *>(&m_cols), sizeof(m_cols));
        out.write(reinterpret_cast<const char *>(m_data.data()), m_data.size() * sizeof(T));
    }

    /**
     * @brief Binary deserialization method.
     */
    void load(std::ifstream &in)
    {
        in.read(reinterpret_cast<char *>(&m_rows), sizeof(m_rows));
        in.read(reinterpret_cast<char *>(&m_cols), sizeof(m_cols));
        m_data.resize(m_rows * m_cols);
        in.read(reinterpret_cast<char *>(m_data.data()), m_data.size() * sizeof(T));
    }
};

#endif // MATRIX_HPP