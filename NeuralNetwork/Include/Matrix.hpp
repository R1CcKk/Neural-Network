#ifndef MATRIX_HPP
#define MATRIX_HPP  

#include<vector>
#include<iostream>
#include<fstream>
#include"Product.hpp"
#include"Helper.hpp"

/**
 * @brief A template-based Matrix class providing fundamental linear algebra operations.
 * * This class manages a 2D matrix stored in a 1D std::vector to ensure contiguous
 * memory allocation and better cache locality. It supports basic arithmetic,
 * sub-matrix extraction, and high-performance multiplication via Strassen's algorithm
 * with automatic padding/unpadding.
 * * @tparam T The numeric type of the elements (e.g., int, float, double).
 */
template<typename T>
class Matrix {
    private:
        int m_rows;
        int m_cols;
        std::vector<T> m_data; 
    public:
        /**
         * @brief Constructs a new Matrix with specified dimensions initialized to zero.
         * @param rows Number of rows.
         * @param cols Number of columns.
         */
        Matrix(int rows, int cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0) {}
        Matrix() : m_rows(0), m_cols(0), m_data(0,0) {}

        Matrix(const std::vector<T> &vec, bool asColumn = true){
            int n = vec.size();
            if (asColumn){
                m_rows = n;
                m_cols = 1;
            }
            else{
                m_rows = 1;
                m_cols = n;
            }
            m_data = vec; 
        }

        int getRows() const{
            return m_rows;
        }

        int getCols() const{
            return m_cols;
        }

        /**
         * @brief Accesses the element at (row, col) for read/write operations.
         * @note Maps 2D coordinates to 1D vector index: [row * m_cols + col].
         */

        T& operator()(int row, int col) {
            return m_data[row * m_cols + col];
        }

        /**
         * @brief Accesses the element at (row, col) for read-only operations on constant matrices.
         */

        const T& operator()(int row, int col) const {
            return m_data[row * m_cols + col];
        }

        /**
         * @brief Performs element-wise addition of two matrices.
         * @throws std::invalid_argument If dimensions do not match.
         */
        Matrix<T> operator+(const Matrix<T>& other) const {
            if(m_rows != other.m_rows || m_cols != other.m_cols){
                throw std::invalid_argument("Matrix dimensions must agree for addition.");
            }
            Matrix<T> result(m_rows, m_cols);
            for(auto i = 0; i < m_rows; ++i){
                for(auto j = 0; j < m_cols; ++j){
                    result(i,j) = (*this)(i,j) + other(i,j);
                }
            }
            return result;
        }

        /**
         * @brief Performs element-wise subtraction of two matrices.
         * @throws std::invalid_argument If dimensions do not match.
         */
        Matrix<T> operator-(const Matrix<T>& other) const{
            if (m_rows != other.m_rows || m_cols != other.m_cols){
                throw std::invalid_argument("Matrix dimensions must agree for subtraction.");
            }
            Matrix<T> result(m_rows, m_cols);
            for(auto i = 0; i < m_rows; ++i){
                for(auto j = 0; j < m_cols; ++j){
                    result(i,j) = (*this)(i,j) - other(i,j);
                }
            }
            return result;
        }

        /**
         * @brief Multiplies two matrices using a hybrid approach (Classical vs Strassen).
         * * If the matrix size is below a specific threshold, it defaults to the
         * cache-optimized classical multiplication. Otherwise, it pads the matrices
         * to the nearest power of two and applies Strassen's algorithm.
         * * @param other The right-hand side matrix.
         * @return The resulting product matrix, unpadded to original dimensions.
         */
        Matrix<T> operator*(const Matrix<T>& other) const{
            if (m_cols != other.m_rows){
                throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
            }
            int treshold = 64; //soglia per passare al metodo classico
            if(m_cols * m_rows < treshold || other.m_cols * other.m_rows < treshold){
                //uso il metodo classico
                return matrixMultiply(*this, other);
            }
            int maxDim = std::max({m_rows, m_cols, other.m_rows, other.m_cols});
            int paddedSize = nextPowerOfTwo(maxDim);

            Matrix<T> APadded = matrixPadding(*this, paddedSize);
            Matrix<T> BPadded = matrixPadding(other, paddedSize);

            Matrix<T> CPadded = strassenMultiply(APadded, BPadded);

            return CPadded.matrixUnpadding(m_rows, other.m_cols);
        }
        Matrix<T> operator*(T scalar) const
        {
            Matrix<T> result(m_rows, m_cols);
            for (int i = 0; i < m_rows; ++i)
            {
                for (int j = 0; j < m_cols; ++j)
                {
                    result(i, j) = (*this)(i, j) * scalar;
                }
            }
            return result;
        }
        // algoritmo estrazione sottomatrici
        Matrix<T> getSubMatrix(int startRow, int startCol, int size) const
        {
            Matrix<T> sub(size, size);
            for (int i = 0; i < sub.getRows(); ++i)
            {
                for (int j = 0; j < sub.getCols(); ++j)
                {
                    sub(i, j) = (*this)(startRow + i, startCol + j);
                }
            }
            return sub;
        }

        // algoritmo combinazione sottomatrici
        void setSubMatrix(int startRow, int startCol, const Matrix<T> &sub)
        {
            for (int i = 0; i < sub.getRows(); ++i)
            {
                for (int j = 0; j < sub.getCols(); ++j)
                {
                    (*this)(startRow + i, startCol + j) = sub(i, j);
                }
            }
        }

        /**
         * @brief Pads the matrix with zeros to a new square dimension.
         * * This ensures that matrices meet the size requirements (powers of two)
         * for recursive partitioning algorithms.
         */
        Matrix<T> matrixPadding(const Matrix<T>& A, int newSize) const{
            // int max_dim = std::max(A.getRows(), A.getCols());
            // int new_size = nextPowerOfTwo(max_dim);

            if(newSize == A.getRows() && newSize == A.getCols()){
                return A; // già potenza di 2
            }

            Matrix<T> padded(newSize, newSize);
            for(auto i = 0; i < newSize; ++i){
                for(auto j = 0; j < newSize; ++j){
                    if(i < A.getRows() && j < A.getCols()){
                        padded(i,j) = A(i,j);
                    } else {
                        padded(i,j) = 0; // padding con zeri
                    }
                }
            }
            return padded;
        }
        
        Matrix<T> matrixUnpadding(int rows, int cols){
            Matrix<T> result(rows, cols);
            for(auto i = 0; i < rows; ++i){
                for(auto j = 0; j < cols; ++j){
                    result(i,j) = (*this)(i,j);
                }
            }
            
            return result;
        }

        Matrix<T> transpose() const{
            Matrix<T> transposed(m_cols, m_rows);
            for(int i = 0; i < m_rows; ++i){
                for(int j = 0; j < m_cols; ++j){
                    transposed(j,i) = (*this)(i,j);
                }
            }
            return transposed;
        }

        /**
         * @brief Factory method to create a Matrix from a structured text file.
         * * Expected file format:
         * [rows] [cols]
         * [v1,1] [v1,2] ...
         * @throws std::runtime_error If file cannot be opened or data is missing.
         */
        static Matrix<T> fromFile(const std::string& filename){
            std::ifstream file(filename);

            if(!file.is_open()){
                throw std::runtime_error("Error: Could not open file " + filename);
            }

            int rows, cols;
            file >> rows >> cols; 
            
            Matrix<T> res(rows,cols);
            for(int i = 0 ; i < rows; ++i){
                for(int j = 0; j < cols; ++j){
                    if(!(file >> res(i,j))){
                        throw std::runtime_error("Error: Insufficient data in file " + filename);
                    }
                }
            }
            return res;
        }

        void toFile(const std::string& filename) const{
            std::ofstream file(filename);
            if(!file.is_open()){
                throw std::runtime_error("Error: Could not create output file.");
            }

            file << std::fixed << std::setprecision(2);

            for(int i = 0; i < m_rows; i++){
                for(int j = 0; j < m_cols; j++){
                    file << (*this)(i,j) << " "; //prende quello a destra di << e lo mette nello stream a sinistra
                }
                file << "\n";
            }
            file.close();
        }

        void printMatrix() {
            for(int i = 0; i < m_rows; ++i){
                for(int j = 0; j < m_cols; ++j){
                    std::cout << (*this)(i,j) << " ";
                }
                std::cout << std::endl;
            }
        }
        Matrix<T>& apply(double (*func)(double)){
            for (int i = 0; i < m_rows * m_cols; ++i)
            {
                m_data[i] = func(m_data[i]);
            }
            return *this;
        }

        Matrix<T> hadamard(const Matrix<T> &other) const
        {
            if (m_rows != other.m_rows || m_cols != other.m_cols)
                throw std::invalid_argument("Dimensions must match for Hadamard product");
            Matrix<T> result(m_rows, m_cols);
            for (int i = 0; i < m_data.size(); ++i)
                result.m_data[i] = m_data[i] * other.m_data[i];
            return result;
        }
};

#endif // MATRIX_HPP