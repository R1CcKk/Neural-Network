#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "Matrix.hpp"
#include <cmath>

namespace Activation
{

    // --- ReLU (Rectified Linear Unit) ---
    // Standard activation for hidden layers.
    inline double relu(double x)
    {
        return (x > 0) ? x : 0.0;
    }

    inline double reluDerivative(double x)
    {
        return (x > 0) ? 1.0 : 0.0;
    }

    // --- Sigmoid ---
    inline double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoidDerivative(double x)
    {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    // --- Softmax ---
    // template <typename T>
    inline void softmax(Matrix<double> &m)
    {
        // Find the maximum value to ensure numerical stability.
        // Subtracting maxVal prevents std::exp() from generating overflow (infinity)
        // for large input values, without altering the final probability distribution.
        // double maxVal = m(0, 0);
        // for (int i = 1; i < m.getRows(); ++i)
        // {
        //     if (m(i, 0) > maxVal)
        //     {
        //         maxVal = m(i, 0);
        //     }
        // }

        // double sumExp = 0.0;
        // for (int i = 0; i < m.getRows(); ++i)
        // {
        //     m(i, 0) = std::exp(m(i, 0) - maxVal);
        //     sumExp += m(i, 0);
        // }

        // for (int i = 0; i < m.getRows(); ++i)
        // {
        //     m(i, 0) /= sumExp;
        // }
        for(int j = 0; j < m.getCols(); ++j){
            double maxVal = m(0,j);
            for(int i = 1; i < m.getRows(); ++i)
            {
                if (m(i, j) > maxVal)
                {
                    maxVal = m(i, j);
                }
            }
            double sumExp = 0.0;
            for(int i = 0; i < m.getRows(); ++i)
            {
                m(i, j) = std::exp(m(i, j) - maxVal);
                sumExp += m(i, j);
            }
            for(int i = 0; i < m.getRows(); ++i)
            {
                m(i, j) /= sumExp;
            }
        }
    }
}

#endif // ACTIVATIONS_HPP
