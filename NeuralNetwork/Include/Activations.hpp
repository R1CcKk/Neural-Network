#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include"Matrix.hpp"
#include<iostream>
#include<cmath>

namespace Activation{

    // --- ReLU ---
    inline double relu(double x)
    {
        return (x > 0) ? x : 0;
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

    // --- Tanh ---
    inline double tanhFunc(double x)
    { 
        return std::tanh(x);
    }

    inline double tanhDerivative(double x)
    {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }

    template <typename T>
    inline void softmax(Matrix<double>& m){
        //maxval per evitare NaN o overflow quando si calcola exp(m(i,0)). 
        //Ad esempio eˆ1000 sarebbe troppo grande per essere rappresentato come double, ma eˆ(1000 - 1000) = eˆ0 = 1 è gestibile.
        //Questo non cambia il risultato finale della softmax, ma garantisce stabilità numerica.   
        double maxVal = m(0,0);
        for(int i = 1; i < m.getRows(); ++i){
            if(m(i,0) > maxVal){
                maxVal = m(i,0);
            }
        }

        double sumExp = 0.0;
        for(int i = 0; i < m.getRows(); ++i){
            m(i,0) = std::exp(m(i,0) - maxVal); // per stabilità numerica
            sumExp += m(i,0);
        }

        for(int i = 0; i < m.getRows(); ++i){
            m(i,0) /= sumExp;
        }
    };
}

#endif // ACTIVATIONS_HPP