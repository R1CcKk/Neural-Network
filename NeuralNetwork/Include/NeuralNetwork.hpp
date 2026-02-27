#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include"Matrix.hpp"
#include"Activations.hpp"
#include<vector>

class NeuralNetwork{
    private:
        std::vector<int> m_topology; //{784,10,10}

        std::vector<Matrix<double>> m_weights;
        std::vector<Matrix<double>> m_biases;

        std::vector<Matrix<double>> m_activations; // memorizza le attivazioni di ogni layer durante il forward pass
        std::vector<Matrix<double>> m_zValues; // memorizza i valori z (pre-activation) di ogni layer durante il forward pass
    
    public:
        NeuralNetwork(const std::vector<int>& topology);

        Matrix<double> forward(const Matrix<double>& input);

        void backPropagate(const Matrix<double>&target);

        void updateParameters(double learningRate);
};


#endif // NEURALNETWORK_HPP