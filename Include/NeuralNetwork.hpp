#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include"Matrix.hpp"
#include"Activations.hpp"
#include<vector>

class NeuralNetwork{
    private:
        std::vector<int> m_topology; //{784,10,10}

        std::vector<Matrix<double>> m_weights; // Weight matrices for each layer transition.
        std::vector<Matrix<double>> m_biases; // Bias vectors for each layer transition.

        std::vector<Matrix<double>> m_activations; // Activations for each layer (including input layer).
        std::vector<Matrix<double>> m_zValues; // Pre-activation values (z) for each layer, used in backpropagation.

        std::vector<Matrix<double>> m_weightGradients; // Gradients for weights, computed during backpropagation.
        std::vector<Matrix<double>> m_biasGradients; // Gradients for biases, computed during backpropagation.
    public:
        NeuralNetwork(const std::vector<int>& topology);

        Matrix<double> forward(const Matrix<double>& input);

        void backPropagate(const Matrix<double>&target);

        void updateParameters(double learningRate);

        void saveModel(const std::string &filename) const;

        void loadModel(const std::string &filename);

        static int getPrediction(const Matrix<double> &output);
};


#endif // NEURALNETWORK_HPP
