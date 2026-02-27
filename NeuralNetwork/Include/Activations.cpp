#include"NeuralNetwork.hpp"
#include"Activations.hpp"
#include"Matrix.hpp"
#include<random>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology) : m_topology(topology){
    for(size_t i = 0; i < topology.size() - 1; ++i){
        Matrix<double> weightMatrix(topology[i+1], topology[i]);
        m_weights.push_back(weightMatrix);

        Matrix<double> biasMatrix(topology[i+1], 1);
        m_biases.push_back(biasMatrix);
    }
}

Matrix<double> NeuralNetwork::forward(const Matrix<double>& input){
    m_activations.clear();
    m_zValues.clear();

    Matrix<double> activation = input;
    m_activations.push_back(activation); // memorizza l'input come prima attivazione

    for(size_t i = 0; i < m_weights.size(); ++i){
        Matrix<double> z = m_weights[i] * activation + m_biases[i];
        m_zValues.push_back(z); // memorizza il valore z (pre-activation)

        if(i == m_weights.size() - 1){
            activation = z.apply(Activation::sigmoid); // output layer con sigmoid
        } else {
            activation = z.apply(Activation::relu); // hidden layers con ReLU
        }
        m_activations.push_back(activation); // memorizza l'attivazione del layer corrente
    }
    return activation;
}

void NeuralNetwork::backPropagate(const Matrix<double>& target){

    Matrix<double> dZ2 = m_activations.back() - target; // errore dell'output layer
    Matrix<double> dW2 = dZ2 * m_activations[m_activations.size() - 2].transpose(); // gradiente pesi output layer
    Matrix<double> dB2 = dZ2; // gradiente bias output layer

    Matrix<double> dZ1 = (m_weights.back().transpose() * dZ2).hadamard(m_zValues[m_zValues.size() - 2].apply(Activation::reluDerivative)); // errore hidden layer
    Matrix<double> dW1 = dZ1 * m_activations[0].transpose(); // gradiente pesi hidden layer
    Matrix<double> dB1 = dZ1; // gradiente bias hidden layer
}