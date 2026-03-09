#include "NeuralNetwork.hpp"
#include "Activations.hpp"
#include "Matrix.hpp"
#include <random>

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology) : m_topology(topology)
{
    for (size_t i = 0; i < topology.size() - 1; ++i)
    {
        Matrix<double> weightMatrix(topology[i + 1], topology[i]);
        m_weights.push_back(weightMatrix);
        m_weights.back().randomize();
        m_weightGradients.push_back(Matrix<double>(topology[i + 1], topology[i]));

        Matrix<double> biasMatrix(topology[i + 1], 1);
        m_biases.push_back(biasMatrix);
        m_biasGradients.push_back(Matrix<double>(topology[i + 1], 1));
    }
}

Matrix<double> NeuralNetwork::forward(const Matrix<double> &input)
{
    m_activations.clear();
    m_zValues.clear();

    Matrix<double> activation = input;
    m_activations.push_back(activation);

    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        Matrix<double> z = m_weights[i] * activation + m_biases[i];
        m_zValues.push_back(z);

        if (i == m_weights.size() - 1)
        {
            activation = z.apply(Activation::sigmoid);
        }
        else
        {
            activation = z.apply(Activation::relu);
        }
        m_activations.push_back(activation);
    }
    return activation;
}

/**
 * @brief Backpropagation algorithm computing gradients via the Chain Rule.
 * * Propagates the error backwards from the output layer to the hidden layers.
 * The function maps the derivative of the activation function onto the pre-activation
 * values (Z) without mutating them, ensuring accurate gradient computation.
 */
void NeuralNetwork::backPropagate(const Matrix<double> &target)
{
    double m_inv = 1.0 / static_cast<double>(target.getCols());

    // 1. Output layer error (dZ)
    Matrix<double> dZ = m_activations.back() - target;

    // 2. Iterate backwards through the layers
    for (int i = m_weights.size() - 1; i >= 0; --i)
    {

        // Compute weight gradients: dZ * A_prev^T
        Matrix<double> dW = dZ * m_activations[i].transpose() * m_inv;
        m_weightGradients[i] = dW;

        // Compute bias gradients: Sum of dZ
        Matrix<double> dB = dZ.sumColumns() * m_inv;
        m_biasGradients[i] = dB;

        // 3. Propagate the error to the previous layer (if not the first layer)
        if (i > 0)
        {
            Matrix<double> derivative = m_zValues[i - 1].map(Activation::reluDerivative);
            dZ = (m_weights[i].transpose() * dZ).hadamard(derivative);
        }
    }
}

void NeuralNetwork::updateParameters(double learningRate)
{
    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        m_weights[i] = m_weights[i] - (m_weightGradients[i] * learningRate);
        m_biases[i] = m_biases[i] - (m_biasGradients[i] * learningRate);
    }
}

void NeuralNetwork::saveModel(const std::string &filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Error: Cannot open file for saving." << std::endl;
        return;
    }

    size_t topologySize = m_topology.size();
    out.write(reinterpret_cast<const char *>(&topologySize), sizeof(topologySize));
    out.write(reinterpret_cast<const char *>(m_topology.data()), topologySize * sizeof(int));

    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        m_weights[i].save(out);
        m_biases[i].save(out);
    }

    out.close();
    std::cout << "Model successfully saved to: " << filename << std::endl;
}

void NeuralNetwork::loadModel(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error: Cannot read file " << filename << std::endl;
        return;
    }

    size_t topologySize;
    in.read(reinterpret_cast<char *>(&topologySize), sizeof(topologySize));
    m_topology.resize(topologySize);
    in.read(reinterpret_cast<char *>(m_topology.data()), topologySize * sizeof(int));

    m_weights.resize(topologySize - 1);
    m_biases.resize(topologySize - 1);

    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        m_weights[i].load(in);
        m_biases[i].load(in);
    }

    in.close();
    std::cout << "Model successfully loaded from: " << filename << std::endl;
}

/**
 * @brief Returns the index of the neuron with the highest activation.
 * This corresponds to the predicted class label (0-9 for MNIST).
 */
int NeuralNetwork::getPrediction(const Matrix<double> &output)
{
    int predictedLabel = 0;
    double maxProb = output(0, 0);

    // Simple linear search for the maximum value in the output vector
    for (int j = 1; j < output.getRows(); ++j)
    {
        if (output(j, 0) > maxProb)
        {
            maxProb = output(j, 0);
            predictedLabel = j;
        }
    }
    return predictedLabel;
}