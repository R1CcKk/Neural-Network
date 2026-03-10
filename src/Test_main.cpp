#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/DataLoader.hpp"

int main()
{
    // Topology must match the saved model
    std::vector<int> topology = {784, 128, 10};
    NeuralNetwork nn(topology);

    std::cout << "Loading model from binary file..." << std::endl;
    nn.loadModel("mnist_model.bin");

    // Load test dataset (10,000 images)
    std::vector<DataPoint> testData = DataLoader::loadMNISTCsv("mnist_test.csv");
    if (testData.empty())
        return 1;

    std::cout << "Starting Inference Test..." << std::endl;
    int testCorrect = 0;

    for (size_t i = 0; i < testData.size(); ++i)
    {
        // Inference pass (only forward, no training)
        Matrix<double> output = nn.forward(testData[i].input);

        if (NeuralNetwork::getPrediction(output) == testData[i].label)
        {
            testCorrect++;
        }
    }

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "LOADED MODEL ACCURACY: " << (static_cast<double>(testCorrect) / testData.size()) * 100.0 << "%" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return 0;
}