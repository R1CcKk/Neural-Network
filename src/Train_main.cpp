#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"
#include "DataLoader.hpp" 

int main()
{
    // 784 inputs (28x28), 128 hidden neurons, 10 outputs (0-9)
    std::vector<int> topology = {784, 128, 10};
    NeuralNetwork nn(topology);

    double learningRate = 0.01;
    int epochs = 10;

    // Load data using the static method from DataLoader class
    std::vector<DataPoint> trainData = DataLoader::loadMNISTCsv("mnist_train.csv");
    if (trainData.empty())
        return 1;

    std::cout << "Starting training on " << trainData.size() << " images..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        int correct = 0;
        for (size_t i = 0; i < trainData.size(); ++i)
        {
            // Forward pass
            Matrix<double> output = nn.forward(trainData[i].input);

            // Backprop and SGD update
            nn.backPropagate(trainData[i].target);
            nn.updateParameters(learningRate);

            // Compare prediction with ground truth label
            if (NeuralNetwork::getPrediction(output) == trainData[i].label)
            {
                correct++;
            }
        }
        std::cout << "Epoch " << epoch + 1 << " Training Accuracy: "
                  << (static_cast<double>(correct) / trainData.size()) * 100.0 << "%" << std::endl;
    }

    // Final check on test set before saving
    std::vector<DataPoint> testData = DataLoader::loadMNISTCsv("mnist_test.csv");
    if (!testData.empty())
    {
        int testCorrect = 0;
        for (size_t i = 0; i < testData.size(); ++i)
        {
            Matrix<double> output = nn.forward(testData[i].input);
            if (NeuralNetwork::getPrediction(output) == testData[i].label)
                testCorrect++;
        }
        std::cout << "Final Test Accuracy: " << (static_cast<double>(testCorrect) / testData.size()) * 100.0 << "%" << std::endl;
    }

    std::cout << "Saving model to binary file..." << std::endl;
    nn.saveModel("mnist_model.bin");

    return 0;
}