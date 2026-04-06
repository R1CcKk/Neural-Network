#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "NeuralNetwork.hpp"
#include "DataLoader.hpp"

std::pair<Matrix<double>, Matrix<double>> buildBatch(const std::vector<DataPoint> &data, 
    const std::vector<int> &indices, int start, int batchSize){
        //60.000/64 = 937 batches, last batch will have 60.000 - 937*64 = 8 samples
        int actualBatch = std::min(batchSize, (int)(indices.size() - start));
        int inputSize = data[0].input.getRows(); //784 
        int outputSize = data[0].target.getRows(); //10

        // rows = features/classes, columns = number of images in batch
        Matrix<double> batchX(inputSize, actualBatch); //(784,64)
        Matrix<double> batchY(outputSize, actualBatch);//(10,64)

        for(int j = 0; j < actualBatch; ++j){
            int idx = indices[start + j];
            for(int r = 0; r < inputSize; ++r){
                batchX(r, j) = data[idx].input(r, 0);
            }
            for(int r = 0; r < outputSize; ++r){
                batchY(r, j) = data[idx].target(r, 0);
            }
        }
        return {batchX, batchY};
    }

    int main(){
        
        std::vector<int> topology = {784, 128, 10};
        NeuralNetwork nn(topology);
        
        double learningRate = 0.1;
        int epochs = 30;
        int batchSize = 64;

        std::vector<DataPoint> trainData = DataLoader::loadMNISTCsv("../mnist_train.csv");
        if(trainData.empty()) return 1;

        std::cout << "Starting training on " << trainData.size() << " images. Batch size: " << batchSize << std::endl;

        std::vector<int> indices(trainData.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(42); // Fixed seed for reproducibility

        
        for(int epoch = 0; epoch < epochs; ++epoch){
            // Shuffle indices at the start of each epoch to prevent the network from learning the order
            std::shuffle(indices.begin(), indices.end(), rng);
            int correct = 0;

            for(size_t start = 0; start < (int)trainData.size(); start += batchSize){
                auto [batchX, batchY] = buildBatch(trainData, indices, start, batchSize);

                Matrix<double> output = nn.forward(batchX);
                nn.backPropagate(batchY);
                nn.updateParameters(learningRate);

                int actualBatch = batchY.getCols();
                for(int j = 0; j < actualBatch; ++j){
                    Matrix<double> col(output.getRows(), 1);
                    for(int r = 0; r < output.getRows(); ++r){
                        col(r, 0) = output(r, j);
                    }
                int trueLabel = 0;
                for(int r = 0; r < batchY.getRows(); ++r){
                    if(batchY(r, j) > batchY(trueLabel, j)){
                        trueLabel = r;
                    }
                }              
                if(NeuralNetwork::getPrediction(col) == trueLabel){
                    correct++;
                }
                
            }
        }
        std::cout << "Epoch " << epoch + 1 << " Training Accuracy: "
                  << (static_cast<double>(correct) / trainData.size()) * 100.0
                  << "%" << std::endl;
    }
    std::vector<DataPoint> testData = DataLoader::loadMNISTCsv("mnist_train.csv");
    if (!testData.empty())
    {
        int testCorrect = 0;
        for (size_t i = 0; i < testData.size(); ++i)
        {
            Matrix<double> output = nn.forward(testData[i].input);
            if (NeuralNetwork::getPrediction(output) == testData[i].label)
                testCorrect++;
        }
        std::cout << "Final Test Accuracy: "
                  << (static_cast<double>(testCorrect) / testData.size()) * 100.0
                  << "%" << std::endl;
    }

    nn.saveModel("mnist_model.bin");
    return 0;
}