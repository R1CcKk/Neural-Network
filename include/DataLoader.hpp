#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "Matrix.hpp"

/**
 * @struct DataPoint
 * @brief Represents a single MNIST observation.
 */
struct DataPoint
{
    Matrix<double> input;
    Matrix<double> target;
    int label;
};

/**
 * @class DataLoader
 * @brief Handles data ingestion and normalization.
 */
class DataLoader
{
public:
    static std::vector<DataPoint> loadMNISTCsv(const std::string &filename)
    {
        std::vector<DataPoint> dataset;
        std::ifstream file(filename);
        if (!file.is_open())
            return dataset;

        std::string line, header;
        std::getline(file, header);

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;
            std::vector<double> inputVec(784);
            std::vector<double> targetVec(10, 0.0);
            int label = -1;

            if (std::getline(ss, value, ','))
            {
                label = std::stoi(value);
                targetVec[label] = 1.0;
            }

            int pixelIndex = 0;
            while (std::getline(ss, value, ','))
            {
                if (pixelIndex < 784)
                {
                    inputVec[pixelIndex] = std::stod(value) / 255.0; // Normalization
                    pixelIndex++;
                }
            }
            dataset.push_back({Matrix<double>(inputVec, true), Matrix<double>(targetVec, true), label});
        }
        return dataset;
    }
};

#endif