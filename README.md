# MNIST Multilayer Perceptron from Scratch (C++)

This project implements a Feedforward Neural Network (Multilayer Perceptron) to classify handwritten digits from the MNIST dataset. The entire engine is written in pure C++ without any external machine learning libraries.

## Technical Highlights
- **Cache-Friendly Linear Algebra**: Used a custom Matrix class with 1D array storage to ensure contiguous memory access. Matrix multiplication is optimized using an `ikj` loop order to minimize CPU cache misses.
- **Custom Binary Serialization**: Implemented a raw binary format for saving and loading the model, bypassing the overhead of CSV/text parsing.
- **Numerical Stability**: Included a Softmax implementation with a max-subtraction trick to prevent floating-point overflow.
- **Modular Design**: Separated data loading, mathematical operations, and network logic into distinct modules.

## Model Architecture
- **Input Layer**: 784 neurons (28x28 pixels).
- **Hidden Layer**: 128 neurons with **ReLU** activation.
- **Output Layer**: 10 neurons with **Softmax** activation (representing digits 0-9).
- **Optimization**: Stochastic Gradient Descent (SGD) with Xavier Initialization.



## Performance
Trained on 60,000 images for 5 epochs:
- **Final Training Accuracy**: 98.88%
- **Final Test Accuracy (Inference)**: 97.65%

## Project Structure
- `Matrix.hpp`: Core linear algebra operations.
- `NeuralNetwork.hpp/cpp`: Forward pass, backpropagation, and model I/O.
- `DataLoader.hpp`: Handles MNIST CSV parsing and data normalization.
- `Activations.hpp`: ReLU, Sigmoid, and Softmax functions.

## How to build and run
Compile using any C++11 compliant compiler with `-O3` optimization for best performance:

```bash
g++ -O3 Train_main.cpp NeuralNetwork.cpp -o train_mnist
./train_mnist
