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

## Performance & Optimization Notes

This implementation focuses on CPU efficiency and memory hierarchy awareness.

### 1. Cache Locality & Memory Layout
Unlike a naive implementation using nested vectors (`std::vector<std::vector<double>>`), this project uses a **1.5D approach**:
* All matrices are stored as a **single contiguous block of memory** (1D `std::vector`).
* This layout ensures that when the CPU fetches a value, the subsequent values are likely already in the **L1/L2 Cache**, drastically reducing **cache misses**.



### 2. Matrix Multiplication Optimization
The core GEMM (General Matrix Multiply) operations are implemented using the **`ikj` loop order**. 
* Standard `ijk` multiplication causes non-sequential memory access in the second matrix.
* The `ikj` variant allows for **stride-1 access patterns**, which is significantly more friendly to the CPU's prefetcher.

## How to build and run
Compile using any C++11 compliant compiler with `-O3` optimization for best performance:

```bash
g++ -O3 Train_main.cpp NeuralNetwork.cpp -o train_mnist
./train_mnist
