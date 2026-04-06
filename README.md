# MNIST Multilayer Perceptron from Scratch (C++)

This project implements a Feedforward Neural Network (Multilayer Perceptron) to classify handwritten digits from the MNIST dataset. The entire engine is written in pure C++ without any external machine learning libraries.

## Technical Highlights

- **Cache-Friendly Linear Algebra**: Custom `Matrix` class with 1D contiguous array storage maximizing L1/L2 cache locality. Matrix multiplication uses an `ikj` loop order to minimize CPU cache misses.
- **Mini-Batch Gradient Descent**: Training uses mini-batches of configurable size (default: 64), averaging gradients over the batch for more stable convergence.
- **Bias Broadcasting**: Custom `addBias()` method on `Matrix` broadcasts a `(n, 1)` bias vector across all columns of a `(n, batchSize)` activation matrix during the forward pass.
- **Column-wise Softmax**: Output layer softmax normalized independently per column (per sample) for correctness with batched inputs.
- **Custom Binary Serialization**: Raw binary format for saving and loading the trained model, bypassing CSV/text parsing overhead.
- **Numerical Stability**: Softmax uses a per-column max-subtraction trick to prevent floating-point overflow.
- **Modular Design**: Data loading, linear algebra, activations, and network logic are separated into distinct modules.

## Model Architecture

- **Input Layer**: 784 neurons (28×28 pixels, flattened and normalized to [0, 1]).
- **Hidden Layer**: 128 neurons with **ReLU** activation.
- **Output Layer**: 10 neurons with **Softmax** activation (representing digits 0–9).
- **Optimization**: Mini-Batch Gradient Descent with **He initialization** (suited for ReLU).
- **Batch Size**: 64 samples per update step.
- **Shuffle**: Training indices are shuffled at the start of every epoch to prevent ordering bias.

## Performance

Trained on 60,000 images for 30 epochs with batch size 64 and learning rate 0.01:

| Epoch | Training Accuracy |
|-------|------------------|
| 1     | 90.30%           |
| 5     | 97.50%           |
| 10    | 98.78%           |
| 15    | 99.34%           |
| 20    | 99.68%           |
| 25    | 99.84%           |
| 30    | 99.91%           |

> Previous SGD baseline (5 epochs): 98.88% training accuracy, 97.65% test accuracy.

## Build and Run (CMake)

Compile in **Release** mode to enable `-O3` optimizations, which are essential for matrix-heavy workloads.

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Dataset

The model is trained and evaluated on the **MNIST** database of handwritten digits.

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels (grayscale, flattened to 784 values)
- **Label Format**: Integer (0–9), one-hot encoded as target vector

**Sources:**

- [Kaggle — MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [OpenML — mnist_784](https://www.openml.org/d/554)

### Preprocessing

1. **Flattening**: Each 28×28 image is converted to a 784-element column vector.
2. **Normalization**: Pixel values scaled from [0, 255] to [0, 1]:

$$x_{\text{norm}} = \frac{x}{255.0}$$

## Performance & Optimization Notes

### Cache Locality & Memory Layout

All matrices use a **single contiguous 1D `std::vector<double>`** internally. Row-major indexing maps `(i, j)` to `i * cols + j`, ensuring sequential memory access during the innermost loop of matrix multiplication.

### Matrix Multiplication (`ikj` loop order)

Standard `ijk` multiplication accesses the second matrix column-by-column (non-sequential). The `ikj` variant hoists `A(i,k)` into a scalar `tmp` and accesses `B` row-by-row (sequential), enabling the CPU prefetcher to work efficiently:

```cpp
for (int i = 0; i < m_rows; ++i)
    for (int k = 0; k < m_cols; ++k)
    {
        T tmp = (*this)(i, k);
        for (int j = 0; j < other.m_cols; ++j)
            result(i, j) += tmp * other(k, j);
    }
```

### OpenMP Parallelism

The outer loop of matrix multiplication is parallelized with `#pragma omp parallel for`, distributing rows across CPU cores. The CMake build links OpenMP automatically.
