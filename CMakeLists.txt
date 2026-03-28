cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(Include)

set(SOURCES
    src/NeuralNetwork.cpp
)

add_executable(train_nn ${SOURCES} src/Train_main.cpp)

add_executable(test_nn ${SOURCES} src/Test_main.cpp)