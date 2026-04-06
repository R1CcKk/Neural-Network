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

add_executable(train_batch_nn ${SOURCES} src/Train_Main_Batch.cpp)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(train_nn PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_nn PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(train_batch_nn PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Open MP founded")
else()
    find_library(LIBOMP_PATH omp HINTS /opt/homebrew/opt/libomp/lib)

    if(LIBOMP_PATH)
        target_link_libraries(train_nn PRIVATE ${LIBOMP_PATH})
        target_link_libraries(test_nn PRIVATE ${LIBOMP_PATH})
        target_link_libraries(train_batch_nn PRIVATE ${LIBOMP_PATH})
        target_compile_options(train_nn PRIVATE -Xpreprocessor -fopenmp)
        target_compile_options(test_nn PRIVATE -Xpreprocessor -fopenmp)
        target_compile_options(train_batch_nn PRIVATE -Xpreprocessor -fopenmp)
        include_directories(/opt/homebrew/opt/libomp/include)
        message(STATUS "OpenMP forced via Homebrew")
    else()
        message(WARNING "OpenMP not found.")
    endif()
endif()