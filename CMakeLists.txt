cmake_minimum_required(VERSION 3.5)
project(tensorflow_cpp)

set(CMAKE_CXX_STANDARD 11)

option(USE_TENSORFLOW_CPP "USE Tensorflow C++ API" ON)
option(USE_TENSORFLOW_C "USE Tensorflow C API" OFF)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

if(USE_TENSORFLOW_CPP)
    # Setup Tensorflow C++
    set(TENSORFLOW_INCLUDES
            /usr/local/include/tf/
            /usr/local/include/tf/bazel-genfiles
            /usr/local/include/tf/tensorflow/
            /usr/local/include/tf/tensorflow/third-party)

    set(TENSORFLOW_LIBS
            /usr/local/lib/libtensorflow_cc.so
            /usr/local/lib/libtensorflow_framework.so)

    include_directories(
            ${TENSORFLOW_INCLUDES}
            ${PROJECT_SOURCE_DIR}/third_party/eigen3
    )

    # Simple Net C++
    add_executable(load_simple_net simple/load_simple_net.cpp)
    target_link_libraries(load_simple_net
            ${TENSORFLOW_LIBS}
            ${OpenCV_LIBS}
            )

    # CNN Net C++
    add_executable(load_predict_cnn_cpp fashion_mnist/load_predict.cpp)
    target_link_libraries(load_predict_cnn_cpp
            ${TENSORFLOW_LIBS}
            ${OpenCV_LIBS}
            )

endif()

if(USE_TENSORFLOW_C)
    # Setup Tensorflow C
    set(TENSORFLOW_INCLUDES
            ${PROJECT_SOURCE_DIR}/third_party/libtensorflow_cpu/include)

    set(TENSORFLOW_LIBS
            ${PROJECT_SOURCE_DIR}/third_party/libtensorflow_cpu/lib/libtensorflow.so)

    # CNN Net C
    add_executable(load_predict_cnn_c fashion_mnist/load_predict_c_api.cc)
    target_link_libraries(load_predict_cnn_c
            ${TENSORFLOW_LIBS}
            ${OpenCV_LIBS}
            )
endif()
