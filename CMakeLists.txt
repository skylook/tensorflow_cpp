cmake_minimum_required(VERSION 3.5)
project(tensorflow_cpp)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

set(TENSORFLOW_INCLUDES
        /usr/local/include/tf/
        /usr/local/include/tf/bazel-genfiles
        /usr/local/include/tf/tensorflow/
        /usr/local/include/tf/tensorflow/third-party)

set(TENSORFLOW_LIBS
        /usr/local/lib/libtensorflow_cc.so
        /usr/local/lib//libtensorflow_framework.so)


include_directories(
        ${TENSORFLOW_INCLUDES}
        ${PROJECT_SOURCE_DIR}/third_party/eigen3
)

# Simple Net
add_executable(load_simple_net simple/load_simple_net.cpp)
target_link_libraries(load_simple_net
        ${TENSORFLOW_LIBS}
        ${OpenCV_LIBS}
        )

# Tfeat
add_executable(load_tfeat tfeat/load_tfeat.cpp)
target_link_libraries(load_tfeat
        ${TENSORFLOW_LIBS}
        ${OpenCV_LIBS}
        )