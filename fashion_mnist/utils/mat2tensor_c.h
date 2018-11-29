//
// Created by liuxiao on 18-11-29.
//

#ifndef TENSORFLOW_CPP_MAT2TENSOR_C_H
#define TENSORFLOW_CPP_MAT2TENSOR_C_H

#include <c_api.h> // TensorFlow C API header
#include <cstdlib>
#include <iostream>
#include <vector>

tensorflow::Tensor Mat2Tensor(cv::Mat &img, float normal = 1/255.0) {

tensorflow::Tensor image_input = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
        {1, img.size().height, img.size().width, img.channels()}));

float *tensor_data_ptr = image_input.flat<float>().data();
cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()), tensor_data_ptr);
img.convertTo(fake_mat, CV_32FC(img.channels()));

fake_mat *= normal;

return image_input;

}

#endif //TENSORFLOW_CPP_MAT2TENSOR_C_H
