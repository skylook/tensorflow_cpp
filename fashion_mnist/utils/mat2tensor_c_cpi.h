//
// Created by liuxiao on 18-11-29.
//

#ifndef TENSORFLOW_CPP_MAT2TENSOR_C_H
#define TENSORFLOW_CPP_MAT2TENSOR_C_H

#include <tensorflow/c/c_api.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "opencv2/core/core.hpp"

TF_Tensor* Mat2Tensor(cv::Mat &img, float normal = 1/255.0) {
    const std::vector<std::int64_t> input_dims = {1, img.size().height, img.size().width, img.channels()};

    // Convert to float 32 and do normalize ops
    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()));
    img.convertTo(fake_mat, CV_32FC(img.channels()));
    fake_mat *= normal;

    TF_Tensor* image_input = TFUtils::CreateTensor(TF_FLOAT,
                        input_dims.data(), input_dims.size(),
                        fake_mat.data, (fake_mat.size().height * fake_mat.size().width * fake_mat.channels() * sizeof(float)));

    return image_input;

}

#endif //TENSORFLOW_CPP_MAT2TENSOR_C_H
