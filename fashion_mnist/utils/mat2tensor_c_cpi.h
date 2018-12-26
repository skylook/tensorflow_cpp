// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// Copyright (c) 2018 Liu Xiao <liuxiao@foxmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
