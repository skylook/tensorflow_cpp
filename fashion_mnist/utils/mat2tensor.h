//
// Created by skylook on 18-10-12.
//

#ifndef TENSORFLOW_CPP_IMG2TENSOR_H
#define TENSORFLOW_CPP_IMG2TENSOR_H

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

//#include "TensorflowObjectDetection.h"
#include <utility>
#include <fstream>
#include <regex>
#include <iostream>
#include <utility>

#include <vector>
#include "opencv2/core/core.hpp"

tensorflow::Tensor Mat2Tensor(cv::Mat &img, float normal = 1/255.0) {

    tensorflow::Tensor image_input = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
            {1, img.size().height, img.size().width, img.channels()}));

    float *tensor_data_ptr = image_input.flat<float>().data();
    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()), tensor_data_ptr);
    img.convertTo(fake_mat, CV_32FC(img.channels()));

    fake_mat *= normal;

    return image_input;

}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
tensorflow::Status Mat2Tensor(const cv::Mat &mat, tensorflow::Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    float *p = outTensor.flat<float>().data();
    cv::Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::vector<tensorflow::Tensor> outTensors;
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return tensorflow::Status::OK();
}

#endif //TENSORFLOW_CPP_IMG2TENSOR_H
