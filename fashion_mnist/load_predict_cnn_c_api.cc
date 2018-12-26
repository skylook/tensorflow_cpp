// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// Copyright (c) 2018 Liu Xiao <liuxiao@foxmail.com> and Daniil Goncharov <neargye@gmail.com>.
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

#include "../utils/TFUtils.hpp"
#include "utils/mat2tensor_c_cpi.h"

#include <iostream>
#include <vector>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//std::string class_names[10] = {'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'};
std::string class_names[] = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

int ArgMax(const std::vector<float> result);

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << std::endl << "Usage: ./project path_to_graph.pb path_to_image.png" << std::endl;
        return 1;
    }

    // Load graph
    std::string graph_path = argv[1];

    // TFUtils init
    TFUtils TFU;
    TFUtils::STATUS status = TFU.LoadModel(graph_path);

    if (status != TFUtils::SUCCESS) {
        std::cerr << "Can't load graph" << std::endl;
        return 1;
    }

    // Load image and convert to tensor
    std::string image_path = argv[2];
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);

    const std::vector<std::int64_t> input_dims = {1, image.size().height, image.size().width, image.channels()};

    TF_Tensor* input_image = Mat2Tensor(image, 1/255.0);

    // Input Tensor/Ops Create
    const std::vector<TF_Tensor*> input_tensors = {input_image};

    const std::vector<TF_Output> input_ops = {TFU.GetOperationByName("input_image_input", 0)};

    // Output Tensor/Ops Create
    const std::vector<TF_Output> output_ops = {TFU.GetOperationByName("output_class/Softmax", 0)};

    std::vector<TF_Tensor*> output_tensors = {nullptr};

    status = TFU.RunSession(input_ops, input_tensors,
                            output_ops, output_tensors);

    if (status == TFUtils::SUCCESS) {
        const std::vector<std::vector<float>> data = TFUtils::GetTensorsData<float>(output_tensors);
        const std::vector<float> result = data[0];

        int pred_index = ArgMax(result);

        // Print test accuracy
        printf("Predict: %d Label: %s", pred_index, class_names[pred_index].c_str());

    } else {
        std::cout << "Error run session";
        return 2;
    }

    TFUtils::DeleteTensors(input_tensors);
    TFUtils::DeleteTensors(output_tensors);

    return 0;
}

int ArgMax(const std::vector<float> result)
{
    float max_value = -1.0;
    int max_index = -1;
    const long count = result.size();
    for (int i = 0; i < count; ++i) {
        const float value = result[i];
        if (value > max_value) {
            max_index = i;
            max_value = value;
        }
        std::cout << "value[" << i << "] = " << value << std::endl;
    }
    return max_index;
}