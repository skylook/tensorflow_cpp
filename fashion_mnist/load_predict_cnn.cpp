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

#include "utils/mat2tensor.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace tensorflow;

//std::string class_names[10] = {'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'};
std::string class_names[] = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

int ArgMax(const tensorflow::TTypes<float, 1>::Tensor& prediction);

/**
 * @brief simple model for click through rate prediction
 * @details [long description]
 *
 * @param argv[1] graph protobuf
 *
 * @return [description]
 */
int main(int argc, char* argv[]) {
    // Initialize a tensorflow session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }

    if (argc != 3)
    {
        std::cerr << std::endl << "Usage: ./project path_to_graph.pb path_to_image.png" << std::endl;
        return 1;
    }

    // Load the protobuf graph
    GraphDef graph_def;
    std::string graph_path = argv[1];
    status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }

    std::string image_path = argv[2];
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);

    Tensor input_image = Mat2Tensor(image, 1/255.0);

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }

    // Setup inputs and outputs:
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "input_image_input:0", input_image }
    };
    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    // Run the session, evaluating our "c" operation from the graph
    status = session->Run(inputs, {"output_class/Softmax:0"}, {}, &outputs);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Run session successfully" << std::endl;
    }

    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.
    // Print the results
    std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 30>

//    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = outputs[0].flat<float>();
    const tensorflow::TTypes<float, 1>::Tensor& prediction = outputs[0].flat_inner_dims<float, 1>();

    int pred_index = ArgMax(prediction);

    // Print test accuracy
    printf("Predict: %d Label: %s", pred_index, class_names[pred_index].c_str());

    // Free any resources used by the session
    session->Close();

    return 0;
}

int ArgMax(const tensorflow::TTypes<float, 1>::Tensor& prediction)
{
    float max_value = -1.0;
    int max_index = -1;
    const long count = prediction.size();
    for (int i = 0; i < count; ++i) {
        const float value = prediction(i);
        if (value > max_value) {
            max_index = i;
            max_value = value;
        }
        std::cout << "value[" << i << "] = " << value << std::endl;
    }
    return max_index;
}
