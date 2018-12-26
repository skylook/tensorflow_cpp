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

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << std::endl << "Usage: ./project path_to_graph.pb" << std::endl;
        return 1;
    }

    std::string graph_path = argv[1];

    // TFUtils init
    TFUtils TFU;
    TFUtils::STATUS status = TFU.LoadModel(graph_path);

    if (status != TFUtils::SUCCESS) {
        std::cerr << "Can't load graph" << std::endl;
        return 1;
    }

    // Input Tensor Create
    const std::vector<std::int64_t> input_a_dims = {1, 1};
    const std::vector<float> input_a_vals = {2.0};
    const std::vector<std::int64_t> input_b_dims = {1, 1};
    const std::vector<float> input_b_vals = {3.0};

    const std::vector<TF_Output> input_ops = {TFU.GetOperationByName("a", 0),
                                             TFU.GetOperationByName("b", 0)};

    const std::vector<TF_Tensor*> input_tensors = {TFUtils::CreateTensor(TF_FLOAT, input_a_dims, input_a_vals),
                                                   TFUtils::CreateTensor(TF_FLOAT, input_b_dims, input_b_vals)};

    // Output Tensor Create
    const std::vector<TF_Output> output_ops = {TFU.GetOperationByName("c", 0)};

    std::vector<TF_Tensor*> output_tensors = {nullptr};

    status = TFU.RunSession(input_ops, input_tensors,
                                              output_ops, output_tensors);

    TFUtils::PrinStatus(status);

    if (status == TFUtils::SUCCESS) {
        const std::vector<std::vector<float>> data = TFUtils::GetTensorsData<float>(output_tensors);
        const std::vector<float> result = data[0];
        std::cout << "Output value: " << result[0] << std::endl;
    } else {
        std::cout << "Error run session";
        return 2;
    }

    TFUtils::DeleteTensors(input_tensors);
    TFUtils::DeleteTensors(output_tensors);

    return 0;
}