#include "../utils/tf_utils.hpp"
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

    TF_Graph* graph = tf_utils::LoadGraphDef(graph_path.c_str());
    if (graph == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        return 1;
    }

    // Load image and convert to tensor
    std::string image_path = argv[2];
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);

    const std::vector<std::int64_t> input_dims = {1, image.size().height, image.size().width, image.channels()};

    TF_Tensor* input_image = Mat2Tensor(image, 1/255.0);

    // Input Tensor/Ops Create
    const std::vector<TF_Tensor*> input_tensors = {input_image};

    const std::vector<TF_Output> input_ops = {{TF_GraphOperationByName(graph, "input_image_input"), 0}};

    // Output Tensor/Ops Create
    const std::vector<TF_Output> output_ops = {{TF_GraphOperationByName(graph, "output_class/Softmax"), 0}};

    std::vector<TF_Tensor*> output_tensors = {nullptr};

    const bool success = tf_utils::RunSession(graph,
                                              input_ops, input_tensors,
                                              output_ops, output_tensors);

    if (success) {
        const std::vector<std::vector<float>> data = tf_utils::TensorsData<float>(output_tensors);
        const std::vector<float> result = data[0];

        int pred_index = ArgMax(result);

        // Print test accuracy
        printf("Predict: %d Label: %s", pred_index, class_names[pred_index].c_str());

    } else {
        std::cout << "Error run session";
        return 2;
    }

    tf_utils::DeleteTensors(input_tensors);
    tf_utils::DeleteTensors(output_tensors);

    TF_DeleteGraph(graph);

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