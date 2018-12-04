#include "../utils/tf_utils.hpp"

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

    TF_Graph* graph = tf_utils::LoadGraphDef(graph_path.c_str());
    if (graph == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        return 1;
    }

    // Input Tensor Create
    const std::vector<std::int64_t> input_a_dims = {1, 1};
    const std::vector<float> input_a_vals = {2.0};
    const std::vector<std::int64_t> input_b_dims = {1, 1};
    const std::vector<float> input_b_vals = {3.0};

    const std::vector<TF_Output> input_ops = {{TF_GraphOperationByName(graph, "a"), 0},
                                             {TF_GraphOperationByName(graph, "b"), 0}};

    const std::vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, input_a_dims, input_a_vals),
                                                   tf_utils::CreateTensor(TF_FLOAT, input_b_dims, input_b_vals)};

    // Output Tensor Create
    const std::vector<TF_Output> output_ops = {{TF_GraphOperationByName(graph, "c"), 0}};

    std::vector<TF_Tensor*> output_tensors = {nullptr};

    const bool success = tf_utils::RunSession(graph,
                                              input_ops, input_tensors,
                                              output_ops, output_tensors);

    if (success) {
        const std::vector<std::vector<float>> data = tf_utils::TensorsData<float>(output_tensors);
        const std::vector<float> result = data[0];
        std::cout << "Output value: " << result[0] << std::endl;
    } else {
        std::cout << "Error run session";
        return 2;
    }

    tf_utils::DeleteTensors(input_tensors);
    tf_utils::DeleteTensors(output_tensors);

    TF_DeleteGraph(graph);

    return 0;
}