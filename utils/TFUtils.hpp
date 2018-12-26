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

#pragma once

#if defined(_MSC_VER)
#  if !defined(COMPILER_MSVC)
#    define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#  endif
#  pragma warning(push)
#  pragma warning(disable : 4190)
#endif
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

class TFUtils {
public:
    enum STATUS
    {
        SUCCESS = 0,
        SESSION_CREATE_FAILED = 1,
        MODEL_LOAD_FAILED = 2,
        FAILED_RUN_SESSION = 3,
        MODEL_NOT_LOADED = 4,
    };
    TFUtils();
    STATUS LoadModel(std::string model_file);
    ~TFUtils();

    TF_Output GetOperationByName(std::string name, int idx);

    STATUS RunSession(const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                    const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors);

    // Static functions
    template <typename T>
    static TF_Tensor* CreateTensor(TF_DataType data_type,
                            const std::vector<std::int64_t>& dims,
                            const std::vector<T>& data) {
        return CreateTensor(data_type,
                            dims.data(), dims.size(),
                            data.data(), data.size() * sizeof(T));
    }


    static void DeleteTensor(TF_Tensor* tensor);

    static void DeleteTensors(const std::vector<TF_Tensor*>& tensors);

    template <typename T>
    static std::vector<std::vector<T>> GetTensorsData(const std::vector<TF_Tensor*>& tensors) {
        std::vector<std::vector<T>> data;
        data.reserve(tensors.size());
        for (const auto t : tensors) {
            data.push_back(GetTensorData<T>(t));
        }

        return data;
    }
    static TF_Tensor* CreateTensor(TF_DataType data_type,
                                   const std::int64_t* dims, std::size_t num_dims,
                                   const void* data, std::size_t len);

    template <typename T>
    static std::vector<T> GetTensorData(const TF_Tensor* tensor) {
        const auto data = static_cast<T*>(TF_TensorData(tensor));
        if (data == nullptr) {
            return {};
        }

        return {data, data + (TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor)))};
    }
//    STATUS GetErrorCode();
    static void PrinStatus(STATUS status);

private:
    TF_Graph* graph_def;
    TF_Session* sess;
    STATUS init_error_code;

private:
    TF_Graph* LoadGraphDef(const char* file);

    TF_Session* CreateSession(TF_Graph* graph);

    bool CloseAndDeleteSession(TF_Session* sess);

    bool RunSession(TF_Session* sess,
                    const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                    const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs);

    bool RunSession(TF_Session* sess,
                    const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                    const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors);




}; // End class TFUtils

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
