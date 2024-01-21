/**
 * @file tnn_api.h
 * @author Kai Yuan
 * @brief
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <torch/extension.h>
#include <torch/script.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string> // Include the necessary header for std::string
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"
#include "encoding.h"
#include "json.hpp"
#include "network_with_encodings.h"
#include "oneapi/mkl.hpp"
using json = nlohmann::json;

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T> void printVector(const std::vector<T> &vec) {
    for (const T &element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

namespace tnn {
template <typename T> class Module {
  public:
    Module(std::string device_name) : m_device(torch::kCPU), m_device_name(device_name) {
        std::cout << "Running on device: " << m_device_name << std::endl;
        if (m_device_name == "cpu") {
            m_device = torch::kCPU;
        } else if (m_device_name == "xpu") {
            m_device = torch::kXPU;
        } else {
            std::cout << "No device name " << device_name << ". Consider falling back to CPU as device. Exiting now"
                      << std::endl;
            exit(1);
        }
    }
    virtual ~Module() {}

    virtual torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params, int use_inference = 0) = 0;

    virtual torch::Tensor backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output,
                                        torch::Tensor params) = 0;
    virtual void initialize_params(float *params_full_precision, int use_easy = 0) = 0;
    virtual void free_memory() = 0;
    virtual int n_params() = 0;
    virtual int n_output_dims() = 0;

  protected:
    sycl::queue sycl_queue;

    torch::Device m_device;
    std::string m_device_name;

    std::vector<T> get_vector_from_tensor(torch::Tensor tensor) {
        static_assert(std::is_same<T, bf16>::value || std::is_same<T, float>::value,
                      "get_vector_from_tensor only accepts bf16 or float types.");

        std::vector<T> array(tensor.numel());

        float *tensor_data = tensor.data_ptr<float>();
        for (int i = 0; i < tensor.numel(); ++i) {
            if (std::is_same<T, bf16>::value) {
                array[i] = bf16(tensor_data[i]);
            } else {
                array[i] = tensor_data[i];
            }
        }
        return array;
    }

    void convert_tensor_to_dev_mem(torch::Tensor &tensor, DeviceMem<T> *device_mem_array) {
        std::vector<T> array = get_vector_from_tensor<T>(tensor);
        if (array.size() != device_mem_array->size()) {
            std::cerr << "Assertion failed: array.size() == device_mem_array.size()\n"
                      << "array.size(): " << array.size() << "\n"
                      << "device_mem_array.size(): " << device_mem_array->size() << std::endl;
            exit(1);
        } // conversion to DeviceMem required by Swiftnet forward_pass

        // copy array to device_mem_array
        device_mem_array->copy_from_host(array, sycl_queue);
    }

    torch::Tensor get_converted_tensor_from_dev_mem(DeviceMem<T> &device_mem_array, int print_out = 0) {
        // Conversion to float array for pybindings
        std::vector<T> list_T(device_mem_array.size());
        device_mem_array.copy_to_host(list_T, sycl_queue);

        // Convert the original vector to a std::vector<float>
        std::vector<float> list_float(list_T.size());
        for (size_t i = 0; i < list_T.size(); ++i) {
            list_float[i] = static_cast<float>(list_T[i]);
        }
        if (print_out) {
            std::cout << "About to convert this " << std::endl;
            printVector(list_float);
        }
        //   convert to torch tensor
        torch::Tensor output_tensor =
            torch::from_blob(list_float.data(), {static_cast<long>(list_float.size())}, torch::kFloat32).clone();
        return output_tensor;
    }

    torch::Tensor get_converted_tensor_from_array(T *array, int size) {
        torch::Tensor tensor = torch::from_blob(array, {size}, torch::kFloat32);
        return tensor;
    }
};

// template <typename T> class EncodingModule : public Module<T> {
//   public:
//     EncodingModule(int input_width, const json &encoding_config, std::string device_name);
//     ~EncodingModule() {}

//     torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params, int use_inference = 0) override;

//     torch::Tensor backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output, torch::Tensor params)
//     override; void initialize_params(float *params_full_precision, int use_easy = 0) override; void free_memory()
//     override; int n_params() override {
//         std::cout << "Encodings don't have params, thus: n_params = 0" << std::endl;
//         return 0;
//     };
//     int n_output_dims() override {
//         // std::cout << "Encoding width: " << encoding->output_width() << std::endl;
//         return encoding->output_width();
//     }

//   private:
//     Encoding<T> *encoding;
//     int m_input_width;
//     sycl::queue sycl_queue;
// };

template <typename T, int WIDTH> class NetworkWithEncodingModule : public Module<T> {
  public:
    NetworkWithEncodingModule(int width, int input_width, int output_width, int n_hidden_layers, Activation activation,
                              Activation output_activation, const json &encoding_config, std::string device_name);
    ~NetworkWithEncodingModule() {}

    torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params, int use_inference = 0) override;

    torch::Tensor backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output, torch::Tensor params) override;
    void initialize_params(float *params_full_precision, int use_easy = 0) override;
    void free_memory() override;
    int n_params() override;
    void set_params(torch::Tensor &params);

    int n_output_dims() {
        std::cout << "Not implemented" << std::endl;
        return -1;
    }

  private:
    std::shared_ptr<NetworkWithEncoding<T>> network;
    int m_input_width;
    int m_output_width;
    int m_width;
    int m_n_hidden_layers;
    sycl::queue sycl_queue;
};

} // namespace tnn
