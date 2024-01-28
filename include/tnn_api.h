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

#include "common.h"
#include "encoding.h"
#include "io.h"
#include "json.hpp"
#include "network_with_encodings.h"
#include "oneapi/mkl.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using json = nlohmann::json;

using bf16 = sycl::ext::oneapi::bfloat16;

// Function to convert torch::Tensor to std::vector
template <typename T> std::vector<T> convertTensorToVector(const torch::Tensor &tensor) {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, int>::value ||
                      std::is_same<T, bf16>::value,
                  "Unsupported data type");

    if (tensor.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("Tensor is not of type bf16 ");
    }

    if constexpr (std::is_same<T, bf16>::value) {
        std::vector<T> result(tensor.numel());
        T *sycl_bf16_ptr = reinterpret_cast<T *>(tensor.data_ptr());
        queue q; // TODO: how handle this? Is there even a better way without memcpy, as this is device to device -> no
                 // conversion to vector, but passing the pointer directly for set_weights
        q.memcpy(result.data(), sycl_bf16_ptr, sizeof(T) * result.size()).wait();

        return result;
    } else {
        // Convert the tensor directly to the target data type, this is only supproted for float, double, and int
        return std::vector<T>(tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel());
    }
}

// Template specialization for float, double, and int
template <typename T> torch::Tensor convertVectorToTensor(const std::vector<T> &vector) {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, int>::value ||
                      std::is_same<T, bf16>::value,
                  "Unsupported data type");

    torch::Tensor tensor = torch::from_blob((void *)vector.data(), {static_cast<int64_t>(vector.size())});
    return tensor.clone(); // Ensure that the tensor is contiguous
}

// Template specialization for torch::kHalf (bf16 and sycl::half)
template <> torch::Tensor convertVectorToTensor<bf16>(const std::vector<bf16> &vector) {
    torch::Tensor tensor = torch::empty({static_cast<int64_t>(vector.size())}, torch::kFloat);
    for (size_t i = 0; i < vector.size(); ++i) {
        tensor[i] = static_cast<float>(vector[i]);
    }
    return tensor.to(torch::kBFloat16);
}

template <typename T> torch::Tensor convertDeviceMatrixToTensor(DeviceMatrix<T> &device_matrix) {
    return convertVectorToTensor(device_matrix.copy_to_host());
}
template <typename T> torch::Tensor convertDeviceMatricesToTensor(DeviceMatrices<T> &device_matrices) {
    return convertVectorToTensor(device_matrices.copy_to_host());
}

namespace tnn {
class Module {
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
    virtual torch::Tensor initialize_params() = 0;

    virtual torch::Tensor initialize_params(const torch::Tensor &tensor) = 0;
    virtual size_t n_params() = 0;
    virtual int n_output_dims() = 0;

  protected:
    sycl::queue sycl_queue;

    torch::Device m_device;
    std::string m_device_name;
};

template <typename T> class EncodingModule : public Module {
  public:
    EncodingModule(int input_width, const json &encoding_config, std::string device_name)
        : Module(device_name), m_input_width(input_width) {
        encoding = create_encoding<T>(encoding_config);
        sycl_queue = sycl::queue();
    }
    ~EncodingModule() {}

    torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params,
                               int use_inference = 0) override { //   assert(input_tensor.sizes() == 2 &&p
        //          "Tensor length for Encoding forward is not equal to 2!");
        //   std::cout << "Input tensor sizes: " << input_tensor.sizes() << std::endl;
        // int batch_size = input_tensor.sizes()[1];
        // DeviceMatrix<float> input_matrix = DeviceMatrix<float>(input_tensor.data_ptr<float>(), m_input_width,
        // batch_size);

        // torch::Tensor output_tensor = torch::empty({encoding->output_width(), batch_size},
        //                                            torch::TensorOptions().dtype(torch::kFloat32).device(m_device));
        // DeviceMatrix<float> output_matrix = DeviceMatrix<float>(output_tensor.data_ptr<float>(), m_input_width,
        // batch_size);

        // std::unique_ptr<Context> model_ctx = encoding->forward_impl(&sycl_queue, input_matrix, &output_matrix);

        // return output_tensor;};
    }

    torch::Tensor backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output, torch::Tensor params) override {}

    torch::Tensor initialize_params() override { throw std::invalid_argument("Not implemented yet."); }

    virtual torch::Tensor initialize_params(const torch::Tensor &tensor) override {
        throw std::invalid_argument("Not implemented yet.");
    }

    size_t n_params() override {
        std::cout << "Encodings don't have params, thus: n_params = 0" << std::endl;
        return 0;
    }

    int n_output_dims() override { return encoding->output_width(); }

  private:
    Encoding<T> *encoding;
    int m_input_width;
    sycl::queue sycl_queue;
};

template <typename T_enc, typename T_net, int WIDTH> class NetworkWithEncodingModule : public Module {
  public:
    NetworkWithEncodingModule(int input_width, int output_width, int n_hidden_layers, Activation activation,
                              Activation output_activation, std::string filename, std::string device_name)
        : Module(device_name), m_input_width(input_width), m_output_width(output_width), m_width(WIDTH),
          m_n_hidden_layers(n_hidden_layers) {

        json encoding_config = loadJsonConfig(filename);
        encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;
        network = create_network_with_encoding<T_enc, T_net, WIDTH>(
            this->sycl_queue, m_width, output_width, n_hidden_layers, activation, output_activation, encoding_config);
    }

    NetworkWithEncodingModule(int input_width, int output_width, int n_hidden_layers, Activation activation,
                              Activation output_activation, const json &encoding_config, std::string device_name)
        : Module(device_name), m_input_width(input_width), m_output_width(output_width), m_width(WIDTH),
          m_n_hidden_layers(n_hidden_layers) {

        network = create_network_with_encoding<T_enc, T_net, WIDTH>(
            this->sycl_queue, m_width, output_width, n_hidden_layers, activation, output_activation, encoding_config);
    }

    ~NetworkWithEncodingModule() {}

    torch::Tensor forward_pass(torch::Tensor input_tensor, torch::Tensor params, int use_inference = 0) override {
        set_params(params);
        int batch_size = input_tensor.sizes()[0];
        DeviceMatrix<T_enc> input_encoding(batch_size, m_input_width, this->sycl_queue);
        this->sycl_queue.memcpy(input_encoding.data(), input_tensor.data_ptr<T_enc>(),
                                input_encoding.size() * sizeof(T_enc));
        this->sycl_queue.wait();

        DeviceMatrix<T_enc> output_encoding(batch_size, m_width, this->sycl_queue);
        DeviceMatrix<T_net> output_network(batch_size, m_output_width, this->sycl_queue);
        DeviceMatrix<T_net> input_network(batch_size, WIDTH, this->sycl_queue);

        if (use_inference) {
            network->inference(input_encoding, input_network, output_encoding, output_network, {});
            return convertDeviceMatrixToTensor(output_network);
        } else {
            DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(),
                                          batch_size, network.get_network_width(), batch_size,
                                          network.get_output_width(), q);
            network->forward_pass(input_encoding, input_network, output_encoding, interm_forw, {});
            return convertDeviceMatricesToTensor(interm_forw);
        }
    }

    torch::Tensor backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output, torch::Tensor params) override {
        // int batch_size = input_tensor.sizes()[1];

        // DeviceMem<bf16> input_backward = DeviceMem<bf16>(batch_size * m_input_width, sycl_queue);
        // DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * m_output_width, sycl_queue);

        // convert_tensor_to_dev_mem(grad_output, &grads);
        // convert_tensor_to_dev_mem(input_tensor, &input_backward);

        // set_params(params);

        // DeviceMem<bf16> *grads_matrices = network->backward_pass(input_backward, grads, forward, batch_size);

        // torch::Tensor grad_loss = get_converted_tensor_from_dev_mem(*grads_matrices);

        // return grad_loss;
    }

    torch::Tensor initialize_params() override {
        std::vector<bf16> network_weights = network->get_network()->get_weights_matrices().copy_to_host();
        torch::Tensor network_weights_tensor = convertVectorToTensor<T_net>(network_weights);
        return network_weights_tensor;
    }

    torch::Tensor initialize_params(const torch::Tensor &tensor) override {
        network->get_network()->set_weights_matrices(convertTensorToVector<T_net>(tensor));
        return tensor;
    }

    size_t n_params() override { return network->get_network()->get_weights_matrices().nelements(); }

    void set_params(torch::Tensor &params) {
        if (this->m_device_name == "cpu") {
            throw std::invalid_argument("CPU currently not supported/tested. Run on XPU please");
        } else if (this->m_device_name == "xpu") {
            network->get_network()->set_weights_matrices(convertTensorToVector<T_net>(params));
        } else {
            std::cout << "No behaviour for device " << this->m_device_name << ". Exiting code." << std::endl;
            exit(1);
        }
    }

    int n_output_dims() override {
        std::cout << "Not implemented" << std::endl;
        return -1;
    }

  private:
    std::shared_ptr<NetworkWithEncoding<T_enc, T_net>> network;
    int m_input_width;
    int m_output_width;
    int m_width;
    int m_n_hidden_layers;
};

} // namespace tnn
