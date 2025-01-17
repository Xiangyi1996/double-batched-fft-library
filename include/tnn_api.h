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
#include "network_with_encodings.h"
#include "oneapi/mkl.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T> void printVector(std::string name, const std::vector<T> &vec) {
    std::cout << name << std::endl;
    for (const T &value : vec) {
        std::cout << value << ", ";
    }
    std::cout << std::endl;
}

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
        queue q;
        /// TODO: how handle this? Is there even a better way without memcpy, as this is device to device ->no
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

    virtual torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params, bool use_inference) = 0;

    virtual torch::Tensor backward_pass(torch::Tensor grad_output, torch::Tensor params) = 0;
    virtual torch::Tensor initialize_params() = 0;

    virtual torch::Tensor initialize_params(const torch::Tensor &tensor) = 0;
    virtual size_t n_params() = 0;

  protected:
    sycl::queue sycl_queue;

    torch::Device m_device;
    std::string m_device_name;
};

template <typename T> class EncodingModule : public Module {
  public:
    EncodingModule(int input_width, std::string filename, std::string device_name)
        : Module(device_name), m_input_width(input_width) {

        json encoding_config = io::loadJsonConfig(filename);
        // encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;
        encoding = create_encoding<T>(encoding_config);
        sycl_queue = sycl::queue();
    }

    EncodingModule(int input_width, const json &encoding_config, std::string device_name)
        : Module(device_name), m_input_width(input_width) {
        const json encoding_config_validated = io::validateAndCopyEncodingConfig(encoding_config);

        // encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;
        encoding = create_encoding<T>(encoding_config_validated);
        sycl_queue = sycl::queue();
    }
    ~EncodingModule() {}

    torch::Tensor forward_pass(torch::Tensor input_tensor, torch::Tensor params, bool use_inference) override {
        int batch_size = input_tensor.sizes()[0];

        DeviceMatrix<T> input_encoding(batch_size, m_input_width, this->sycl_queue);
        this->sycl_queue.memcpy(input_encoding.data(), input_tensor.data_ptr<T>(), input_encoding.size() * sizeof(T));
        this->sycl_queue.wait();

        DeviceMatrix<T> output_encoding(batch_size, encoding->output_width(), this->sycl_queue);
        std::unique_ptr<Context> model_ctx =
            encoding->forward_impl(&this->sycl_queue, input_encoding, &output_encoding);
        this->sycl_queue.wait();
        return convertDeviceMatrixToTensor(output_encoding);
    }

    torch::Tensor backward_pass(torch::Tensor grad_output, torch::Tensor params) override {}

    torch::Tensor initialize_params() override { throw std::invalid_argument("Not implemented yet."); }

    torch::Tensor initialize_params(const torch::Tensor &tensor) override {
        throw std::invalid_argument("Not implemented yet.");
    }
    size_t n_params() override { return encoding->n_params(); }

  private:
    std::shared_ptr<Encoding<T>> encoding;
    int m_input_width;
    sycl::queue sycl_queue;
};

template <typename T_enc, typename T_net, int WIDTH> class NetworkWithEncodingModule : public Module {
  public:
    NetworkWithEncodingModule(int input_width, int output_width, int n_hidden_layers, Activation activation,
                              Activation output_activation, std::string filename, std::string device_name)
        : Module(device_name), m_input_width(input_width), m_output_width(output_width), m_width(WIDTH),
          m_n_hidden_layers(n_hidden_layers) {

        json encoding_config = io::loadJsonConfig(filename);
        encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;
        network = create_network_with_encoding<T_enc, T_net, WIDTH>(this->sycl_queue, output_width, n_hidden_layers,
                                                                    activation, output_activation, encoding_config);
    }

    NetworkWithEncodingModule(int input_width, int output_width, int n_hidden_layers, Activation activation,
                              Activation output_activation, const json &encoding_config, std::string device_name)
        : Module(device_name), m_input_width(input_width), m_output_width(output_width), m_width(WIDTH),
          m_n_hidden_layers(n_hidden_layers) {
        const json encoding_config_validated = io::validateAndCopyEncodingConfig(encoding_config);
        network = create_network_with_encoding<T_enc, T_net, WIDTH>(
            this->sycl_queue, output_width, n_hidden_layers, activation, output_activation, encoding_config_validated);
    }

    ~NetworkWithEncodingModule() { delete interm_forw; }

    torch::Tensor forward_pass(torch::Tensor input_tensor, torch::Tensor params, const bool use_inference) override {
        set_params(params);
        int batch_size = input_tensor.sizes()[0];
        input_encoding = new DeviceMatrix<T_enc>(batch_size, m_input_width, this->sycl_queue);
        this->sycl_queue.memcpy(input_encoding->data(), input_tensor.data_ptr<T_enc>(),
                                input_encoding->size() * sizeof(T_enc));
        this->sycl_queue.wait();

        DeviceMatrix<T_enc> output_encoding(batch_size, m_width, this->sycl_queue);
        DeviceMatrix<T_net> output_network(batch_size, m_output_width, this->sycl_queue);
        DeviceMatrix<T_net> input_network(batch_size, WIDTH, this->sycl_queue);
        if (use_inference) {
            network->inference(*input_encoding, input_network, output_encoding, output_network, {});
            return convertDeviceMatrixToTensor(output_network);
        } else {
            interm_forw = new DeviceMatrices<T_net>(network->get_network()->get_n_hidden_layers() + 2, batch_size,
                                                    network->get_network()->get_input_width(), batch_size,
                                                    network->get_network()->get_network_width(), batch_size,
                                                    network->get_network()->get_output_width(), this->sycl_queue);
            network->forward_pass(*input_encoding, input_network, output_encoding, *interm_forw, {});

            std::vector<T_net> interm_forw_vec = interm_forw->copy_to_host();
            std::vector<T_net> output_network_vec(interm_forw_vec.end() - (batch_size * m_output_width),
                                                  interm_forw_vec.end());
            return convertVectorToTensor(output_network_vec);
        }
    }

    torch::Tensor backward_pass(torch::Tensor grad_output, torch::Tensor params) override {
        set_params(params);

        int batch_size = grad_output.sizes()[0];
        DeviceMatrix<T_net> dL_doutput(batch_size, network->get_network()->get_output_width(), this->sycl_queue);
        // we need to pad grad_output to network_output_width firstsd
        int64_t pad_right = network->get_network()->get_output_width() - grad_output.size(1);

        if (pad_right > 0) {
            // Applying padding, only on the right side (column wise)

            // std::cout << "Grad before: " << grad_output << std::endl;
            torch::nn::functional::PadFuncOptions pad_options({0, pad_right, 0, 0});
            grad_output = torch::nn::functional::pad(grad_output, pad_options);
            // std::cout << "Grad after: " << grad_output << std::endl;
        }

        dL_doutput.copy_from_device(grad_output.data_ptr<float>());

        DeviceMatrices<T_net> grads(network->get_network()->get_n_hidden_layers() + 1,
                                    network->get_network()->get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                                    network->get_network()->get_output_width(), this->sycl_queue);
        grads.fill(0.0f).wait();
        DeviceMatrices<T_net> interm_backw(network->get_network()->get_n_hidden_layers() + 1, batch_size,
                                           network->get_network()->get_network_width(), batch_size,
                                           network->get_network()->get_network_width(), batch_size,
                                           network->get_network()->get_output_width(), this->sycl_queue);
        interm_backw.fill(0.0f).wait();

        DeviceMatrix<T_net> dL_dinput(batch_size, network->get_network()->get_input_width(), this->sycl_queue);
        network->backward_pass(dL_doutput, grads, interm_backw, *interm_forw, {}, *input_encoding, dL_dinput);

        return convertDeviceMatricesToTensor(grads);
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

  private:
    std::shared_ptr<NetworkWithEncoding<T_enc, T_net>> network;
    int m_input_width;
    int m_output_width;
    int m_width;
    int m_n_hidden_layers;
    DeviceMatrices<T_net> *interm_forw = nullptr;  // allocated in forward_pass, deallocated in deconstructor
    DeviceMatrix<T_enc> *input_encoding = nullptr; // allocated in forward_pass, deallocated in deconstructor
};

} // namespace tnn
