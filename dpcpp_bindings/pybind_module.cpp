/**
 * @file pybind_module.cpp
 * @author Kai Yuan
 * @brief
 * @version 0.1
 * @date 2024-01-22
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <iostream>

#include "tnn_api.h"

// C++ interface

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_XPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x)

template <typename T> class Module {
  public:
    Module(tnn::Module<T> *module) : m_module{module} {}
    //   Module() {}

    torch::Tensor fwd(torch::Tensor input, torch::Tensor params) {
        // CHECK_INPUT(input);
        // CHECK_INPUT(params);
        return m_module->forward_pass(input, params, 0);
    }

    torch::Tensor inference(torch::Tensor input, torch::Tensor params) {
        // CHECK_INPUT(input);
        // CHECK_INPUT(params);
        return m_module->forward_pass(input, params, 1);
    }

    torch::Tensor bwd(torch::Tensor input_tensor, torch::Tensor grad_output, torch::Tensor params) {
        // CHECK_INPUT(input_tensor);
        // CHECK_INPUT(grad_output);
        // CHECK_INPUT(params);
        return m_module->backward_pass(input_tensor, grad_output, params);
    }

    torch::Tensor initial_params(int use_easy = 0) {
        torch::Tensor output = torch::zeros({n_params()}, torch::TensorOptions().dtype(torch::kFloat32));
        m_module->initialize_params(output.data_ptr<float>(), use_easy);
        return output;
    }

    uint32_t n_params() const { return (uint32_t)m_module->n_params(); }

    uint32_t n_output_dims() const { return (uint32_t)m_module->n_output_dims(); }

    void free_memory() { m_module->free_memory(); }

  private:
    std::unique_ptr<tnn::Module<bf16>> m_module;
};

// template <typename T>
// Module<T> create_network_module(const int width, int input_width, int output_width, int n_hidden_layers,
//                                 Activation activation, Activation output_activation, std::string device_name) {

//     // Define the parameters for creating IdentityEncoding
//     const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
//                                {EncodingParams::SCALE, 1.0},
//                                {EncodingParams::OFFSET, 0.0},
//                                {EncodingParams::ENCODING, EncodingNames::IDENTITY}};

//     tnn::NetworkWithEncodingModule<T> *networkwithencoding_module = new tnn::NetworkWithEncodingModule<T>(
//         width, input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config,
//         device_name);
//     return Module{networkwithencoding_module};
// }

// template <typename T>
// Module<T> create_encoding_module(int input_width, std::string encoding_name, const json &encoding_config,
//                                  std::string device_name) {
//     tnn::EncodingModule<T> *encoding_module = new tnn::EncodingModule<T>(input_width, encoding_config, device_name);
//     return Module{encoding_module};
// }

// template <typename T>
// Module<T> create_networkwithencoding_module(int width, int input_width, int output_width, int n_hidden_layers,
//                                             Activation activation, Activation output_activation,
//                                             const json &encoding_config, std::string device_name) {
//     tnn::NetworkWithEncodingModule<T> *networkwithencoding_module = new tnn::NetworkWithEncodingModule<T>(
//         width, input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config,
//         device_name);
//     return Module{networkwithencoding_module};
// }

PYBIND11_MODULE(tiny_nn, m) {
    pybind11::enum_<Activation>(m, "Activation")
        .value("ReLU", Activation::ReLU)
        .value("LeakyReLU", Activation::LeakyReLU)
        .value("Exponential", Activation::Exponential)
        .value("Sine", Activation::Sine)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Squareplus", Activation::Squareplus)
        .value("Softplus", Activation::Softplus)
        .value("Tanh", Activation::Tanh)
        .value("Linear", Activation::None)
        .export_values();

    pybind11::class_<Module<bf16>>(m, "Module")
        .def("fwd", &Module<bf16>::fwd)
        .def("inference", &Module<bf16>::inference)
        .def("bwd", &Module<bf16>::bwd)
        .def("initial_params", &Module<bf16>::initial_params)
        .def("n_params", &Module<bf16>::n_params)
        .def("n_output_dims", &Module<bf16>::n_output_dims)
        .def("free_memory", &Module<bf16>::free_memory);

    // m.def("create_network", &create_network_module<bf16>);
    // m.def("create_encoding", &create_encoding_module<bf16>);
    // m.def("create_networkwithencoding", &create_networkwithencoding_module<bf16>);
}
