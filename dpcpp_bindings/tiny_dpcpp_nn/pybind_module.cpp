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

#include "tnn_api.h" // need to include json before pybind11_json
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic conversion
#include <pybind11_json.hpp>
#include <torch/extension.h>

using json = nlohmann::json;
// C++ interface

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_XPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x)

class PybindingModule {
  public:
    PybindingModule(tnn::Module *module) : m_module{std::unique_ptr<tnn::Module>(module)} {}

    torch::Tensor fwd(torch::Tensor input, torch::Tensor params) {
        CHECK_INPUT(input);
        CHECK_INPUT(params);
        return m_module->forward_pass(input, params, 0);
    }

    torch::Tensor inference(torch::Tensor input, torch::Tensor params) {
        CHECK_INPUT(input);
        CHECK_INPUT(params);
        return m_module->forward_pass(input, params, 1);
    }

    torch::Tensor bwd(torch::Tensor grad_output, torch::Tensor params) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(params);
        return m_module->backward_pass(grad_output, params);
    }

    torch::Tensor initial_params() { return m_module->initialize_params(); }

    torch::Tensor initial_params(const torch::Tensor &tensor) { return m_module->initialize_params(tensor); }

    uint32_t n_params() const { return (uint32_t)m_module->n_params(); }

  private:
    std::unique_ptr<tnn::Module> m_module;
};

template <typename T, int WIDTH>
PybindingModule create_network_module(int input_width, int output_width, int n_hidden_layers, Activation activation,
                                      Activation output_activation, std::string device_name) {
    // Define the parameters for creating IdentityEncoding
    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                               {EncodingParams::SCALE, 1.0},
                               {EncodingParams::OFFSET, 0.0},
                               {EncodingParams::ENCODING, EncodingNames::IDENTITY}};

    tnn::NetworkWithEncodingModule<float, T, WIDTH> *networkwithencoding_module =
        new tnn::NetworkWithEncodingModule<float, T, WIDTH>(input_width, output_width, n_hidden_layers, activation,
                                                            output_activation, encoding_config, device_name);
    return PybindingModule{networkwithencoding_module};
}
PybindingModule create_network_factory(int input_width, int output_width, int n_hidden_layers, Activation activation,
                                       Activation output_activation, std::string device_name, int width) {
    if (width == 16) {
        return create_network_module<bf16, 16>(input_width, output_width, n_hidden_layers, activation,
                                               output_activation, device_name);
    } else if (width == 32) {
        return create_network_module<bf16, 32>(input_width, output_width, n_hidden_layers, activation,
                                               output_activation, device_name);
    } else if (width == 64) {
        return create_network_module<bf16, 64>(input_width, output_width, n_hidden_layers, activation,
                                               output_activation, device_name);
    } else if (width == 128) {
        return create_network_module<bf16, 128>(input_width, output_width, n_hidden_layers, activation,
                                                output_activation, device_name);
    } else {
        throw std::runtime_error("Unsupported width.");
    }
}
template <typename T>
PybindingModule create_encoding_module(int input_width, std::string encoding_name, const json &encoding_config,
                                       std::string device_name) {
    tnn::EncodingModule<T> *encoding_module = new tnn::EncodingModule<T>(input_width, encoding_config, device_name);
    return PybindingModule{encoding_module};
}

template <typename T_enc, typename T_net, int WIDTH>
PybindingModule create_networkwithencoding_module(int input_width, int output_width, int n_hidden_layers,
                                                  Activation activation, Activation output_activation,
                                                  const json &encoding_config, std::string device_name) {
    tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH> *networkwithencoding_module =
        new tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH>(input_width, output_width, n_hidden_layers, activation,
                                                                output_activation, encoding_config, device_name);
    return PybindingModule{networkwithencoding_module};
}

// C++ factory function
PybindingModule create_networkwithencoding_factory(int input_width, int output_width, int n_hidden_layers,
                                                   Activation activation, Activation output_activation,
                                                   const json &encoding_config, std::string device_name, int width) {
    if (width == 16) {
        return create_networkwithencoding_module<float, bf16, 16>(
            input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config, device_name);
    } else if (width == 32) {
        return create_networkwithencoding_module<float, bf16, 32>(
            input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config, device_name);
    } else if (width == 64) {
        return create_networkwithencoding_module<float, bf16, 64>(
            input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config, device_name);
    } else if (width == 128) {
        return create_networkwithencoding_module<float, bf16, 128>(
            input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config, device_name);
    } else {
        throw std::runtime_error("Unsupported width.");
    }
}
PYBIND11_MODULE(tiny_dpcpp_nn_pybind_module, m) {
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

    pybind11::class_<PybindingModule>(m, "Module")
        .def("fwd", &PybindingModule::fwd)
        .def("inference", &PybindingModule::inference)
        .def("bwd", &PybindingModule::bwd)
        .def("initial_params", (torch::Tensor(PybindingModule::*)()) & PybindingModule::initial_params)
        .def("initial_params",
             (torch::Tensor(PybindingModule::*)(const torch::Tensor &)) & PybindingModule::initial_params)
        .def("n_params", &PybindingModule::n_params);
    m.def("create_network", &create_network_factory, pybind11::arg("input_width"), pybind11::arg("output_width"),
          pybind11::arg("n_hidden_layers"), pybind11::arg("activation"), pybind11::arg("output_activation"),
          pybind11::arg("device_name"), pybind11::arg("width"));
    m.def("create_encoding", &create_encoding_module<float>);
    m.def("create_networkwithencoding", &create_networkwithencoding_factory, pybind11::arg("input_width"),
          pybind11::arg("output_width"), pybind11::arg("n_hidden_layers"), pybind11::arg("activation"),
          pybind11::arg("output_activation"), pybind11::arg("encoding_config"), pybind11::arg("device_name"),
          pybind11::arg("width"));
}
