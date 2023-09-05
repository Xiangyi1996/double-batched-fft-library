#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <iostream>

#include "tnn_api.h"

// C++ interface

#define CHECK_XPU(x) \
  TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_XPU(x);        \
  CHECK_CONTIGUOUS(x)

class Module {
 public:
  Module(tnn::SwiftNetModule* module) : m_module{module} {}
  //   Module() {}

  torch::Tensor fwd(torch::Tensor input, torch::Tensor params) {
    // CHECK_INPUT(input);
    // CHECK_INPUT(params);
    return m_module->forward_pass(input, params);
  }

  torch::Tensor bwd(torch::Tensor input_tensor, torch::Tensor grad_output,
                    torch::Tensor params) {
    // CHECK_INPUT(input_tensor);
    // CHECK_INPUT(grad_output);
    // CHECK_INPUT(params);
    return m_module->backward_pass(input_tensor, grad_output, params);
  }

  torch::Tensor initial_params() {
    torch::Tensor output = torch::zeros(
        {n_params()}, torch::TensorOptions().dtype(torch::kFloat32));
    m_module->initialize_params(output.data_ptr<float>());
    return output;
  }

  uint32_t n_params() const { return (uint32_t)m_module->n_params(); }

  void free_memory() { m_module->free_memory(); }

 private:
  std::unique_ptr<tnn::SwiftNetModule> m_module;
};

Module create_network(const int width, int input_width, int output_width,
                      int n_hidden_layers, Activation activation,
                      Activation output_activation, const int batch_size,
                      std::string device_name) {
  tnn::SwiftNetModule* network_module = tnn::create_network(
      width, input_width, output_width, n_hidden_layers, activation,
      output_activation, batch_size, device_name);

  return Module{network_module};
}

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

  pybind11::class_<Module>(m, "Module")
      .def(pybind11::init<tnn::SwiftNetModule*>())
      .def("fwd", &Module::fwd)
      .def("bwd", &Module::bwd)
      .def("initial_params", &Module::initial_params)
      .def("n_params", &Module::n_params)
      .def("free_memory", &Module::free_memory);

  m.def("create_network", &create_network);
}
