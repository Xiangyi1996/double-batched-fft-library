#pragma once  // Use pragma once for include guards

#include <torch/extension.h>

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>  // Include the necessary header for std::string
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "oneapi/mkl.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

namespace tnn {

class SwiftNetMLPFactory {
 public:
  static std::unique_ptr<Network> create(queue q, int width, int input_width,
                                         int output_width, int n_hidden_layers,
                                         Activation activation,
                                         Activation output_activation,
                                         int batch_size) {
    switch (width) {
      case 64:
        return std::make_unique<SwiftNetMLP<64>>(q, input_width, output_width,
                                                 n_hidden_layers, activation,
                                                 output_activation, batch_size);
      default:
        throw std::runtime_error(
            "SwiftNetMLP not implemented for the specified width: " +
            std::to_string(width));
    }
  }
};

class SwiftNetModule {
 public:
  SwiftNetModule(const int width, int input_width, int output_width,
                 int n_hidden_layers, Activation activation,
                 Activation output_activation, const int batch_size,
                 std::string device_name);

  torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params,
                             int use_inference = 0);

  torch::Tensor backward_pass(torch::Tensor input_tensor,
                              torch::Tensor grad_output, torch::Tensor params);
  void initialize_params(float *params_full_precision);
  int n_params();
  void free_memory();
  torch::Device m_device;
  std::string m_device_name;

 private:
  template <typename T>
  torch::Tensor get_converted_tensor_from_array(T *array, int size);
  std::vector<bf16> get_vector_from_tensor(torch::Tensor tensor);
  void convert_tensor_to_dev_mem(torch::Tensor tensor,
                                 DeviceMem<bf16> device_mem_array);
  template <typename T>
  torch::Tensor get_converted_tensor_from_dev_mem(DeviceMem<T> device_mem_array,
                                                  int print_out = 0);
  std::unique_ptr<Network> network;

  sycl::queue sycl_queue;

  DeviceMem<bf16> input;
  DeviceMem<bf16> input_backward;
  DeviceMem<bf16> grads;
  DeviceMem<float> output;
  DeviceMem<bf16> deltas;

  int forward_size;
  float *forward;

  float *A_forward;
  float *B_forward;
  float *C_forward;

  float *out_inter;
  float *delta_temp;
  float *A_backward;
  float *B_backward;
  float *C_backward;
  float *A_backward_last_layer;
  float *B_backward_last_layer;
  float *C_backward_last_layer;
  float *D_backward_last_layer;
  float *E_backward_last_layer;
  float *F_backward_last_layer;
  float *A_dgemm;
  float *B_dgemm;
  float *C_dgemm;
};

tnn::SwiftNetModule *create_network(const int width, int input_width,
                                    int output_width, int n_hidden_layers,
                                    Activation activation,
                                    Activation output_activation,
                                    const int batch_size,
                                    std::string device_name);

}  // namespace tnn
