#pragma once  // Use pragma once for include guards

#include <torch/extension.h>
#include <torch/script.h>

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>  // Include the necessary header for std::string
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "encoding.h"
#include "network_with_encodings.h"
#include "oneapi/mkl.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T>
void printVector(const std::vector<T>& vec) {
  for (const T& element : vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

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

class Module {
 public:
  Module(std::string device_name)
      : m_device(torch::kCPU), m_device_name(device_name) {
    std::cout << "Running on device: " << m_device_name << std::endl;
    if (m_device_name == "cpu") {
      m_device = torch::kCPU;
    } else if (m_device_name == "xpu") {
      m_device = torch::kXPU;
    } else {
      std::cout << "No device name " << device_name
                << ". Consider falling back to CPU as device. Exiting now"
                << std::endl;
      exit(1);
    }
  }
  virtual ~Module() {}

  virtual torch::Tensor forward_pass(torch::Tensor input_list,
                                     torch::Tensor params,
                                     int use_inference = 0) = 0;

  virtual void forward_pass(int use_inference) = 0;
  virtual torch::Tensor backward_pass(torch::Tensor input_tensor,
                                      torch::Tensor grad_output,
                                      torch::Tensor params) = 0;
  virtual void initialize_params(float* params_full_precision,
                                 int use_easy = 0) = 0;
  virtual void free_memory() = 0;
  virtual int n_params() = 0;

 protected:
  torch::Device m_device;
  std::string m_device_name;

  sycl::queue sycl_queue;

  template <typename T>
  void set_input(torch::Tensor& input_tensor, DeviceMem<T>* input_device_mem) {
    if (m_device_name == "cpu") {
      convert_tensor_to_dev_mem(input_tensor, input_device_mem);
    } else if (m_device_name == "xpu") {
      float* input_data = input_tensor.data_ptr<float>();
      auto p = input_device_mem->data();
      int s = input_device_mem->size();
      sycl_queue
          .parallel_for<>(range<1>(s),
                          [=](id<1> idx) { p[idx] = T(input_data[idx]); })
          .wait();
    } else {
      std::cout << "No behaviour for device " << m_device_name
                << ". Exiting code." << std::endl;
      exit(1);
    }
  }

  template <typename T>
  std::vector<T> get_vector_from_tensor(torch::Tensor tensor) {
    static_assert(std::is_same<T, bf16>::value || std::is_same<T, float>::value,
                  "get_vector_from_tensor only accepts bf16 or float types.");

    std::vector<T> array(tensor.numel());

    float* tensor_data = tensor.data_ptr<float>();
    for (int i = 0; i < tensor.numel(); ++i) {
      if (std::is_same<T, bf16>::value) {
        array[i] = bf16(tensor_data[i]);
      } else {
        array[i] = tensor_data[i];
      }
    }
    return array;
  }

  template <typename T>
  void convert_tensor_to_dev_mem(torch::Tensor& tensor,
                                 DeviceMem<T>* device_mem_array) {
    std::vector<T> array = get_vector_from_tensor<T>(tensor);
    if (array.size() != device_mem_array->size()) {
      std::cerr << "Assertion failed: array.size() == device_mem_array.size()\n"
                << "array.size(): " << array.size() << "\n"
                << "device_mem_array.size(): " << device_mem_array->size()
                << std::endl;
      exit(1);
    }  // conversion to DeviceMem required by Swiftnet forward_pass

    // copy array to device_mem_array
    device_mem_array->copy_from_host(array, sycl_queue);
  }

  template <typename T>
  torch::Tensor get_converted_tensor_from_dev_mem(
      DeviceMem<T>* device_mem_array, int print_out = 0) {
    // Conversion to float array for pybindings
    std::vector<T> list_T(device_mem_array->size());
    device_mem_array->copy_to_host(list_T, sycl_queue);

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
        torch::from_blob(list_float.data(),
                         {static_cast<long>(list_float.size())},
                         torch::kFloat32)
            .clone();
    return output_tensor;
  }

  template <typename T>
  torch::Tensor get_converted_tensor_from_array(T* array, int size) {
    torch::Tensor tensor = torch::from_blob(array, {size}, torch::kFloat32);
    return tensor;
  }
  DeviceMem<float> input_float;
};

class SwiftNetModule : public Module {
 public:
  SwiftNetModule(const int width, int input_width, int output_width,
                 int n_hidden_layers, Activation activation,
                 Activation output_activation, const int batch_size,
                 std::string device_name);

  torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params,
                             int use_inference = 0) override;
  void forward_pass(int use_inference) override;

  torch::Tensor backward_pass(torch::Tensor input_tensor,
                              torch::Tensor grad_output,
                              torch::Tensor params) override;
  void initialize_params(float* params_full_precision,
                         int use_easy = 0) override;
  void free_memory() override;
  int n_params() override;

  void set_params(torch::Tensor& params);

  DeviceMem<float> output;

 private:
  std::unique_ptr<Network> network;

  DeviceMem<bf16> input_bf16;
  DeviceMem<bf16> input_backward;
  DeviceMem<bf16> grads;
  DeviceMem<bf16> deltas;

  int forward_size;
  float* forward;

  float* A_forward;
  float* B_forward;
  float* C_forward;

  float* out_inter;
  float* delta_temp;
  float* A_backward;
  float* B_backward;
  float* C_backward;
  float* A_backward_last_layer;
  float* B_backward_last_layer;
  float* C_backward_last_layer;
  float* D_backward_last_layer;
  float* E_backward_last_layer;
  float* F_backward_last_layer;
  float* A_dgemm;
  float* B_dgemm;
  float* C_dgemm;
};

class EncodingModule : public Module {
 public:
  EncodingModule(int input_width, int batch_size, int output_width, int scale,
                 int offset, std::string device_name);
  ~EncodingModule() {}

  torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params,
                             int use_inference = 0) override;

  void forward_pass(int use_inference);

  torch::Tensor backward_pass(torch::Tensor input_tensor,
                              torch::Tensor grad_output,
                              torch::Tensor params) override;
  void initialize_params(float* params_full_precision,
                         int use_easy = 0) override;
  void free_memory() override;
  int n_params() override{};

  GPUMatrix<bf16> output_matrix;

 private:
  torch::Tensor forward_impl(int use_inference);

  Encoding<bf16>* encoding;

  DeviceMem<bf16> output;
  DeviceMem<bf16> target;
  GPUMatrix<float> input_matrix;
  GPUMatrix<bf16> target_matrix;
};

class NetworkWithEncodingModule : public Module {
 public:
  NetworkWithEncodingModule(
      int width, int input_width, int output_width, int n_hidden_layers,
      Activation activation, Activation output_activation, const int batch_size,
      std::string encoding_name,
      const std::unordered_map<std::string, std::string>& encoding_config,
      std::string device_name);
  ~NetworkWithEncodingModule() {}

  torch::Tensor forward_pass(torch::Tensor input_list, torch::Tensor params,
                             int use_inference = 0) override;
  void forward_pass(int use_inference) override{};

  torch::Tensor backward_pass(torch::Tensor input_tensor,
                              torch::Tensor grad_output,
                              torch::Tensor params) override;
  void initialize_params(float* params_full_precision,
                         int use_easy = 0) override;
  void free_memory() override;
  int n_params() override;
  void set_params(torch::Tensor& params);

 private:
  NetworkWithEncoding* network;
  GPUMatrix<float> input_matrix;

  // for backward pass
  DeviceMem<bf16> input_backward;
  DeviceMem<bf16> grads;
};

}  // namespace tnn
