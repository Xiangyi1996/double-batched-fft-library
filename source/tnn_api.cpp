#include "tnn_api.h"

#include <torch/script.h>

template <typename T>
void printVector(const std::vector<T> &vec) {
  for (const T &element : vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

namespace tnn {

SwiftNetModule::SwiftNetModule(const int width, int input_width,
                               int output_width, int n_hidden_layers,
                               Activation activation,
                               Activation output_activation,
                               const int batch_size, std::string device_name)
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

  sycl_queue = sycl::queue();

  network = SwiftNetMLPFactory::create(
      sycl_queue, width, input_width, output_width, n_hidden_layers, activation,
      output_activation, batch_size);

  input = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  input_backward = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  output = DeviceMem<float>(batch_size * output_width, sycl_queue);
  deltas = DeviceMem<bf16>(batch_size * output_width, sycl_queue);
  grads = DeviceMem<bf16>(batch_size * output_width, sycl_queue);

  input.initialize_constant(0.1f, sycl_queue);
  input_backward.initialize_constant(0.1f, sycl_queue);
  output.initialize_constant(0.0f, sycl_queue);
  grads.initialize_constant(bf16(0.0f), sycl_queue);
}

std::vector<bf16> SwiftNetModule::get_vector_from_tensor(torch::Tensor tensor) {
  std::vector<bf16> array_bf16(tensor.numel());

  float *tensor_data = tensor.data_ptr<float>();
  for (int i = 0; i < tensor.numel(); ++i) {
    array_bf16[i] = bf16(tensor_data[i]);
  }
  return array_bf16;
}

void SwiftNetModule::convert_tensor_to_dev_mem(
    torch::Tensor tensor, DeviceMem<bf16> device_mem_array) {
  std::vector<bf16> array_bf16 = get_vector_from_tensor(tensor);
  if (array_bf16.size() != device_mem_array.size()) {
    std::cerr
        << "Assertion failed: array_bf16.size() == device_mem_array.size()\n"
        << "array_bf16.size(): " << array_bf16.size() << "\n"
        << "device_mem_array.size(): " << device_mem_array.size() << std::endl;
    exit(1);
  }  // conversion to DeviceMem required by Swiftnet forward_pass

  // copy array_bf16 to device_mem_array
  device_mem_array.copy_from_host(array_bf16, sycl_queue);
}

template <typename T>
torch::Tensor SwiftNetModule::get_converted_tensor_from_dev_mem(
    DeviceMem<T> device_mem_array, int print_out) {
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
      torch::from_blob(list_float.data(),
                       {static_cast<long>(list_float.size())}, torch::kFloat32)
          .clone();
  return output_tensor;
}

template <typename T>
torch::Tensor SwiftNetModule::get_converted_tensor_from_array(T *array,
                                                              int size) {
  torch::Tensor tensor = torch::from_blob(array, {size}, torch::kFloat32);
  return tensor;
}

torch::Tensor SwiftNetModule::forward_pass(torch::Tensor input_tensor,
                                           torch::Tensor params,
                                           int use_inference = 0) {
  //   std::cout << "Input tensor: " << input_tensor << std::endl;
  if (m_device_name == "cpu") {
    std::vector<bf16> params_bf16 = get_vector_from_tensor(params);
    network->set_params(params_bf16);
    convert_tensor_to_dev_mem(input_tensor, input);
  } else if (m_device_name == "xpu") {
    float *tensor_data = params.data_ptr<float>();
    network->set_params(tensor_data);
    // TODO: refactor and make own function
    // set the DeviceMem input vals to the ones from input_tensor. Because it's
    // on device, we need to sycl_queue it
    float *input_data = input_tensor.data_ptr<float>();
    auto p = input.data();
    int s = input.size();
    sycl_queue
        .parallel_for<>(range<1>(s),
                        [=](id<1> idx) { p[idx] = bf16(input_data[idx]); })
        .wait();
  } else {
    std::cout << "No behaviour for device " << m_device_name
              << ". Exiting code." << std::endl;
    exit(1);
  }
  if (use_inference) {
    // Calling forward pass of Swiftnet
    network->inference(input, network->m_forward, network->m_A_forward,
                       network->m_B_forward, network->m_C_forward, output);

  } else {
    // Calling forward pass of Swiftnet
    network->forward_pass(input, network->m_forward, network->m_A_forward,
                          network->m_B_forward, network->m_C_forward, output);
  }

  torch::Tensor output_tensor = get_converted_tensor_from_dev_mem(output);

  return output_tensor;
}

torch::Tensor SwiftNetModule::backward_pass(torch::Tensor input_tensor,
                                            torch::Tensor grad_output,
                                            torch::Tensor params) {
  //   std::cout << "Grad output " << grad_output << std::endl;
  convert_tensor_to_dev_mem(grad_output, grads);
  convert_tensor_to_dev_mem(input_tensor, input_backward);

  std::vector<bf16> params_bf16 = get_vector_from_tensor(params);
  network->set_params(params_bf16);

  network->backward_pass(
      input_backward, grads, network->m_out_inter, network->m_deltas_temp,
      network->m_deltas, network->m_A_backward, network->m_B_backward,
      network->m_C_backward, network->m_A_backward_last_layer,
      network->m_B_backward_last_layer, network->m_C_backward_last_layer,
      network->m_D_backward_last_layer, network->m_E_backward_last_layer,
      network->m_F_backward_last_layer, network->m_A_dgemm, network->m_B_dgemm,
      network->m_C_dgemm, network->m_forward);

  DeviceMem<bf16> grads_matrices = *(network->get_grads_matrices());

  torch::Tensor grad_loss =
      get_converted_tensor_from_dev_mem(grads_matrices, 0);
  //   std::cout << "Tensor grad_loss: " << grad_loss << std::endl;
  return grad_loss;
}

void SwiftNetModule::initialize_params(float *params_full_precision) {
  network->initialize_params();
  std::vector<bf16> params_full_precision_list(
      network->m_weights_matrices.size());
  network->m_weights_matrices.copy_to_host(params_full_precision_list,
                                           sycl_queue);

  for (int i = 0; i < params_full_precision_list.size(); i++) {
    params_full_precision[i] = float(params_full_precision_list[i]);
  }
}

int SwiftNetModule::n_params() {
  return network->get_weights_matrices()->size();
}

void SwiftNetModule::free_memory() {
  input.free_mem(sycl_queue);
  input_backward.free_mem(sycl_queue);
  output.free_mem(sycl_queue);
  deltas.free_mem(sycl_queue);
  grads.free_mem(sycl_queue);
  network->free_mem(sycl_queue);
}

SwiftNetModule *create_network(const int width, int input_width,
                               int output_width, int n_hidden_layers,
                               Activation activation,
                               Activation output_activation,
                               const int batch_size, std::string device_name) {
  return new SwiftNetModule(width, input_width, output_width, n_hidden_layers,
                            activation, output_activation, batch_size,
                            device_name);
}
}  // namespace tnn