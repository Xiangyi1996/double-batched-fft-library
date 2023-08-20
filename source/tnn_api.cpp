#include "tnn_api.h"

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
                               const int batch_size) {
  sycl_queue = sycl::queue();

  network = SwiftNetMLPFactory::create(
      sycl_queue, width, input_width, output_width, n_hidden_layers, activation,
      output_activation, batch_size);
  const size_t alignment = 4096;

  const int n_hidden_matrices = n_hidden_layers - 1;
  const int layer_length = width * batch_size;

  int shmem_size = batch_size * width * 5;

  forward_size =
      batch_size * (width + output_width + width * n_hidden_matrices);
  forward = malloc_device<float>(forward_size, sycl_queue);

  input = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  input_backward = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  output = DeviceMem<float>(batch_size * output_width, sycl_queue);
  deltas = DeviceMem<bf16>(batch_size * output_width, sycl_queue);
  grads = DeviceMem<bf16>(batch_size * output_width, sycl_queue);

  input.initialize_constant(0.1f, sycl_queue);
  input_backward.initialize_constant(0.1f, sycl_queue);
  output.initialize_constant(0.0f, sycl_queue);
  grads.initialize_constant(bf16(0.0f), sycl_queue);

  A_forward =
      sycl::aligned_alloc_device<float>(alignment, layer_length, sycl_queue);
  B_forward = sycl::aligned_alloc_device<float>(alignment, output_width * 64,
                                                sycl_queue);
  C_forward = sycl::aligned_alloc_device<float>(
      alignment, output_width * batch_size, sycl_queue);

  out_inter = malloc_device<float>(batch_size * width * (n_hidden_matrices + 1),
                                   sycl_queue);
  delta_temp = sycl::aligned_alloc_device<float>(
      alignment, output_width * batch_size, sycl_queue);

  A_backward = sycl::aligned_alloc_device<float>(alignment, width * batch_size,
                                                 sycl_queue);
  B_backward = sycl::aligned_alloc_device<float>(
      alignment, batch_size * output_width, sycl_queue);
  C_backward = sycl::aligned_alloc_device<float>(
      alignment, width * output_width, sycl_queue);
  A_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, grads.size(), sycl_queue);
  B_backward_last_layer = sycl::aligned_alloc_device<float>(
      alignment, output_width * width, sycl_queue);
  C_backward_last_layer = sycl::aligned_alloc_device<float>(
      alignment, width * batch_size, sycl_queue);
  D_backward_last_layer = sycl::aligned_alloc_device<float>(
      alignment, width * batch_size, sycl_queue);
  E_backward_last_layer = sycl::aligned_alloc_device<float>(
      alignment, batch_size * width, sycl_queue);
  F_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, width * width, sycl_queue);

  A_dgemm = sycl::aligned_alloc_device<float>(alignment, batch_size * width,
                                              sycl_queue);
  B_dgemm = sycl::aligned_alloc_device<float>(alignment, batch_size * width,
                                              sycl_queue);
  C_dgemm =
      sycl::aligned_alloc_device<float>(alignment, width * width, sycl_queue);
}

std::vector<bf16> SwiftNetModule::get_vector_from_tensor(torch::Tensor tensor) {
  std::vector<bf16> array_bf16(tensor.numel());
  const float *tensor_data = tensor.data_ptr<float>();
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
    DeviceMem<T> device_mem_array) {
  // Conversion to float array for pybindings
  std::vector<T> list_T(device_mem_array.size());
  device_mem_array.copy_to_host(list_T, sycl_queue);

  // Convert the original vector to a std::vector<float>
  std::vector<float> list_float(list_T.size());
  for (size_t i = 0; i < list_T.size(); ++i) {
    list_float[i] = static_cast<float>(list_T[i]);
  }
  //   std::cout << "About to convert this " << std::endl;
  //   printVector(list_float);
  // convert to torch tensor
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
                                           torch::Tensor params) {
  convert_tensor_to_dev_mem(input_tensor, input);
  std::vector<bf16> params_bf16 = get_vector_from_tensor(params);
  network->set_params(params_bf16);

  // Calling forward pass of Swiftnet
  network->forward_pass(input, forward, A_forward, B_forward, C_forward,
                        output);

  torch::Tensor output_tensor = get_converted_tensor_from_dev_mem(output);

  return output_tensor;
}

torch::Tensor SwiftNetModule::backward_pass(torch::Tensor input_tensor,
                                            torch::Tensor grad_output,
                                            torch::Tensor params) {
  convert_tensor_to_dev_mem(grad_output, grads);
  convert_tensor_to_dev_mem(input_tensor, input_backward);

  std::vector<bf16> params_bf16 = get_vector_from_tensor(params);
  network->set_params(params_bf16);

  network->backward_pass(
      input_backward, grads, out_inter, delta_temp, deltas, A_backward,
      B_backward, C_backward, A_backward_last_layer, B_backward_last_layer,
      C_backward_last_layer, D_backward_last_layer, E_backward_last_layer,
      F_backward_last_layer, A_dgemm, B_dgemm, C_dgemm, forward);

  DeviceMem<bf16> grads_matrices = *(network->get_grads_matrices());

  torch::Tensor grad_loss = get_converted_tensor_from_dev_mem(grads_matrices);
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
  grads.free_mem(sycl_queue);

  free(forward, sycl_queue);
  free(A_forward, sycl_queue);
  free(B_forward, sycl_queue);
  free(C_forward, sycl_queue);

  free(out_inter, sycl_queue);
  free(delta_temp, sycl_queue);

  //   deltas.free_mem(q);

  free(A_backward, sycl_queue);
  free(B_backward, sycl_queue);
  free(C_backward, sycl_queue);

  free(A_backward_last_layer, sycl_queue);
  free(B_backward_last_layer, sycl_queue);
  free(C_backward_last_layer, sycl_queue);
  free(D_backward_last_layer, sycl_queue);
  free(E_backward_last_layer, sycl_queue);
  free(F_backward_last_layer, sycl_queue);

  free(A_dgemm, sycl_queue);
  free(B_dgemm, sycl_queue);
  free(C_dgemm, sycl_queue);
}

SwiftNetModule *create_network(const int width, int input_width,
                               int output_width, int n_hidden_layers,
                               Activation activation,
                               Activation output_activation,
                               const int batch_size) {
  return new SwiftNetModule(width, input_width, output_width, n_hidden_layers,
                            activation, output_activation, batch_size);
}
}  // namespace tnn