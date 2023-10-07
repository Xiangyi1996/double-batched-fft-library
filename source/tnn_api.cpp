#include "tnn_api.h"

#include "Encodings/identity.h"

namespace tnn {

SwiftNetModule::SwiftNetModule(const int width, int input_width,
                               int output_width, int n_hidden_layers,
                               Activation activation,
                               Activation output_activation,
                               const int batch_size, std::string device_name)
    : Module(device_name) {
  sycl_queue = sycl::queue();

  network = SwiftNetMLPFactory::create(
      sycl_queue, width, input_width, output_width, n_hidden_layers, activation,
      output_activation, batch_size);

  input_bf16 = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  input_backward = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  output = DeviceMem<float>(batch_size * output_width, sycl_queue);
  deltas = DeviceMem<bf16>(batch_size * output_width, sycl_queue);
  grads = DeviceMem<bf16>(batch_size * output_width, sycl_queue);

  input_bf16.initialize_constant(0.1f, sycl_queue);
  input_backward.initialize_constant(0.1f, sycl_queue);
  output.initialize_constant(0.0f, sycl_queue);
  grads.initialize_constant(bf16(0.0f), sycl_queue);
}

void SwiftNetModule::set_params(torch::Tensor &params) {
  if (m_device_name == "cpu") {
    std::vector<bf16> params_bf16 = get_vector_from_tensor<bf16>(params);
    network->set_params(params_bf16);
  } else if (m_device_name == "xpu") {
    float *tensor_data = params.data_ptr<float>();
    network->set_params(tensor_data);
  } else {
    std::cout << "No behaviour for device " << m_device_name
              << ". Exiting code." << std::endl;
    exit(1);
  }
}

torch::Tensor SwiftNetModule::forward_pass(torch::Tensor input_tensor,
                                           torch::Tensor params,
                                           int use_inference) {
  //   std::cout << "Input tensor: " << input_tensor << std::endl;
  set_params(params);
  set_input(input_tensor, &input_bf16);
  forward_pass(use_inference);
  return get_converted_tensor_from_dev_mem(&output);
}

void SwiftNetModule::forward_pass(int use_inference) {
  if (use_inference) {
    // Calling forward pass of Swiftnet
    network->inference(input_bf16, network->m_forward, network->m_A_forward,
                       network->m_B_forward, network->m_C_forward, output);

  } else {
    // Calling forward pass of Swiftnet
    network->forward_pass(input_bf16, network->m_forward, network->m_A_forward,
                          network->m_B_forward, network->m_C_forward, output);
  }
}

torch::Tensor SwiftNetModule::backward_pass(torch::Tensor input_tensor,
                                            torch::Tensor grad_output,
                                            torch::Tensor params) {
  //   std::cout << "Params " << params << std::endl;

  convert_tensor_to_dev_mem(grad_output, &grads);
  convert_tensor_to_dev_mem(input_tensor, &input_backward);

  std::vector<bf16> params_bf16 = get_vector_from_tensor<bf16>(params);
  network->set_params(params_bf16);

  network->backward_pass(
      input_backward, grads, network->m_out_inter, network->m_deltas,
      network->m_A_backward, network->m_B_backward, network->m_C_backward,
      network->m_A_backward_last_layer, network->m_B_backward_last_layer,
      network->m_C_backward_last_layer, network->m_D_backward_last_layer,
      network->m_E_backward_last_layer, network->m_F_backward_last_layer,
      network->m_A_dgemm, network->m_B_dgemm, network->m_C_dgemm,
      network->m_forward);

  DeviceMem<bf16> *grads_matrices = (network->get_grads_matrices());

  torch::Tensor grad_loss =
      get_converted_tensor_from_dev_mem(grads_matrices, 0);
  //   std::cout << "Tensor grad_loss: " << grad_loss << std::endl;
  //   std::cout << "Flush the printfs on kernel" << std::endl;
  return grad_loss;
}

void SwiftNetModule::initialize_params(float *params_full_precision,
                                       int use_easy) {
  network->initialize_params(use_easy);
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
  input_bf16.free_mem(sycl_queue);
  input_backward.free_mem(sycl_queue);
  output.free_mem(sycl_queue);
  deltas.free_mem(sycl_queue);
  grads.free_mem(sycl_queue);
  network->free_mem(sycl_queue);
}

// EncodingModule::EncodingModule(
//     int input_width, std::string encoding_name,
//     const std::unordered_map<std::string, std::string> &encoding_config,
//     std::string device_name)
//     : Module(device_name) {
//   sycl_queue = sycl::queue();

//   input = DeviceMem<float>(
//       batch_size * input_width,
//       sycl_queue);  // using this instead of directly in GPUMatrix because we
//                     // pass DeviceMems around for NetworkWithEncoding
//   output = DeviceMem<bf16>(batch_size * output_width, sycl_queue);
//   target = DeviceMem<bf16>(batch_size * output_width, sycl_queue);

//   input.initialize_constant(0.01f, sycl_queue);
//   output.initialize_constant(0.0f, sycl_queue);
//   target.initialize_constant(1.0f, sycl_queue);

//   input_matrix = GPUMatrix<float>(input.data(), batch_size * input_width,
//                                   batch_size * output_width);
//   output_matrix = GPUMatrix<bf16>(output.data(), batch_size * input_width,
//                                   batch_size * output_width);
//   target_matrix = GPUMatrix<bf16>(target.data(), batch_size * input_width,
//                                   batch_size * output_width);

//   encoding = new IdentityEncoding<bf16>(input_width, scale, offset);
// }

// void EncodingModule::initialize_params(float *params_full_precision,
//                                        int use_easy) {
//   encoding->initialize_params();
// }

// torch::Tensor EncodingModule::forward_pass(torch::Tensor input_tensor,
//                                            torch::Tensor params,
//                                            int use_inference) {
//   set_input<bf16>(input_tensor);
//   std::unique_ptr<Context> model_ctx =
//       encoding->forward_impl(&sycl_queue, input_matrix, &output_matrix);

//   torch::Tensor output_tensor = get_converted_tensor_from_dev_mem(output);

//   return output_tensor;
// }

// torch::Tensor EncodingModule::backward_pass(torch::Tensor input_tensor,
//                                             torch::Tensor grad_output,
//                                             torch::Tensor params) {}

// void EncodingModule::free_memory() {}
// EncodingModule *create_encoding_module(int input_width, int batch_size,
//                                        int output_width, int scale, int
//                                        offset, std::string device_name) {
//   return new EncodingModule(input_width, batch_size, output_width, scale,
//                             offset, device_name);
// }

NetworkWithEncodingModule::NetworkWithEncodingModule(
    int width, int input_width, int output_width, int n_hidden_layers,
    Activation activation, Activation output_activation, const int batch_size,
    std::string encoding_name,
    const std::unordered_map<std::string, std::string> &encoding_config,
    std::string device_name)
    : Module(device_name) {
  //   std::cout << "Input width: " << input_width
  //             << ", output_width: " << output_width
  //             << ", batch_size: " << batch_size
  //             << ", encoding name: " << encoding_name << std::endl;
  network = new NetworkWithEncoding(input_width, output_width, n_hidden_layers,
                                    activation, output_activation, batch_size,
                                    encoding_name, encoding_config);

  input_float = DeviceMem<float>(batch_size * input_width, sycl_queue);
  input_backward = DeviceMem<bf16>(batch_size * input_width, sycl_queue);
  grads = DeviceMem<bf16>(batch_size * output_width, sycl_queue);

  //   std::cout << "Input devicemem: " << input_float.size() << std::endl;
  // using this instead of directly in GPUMatrix because we
  // pass DeviceMems around for NetworkWithEncoding

  input_float.initialize_constant(0.01f, sycl_queue);

  input_matrix = GPUMatrix<float>(input_float.data(), input_width, batch_size);
}

torch::Tensor NetworkWithEncodingModule::forward_pass(
    torch::Tensor input_tensor, torch::Tensor params, int use_inference) {
  set_params(params);
  set_input(input_tensor,
            &input_float);  // sets input_tensor -> input, and
                            // input.data() is member of input_matrix

  network->forward_pass(input_matrix, use_inference);

  torch::Tensor output_tensor =
      get_converted_tensor_from_dev_mem(network->get_output());
  //   std::cout << "return: " << output_tensor return output_tensor;
  return output_tensor;
}

torch::Tensor NetworkWithEncodingModule::backward_pass(
    torch::Tensor input_tensor, torch::Tensor grad_output,
    torch::Tensor params) {
  convert_tensor_to_dev_mem(grad_output, &grads);
  convert_tensor_to_dev_mem(input_tensor, &input_backward);

  set_params(params);

  DeviceMem<bf16> *grads_matrices =
      network->backward_pass(input_backward, grads);

  torch::Tensor grad_loss = get_converted_tensor_from_dev_mem(grads_matrices);

  return grad_loss;
}

void NetworkWithEncodingModule::set_params(torch::Tensor &params) {
  if (m_device_name == "cpu") {
    std::vector<bf16> params_bf16 = get_vector_from_tensor<bf16>(params);
    network->set_params(params_bf16);
  } else if (m_device_name == "xpu") {
    float *tensor_data = params.data_ptr<float>();
    network->set_params(tensor_data);
  } else {
    std::cout << "No behaviour for device " << m_device_name
              << ". Exiting code." << std::endl;
    exit(1);
  }
}

void NetworkWithEncodingModule::initialize_params(float *params_full_precision,
                                                  int use_easy) {
  network->initialize_params(use_easy);
  std::vector<bf16> params_full_precision_list(
      network->get_network()->get_weights_matrices()->size());
  network->get_network()->get_weights_matrices()->copy_to_host(
      params_full_precision_list, sycl_queue);

  for (int i = 0; i < params_full_precision_list.size(); i++) {
    params_full_precision[i] = float(params_full_precision_list[i]);
  }
}

void NetworkWithEncodingModule::free_memory() { network->free_memory(); }

int NetworkWithEncodingModule::n_params() {
  std::cout << "N params: "
            << network->get_network()->get_weights_matrices()->size()
            << std::endl;
  return network->get_network()->get_weights_matrices()->size();
}

}  // namespace tnn
