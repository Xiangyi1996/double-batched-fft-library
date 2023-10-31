#include "tnn_api.h"

#include "Encodings/identity.h"

namespace tnn {

EncodingModule::EncodingModule(
    int input_width, std::string encoding_name,
    const std::unordered_map<std::string, std::string> &encoding_config,
    std::string device_name)
    : Module(device_name), m_input_width(input_width) {
  encoding = create_encoding<float>(encoding_name, encoding_config);
  sycl_queue = sycl::queue();
}

void EncodingModule::initialize_params(float *params_full_precision,
                                       int use_easy) {
  encoding->initialize_params();
}

torch::Tensor EncodingModule::forward_pass(torch::Tensor input_tensor,
                                           torch::Tensor params,
                                           int use_inference) {
  //   assert(input_tensor.sizes() == 2 &&
  //          "Tensor length for Encoding forward is not equal to 2!");
  //   std::cout << "Input tensor sizes: " << input_tensor.sizes() << std::endl;
  int batch_size = input_tensor.sizes()[1];
  GPUMatrix<float> input_matrix = GPUMatrix<float>(
      input_tensor.data_ptr<float>(), m_input_width, batch_size);

  torch::Tensor output_tensor = torch::empty(
      {encoding->output_width(), batch_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(m_device));
  GPUMatrix<float> output_matrix = GPUMatrix<float>(
      output_tensor.data_ptr<float>(), m_input_width, batch_size);

  std::unique_ptr<Context> model_ctx =
      encoding->forward_impl(&sycl_queue, input_matrix, &output_matrix);

  return output_tensor;
}

torch::Tensor EncodingModule::backward_pass(torch::Tensor input_tensor,
                                            torch::Tensor grad_output,
                                            torch::Tensor params) {}

void EncodingModule::free_memory() {}

NetworkWithEncodingModule::NetworkWithEncodingModule(
    int width, int input_width, int output_width, int n_hidden_layers,
    Activation activation, Activation output_activation,
    std::string encoding_name,
    const std::unordered_map<std::string, std::string> &encoding_config,
    std::string device_name)
    : Module(device_name),
      m_input_width(input_width),
      m_output_width(output_width),
      m_width(width),
      m_n_hidden_layers(n_hidden_layers) {
  // grid encoding only works with float/not with bf16, for simplicity, always
  // using float, otherwise, we'd need another factory...
  network = new NetworkWithEncoding(input_width, output_width, n_hidden_layers,
                                    activation, output_activation,
                                    encoding_name, encoding_config);
}

torch::Tensor NetworkWithEncodingModule::forward_pass(
    torch::Tensor input_tensor, torch::Tensor params, int use_inference) {
  set_params(params);
  int batch_size = input_tensor.sizes()[1];

  sycl::queue q;
  forward = malloc_device<float>(batch_size * (m_input_width + m_output_width +
                                               m_width * m_n_hidden_layers),
                                 sycl_queue);
  DeviceMem<float> network_output =
      DeviceMem<float>(m_output_width * batch_size, sycl_queue);

  GPUMatrix<float> input_matrix = GPUMatrix<float>(
      input_tensor.data_ptr<float>(), m_input_width, batch_size);
  network->forward_pass(input_matrix, use_inference, network_output, forward);
  torch::Tensor output_tensor =
      get_converted_tensor_from_dev_mem(network_output);

  return output_tensor;
}

torch::Tensor NetworkWithEncodingModule::backward_pass(
    torch::Tensor input_tensor, torch::Tensor grad_output,
    torch::Tensor params) {
  int batch_size = input_tensor.sizes()[1];

  DeviceMem<bf16> input_backward =
      DeviceMem<bf16>(batch_size * m_input_width, sycl_queue);
  DeviceMem<bf16> grads =
      DeviceMem<bf16>(batch_size * m_output_width, sycl_queue);

  convert_tensor_to_dev_mem(grad_output, &grads);
  convert_tensor_to_dev_mem(input_tensor, &input_backward);

  set_params(params);

  DeviceMem<bf16> *grads_matrices =
      network->backward_pass(input_backward, grads, forward, batch_size);

  torch::Tensor grad_loss = get_converted_tensor_from_dev_mem(*grads_matrices);

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
  return network->get_network()->get_weights_matrices()->size();
}

}  // namespace tnn
