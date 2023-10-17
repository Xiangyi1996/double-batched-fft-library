#include "network_with_encodings.h"

DeviceMem<float>* NetworkWithEncoding::forward_pass(GPUMatrix<float>& input,
                                                    int run_inference) {
  //   std::cout << "Input" << std::endl;
  //   input.print();
  encoding->forward_impl(&m_q, input, &encoding_output);
  //   std::cout << "Output encoding: " << std::endl;
  //   encoding_output.print();
  network_input.set_values(encoding_output.n_elements(), encoding_output.data(),
                           m_q);
  if (run_inference) {
    network->inference(network_input, network->m_forward, network->m_A_forward,
                       network->m_B_forward, network->m_C_forward,
                       network_output);
  } else {
    network->forward_pass(network_input, network->m_forward,
                          network->m_A_forward, network->m_B_forward,
                          network->m_C_forward, network_output);
  }
  return &network_output;
}

DeviceMem<bf16>* NetworkWithEncoding::backward_pass(
    DeviceMem<bf16>& input_backward, DeviceMem<bf16>& grad_output) {
  // no encoding bwd, as their gradients are handled individually

  network->backward_pass(
      input_backward, grad_output, network->m_out_inter, network->m_deltas,
      network->m_A_backward, network->m_B_backward, network->m_C_backward,
      network->m_A_backward_last_layer, network->m_B_backward_last_layer,
      network->m_C_backward_last_layer, network->m_D_backward_last_layer,
      network->m_E_backward_last_layer, network->m_F_backward_last_layer,
      network->m_A_dgemm, network->m_B_dgemm, network->m_C_dgemm,
      network->m_forward);

  return (network->get_grads_matrices());
}

void NetworkWithEncoding::set_params(float* params) {  // for xpu
  network->set_params(params);
}

void NetworkWithEncoding::set_params(std::vector<bf16> params) {  // for cpu
  network->set_params(params);
}

void NetworkWithEncoding::initialize_params(int use_easy) {
  network->initialize_params(use_easy);
  encoding->initialize_params();  // this is an empty call
}

void NetworkWithEncoding::free_memory() { network->free_mem(m_q); }

NetworkWithEncoding* create_networkwith_encoding(
    int input_width, int output_width, int n_hidden_layers,
    Activation activation, Activation output_activation, const int batch_size,
    std::string encoding_name,
    const std::unordered_map<std::string, std::string>& encoding_config) {
  return new NetworkWithEncoding(input_width, output_width, n_hidden_layers,
                                 activation, output_activation, batch_size,
                                 encoding_name, encoding_config);
}