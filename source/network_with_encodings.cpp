#include "network_with_encodings.h"

#define NETWORK_INPUT_WIDTH 64

NetworkWithEncoding::NetworkWithEncoding(
    int input_width, int output_width, int n_hidden_layers,
    Activation activation, Activation output_activation, const int batch_size,
    int encoding_scale, int encoding_offset) {
  m_q = sycl::queue();
  encoding =
      new IdentityEncoding<bf16>(input_width, encoding_scale, encoding_offset);
  encoding->set_padded_output_width(WIDTH);
  assert(encoding->padded_output_width() == WIDTH);
  network = new SwiftNetMLP<WIDTH>(m_q, encoding->padded_output_width(),
                                   output_width, n_hidden_layers, activation,
                                   output_activation, batch_size);

  network_input =
      DeviceMem<bf16>(encoding->padded_output_width() * batch_size, m_q);
  network_output = DeviceMem<float>(output_width * batch_size, m_q);
  encoding_output = GPUMatrix<bf16>(
      network_input.data(), encoding->padded_output_width(), batch_size);
}

DeviceMem<float>* NetworkWithEncoding::forward_pass(GPUMatrix<float>& input,
                                                    int run_inference) {
  encoding->forward_impl(&m_q, input, &encoding_output);

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

void NetworkWithEncoding::initialize_params(int use_easy) {
  network->initialize_params(use_easy);
  encoding->initialize_params();  // this is an empty call
}

void NetworkWithEncoding::free_memory() { network->free_mem(m_q); }