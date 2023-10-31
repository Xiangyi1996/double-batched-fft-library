#include "network_with_encodings.h"

void NetworkWithEncoding::forward_pass(GPUMatrix<float>& input,
                                       int run_inference,
                                       DeviceMem<float>& network_output,
                                       float* forward) {
  int batch_size = input.n();

  //   std::cout << "Input" << std::endl;
  //   input.print();
  // Declare all memory here. Bit ugly, but no other way
  int net_width = network->get_net_width();
  int input_width = network->get_inputs_width();
  int n_hidden_layers = network->get_n_hidden_layers();
  int output_width = network->get_output_width();
  int m_alignment = network->m_alignment;

  // Allocate and initialize various memory buffers

  float* A_forward = sycl::aligned_alloc_device<float>(
      m_alignment, input_width * net_width, m_q);
  float* B_forward = sycl::aligned_alloc_device<float>(
      m_alignment, output_width * net_width, m_q);
  float* C_forward = sycl::aligned_alloc_device<float>(
      m_alignment, output_width * batch_size, m_q);

  std::cout << "FWD pass in network with ecndoings.cpp Batch size: "
            << batch_size << std::endl;

  DeviceMem<bf16> network_input =
      DeviceMem<bf16>(m_encoding_output_width * batch_size, m_q);
  GPUMatrix<float> encoding_output =
      GPUMatrix<float>(m_encoding_output_width, batch_size);

  encoding->forward_impl(&m_q, input, &encoding_output);
  //   std::cout << "Output encoding: " << std::endl;
  //   encoding_output.print();
  network_input.set_values(encoding_output.n_elements(), encoding_output.data(),
                           m_q);
  if (run_inference) {
    network->inference(network_input, forward, A_forward, B_forward, C_forward,
                       network_output, batch_size);
  } else {
    network->forward_pass(network_input, forward, A_forward, B_forward,
                          C_forward, network_output, batch_size);
  }

  //   free(forward, m_q);
  free(A_forward, m_q);
  free(B_forward, m_q);
  free(C_forward, m_q);
}

DeviceMem<bf16>* NetworkWithEncoding::backward_pass(
    DeviceMem<bf16>& input_backward, DeviceMem<bf16>& grad_output,
    float* forward, int batch_size) {
  // no encoding bwd, as their gradients are handled individually
  int input_width = network->get_inputs_width();
  int net_width = network->get_net_width();
  int n_hidden_matrices = network->get_n_hidden_matrices();
  int m_alignment = network->m_alignment;
  int output_width = network->get_output_width();

  float* out_inter =
      malloc_device<float>(batch_size * net_width * (n_hidden_matrices), m_q);

  DeviceMem<bf16> deltas;
  deltas.allocate2(net_width * batch_size, m_q);

  float* A_backward = sycl::aligned_alloc_device<float>(
      m_alignment, net_width * batch_size, m_q);
  float* B_backward = sycl::aligned_alloc_device<float>(
      m_alignment, batch_size * output_width, m_q);
  float* C_backward = sycl::aligned_alloc_device<float>(
      m_alignment, net_width * output_width, m_q);

  float* A_backward_last_layer = sycl::aligned_alloc_device<float>(
      m_alignment, batch_size * output_width, m_q);
  float* B_backward_last_layer = sycl::aligned_alloc_device<float>(
      m_alignment, output_width * net_width, m_q);
  float* C_backward_last_layer = sycl::aligned_alloc_device<float>(
      m_alignment, net_width * batch_size, m_q);
  float* D_backward_last_layer = sycl::aligned_alloc_device<float>(
      m_alignment, net_width * batch_size, m_q);
  float* E_backward_last_layer = sycl::aligned_alloc_device<float>(
      m_alignment, batch_size * net_width, m_q);

  float* F_backward_last_layer;
  if (n_hidden_matrices == 0) {
    // in this case, the penultimate layer is the input layer
    F_backward_last_layer = sycl::aligned_alloc_device<float>(
        m_alignment, input_width * net_width, m_q);
  } else {
    F_backward_last_layer = sycl::aligned_alloc_device<float>(
        m_alignment, net_width * net_width, m_q);
  }

  float* A_dgemm = sycl::aligned_alloc_device<float>(
      m_alignment, batch_size * net_width, m_q);
  float* B_dgemm = sycl::aligned_alloc_device<float>(
      m_alignment, batch_size * net_width, m_q);
  // net_width * net_width is the maximum, for the first layer, it's
  // technically input_width * net_width
  float* C_dgemm = sycl::aligned_alloc_device<float>(
      m_alignment, net_width * net_width, m_q);

  network->backward_pass(
      input_backward, grad_output, out_inter, deltas, A_backward, B_backward,
      C_backward, A_backward_last_layer, B_backward_last_layer,
      C_backward_last_layer, D_backward_last_layer, E_backward_last_layer,
      F_backward_last_layer, A_dgemm, B_dgemm, C_dgemm, forward, batch_size);

  //   Free memory for DeviceMem<bf16> arrays using their free_mem member
  //   function
  deltas.free_mem(m_q);

  free(A_backward, m_q);
  free(B_backward, m_q);
  free(C_backward, m_q);
  free(A_backward_last_layer, m_q);
  free(B_backward_last_layer, m_q);
  free(C_backward_last_layer, m_q);
  free(D_backward_last_layer, m_q);
  free(E_backward_last_layer, m_q);
  free(F_backward_last_layer, m_q);
  free(A_dgemm, m_q);
  free(B_dgemm, m_q);
  free(C_dgemm, m_q);
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
                                 activation, output_activation, encoding_name,
                                 encoding_config);
}