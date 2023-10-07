#pragma once
#include "DeviceMem.h"
#include "SwiftNetMLP.h"
#include "encoding_factory.h"
#define WIDTH 64

class NetworkWithEncoding {
 public:
  NetworkWithEncoding(
      int input_width, int output_width, int n_hidden_layers,
      Activation activation, Activation output_activation, const int batch_size,
      std::string encoding_name,
      const std::unordered_map<std::string, std::string>& encoding_config) {
    m_q = sycl::queue();
    encoding =
        create_encoding<bf16>(input_width, encoding_name, encoding_config);
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

  ~NetworkWithEncoding() {}

  DeviceMem<float>* forward_pass(GPUMatrix<float>& input,
                                 int use_inference = 0);

  DeviceMem<bf16>* backward_pass(DeviceMem<bf16>& input_backward,
                                 DeviceMem<bf16>& grad_output);

  void initialize_params(int use_easy = 0);

  void free_memory();
  sycl::queue get_queue() { return m_q; }

  void set_params(float* params);

  void set_params(std::vector<bf16> params);

  Network* get_network() { return network; }
  DeviceMem<float>* get_output() { return &network_output; }

 private:
  Encoding<bf16>* encoding;
  Network* network;
  queue m_q;
  GPUMatrix<bf16> encoding_output;
  DeviceMem<float> network_output;

  DeviceMem<bf16> network_input;
};

NetworkWithEncoding* create_networkwith_encoding(
    int input_width, int output_width, int n_hidden_layers,
    Activation activation, Activation output_activation, const int batch_size,
    std::string encoding_name,
    const std::unordered_map<std::string, std::string>& encoding_config);
