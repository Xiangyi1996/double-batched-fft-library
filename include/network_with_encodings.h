#pragma once
#include "DeviceMem.h"
#include "Network.h"
#include "encoding_factory.h"

class NetworkWithEncoding {
  public:
    NetworkWithEncoding(int input_width, int output_width, int n_hidden_layers, Activation activation,
                        Activation output_activation, std::string encoding_name,
                        const std::unordered_map<std::string, std::string> &encoding_config);
    ~NetworkWithEncoding() {}

    void forward_pass(GPUMatrix<float> &input, int run_inference, DeviceMem<bf16> &network_output, float *forward);

    DeviceMem<bf16> *backward_pass(DeviceMem<bf16> &input_backward, DeviceMem<bf16> &grad_output, float *forward,
                                   int batch_size);

    void initialize_params(int use_easy = 0);

    void free_memory();
    sycl::queue get_queue() { return m_q; }

    void set_params(float *params);

    void set_params(std::vector<bf16> params);

    Network *get_network() { return network; }
    //   DeviceMem<float>* get_output() { return &network_output; }

  private:
    int m_encoding_output_width;

    //   Encoding<float>* encoding_grid;
    Encoding<float> *encoding;

    Network *network;
    queue m_q;
};

NetworkWithEncoding *create_networkwith_encoding(int input_width, int output_width, int n_hidden_layers,
                                                 Activation activation, Activation output_activation,
                                                 const int batch_size, std::string encoding_name,
                                                 const std::unordered_map<std::string, std::string> &encoding_config);
