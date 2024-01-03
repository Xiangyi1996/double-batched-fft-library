#pragma once
#include "DeviceMem.h"
#include "Network.h"
#include "encoding_factory.h"

class NetworkWithEncoding {
  public:
    NetworkWithEncoding(const int input_width, const int output_width, const int n_hidden_layers, Activation activation,
                        Activation output_activation, const std::string& encoding_name,
                        const std::unordered_map<std::string, std::string> &encoding_config);
    ~NetworkWithEncoding() {}

    void forward_pass(GPUMatrix<float> &input, const int run_inference, DeviceMem<bf16> &network_output, DeviceMem<bf16> &intermediate_forward);

    DeviceMem<bf16> *backward_pass(DeviceMem<bf16> &input_backward, DeviceMem<bf16> &grad_output, DeviceMem<bf16> &intermediate_forward,
                                   const int batch_size);

    void initialize_params(int use_easy = 0);

    void free_memory();
    sycl::queue get_queue() const { return m_q; }

    void set_params(float *params);

    void set_params(std::vector<bf16> params);

    Network<bf16> const * const get_network() const { return network_; }

  private:
    int m_encoding_output_width;

    //   Encoding<float>* encoding_grid;
    Encoding<float> *encoding_;

    Network<bf16> *network_;
    sycl::queue m_q;
};

NetworkWithEncoding *create_networkwith_encoding(int input_width, int output_width, int n_hidden_layers,
                                                 Activation activation, Activation output_activation,
                                                 const int batch_size, std::string encoding_name,
                                                 const std::unordered_map<std::string, std::string> &encoding_config);
