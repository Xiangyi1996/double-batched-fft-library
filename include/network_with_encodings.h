#pragma once
#include "DeviceMem.h"
#include "SwiftNetMLP.h"
#include "encoding_factory.h"
#define WIDTH 64

class NetworkWithEncoding {
  public:
    NetworkWithEncoding(int input_width, int output_width, int n_hidden_layers, Activation activation,
                        Activation output_activation, std::string encoding_name,
                        const std::unordered_map<std::string, std::string> &encoding_config) {
        m_q = sycl::queue();

        // if (encoding_name.find("Grid") !=
        //     std::string::npos) {  // ugly if else (change to Factory later)
        //     because
        //                           // we can only run grids with float
        //   encoding_grid = create_encoding<float>(encoding_name, encoding_config);
        // } else {
        //   encoding = create_encoding<bf16>(encoding_name, encoding_config);
        encoding = create_encoding<float>(encoding_name, encoding_config);
        // }

        if (input_width > WIDTH) {
            std::cout << "Input width (" << input_width << ") is larger than WIDTH: " << WIDTH
                      << ". This leads to slower runs as oneMKL gemms are used" << std::endl;
            m_encoding_output_width = input_width;
        } else {
            m_encoding_output_width = WIDTH;
        }

        encoding->set_padded_output_width(m_encoding_output_width);
        // assert(encoding->padded_output_width() == WIDTH);
        network = new SwiftNetMLP<WIDTH>(m_q, m_encoding_output_width, output_width, n_hidden_layers, activation,
                                         output_activation);

        // encoding_output = GPUMatrix<float>(network_input.data(),
        //                                   m_encoding_output_width, batch_size);
    }

    ~NetworkWithEncoding() {}

    void forward_pass(GPUMatrix<float> &input, int run_inference, DeviceMem<float> &network_output, float *forward);

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
