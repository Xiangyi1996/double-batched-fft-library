#pragma once
#include "DeviceMem.h"
#include "Network.h"
#include "encoding_factory.h"

constexpr int WIDTH = 64;
using bf16 = sycl::ext::oneapi::bfloat16;

class NetworkWithEncoding {
  public:
    NetworkWithEncoding(const int input_width, const int output_width, const int n_hidden_layers, Activation activation,
                        Activation output_activation, const std::string &encoding_name,
                        const std::unordered_map<std::string, std::string> &encoding_config)
        : m_q() {

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
        network_ = new SwiftNetMLP<bf16, WIDTH>(m_q, m_encoding_output_width, output_width, n_hidden_layers, activation,
                                                output_activation);

        // encoding_output = GPUMatrix<float>(network_input.data(),
        //                                   m_encoding_output_width, batch_size);
    }
    ~NetworkWithEncoding() {
        delete network_;
        delete encoding_;
    }

    void forward_pass(GPUMatrix<float> &input, const int run_inference, DeviceMem<bf16> &network_output,
                      DeviceMem<bf16> &intermediate_forward) {
        int batch_size = input.n();

        //   std::cout << "Input" << std::endl;
        //   input.print();
        // Declare all memory here. Bit ugly, but no other way
        const int input_width = network->get_inputs_width();
        const int n_hidden_layers = network->get_n_hidden_layers();

        std::cout << "FWD pass in network with ecndoings.cpp Batch size: " << batch_size << std::endl;

        DeviceMem<bf16> network_input(m_encoding_output_width * batch_size, m_q);
        GPUMatrix<float> encoding_output(m_encoding_output_width, batch_size);

        encoding->forward_impl(&m_q, input, &encoding_output);
        // std::cout << "Output encoding: " << std::endl;
        // encoding_output.print();
        DeviceMem<bf16>::copy_from_device(network_input, encoding_output.data(), m_q).wait();
        if (run_inference) {
            network->inference(network_input, intermediate_forward, batch_size, {});
            m_q.wait();
            m_q.memcpy(network_output.data(), intermediate_forward.data(), sizeof(bf16) * network_output.size()).wait();
        } else {
            network->forward_pass(network_input, intermediate_forward, batch_size, {});
            m_q.wait();
            const size_t output_offset = input_width * batch_size + WIDTH * batch_size * n_hidden_layers;
            m_q.memcpy(network_output.data(), intermediate_forward.data() + output_offset,
                       sizeof(bf16) * network_output.size())
                .wait();
        }
    }

    std::vector<sycl::event> backward_pass(const DeviceMem<bf16> &input_backward, DeviceMem<bf16> &output,
                                           DeviceMem<bf16> &intermediate_backward,
                                           const DeviceMem<bf16> &intermediate_forward, const int batch_size,
                                           const std::vector<sycl::event> &deps) {
        return network->backward_pass(input_backward, output, intermediate_backward, intermediate_forward, batch_size,
                                      deps);
    }

    void initialize_params(const int use_easy = 0) {
        network->initialize_weights_matrices(use_easy);
        encoding->initialize_params(); // this is an empty call
    }

    sycl::queue &get_queue() { return m_q; }

    void set_weights_matrices(const std::vector<bf16> &weights) { network->set_weights_matrices(weights); }

    Network<bf16> const *const get_network() const { return network_; }

  private:
    sycl::queue &m_q;
    int m_encoding_output_width;

    //   Encoding<float>* encoding_grid;
    Encoding<float> *encoding_;

    Network<bf16> *network_;
};

NetworkWithEncoding *create_networkwith_encoding(int input_width, int output_width, int n_hidden_layers,
                                                 Activation activation, Activation output_activation,
                                                 const int batch_size, std::string encoding_name,
                                                 const std::unordered_map<std::string, std::string> &encoding_config) {
    return new NetworkWithEncoding(input_width, output_width, n_hidden_layers, activation, output_activation,
                                   encoding_name, encoding_config);
}
