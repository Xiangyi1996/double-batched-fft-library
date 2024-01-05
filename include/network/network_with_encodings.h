#pragma once
#include "DeviceMem.h"
#include "Network.h"
#include "encoding_factory.h"

constexpr int WIDTH = 64;
using bf16 = sycl::ext::oneapi::bfloat16;

///@brief wrapper around a network+encoding such that in each forward pass/inference first the
/// encoding is applied, then the network forward pass/inference
/// TODO: derive this from network class. Network class then needs to allow
/// arbitrary input and output widths. Automatic padding then only happens in the
/// swiftnet class.
class NetworkWithEncoding {
  public:
    NetworkWithEncoding(sycl::queue &q, const int network_input_width, const int network_output_width,
                        const int n_hidden_layers, Activation activation, Activation output_activation,
                        const std::string &encoding_name,
                        const std::unordered_map<std::string, std::string> &encoding_config)
        : m_q(q), network_(new SwiftNetMLP<bf16, WIDTH>(m_q, network_input_width, network_output_width, n_hidden_layers,
                                                        activation, output_activation)),
          encoding_(create_encoding<float>(encoding_name, encoding_config)) {

        encoding_->set_padded_output_width(network_->get_inputs_width());
    }

    ~NetworkWithEncoding() {
        delete encoding_;
        delete network_;
    }

    std::vector<sycl::event> inference(GPUMatrix<float> &input, DeviceMem<bf16> &network_output,
                                       const std::vector<sycl::event> &deps) {
        const int batch_size = input.n();

        const int network_input_width = network_->get_inputs_width();
        const int n_hidden_layers = network_->get_n_hidden_layers();

        DeviceMem<bf16> network_input(network_input_width * batch_size, m_q);
        GPUMatrix<float> encoding_output(network_input_width, batch_size);

        encoding_->forward_impl(&m_q, input, &encoding_output);
        m_q.wait();

        DeviceMem<bf16>::copy_from_device(network_input, encoding_output.data(), m_q).wait();
        return network_->inference(network_input, network_output, batch_size, deps);
    }

    std::vector<sycl::event> forward_pass(GPUMatrix<float> &input, DeviceMem<bf16> &network_output,
                                          DeviceMem<bf16> &intermediate_forward, const std::vector<sycl::event> &deps) {
        const int batch_size = input.n();
        const int network_input_width = network_->get_inputs_width();
        const int n_hidden_layers = network_->get_n_hidden_layers();

        DeviceMem<bf16> network_input(network_input_width * batch_size, m_q);
        GPUMatrix<float> encoding_output(network_input_width, batch_size);

        encoding_->forward_impl(&m_q, input, &encoding_output);
        m_q.wait();

        DeviceMem<bf16>::copy_from_device(network_input, encoding_output.data(), m_q).wait();
        std::vector<sycl::event> new_deps =
            network_->forward_pass(network_input, intermediate_forward, batch_size, deps);
        m_q.wait();

        const size_t output_offset = network_input_width * batch_size + WIDTH * batch_size * n_hidden_layers;
        m_q.memcpy(network_output.data(), intermediate_forward.data() + output_offset,
                   sizeof(bf16) * network_output.size())
            .wait();

        return new_deps;
    }

    std::vector<sycl::event> backward_pass(const DeviceMem<bf16> &input_backward, DeviceMem<bf16> &output,
                                           DeviceMem<bf16> &intermediate_backward,
                                           const DeviceMem<bf16> &intermediate_forward, const int batch_size,
                                           const std::vector<sycl::event> &deps) {
        return network_->backward_pass(input_backward, output, intermediate_backward, intermediate_forward, batch_size,
                                       deps);
    }

    // void initialize_params(const int use_easy = 0) {
    //     network_->initialize_weights_matrices(use_easy);
    //     encoding_->initialize_params(); // this is an empty call
    // }

    sycl::queue &get_queue() { return m_q; }

    void set_weights_matrices(const std::vector<bf16> &weights) { network_->set_weights_matrices(weights); }

    Network<bf16> const *const get_network() const { return network_; }

  private:
    sycl::queue &m_q;
    Network<bf16> *network_;
    Encoding<float> *encoding_;
};

NetworkWithEncoding *create_networkwith_encoding(sycl::queue &q, int input_width, int output_width, int n_hidden_layers,
                                                 Activation activation, Activation output_activation,
                                                 const int batch_size, std::string encoding_name,
                                                 const std::unordered_map<std::string, std::string> &encoding_config) {
    return new NetworkWithEncoding(q, input_width, output_width, n_hidden_layers, activation, output_activation,
                                   encoding_name, encoding_config);
}
