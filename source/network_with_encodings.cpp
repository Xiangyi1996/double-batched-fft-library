#include "network_with_encodings.h"
#include "SwiftNetMLP.h"

constexpr int WIDTH = 64;
NetworkWithEncoding::NetworkWithEncoding(int input_width, int output_width, int n_hidden_layers, Activation activation,
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

void NetworkWithEncoding::forward_pass(GPUMatrix<float> &input, int run_inference, DeviceMem<bf16> &network_output,
                                       float *forward) {
    int batch_size = input.n();

    //   std::cout << "Input" << std::endl;
    //   input.print();
    // Declare all memory here. Bit ugly, but no other way
    int net_width = network->get_net_width();
    int input_width = network->get_inputs_width();
    int n_hidden_layers = network->get_n_hidden_layers();
    int output_width = network->get_output_width();
    int output_width_padded = network->get_padded_output_width();
    // Allocate and initialize various memory buffers

    std::cout << "FWD pass in network with ecndoings.cpp Batch size: " << batch_size << std::endl;

    DeviceMem<bf16> network_input = DeviceMem<bf16>(m_encoding_output_width * batch_size, m_q);
    GPUMatrix<float> encoding_output = GPUMatrix<float>(m_encoding_output_width, batch_size);

    encoding->forward_impl(&m_q, input, &encoding_output);
    // std::cout << "Output encoding: " << std::endl;
    // encoding_output.print();
    network_input.set_values(encoding_output.n_elements(), encoding_output.data(), m_q);
    if (run_inference) {
        network->inference(network_input, forward, batch_size, {});
    } else {
        network->forward_pass(network_input, forward, batch_size, {});
    }

    network->get_queue()
        .memcpy(network_output.data(), network->GetOutput(forward, batch_size), sizeof(bf16) * network_output.size())
        .wait();
}

DeviceMem<bf16> *NetworkWithEncoding::backward_pass(DeviceMem<bf16> &input_backward, DeviceMem<bf16> &grad_output,
                                                    float *forward, int batch_size) {
    // no encoding bwd, as their gradients are handled individually
    int input_width = network->get_inputs_width();
    int net_width = network->get_net_width();
    int n_hidden_matrices = network->get_n_hidden_matrices();
    int output_width = network->get_output_width();

    float *out_inter = malloc_device<float>(batch_size * net_width * (n_hidden_matrices), m_q);

    /// Note CB: do we need grad_output or input_backward here as first parameter?
    network->backward_pass(grad_output, out_inter, forward, batch_size, {});

    return (network->get_grads_matrices());
}

void NetworkWithEncoding::set_params(float *params) { // for xpu
    network->set_params(params);
}

void NetworkWithEncoding::set_params(std::vector<bf16> params) { // for cpu
    network->set_params(params);
}

void NetworkWithEncoding::initialize_params(int use_easy) {
    network->initialize_params(use_easy);
    encoding->initialize_params(); // this is an empty call
}

void NetworkWithEncoding::free_memory() { network->free_mem(m_q); }

NetworkWithEncoding *create_networkwith_encoding(int input_width, int output_width, int n_hidden_layers,
                                                 Activation activation, Activation output_activation,
                                                 const int batch_size, std::string encoding_name,
                                                 const std::unordered_map<std::string, std::string> &encoding_config) {
    return new NetworkWithEncoding(input_width, output_width, n_hidden_layers, activation, output_activation,
                                   encoding_name, encoding_config);
}