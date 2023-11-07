// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// #include "doctest/doctest.h"
#include <iostream>
#include <vector>

#include "activation.h"
#include "network_with_encodings.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 3
#define OUTPUT_WIDTH 20
#define HIDDEN_LAYERS 4
#define NET_WIDTH 64
void test_network_with_encoding() {
    // SWIFTNET
    const int batch_size = 128;
    const int output_width = OUTPUT_WIDTH;
    const int m_n_hidden_layers = HIDDEN_LAYERS;

    GPUMatrix<float> input(INPUT_WIDTH, batch_size);
    //   GPUMatrix<float> input(batch_size, INPUT_WIDTH);
    input.initialize_constant(0.1f);

    //   Define the parameters for creating IdentityEncoding
    std::unordered_map<std::string, std::string> encoding = {
        {"n_dims_to_encode", std::to_string(INPUT_WIDTH)}, {"scale", "1.0"}, {"offset", "0.0"}};
    std::string encoding_name = "Identity";

    //   std::unordered_map<std::string, std::string> encoding = {
    //       {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
    //       {"degree", std::to_string(4)}};
    //   std::string encoding_name = "SphericalHarmonics";
    NetworkWithEncoding network = NetworkWithEncoding(INPUT_WIDTH, OUTPUT_WIDTH, m_n_hidden_layers, Activation::ReLU,
                                                      Activation::None, encoding_name, encoding);
    network.initialize_params(1);

    float *forward = malloc_device<float>(batch_size * (INPUT_WIDTH + OUTPUT_WIDTH + NET_WIDTH * m_n_hidden_layers),
                                          network.get_queue());
    sycl::queue q;
    DeviceMem<float> network_output = DeviceMem<float>(OUTPUT_WIDTH * batch_size, q);

    network.forward_pass(input, 0, network_output, forward);

    std::vector<float> out(batch_size * (OUTPUT_WIDTH));
    network_output.copy_to_host(out, network.get_queue());

    for (int j = 0; j < batch_size * OUTPUT_WIDTH; j++) {
        std::cout << j << ": " << out[j] << std::endl;
    }

    network.free_memory();
}

int main() {
    test_network_with_encoding();
    return 0;
}